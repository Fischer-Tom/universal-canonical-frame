from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler

from lib.logging import AdvancedLogger
from lib.optim import construct_optimizer, freeze_lora
from tools.annotate import Annotator
from tools.ddp_tools import (
    ddp_all_skip,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_tensor,
)
from tools.vis import save_correspondence_grid

_LOSS_KEYS = (
    "loss_total",
    "loss_mesh",
    "loss_mask",
    "loss_aug_consistency",
    "loss_entropy",
    "entropy_ratio",
    "alignment_cos",
    "alignment_angle",
    "ransac_inlier_ratio",
    "snap_ang_deg",
    "aug_consistency_weight",
    "aug_consistency_pixels",
)


@dataclass
class TrainerState:
    iteration: int = 0
    epoch: int = 0
    lora_frozen: bool = False


def save_checkpoint(
    path, model, optimizer, lr_scheduler, scaler, state: TrainerState, logger, cfg
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "iteration": state.iteration,
            "epoch": state.epoch,
            "criterion_c_iter": model.criterion.c_iter,
            "logger_state": logger.get_checkpoint_state(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if (
        "cuda" in rng_state
        and rng_state["cuda"] is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def _rewrap_ddp(model_without_ddp, device: torch.device, *, compile: bool = False):
    m = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp,
        device_ids=[device.index],
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    if compile:
        m = torch.compile(m, dynamic=False)
    return m


def train(
    cfg: DictConfig,
    model,
    model_without_ddp,
    annotator: Annotator,
    fetcher,
    dataset,
    device: torch.device,
    logger: AdvancedLogger,
    total_iters: int,
    resume_checkpoint: Optional[Dict[str, Any]] = None,
    resume_model_only: bool = False,
    resume_from_iteration: Optional[int] = None,
) -> None:
    optimizer, lr_scheduler = construct_optimizer(
        model_without_ddp, total_iters, cfg.optimizer
    )
    use_mixed = bool(cfg.training.get("use_mixed_precision", True))
    scaler = GradScaler("cuda", enabled=use_mixed)
    state = TrainerState()
    distributed = is_dist_avail_and_initialized()
    use_compile = bool(cfg.training.get("use_torch_compile", True))
    batches_per_epoch = max(1, len(fetcher))
    resume_skip_batches = 0

    if resume_checkpoint is not None:
        model_without_ddp.load_state_dict(resume_checkpoint["model_state_dict"])
        logger.info("Model weights restored")
        if not resume_model_only:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(resume_checkpoint["lr_scheduler_state_dict"])
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            state.iteration = resume_checkpoint["iteration"]
            state.epoch = resume_checkpoint["epoch"]
            if resume_from_iteration is not None:
                state.iteration = resume_from_iteration
            state.epoch = min(
                int(state.iteration // batches_per_epoch), int(cfg.training.epochs)
            )
            resume_skip_batches = int(state.iteration % batches_per_epoch)
            if "criterion_c_iter" in resume_checkpoint:
                model_without_ddp.criterion.c_iter = resume_checkpoint[
                    "criterion_c_iter"
                ]
            if "logger_state" in resume_checkpoint:
                logger.restore_from_checkpoint(resume_checkpoint["logger_state"])
            if "rng_state" in resume_checkpoint:
                restore_rng_state(resume_checkpoint["rng_state"])
            logger.info(
                f"Resumed full state @ iter {state.iteration}, epoch {state.epoch}"
            )

    freeze_at = cfg.optimizer.get("freeze_lora_after", None)
    if freeze_at and state.iteration >= freeze_at:
        freeze_lora(model_without_ddp)
        state.lora_frozen = True
        if distributed:
            model = _rewrap_ddp(model_without_ddp, device, compile=use_compile)
        logger.info(f"LoRA frozen on resume (iter {state.iteration} >= {freeze_at})")

    checkpoint_interval = cfg.training.get("checkpoint_interval", 5000)
    keep_interval_checkpoints = bool(
        cfg.training.get("keep_interval_checkpoints", False)
    )
    log_interval = int(cfg.logging.get("log_interval", 100))
    vis_interval = int(cfg.logging.get("vis_interval", log_interval * 1))
    clip_grad_norm = float(cfg.optimizer.get("clip_grad_norm", 0.0))
    logger.info("Starting training loop")
    model.train()

    resume_epoch = state.epoch
    for epoch in range(state.epoch, cfg.training.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(fetcher, "set_epoch"):
            fetcher.set_epoch(epoch)

        fetcher_iter = iter(fetcher)
        for batch_idx in range(len(fetcher)):
            if hasattr(fetcher, "set_iteration"):
                fetcher.set_iteration(state.iteration)
            try:
                batch = next(fetcher_iter)
            except StopIteration:
                break

            if epoch == resume_epoch and batch_idx < resume_skip_batches:
                continue

            with torch.no_grad():
                batch.annotations = annotator(batch, deform=False)

            skip_local = False
            forward_completed = False
            loss_dict = None
            try:
                with torch.autocast(
                    device_type=device.type, enabled=use_mixed, dtype=torch.float16
                ):
                    loss_dict = model(batch, batch.annotations)
                forward_completed = True
                loss = loss_dict["loss_total"]
                if not torch.isfinite(loss):
                    skip_local = True
            except Exception as e:
                if is_main_process():
                    logger.info(f"Forward failed @ iter {state.iteration}: {e}")
                skip_local = True

            skip_iter = ddp_all_skip(skip_local, device) if distributed else skip_local
            if skip_iter:
                if not forward_completed:
                    model_without_ddp.criterion.c_iter += 1
                state.iteration += 1
                if hasattr(fetcher, "set_iteration"):
                    fetcher.set_iteration(state.iteration)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            if freeze_at and not state.lora_frozen and state.iteration >= freeze_at:
                freeze_lora(model_without_ddp)
                state.lora_frozen = True
                if distributed:
                    model = _rewrap_ddp(model_without_ddp, device, compile=use_compile)
                logger.info(f"LoRA frozen @ iter {state.iteration}")

            logs = {}
            for k in _LOSS_KEYS:
                if k not in loss_dict:
                    continue
                v = loss_dict[k].detach()
                if distributed:
                    v = reduce_tensor(v)
                logs[k] = v.item()
            logger.step(state.iteration, logs)

            state.iteration += 1
            if hasattr(fetcher, "set_iteration"):
                fetcher.set_iteration(state.iteration)

            if (
                vis_interval > 0
                and state.iteration % vis_interval == 0
                and is_main_process()
            ):
                try:
                    vis_path = str(logger.vis_dir / f"iter_{state.iteration:07d}.png")
                    save_correspondence_grid(
                        model_without_ddp._last_img,
                        model_without_ddp._last_geom_mask,
                        model_without_ddp._last_logits,
                        model_without_ddp.V,
                        model_without_ddp._last_feat_hw,
                        vis_path,
                    )
                except Exception as e:
                    logger.info(f"Vis save failed @ iter {state.iteration}: {e}")

            if state.iteration % checkpoint_interval == 0 and is_main_process():
                latest_path = logger.checkpoints_dir / "latest.pth"
                save_checkpoint(
                    str(latest_path),
                    model_without_ddp,
                    optimizer,
                    lr_scheduler,
                    scaler,
                    state,
                    logger,
                    cfg,
                )
                logger.info(f"Saved {latest_path}")
                if keep_interval_checkpoints:
                    iter_path = (
                        logger.checkpoints_dir / f"iter_{state.iteration:07d}.pth"
                    )
                    save_checkpoint(
                        str(iter_path),
                        model_without_ddp,
                        optimizer,
                        lr_scheduler,
                        scaler,
                        state,
                        logger,
                        cfg,
                    )
                    logger.info(f"Saved {iter_path}")

        state.epoch = epoch + 1
        resume_skip_batches = 0

    if is_main_process():
        path = logger.checkpoints_dir / "final.pth"
        save_checkpoint(
            str(path),
            model_without_ddp,
            optimizer,
            lr_scheduler,
            scaler,
            state,
            logger,
            cfg,
        )
        logger.info(f"Saved final {path}")
