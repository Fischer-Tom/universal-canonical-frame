import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dataset.data_loader import build_loader
from engine.trainer import load_checkpoint, train
from lib.logging import AdvancedLogger
from models.criterion import healpix_cube_mesh, healpix_sphere_mesh
from models.model import Model
from tools.annotate import Annotator
from tools.ddp_tools import (
    get_rank,
    get_world_size,
    init_ddp,
    is_dist_avail_and_initialized,
)


def _seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_resume_path(cfg: DictConfig) -> str | None:
    manual = cfg.training.get("resume_from", None)
    if manual is None:
        return None

    path = Path(to_absolute_path(str(manual))).expanduser()
    if path.is_file():
        return str(path)
    if path.is_dir():
        checkpoints_subdir = cfg.logging.get("checkpoints_subdir", "checkpoints")
        candidates = [
            path / checkpoints_subdir / "latest.pth",
            path / checkpoints_subdir / "final.pth",
            path / "latest.pth",
            path / "checkpoint.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"No checkpoint found under resume directory: {path}")
    raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")


def _save_resolved_config(cfg: DictConfig, run_dir: Path) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(config_dir / "config_resolved.yaml"), resolve=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Fixed input shapes across the pipeline make cudnn.benchmark a free win,
    # and TF32 matmul is a no-brainer on Ampere+.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    device = init_ddp()
    distributed = is_dist_avail_and_initialized()
    world_size = get_world_size()
    rank = get_rank()
    _seed_everything(cfg.get("seed", 0), rank)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    resume_checkpoint = None
    resume_path = _resolve_resume_path(cfg)
    resume_model_only = cfg.training.get("resume_model_only", False)
    resume_from_iteration = cfg.training.get("resume_from_iteration", None)
    if resume_path is not None:
        resume_checkpoint = load_checkpoint(resume_path, torch.device("cpu"))

    dataset = instantiate(cfg.dataset, split="train")
    fetcher = build_loader(
        dataset,
        cfg.training,
        device,
        training=True,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
    )
    total_iters = cfg.training.epochs * len(fetcher)

    logger = AdvancedLogger(
        total_iters=total_iters,
        run_dir=str(run_dir),
        experiment_name=cfg.get("experiment_name", "universal_canonical_frame"),
        save_frequency=cfg.logging.get("log_interval", 100),
        log_subdir=cfg.logging.get("log_subdir", "logs"),
        vis_subdir=cfg.logging.get("vis_subdir", "vis"),
        checkpoints_subdir=cfg.logging.get("checkpoints_subdir", "checkpoints"),
    )
    logger.info(f"Device: {device}, world_size: {world_size}, rank: {rank}")
    logger.info(f"Total iterations: {total_iters}, dataset size: {len(dataset)}")
    if resume_path is not None:
        logger.info(f"Resume checkpoint: {resume_path}")

    if rank == 0:
        _save_resolved_config(cfg, run_dir)

    if cfg.representation.mesh_type == "sphere":
        V, F = healpix_sphere_mesh(int(cfg.representation.get("nside", 8)))
    else:
        V, F = healpix_cube_mesh(int(cfg.representation.get("subdivisions", 16)))
    logger.info(f"Mesh: {V.shape[0]} vertices")

    V_t = torch.from_numpy(V).to(device)
    F_t = torch.from_numpy(F).to(device)
    annotator = Annotator(V_t, F_t, list(cfg.model.output_size), device=device)

    model = Model(cfg, V_t, F_t, total_iters).to(device)
    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        model_without_ddp = model.module
    if cfg.training.get("use_torch_compile", True):
        model = torch.compile(model, dynamic=False)

    train(
        cfg=cfg,
        model=model,
        model_without_ddp=model_without_ddp,
        annotator=annotator,
        fetcher=fetcher,
        dataset=dataset,
        device=device,
        logger=logger,
        total_iters=total_iters,
        resume_checkpoint=resume_checkpoint,
        resume_model_only=resume_model_only,
        resume_from_iteration=resume_from_iteration,
    )


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("UCO3D_DATASET_ROOT", "/uco3d/uco3d")
    main()
