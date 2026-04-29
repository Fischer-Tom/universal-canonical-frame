from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from cv2 import setRNGSeed
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from engine.evaluator import evaluate
from engine.trainer import load_checkpoint
from models.criterion import healpix_cube_mesh, healpix_sphere_mesh
from models.model import Model
from tools.annotate import Annotator


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    setRNGSeed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(cfg: DictConfig) -> str:
    checkpoint = cfg.evaluation.get("checkpoint", None)
    if checkpoint is None and "checkpoint" in cfg:
        checkpoint = cfg.checkpoint
    if checkpoint is None:
        raise ValueError("Set evaluation.checkpoint to the checkpoint path")
    path = Path(to_absolute_path(str(checkpoint))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)


def _cfg_from_checkpoint(base_cfg: DictConfig, checkpoint: dict) -> DictConfig:
    eval_cfg = OmegaConf.create(base_cfg.get("evaluation", {}))
    checkpoint_cfg = checkpoint.get("config", None)
    if checkpoint_cfg is None:
        cfg = OmegaConf.create(base_cfg)
    else:
        cfg = OmegaConf.create(checkpoint_cfg)
    cfg = OmegaConf.merge(cfg, {"evaluation": eval_cfg})
    return cfg


def _build_mesh(cfg: DictConfig):
    if cfg.representation.mesh_type == "sphere":
        return healpix_sphere_mesh(int(cfg.representation.get("nside", 8)))
    return healpix_cube_mesh(int(cfg.representation.get("subdivisions", 16)))


def _save_eval_config(cfg: DictConfig, run_dir: Path) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(config_dir / "eval_config_resolved.yaml"), resolve=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(base_cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_path = _checkpoint_path(base_cfg)
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    cfg = _cfg_from_checkpoint(base_cfg, checkpoint)
    cfg.evaluation.checkpoint = checkpoint_path

    seed = int(cfg.evaluation.get("seed", cfg.get("seed", 0)))
    _seed_everything(seed)
    requested_device = str(cfg.evaluation.get("device", cfg.training.get("device", "cuda")))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    V, F = _build_mesh(cfg)
    V_t = torch.from_numpy(V).to(device)
    F_t = torch.from_numpy(F).to(device)
    annotator = Annotator(V_t, F_t, list(cfg.model.output_size), device=device)

    model = Model(cfg, V_t, F_t, total_iters=0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    chkpt_stem = Path(checkpoint_path).stem
    output_root = Path(to_absolute_path(str(cfg.evaluation.get("output_root", "eval_logs"))))
    run_dir = output_root / f"{chkpt_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_eval_config(cfg, run_dir)
    print(f"Eval run directory: {run_dir}")

    evaluate(cfg, model, annotator, device, run_dir)


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("UCO3D_DATASET_ROOT", "/uco3d/uco3d")
    main()
