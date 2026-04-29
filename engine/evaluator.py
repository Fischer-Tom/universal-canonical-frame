from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import einsum
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from dataset.data_loader import build_loader
from tools.alignment import estimate_T_n2c_for_frame, rot_err_deg, rot_err_deg_sym


def valid_rotations_cube(device, dtype, gravity: bool = False):
    Rs = []
    seen = set()
    eye = torch.eye(3, device=device, dtype=dtype)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        P = eye[:, perm]
        for signs in (
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (-1, -1, -1),
        ):
            S = torch.diag(torch.tensor(signs, device=device, dtype=dtype))
            R = P @ S
            if int(round(torch.det(R).item())) == 1:
                key = tuple((R.cpu().numpy() * 100).round().astype(int).flatten())
                if key not in seen:
                    seen.add(key)
                    Rs.append(R)
    Rset = torch.stack(Rs, dim=0)
    assert Rset.shape[0] == 24
    if gravity:
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        mask = torch.tensor(
            [bool(torch.allclose(R[:, 1], y_axis, atol=1e-6)) for R in Rset],
            device=device,
        )
        Rset = Rset[mask]
    return Rset


def conjugate_rotation_average(gt_Rs, pred_Rs, n_faces: int = 1, gravity: bool = False):
    best_mean_accuracy = float("-inf")
    R_align = torch.eye(3, device=gt_Rs.device, dtype=gt_Rs.dtype)
    Rset = valid_rotations_cube(gt_Rs.device, gt_Rs.dtype, gravity=gravity)
    rot_error_fn = (
        (lambda p, g: rot_err_deg_sym(p, g, n_faces)) if n_faces != 1 else rot_err_deg
    )
    invalid = torch.any(pred_Rs.isnan().flatten(1), dim=1)
    pred_Rs = pred_Rs[~invalid]
    gt_Rs = gt_Rs[~invalid]
    for R in Rset:
        rotated_pred_Rs = einsum(R, pred_Rs, "i j, b j k -> b i k")
        ang_errors = [
            rot_error_fn(rT_gt, rT_est) for rT_gt, rT_est in zip(gt_Rs, rotated_pred_Rs)
        ]
        mean_accuracy = torch.stack(ang_errors).lt(30).float().mean().item()
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            R_align = R
    return R_align


def _procrustes(A, B, weights=None):
    N = A.shape[0]
    if weights is None:
        M = einsum(B, A, "n i j, n k j -> i k")
    else:
        w = weights.reshape(N, 1, 1)
        M = einsum(w * B, A, "n i j, n k j -> i k")

    U, _S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vh
    return R


def align_rotation_sets(
    A: torch.Tensor,
    B: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_faces: int = 1,
    gravity: bool = False,
):
    assert A.shape == B.shape and A.shape[-2:] == (3, 3)
    if n_faces == 1:
        return _procrustes(A, B, weights)

    Rset = valid_rotations_cube(A.device, A.dtype, gravity=gravity)
    rot_error_fn = (
        (lambda p, g: rot_err_deg_sym(p, g, n_faces)) if n_faces != 1 else rot_err_deg
    )
    invalid = torch.any(A.isnan().flatten(1), dim=1)
    A_valid = A[~invalid]
    B_valid = B[~invalid]
    best_R = torch.eye(3, device=A.device, dtype=A.dtype)
    best_acc = -1.0

    for S in Rset:
        A_sym = einsum(S, A_valid, "i j, n j k -> n i k")
        R_align = _procrustes(A_sym, B_valid, weights)
        aligned = einsum(R_align, A_sym, "i j, n j k -> n i k")
        errors = torch.stack([rot_error_fn(b, a) for a, b in zip(B_valid, aligned)])
        acc = errors.lt(30).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_R = R_align @ S
    return best_R


def load_symmetries_csv(path: str | None) -> dict[str, int]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out: dict[str, int] = {}
    for row in rows:
        category = row.get("category") or row.get("name") or row.get("class")
        value = (
            row.get("n_faces")
            or row.get("symmetry")
            or row.get("symmetries")
            or row.get("n_symmetries")
        )
        if category is None or value in (None, ""):
            continue
        out[str(category)] = int(value)
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _dataset_cfg_for_name(cfg: DictConfig, name: str) -> DictConfig:
    if name in {"dataset", "default", "configured"}:
        return cfg.dataset
    return OmegaConf.load(f"configs/dataset/{name}.yaml")


def _category_names(dataset, dataset_cfg: DictConfig) -> list[str]:
    cfg_categories = _as_list(
        dataset_cfg.get("categories", None) or dataset_cfg.get("cate", None)
    )
    if cfg_categories:
        return [str(c) for c in cfg_categories]
    ds_categories = _as_list(getattr(dataset, "categories", None))
    if ds_categories:
        return [str(c) for c in ds_categories]
    return ["__all__"]


def _instantiate_dataset(dataset_cfg: DictConfig, split: str, category: str | None = None):
    if category is None or category == "__all__":
        return instantiate(dataset_cfg, split=split)
    field = "cate" if "cate" in dataset_cfg else "categories"
    cat_cfg = OmegaConf.merge(dataset_cfg, {field: [category]})
    return instantiate(cat_cfg, split=split)


def _evaluate_category(
    *,
    cfg: DictConfig,
    model,
    annotator,
    dataset,
    category: str,
    n_faces: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor, torch.Tensor]:
    if hasattr(dataset, "setup_samples"):
        dataset.setup_samples([category])

    loader = build_loader(
        dataset,
        cfg.training,
        device,
        training=False,
        distributed=False,
    )
    all_pred_Rs, all_gt_Rs = [], []
    timings: list[float] = []

    for batch in loader:
        with torch.no_grad():
            batch.annotations = annotator(batch, deform=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            inf_dict = model.infer(
                batch,
                batch.annotations,
                use_gt_mask=bool(cfg.evaluation.get("use_gt_mask", False)),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

        cams = batch.cameras
        timings.extend([(t1 - t0) / len(cams)] * len(cams))
        feat_hw = inf_dict.get("feat_hw", tuple(cfg.model.output_size))
        Tn2c_list = []
        for frame_idx in range(len(cams)):
            Tn2c, _size = estimate_T_n2c_for_frame(
                inf_dict, cams, frame_idx, feat_hw
            )
            Tn2c_list.append(Tn2c)

        pred_Rs = cams.R.clone()
        for i, Tn2c in enumerate(Tn2c_list):
            R_abs_col = Tn2c[:3, :3]
            pred_Rs[i] = R_abs_col.transpose(-1, -2)
        all_pred_Rs.append(pred_Rs.detach())
        all_gt_Rs.append(cams.R.detach().clone())

    if not all_pred_Rs:
        empty = torch.empty(0)
        metrics = {
            "median_ang_error": float("nan"),
            "median_ang_sym_error": float("nan"),
            "acc_ang_15": float("nan"),
            "acc_ang_sym_15": float("nan"),
            "acc_ang_30": float("nan"),
            "acc_ang_sym_30": float("nan"),
            "n_samples": 0,
            "avg_time_ms": 0.0,
        }
        return metrics, np.eye(3, dtype=np.float32), empty, empty

    all_gt_Rs_tensor = torch.cat(all_gt_Rs)
    all_pred_Rs_tensor = torch.cat(all_pred_Rs)
    gravity = bool(cfg.representation.get("gravity", False))
    if cfg.representation.mesh_type == "sphere":
        R_align = align_rotation_sets(
            all_pred_Rs_tensor,
            all_gt_Rs_tensor,
            n_faces=n_faces,
            gravity=gravity,
        )
    else:
        R_align = conjugate_rotation_average(
            all_gt_Rs_tensor, all_pred_Rs_tensor, n_faces, gravity=gravity
        )

    aligned_pred_Rs = [R_align @ R for R in all_pred_Rs_tensor]
    ang_error_vals = [
        rot_err_deg(rT_gt, rT_est)
        for rT_gt, rT_est in zip(all_gt_Rs_tensor, aligned_pred_Rs)
    ]
    if n_faces != 1:
        ang_sym_error_vals = [
            rot_err_deg_sym(rT_gt, rT_est, n_faces)
            for rT_gt, rT_est in zip(all_gt_Rs_tensor, aligned_pred_Rs)
        ]
    else:
        ang_sym_error_vals = ang_error_vals

    ang_error_tensor = torch.stack(ang_error_vals)
    ang_sym_error_tensor = torch.stack(ang_sym_error_vals)
    ang_valid = ang_error_tensor[~torch.isnan(ang_error_tensor)]
    ang_sym_valid = ang_sym_error_tensor[~torch.isnan(ang_sym_error_tensor)]
    avg_time_ms = 1000.0 * sum(timings) / len(timings) if timings else 0.0
    metrics = {
        "median_ang_error": ang_valid.median().item() if len(ang_valid) else float("nan"),
        "median_ang_sym_error": ang_sym_valid.median().item()
        if len(ang_sym_valid)
        else float("nan"),
        "acc_ang_15": ang_valid.lt(15.0).float().mean().item()
        if len(ang_valid)
        else float("nan"),
        "acc_ang_sym_15": ang_sym_valid.lt(15.0).float().mean().item()
        if len(ang_sym_valid)
        else float("nan"),
        "acc_ang_30": ang_valid.lt(30.0).float().mean().item()
        if len(ang_valid)
        else float("nan"),
        "acc_ang_sym_30": ang_sym_valid.lt(30.0).float().mean().item()
        if len(ang_sym_valid)
        else float("nan"),
        "n_samples": len(ang_error_vals),
        "avg_time_ms": avg_time_ms,
    }
    return metrics, R_align.cpu().numpy(), ang_valid.cpu(), ang_sym_valid.cpu()


def evaluate(cfg: DictConfig, model, annotator, device: torch.device, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    symmetries = load_symmetries_csv(cfg.evaluation.get("symmetries_csv", None))
    dataset_names = _as_list(cfg.evaluation.get("datasets", None)) or ["dataset"]
    split = str(cfg.evaluation.get("split", "val"))
    summary_rows: dict[str, dict[str, float]] = {}

    for ds_name in dataset_names:
        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {ds_name}")
        print(f"{'=' * 60}")

        dataset_cfg = _dataset_cfg_for_name(cfg, str(ds_name))
        dataset_split = (
            split
            if str(ds_name) in {"dataset", "default", "configured"}
            else str(dataset_cfg.get("split", split))
        )
        probe_dataset = None
        categories = _as_list(
            dataset_cfg.get("categories", None) or dataset_cfg.get("cate", None)
        )
        if not categories:
            probe_dataset = instantiate(dataset_cfg, split=dataset_split)
            categories = _category_names(probe_dataset, dataset_cfg)
        categories = [str(c) for c in categories] or ["__all__"]

        error_per_category = {}
        align_per_category = {}
        all_ang_error_tensor = None
        all_ang_error_sym_tensor = None

        for category in categories:
            dataset = (
                probe_dataset
                if probe_dataset is not None
                and (hasattr(probe_dataset, "setup_samples") or category == "__all__")
                else _instantiate_dataset(dataset_cfg, dataset_split, category)
            )
            n_faces = int(symmetries.get(category, 1))
            print(f"  Category: {category}, #sequences: {len(dataset)}")
            metrics, R_align, ang_valid, ang_sym_valid = _evaluate_category(
                cfg=cfg,
                model=model,
                annotator=annotator,
                dataset=dataset,
                category=category,
                n_faces=n_faces,
                device=device,
            )
            error_per_category[category] = metrics
            align_per_category[category] = R_align
            all_ang_error_tensor = (
                torch.cat((all_ang_error_tensor, ang_valid), dim=0)
                if all_ang_error_tensor is not None
                else ang_valid
            )
            all_ang_error_sym_tensor = (
                torch.cat((all_ang_error_sym_tensor, ang_sym_valid), dim=0)
                if all_ang_error_sym_tensor is not None
                else ang_sym_valid
            )
            print(
                f"    Median: {metrics['median_ang_error']:.2f} deg "
                f"(sym: {metrics['median_ang_sym_error']:.2f} deg)  "
                f"Acc@15: {metrics['acc_ang_15']:.3f}/{metrics['acc_ang_sym_15']:.3f}  "
                f"Acc@30: {metrics['acc_ang_30']:.3f}/{metrics['acc_ang_sym_30']:.3f}  "
                f"Avg time: {metrics['avg_time_ms']:.1f}ms/image"
            )

        align_path = run_dir / f"{ds_name}_alignments.npz"
        np.savez(align_path, **align_per_category)
        print(f"  Saved alignment rotations to {align_path}")

        if all_ang_error_tensor is None or len(all_ang_error_tensor) == 0:
            global_row = {
                "median_ang_error": float("nan"),
                "median_ang_sym_error": float("nan"),
                "acc_ang_15": float("nan"),
                "acc_ang_sym_15": float("nan"),
                "acc_ang_30": float("nan"),
                "acc_ang_sym_30": float("nan"),
                "n_samples": 0,
            }
        else:
            global_row = {
                "median_ang_error": all_ang_error_tensor.median().item(),
                "median_ang_sym_error": all_ang_error_sym_tensor.median().item(),
                "acc_ang_15": all_ang_error_tensor.lt(15.0).float().mean().item(),
                "acc_ang_sym_15": all_ang_error_sym_tensor.lt(15.0).float().mean().item(),
                "acc_ang_30": all_ang_error_tensor.lt(30.0).float().mean().item(),
                "acc_ang_sym_30": all_ang_error_sym_tensor.lt(30.0).float().mean().item(),
                "n_samples": len(all_ang_error_tensor),
            }

        df_cats = pd.DataFrame.from_dict(error_per_category, orient="index")
        cat_avg_row = (
            df_cats.drop(columns=["n_samples"], errors="ignore").mean().to_dict()
        )
        cat_avg_row["n_samples"] = df_cats["n_samples"].sum()
        error_per_category["__global__"] = global_row
        error_per_category["__category_avg__"] = cat_avg_row

        df = pd.DataFrame.from_dict(error_per_category, orient="index")
        df.index.name = "category"
        csv_path = run_dir / f"{ds_name}.csv"
        df.to_csv(csv_path)
        summary_rows[str(ds_name)] = {
            "median_ang_error": cat_avg_row["median_ang_error"],
            "median_ang_sym_error": cat_avg_row["median_ang_sym_error"],
            "acc_ang_15": cat_avg_row["acc_ang_15"],
            "acc_ang_sym_15": cat_avg_row["acc_ang_sym_15"],
            "acc_ang_30": cat_avg_row["acc_ang_30"],
            "acc_ang_sym_30": cat_avg_row["acc_ang_sym_30"],
            "n_samples": cat_avg_row["n_samples"],
        }
        print(f"  Saved to {csv_path}")

    if summary_rows:
        summary_df = pd.DataFrame.from_dict(summary_rows, orient="index")
        summary_df.index.name = "dataset"
        summary_path = run_dir / "summary.csv"
        summary_df.to_csv(summary_path)
        print(f"\n{'=' * 60}")
        print(f"Summary saved to {summary_path}")
        print(summary_df.to_string())
