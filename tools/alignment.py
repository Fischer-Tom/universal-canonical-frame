from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import torch


def se3_from_Rt(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        t = t.squeeze(-1)
    T = torch.eye(4, device=R.device, dtype=R.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_err_deg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    R_delta = pred.transpose(-1, -2) @ gt
    tr = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
    cos = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def rot_err_deg_sym(
    pred: torch.Tensor, gt: torch.Tensor, n_faces: int = 1
) -> torch.Tensor:
    """Symmetry-aware rotation error in degrees.

    n_faces:
      1 = no symmetry
      0 = continuous Y-axis symmetry
      2 = 180-degree Y-axis ambiguity
      4 = 90-degree Y-axis ambiguity
    """
    if n_faces == 1:
        return rot_err_deg(pred, gt)

    if n_faces == 0:
        y = torch.tensor([0.0, 1.0, 0.0], device=pred.device, dtype=pred.dtype)
        y1 = pred @ y
        y2 = gt @ y
        cos = torch.dot(y1, y2) / (y1.norm() * y2.norm() + 1e-9)
        cos = torch.clamp(cos, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cos))

    best = None
    for k in range(n_faces):
        angle = 2.0 * math.pi * k / n_faces
        c, s = math.cos(angle), math.sin(angle)
        R_y = torch.tensor(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            device=pred.device,
            dtype=pred.dtype,
        )
        err = rot_err_deg(pred @ R_y, gt)
        best = err if best is None else torch.minimum(best, err)
    return best


def _single_frame_alignment(inf_dict, cams, frame_idx: int, image_size):
    from poselib import estimate_absolute_pose

    nocs_sel = inf_dict["m_v3d"]
    yx_sel = inf_dict["yx_sel"]
    b_sel = inf_dict["b_sel"]
    H, W = image_size

    mask = b_sel == frame_idx
    if mask.sum() < 10:
        R = torch.eye(3, device=nocs_sel.device, dtype=nocs_sel.dtype)
        t = torch.zeros(3, device=nocs_sel.device, dtype=nocs_sel.dtype)
        return R, t, torch.ones_like(t)

    nocs_f = nocs_sel[mask]
    yx_f = yx_sel[mask]
    cam_f = deepcopy(cams[[frame_idx]])
    negative_focal = (cam_f.focal_length < 0).any()
    cam_f.focal_length = torch.abs(cam_f.focal_length)
    flip_matrix = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=nocs_f.device,
        dtype=nocs_f.dtype,
    )

    proc_obj_pts = (nocs_f @ flip_matrix).cpu().numpy().astype(np.float64)
    proc_im_pts = yx_f[:, [1, 0]].cpu().numpy().astype(np.float64)
    fx_ndc, fy_ndc = cam_f.focal_length[0].tolist()
    px_ndc, py_ndc = cam_f.principal_point[0].tolist()

    fx = fx_ndc * (W / 2.0)
    fy = fy_ndc * (H / 2.0)
    cx = (px_ndc + 1.0) * (W / 2.0)
    cy = (1.0 - py_ndc) * (H / 2.0)

    camera = {
        "model": "PINHOLE",
        "width": int(W),
        "height": int(H),
        "params": [fx, fy, cx, cy],
    }
    bundle_opt = {
        "max_iterations": 100,
        "loss_type": "CAUCHY",
        "loss_scale": 2.0,
        "verbose": False,
    }
    ransac_opt = {
        "max_reproj_error": 4.0,
        "success_prob": 0.999,
        "max_iterations": 20000,
    }
    pose, _info = estimate_absolute_pose(
        proc_im_pts, proc_obj_pts, camera, ransac_opt=ransac_opt, bundle_opt=bundle_opt
    )
    R_cv = pose.R
    t_cv = pose.t
    S = np.diag([-1.0, -1.0, 1.0])
    R_p3d = (S @ R_cv).T
    R_p3d[:2, :] = R_p3d[:2, :] * -1.0
    t_p3d = (t_cv.reshape(1, 3) @ S).ravel()
    if negative_focal:
        R_p3d[:, :1] *= -1.0
    t_p3d[0] *= -1.0

    R = torch.from_numpy(R_p3d).to(nocs_f.device).to(nocs_f.dtype)
    if torch.any(torch.isnan(R)):
        R = torch.eye(3, device=nocs_f.device, dtype=nocs_f.dtype)
    t = torch.from_numpy(t_p3d).to(nocs_f.device).to(nocs_f.dtype)
    size = torch.ones_like(t)
    return R.transpose(-2, -1), t, size


def estimate_T_n2c_for_frame(inf_dict, cams, frame_idx: int, image_size):
    R_n2c, t_n2c, size = _single_frame_alignment(inf_dict, cams, frame_idx, image_size)
    return se3_from_Rt(R_n2c, t_n2c), size
