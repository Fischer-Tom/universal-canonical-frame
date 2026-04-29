from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch


@dataclass
class Camera:
    """
    PyTorch3D-compatible PerspectiveCameras replacement (row-major, NDC).

    Conventions match pytorch3d.renderer.PerspectiveCameras:
      - Row-vector convention: v_view = v_world @ R + T
      - NDC space: +X left, +Y up, +Z away from viewer
      - focal_length / principal_point in NDC units

    Batched: every tensor field has leading dim N.
    """

    R: torch.Tensor                          # (N, 3, 3)
    T: torch.Tensor                          # (N, 3)
    focal_length: torch.Tensor               # (N, 2)
    principal_point: torch.Tensor            # (N, 2)
    image_size: Optional[torch.Tensor] = None  # (N, 2) as (H, W)

    def __len__(self) -> int:
        return int(self.R.shape[0])

    @property
    def device(self) -> torch.device:
        return self.R.device

    @property
    def dtype(self) -> torch.dtype:
        return self.R.dtype

    def __getitem__(self, idx) -> "Camera":
        if isinstance(idx, int):
            idx = [idx]
        return Camera(
            R=self.R[idx],
            T=self.T[idx],
            focal_length=self.focal_length[idx],
            principal_point=self.principal_point[idx],
            image_size=None if self.image_size is None else self.image_size[idx],
        )

    def clone(self) -> "Camera":
        return Camera(
            R=self.R.clone(),
            T=self.T.clone(),
            focal_length=self.focal_length.clone(),
            principal_point=self.principal_point.clone(),
            image_size=None if self.image_size is None else self.image_size.clone(),
        )

    def to(self, device=None, non_blocking: bool = False) -> "Camera":
        def mv(t):
            return None if t is None else t.to(device=device, non_blocking=non_blocking)

        return Camera(
            R=mv(self.R),
            T=mv(self.T),
            focal_length=mv(self.focal_length),
            principal_point=mv(self.principal_point),
            image_size=mv(self.image_size),
        )

    def world_to_view_matrix(self) -> torch.Tensor:
        """(N, 4, 4) row-major world->view. v_view_h = v_world_h @ W."""
        N = len(self)
        W = torch.zeros((N, 4, 4), device=self.device, dtype=self.dtype)
        W[:, :3, :3] = self.R
        W[:, 3, :3] = self.T
        W[:, 3, 3] = 1.0
        return W

    def projection_matrix(self) -> torch.Tensor:
        """
        (N, 4, 4) row-major view->NDC perspective projection.
        v_ndc_h = v_view_h @ K; after divide by w, (x, y) in NDC.
        Matches pytorch3d PerspectiveCameras NDC formula.
        """
        N = len(self)
        fx = self.focal_length[:, 0]
        fy = self.focal_length[:, 1]
        px = self.principal_point[:, 0]
        py = self.principal_point[:, 1]

        K = torch.zeros((N, 4, 4), device=self.device, dtype=self.dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 2, 0] = px
        K[:, 2, 1] = py
        K[:, 3, 2] = 1.0
        K[:, 2, 3] = 1.0
        return K

    def full_projection_matrix(self) -> torch.Tensor:
        """(N, 4, 4) world->NDC. v_ndc_h = v_world_h @ M."""
        return self.world_to_view_matrix() @ self.projection_matrix()

    def transform_points(self, points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Project world-space points to NDC.

        Args:
            points: (N, P, 3) or (P, 3) — if (P, 3), broadcast across N cameras.

        Returns:
            (N, P, 3) with (x_ndc, y_ndc, z_view).
        """
        if points.dim() == 2:
            points = points.unsqueeze(0).expand(len(self), -1, -1)
        ones = torch.ones_like(points[..., :1])
        pts_h = torch.cat([points, ones], dim=-1)           # (N, P, 4)
        M = self.full_projection_matrix()                    # (N, 4, 4)
        out = torch.bmm(pts_h, M)                            # (N, P, 4)
        w = out[..., 3:4].clamp(min=eps)
        return out[..., :3] / w

    @classmethod
    def from_uco3d(cls, cams) -> "Camera":
        """Convert a uco3d `Cameras` (pytorch3d PerspectiveCameras-compatible) to Camera."""
        return cls(
            R=cams.R,
            T=cams.T,
            focal_length=cams.focal_length,
            principal_point=cams.principal_point,
            image_size=getattr(cams, "image_size", None),
        )


def concat_cameras(cams: list[Camera]) -> Camera:
    def _cat(attr):
        vals = [getattr(c, attr) for c in cams]
        if any(v is None for v in vals):
            return None
        return torch.cat(vals, dim=0)

    return Camera(
        R=torch.cat([c.R for c in cams], dim=0),
        T=torch.cat([c.T for c in cams], dim=0),
        focal_length=_cat("focal_length"),
        principal_point=_cat("principal_point"),
        image_size=_cat("image_size"),
    )
