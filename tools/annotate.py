import math
from dataclasses import dataclass
from typing import Any

# Using nvdiffrast instead of PyTorch3D
import nvdiffrast.torch as dr
import torch
from einops import rearrange


@dataclass
class AnnotationResult:
    nocs: torch.Tensor
    depth: torch.Tensor
    geom_mask: torch.Tensor
    render_mask: torch.Tensor
    valid_mask: torch.Tensor
    obj_xyz: torch.Tensor
    obj_idx: torch.Tensor
    cameras: Any  # Contains transformed/rescaled cameras

    def to(self, device=None, non_blocking: bool = False) -> "AnnotationResult":
        def mv(t):
            return t.to(device=device, non_blocking=non_blocking) if isinstance(t, torch.Tensor) else t

        cams = self.cameras
        if cams is not None:
            cams = cams.to(device=device, non_blocking=non_blocking)
        return AnnotationResult(
            nocs=mv(self.nocs),
            depth=mv(self.depth),
            geom_mask=mv(self.geom_mask),
            render_mask=mv(self.render_mask),
            valid_mask=mv(self.valid_mask),
            obj_xyz=mv(self.obj_xyz),
            obj_idx=mv(self.obj_idx),
            cameras=cams,
        )


def vertices_to_nocs01(verts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    verts: (V, 3) in object coordinates.
    returns: (V, 3) NOCS in [0, 1] by bbox normalization.
    """
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    scale = (vmax - vmin).clamp_min(eps)
    return (verts - vmin) / scale


class Annotator:
    def __init__(
        self,
        V,
        F,
        image_size,
        device=torch.device("cpu"),
        background_color=(1.0, 1.0, 1.0),
    ):
        """
        Initializes the nvdiffrast-based annotator.
        """
        self.device = device

        self.V_obj = torch.as_tensor(
            V, dtype=torch.float32, device=self.device
        )  # (V, 3)
        # nvdiffrast requires faces to be int32
        self.F = torch.as_tensor(F, dtype=torch.int32, device=self.device)  # (F, 3)

        self.size = (
            torch.max(self.V_obj, dim=0).values - torch.min(self.V_obj, dim=0).values
        )

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.bg_color = torch.tensor(
            background_color, dtype=torch.float32, device=self.device
        )

        # Initialize the nvdiffrast context
        # Note: RasterizeCudaContext is generally faster if using CUDA
        self.glctx = (
            dr.RasterizeCudaContext(device=self.device)
            if self.device.type == "cuda"
            else dr.RasterizeGLContext(device=self.device)
        )

        # Precompute NOCS features
        self.nocs0 = vertices_to_nocs01(self.V_obj)

    @torch.no_grad()
    def __call__(
        self, multi_frame, deform=False, rescale_T=True, cameras_override=None
    ) -> AnnotationResult:
        src_cams = cameras_override if cameras_override is not None else multi_frame.cameras
        cameras = src_cams.clone()
        batch_size = len(cameras)

        seq_lengths = torch.tensor(multi_frame.seq_lengths, device=self.device)
        per_frame_obj_idx = torch.arange(
            len(seq_lengths), device=self.device
        ).repeat_interleave(seq_lengths)

        obj_sizes = multi_frame.obj_sizes.to(self.device)
        obj_centers = multi_frame.obj_centers.to(self.device)
        if obj_sizes.shape[0] == batch_size:
            sizes = obj_sizes
            centers = obj_centers
        elif obj_sizes.shape[0] == len(seq_lengths):
            sizes = obj_sizes[per_frame_obj_idx]
            centers = obj_centers[per_frame_obj_idx]
        else:
            raise ValueError(
                "obj_sizes/obj_centers must be either per-frame or per-sequence: "
                f"got {obj_sizes.shape[0]} rows for {batch_size} frames and "
                f"{len(seq_lengths)} sequences"
            )

        scales = torch.linalg.norm(sizes, dim=-1)
        deformation = sizes / (scales[:, None] * self.size)

        if not deform:
            deformation = torch.ones_like(deformation)

        # Deform the object vertices
        V_deformed_batch = self.V_obj[None] * deformation[:, None, :]  # (B, V, 3)

        Rc = torch.bmm(centers[:, None, :], cameras.R).squeeze(1)
        if rescale_T:
            cameras.T = (cameras.T + Rc) / scales[:, None]

        # 1. Transform vertices to clip space (Homogeneous coordinates) for nvdiffrast
        # Row-vector convention: v' = v @ M
        proj_matrix = cameras.full_projection_matrix()  # (B, 4, 4)

        V_homog = torch.cat(
            [V_deformed_batch, torch.ones_like(V_deformed_batch[..., :1])], dim=-1
        )
        V_clip = torch.bmm(V_homog, proj_matrix)  # (B, V, 4)

        # Invert Y and X axes to match OpenGL/nvdiffrast NDC conventions if coming from PyTorch3D
        # PyTorch3D's NDC is +X left, +Y up. OpenGL's is +X right, +Y up.
        # We negate X and Y to ensure rasterization matches PyTorch3D screen space exactly.
        V_clip[..., 0] *= -1.0
        V_clip[..., 1] *= -1.0

        # 2. Rasterize
        # Resolution is passed as [height, width]
        rast, rast_db = dr.rasterize(
            self.glctx, V_clip, self.F, resolution=list(self.image_size)
        )

        # Face ID mask (rast[..., 3] > 0 means a face was hit)
        mask = rast[..., 3] > 0

        # 3. Interpolate attributes
        # NOCS
        nocs_rgb, _ = dr.interpolate(self.nocs0.contiguous(), rast, self.F)

        # Original undeformed 3D coordinates
        obj_xyz, _ = dr.interpolate(self.V_obj.contiguous(), rast, self.F)

        # Depth (We need true camera-space depth, not NDC depth)
        # Row-vector world->view: v_view_h = v_world_h @ W
        W_mat = cameras.world_to_view_matrix()  # (B, 4, 4)
        V_view = torch.bmm(V_homog, W_mat)  # (B, V, 4)
        V_cam_z = V_view[..., 2:3]  # (B, V, 1) Z-depth

        depth, _ = dr.interpolate(V_cam_z.contiguous(), rast, self.F)
        depth = depth[..., 0]  # (B, H, W)

        # 4. Blend Background
        bg = self.bg_color.view(1, 1, 1, 3).expand(
            batch_size, self.image_size[0], self.image_size[1], 3
        )
        nocs_rgb = torch.where(mask[..., None], nocs_rgb, bg)

        # 5. Handle user-provided valid_masks
        if hasattr(multi_frame, "valid_masks") and multi_frame.valid_masks is not None:
            # valid_masks: (N, 1, H, W) uint8 at full image resolution — downsample to render size
            Hr, Wr = self.image_size
            valid_ds = torch.nn.functional.interpolate(
                multi_frame.valid_masks.float(), size=(Hr, Wr), mode="nearest"
            )
            valid_mask = valid_ds.squeeze(1).bool()  # (N, H, W)
            geom_mask = mask & valid_mask
            depth = depth * valid_mask
            nocs_rgb[..., :3] = nocs_rgb[..., :3] * valid_mask[..., None]
            obj_xyz[..., :3] = obj_xyz[..., :3] * valid_mask[..., None]
        else:
            valid_mask = torch.ones_like(mask)
            geom_mask = mask

        # 6. Return AnnotationResult
        return AnnotationResult(
            nocs=rearrange(nocs_rgb, "b h w c -> b c h w"),
            depth=depth,
            geom_mask=geom_mask,
            render_mask=mask,
            valid_mask=valid_mask,
            obj_xyz=obj_xyz,
            obj_idx=per_frame_obj_idx,
            cameras=cameras,
        )

    @torch.no_grad()
    def rerender_aligned(
        self,
        multi_frame,
        R_obj: torch.Tensor,
        obj_id_per_frame: torch.Tensor = None,
        deform=False,
    ) -> AnnotationResult:
        if obj_id_per_frame is None:
            seq_lengths = torch.tensor(multi_frame.seq_lengths, device=self.device)
            obj_id_per_frame = torch.arange(len(seq_lengths), device=self.device).repeat_interleave(seq_lengths)

        cameras = multi_frame.cameras.clone()
        cameras.R = R_obj[obj_id_per_frame] @ cameras.R
        return self.__call__(
            multi_frame, deform=deform, rescale_T=False, cameras_override=cameras
        )
