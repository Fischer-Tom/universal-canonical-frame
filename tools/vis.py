from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    m = IMAGENET_MEAN.to(img.device, img.dtype)
    s = IMAGENET_STD.to(img.device, img.dtype)
    return (img * s + m).clamp(0, 1)


def correspondence_rgb(logits_or_idx: torch.Tensor, V: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Map either logits or precomputed vertex indices to XYZ-normalized RGB.

    logits_or_idx: (B, Q, H*W) logits or (B, H*W) vertex indices.
    V: (Q, 3) vertices.
    returns: (B, 3, H, W)
    """
    if logits_or_idx.dim() == 2:
        idx = logits_or_idx.long()
    else:
        idx = logits_or_idx.float().argmax(dim=1)
    V = V.to(device=idx.device)
    xyz = V[idx]
    xyz = xyz.view(-1, H, W, 3).permute(0, 3, 1, 2).contiguous()
    mn = V.amin(0).view(1, 3, 1, 1)
    mx = V.amax(0).view(1, 3, 1, 1)
    return ((xyz - mn) / (mx - mn + 1e-9)).clamp(0, 1)


def mask_rgb(mask: torch.Tensor) -> torch.Tensor:
    """Binary mask rendered as a 3-channel image."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask.to(dtype=torch.float32).clamp(0, 1).repeat(1, 3, 1, 1)


@torch.no_grad()
def save_correspondence_grid(
    img_rgb: torch.Tensor,
    geom_mask: torch.Tensor,
    logits: torch.Tensor,
    V: torch.Tensor,
    feat_hw: tuple[int, int],
    out_path: str,
    max_n: int = 8,
    unnormalize: bool = True,
) -> None:
    """Save a three-row grid: input images, geometry masks, masked correspondence RGB."""
    n = min(img_rgb.shape[0], max_n)
    img = _unnormalize(img_rgb[:n]) if unnormalize else img_rgb[:n].clamp(0, 1)
    H, W = feat_hw
    mask = mask_rgb(geom_mask[:n])
    mask = F.interpolate(mask, size=img.shape[-2:], mode="nearest")
    corr = correspondence_rgb(logits[:n], V, H, W)
    corr = F.interpolate(corr, size=img.shape[-2:], mode="nearest")
    corr = corr * mask
    grid = torch.cat([img.cpu(), mask.cpu(), corr.cpu()], dim=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(grid, out_path, nrow=n)
