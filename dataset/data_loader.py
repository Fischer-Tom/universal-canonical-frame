from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TVF
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from utils.camera import Camera

from .collate import safe_collate
from .schema import Batch, SequenceSample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class GaussianBlur(v2.RandomApply):
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 1.0):
        super().__init__(transforms=[v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))], p=p)


class SharedIteration:
    def __init__(self, iteration: int = 0):
        self._value = mp.Value("q", int(iteration))

    def get(self) -> int:
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, iteration: int) -> None:
        with self._value.get_lock():
            self._value.value = int(iteration)


def _rotz_batch(deg: torch.Tensor) -> torch.Tensor:
    rad = deg * (math.pi / 180.0)
    c, s = torch.cos(rad), torch.sin(rad)
    R = torch.zeros((deg.shape[0], 3, 3), device=deg.device, dtype=torch.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def _rotate_on_spot(R: torch.Tensor, T: torch.Tensor, Rz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-convention camera rotation about its own center.

    Assuming pytorch3d row-vector convention where `v_view = v_world @ R + T`,
    rotating camera on-spot by Rz applies `v_view' = v_view @ Rz`, giving
    `R_new = R @ Rz`, `T_new = T @ Rz`.
    """
    return R @ Rz, (T.unsqueeze(1) @ Rz).squeeze(1)


def _rotate_principal_point_on_image_(cams: Camera, Rz: torch.Tensor) -> None:
    if cams.principal_point is None:
        return
    R2 = Rz[:, :2, :2].to(device=cams.device, dtype=cams.principal_point.dtype)
    cams.principal_point = (cams.principal_point.unsqueeze(1) @ R2).squeeze(1)


def _sample_recrop_hw(H, W, scale_range, ratio_range, device, min_size=32, num_tries=10):
    area = float(H * W)
    smin, smax = scale_range
    rmin, rmax = ratio_range
    for _ in range(num_tries):
        ta = area * torch.empty((), device=device).uniform_(smin, smax).item()
        lr = torch.empty((), device=device).uniform_(math.log(rmin), math.log(rmax)).item()
        ratio = math.exp(lr)
        h = int(round(math.sqrt(ta / ratio)))
        w = int(round(math.sqrt(ta * ratio)))
        if min_size <= h <= H and min_size <= w <= W:
            y1 = int(torch.randint(0, H - h + 1, (1,), device=device).item())
            x1 = int(torch.randint(0, W - w + 1, (1,), device=device).item())
            return y1, x1, y1 + h, x1 + w
    h = max(min_size, int(round(min(H, W) * 0.9)))
    w = h
    return (H - h) // 2, (W - w) // 2, (H - h) // 2 + h, (W - w) // 2 + w


def _get_mask_bbox(mask: torch.Tensor, padding: float = 0.0) -> Tuple[int, int, int, int]:
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    H, W = mask.shape
    nz = mask.nonzero(as_tuple=True)
    if len(nz[0]) == 0:
        return 0, 0, H, W
    y1, y2 = nz[0].min().item(), nz[0].max().item()
    x1, x2 = nz[1].min().item(), nz[1].max().item()
    if padding > 0:
        pad_h = int((y2 - y1) * padding)
        pad_w = int((x2 - x1) * padding)
        y1 = max(0, y1 - pad_h)
        y2 = min(H, y2 + pad_h)
        x1 = max(0, x1 - pad_w)
        x2 = min(W, x2 + pad_w)
    return y1, x1, y2 + 1, x2 + 1


def _scale_to_fit_and_blur_pad(x, Ht, Wt, mode="bilinear", bg_down: int = 16):
    N, C, H, W = x.shape
    s = min(Ht / H, Wt / W)
    Hr, Wr = int(round(H * s)), int(round(W * s))
    ac = False if mode == "bilinear" else None
    x_fit = torch.nn.functional.interpolate(x, size=(Hr, Wr), mode=mode, align_corners=ac)
    # Blurry background approximated by hard area-downsample then upsample.
    # Constant cost (independent of input size) and visually equivalent to a
    # wide-sigma gaussian for the purpose of hiding hard pad edges.
    Hs = max(1, Ht // bg_down)
    Ws = max(1, Wt // bg_down)
    x_small = torch.nn.functional.interpolate(x, size=(Hs, Ws), mode="area")
    x_bg = torch.nn.functional.interpolate(x_small, size=(Ht, Wt), mode=mode, align_corners=ac)
    pt, pl = (Ht - Hr) // 2, (Wt - Wr) // 2
    x_bg[:, :, pt:pt + Hr, pl:pl + Wr] = x_fit
    return x_bg, s, (pl, pt), (Hr, Wr)


def _scale_to_fit_and_zero_pad(x, Ht, Wt, mode):
    N, C, H, W = x.shape
    s = min(Ht / H, Wt / W)
    Hr, Wr = int(round(H * s)), int(round(W * s))
    ac = False if mode == "bilinear" else None
    x_fit = torch.nn.functional.interpolate(x, size=(Hr, Wr), mode=mode, align_corners=ac)
    out = torch.zeros(N, C, Ht, Wt, device=x.device, dtype=x.dtype)
    pt, pl = (Ht - Hr) // 2, (Wt - Wr) // 2
    out[:, :, pt:pt + Hr, pl:pl + Wr] = x_fit
    return out, s, (pl, pt), (Hr, Wr)


class RandomPatchOcclusionInBox(torch.nn.Module):
    def __init__(self, p=0.5, num_patches=(1, 2), patch_size=(0.06, 0.25), fill_value="zero", min_patch_px=4):
        super().__init__()
        self.p = float(p)
        self.num_patches = (int(num_patches[0]), int(num_patches[1]))
        self.patch_size = (float(patch_size[0]), float(patch_size[1]))
        self.fill_value = fill_value
        self.min_patch_px = int(min_patch_px)

    def forward(self, img, content_box=None):
        C, H, W = img.shape
        keep = torch.ones((1, H, W), device=img.device, dtype=torch.uint8)
        if content_box is None:
            x0, y0, bw, bh = 0, 0, W, H
        else:
            x0, y0, bw, bh = content_box
        x0 = max(0, min(int(x0), W - 1))
        y0 = max(0, min(int(y0), H - 1))
        bw = max(1, min(int(bw), W - x0))
        bh = max(1, min(int(bh), H - y0))
        if torch.rand((), device=img.device).item() > self.p:
            return img, keep
        img = img.clone()
        num = int(torch.randint(self.num_patches[0], self.num_patches[1] + 1, (1,), device=img.device).item())
        base = float(min(bw, bh))
        for _ in range(num):
            fh = float(torch.empty((), device=img.device).uniform_(*self.patch_size).item())
            fw = float(torch.empty((), device=img.device).uniform_(*self.patch_size).item())
            ph = min(max(self.min_patch_px, int(round(base * fh))), bh)
            pw = min(max(self.min_patch_px, int(round(base * fw))), bw)
            y = int(torch.randint(y0, y0 + max(1, bh - ph + 1), (1,), device=img.device).item())
            x = int(torch.randint(x0, x0 + max(1, bw - pw + 1), (1,), device=img.device).item())
            if self.fill_value == "zero":
                img[:, y:y + ph, x:x + pw] = 0
            elif self.fill_value == "random":
                img[:, y:y + ph, x:x + pw] = torch.rand((C, ph, pw), device=img.device, dtype=img.dtype)
            elif self.fill_value == "mean":
                img[:, y:y + ph, x:x + pw] = img.mean()
            else:
                raise ValueError(self.fill_value)
            keep[:, y:y + ph, x:x + pw] = 0
        return img, keep


# ---------------------------------------------------------------------------
# Camera intrinsic adjustment
# ---------------------------------------------------------------------------
def _set_cameras_image_size_(cams: Camera, new_size_hw: Tuple[int, int]):
    H, W = new_size_hw
    cams.image_size = torch.tensor([H, W], dtype=torch.float32, device=cams.device).repeat(cams.R.shape[0], 1)


def _adjust_cameras_for_crop_scale_pad(
    cams: Camera,
    crop_offsets: List[Tuple[int, int]],
    scales: List[float],
    padding_offsets: List[Tuple[int, int]],
    new_size_hw: Tuple[int, int],
    flip_mask: Optional[torch.Tensor] = None,
):
    if cams.focal_length is None or cams.principal_point is None or cams.image_size is None:
        _set_cameras_image_size_(cams, new_size_hw)
        return
    N = cams.R.shape[0]
    dev = cams.device
    scales_t = torch.tensor(scales, dtype=torch.float32, device=dev)
    pad_left_t = torch.tensor([p[0] for p in padding_offsets], dtype=torch.float32, device=dev)
    pad_top_t = torch.tensor([p[1] for p in padding_offsets], dtype=torch.float32, device=dev)
    crop_x_t = torch.tensor([c[0] for c in crop_offsets], dtype=torch.float32, device=dev)
    crop_y_t = torch.tensor([c[1] for c in crop_offsets], dtype=torch.float32, device=dev)

    old_sizes_wh = cams.image_size.flip(dims=[1])
    half_old = old_sizes_wh / 2.0
    rescale_old = half_old.min(dim=1, keepdim=True).values
    focal_px = cams.focal_length * rescale_old
    pp_px = half_old - cams.principal_point * rescale_old

    pp_px[:, 0] -= crop_x_t
    pp_px[:, 1] -= crop_y_t
    focal_px = focal_px * scales_t.unsqueeze(1)
    pp_px = pp_px * scales_t.unsqueeze(1)
    pp_px[:, 0] += pad_left_t
    pp_px[:, 1] += pad_top_t

    Ht, Wt = new_size_hw
    new_wh = torch.tensor([Wt, Ht], dtype=torch.float32, device=dev).unsqueeze(0)
    half_new = new_wh / 2.0
    rescale_new = half_new.min(dim=1, keepdim=True).values
    focal_ndc = focal_px / rescale_new
    pp_ndc = (half_new - pp_px) / rescale_new
    if flip_mask is not None and flip_mask.any():
        pp_ndc[flip_mask, 0] = -pp_ndc[flip_mask, 0]

    cams.focal_length = focal_ndc
    cams.principal_point = pp_ndc
    cams.image_size = torch.tensor([Ht, Wt], dtype=torch.float32, device=dev).repeat(N, 1)


# ---------------------------------------------------------------------------
# Per-sample transform (CPU worker) — produces uniform-sized SequenceSample
# ---------------------------------------------------------------------------
@dataclass
class SampleTransformCfg:
    size_hw: Tuple[int, int] = (384, 384)
    flip_p: float = 0.0
    crop_to_mask: bool = True
    crop_padding: Tuple[float, float] = (0.1, 0.25)
    geometric_start_iter: int = 0
    recrop_p: float = 0.7
    recrop_scale: Tuple[float, float] = (0.9, 1.0)
    recrop_ratio: Tuple[float, float] = (0.9, 1.1)
    recrop_min_size: int = 48
    rotate_p: float = 0.3
    rotate_deg: float = 30.0
    occlusion_start_iter: int = 0
    patch_mask_p: float = 0.5
    patch_mask_num: Tuple[int, int] = (1, 2)
    patch_mask_size: Tuple[float, float] = (0.05, 0.30)
    blur_p: float = 0.4
    solarize_p: float = 0.0
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    normalize: bool = True
    training: bool = True


class SampleTransform:
    """Apply per-frame augment + resize to uniform (Ht,Wt). Runs on CPU in worker.

    Produces a new `SequenceSample` with uniform image/mask size and updated
    camera intrinsics (crop+scale+pad+flip composed).
    """

    def __init__(self, cfg: SampleTransformCfg, iteration_state: Optional[SharedIteration] = None):
        self.cfg = cfg
        self.iteration_state = iteration_state or SharedIteration(0)
        if cfg.training:
            self.color_jitter = v2.Compose([
                v2.RandomApply(
                    [v2.ColorJitter(cfg.brightness, cfg.contrast, cfg.saturation, cfg.hue)], p=0.8
                ),
                v2.RandomGrayscale(p=0.2),
            ])
            self.extra = v2.Compose([
                GaussianBlur(p=cfg.blur_p),
                v2.RandomSolarize(threshold=0.5, p=cfg.solarize_p),
            ])
            self.occluder = (
                RandomPatchOcclusionInBox(
                    p=cfg.patch_mask_p,
                    num_patches=cfg.patch_mask_num,
                    patch_size=cfg.patch_mask_size,
                ) if cfg.patch_mask_p > 0 else None
            )
        else:
            self.color_jitter = None
            self.extra = None
            self.occluder = None

    def set_iteration(self, iteration: int) -> None:
        self.iteration_state.set(iteration)

    def _geometric_enabled(self) -> bool:
        return self.cfg.training and self.iteration_state.get() >= self.cfg.geometric_start_iter

    def _occlusion_enabled(self) -> bool:
        return self.cfg.training and self.iteration_state.get() >= self.cfg.occlusion_start_iter

    @torch.no_grad()
    def __call__(self, sample: SequenceSample) -> SequenceSample:
        cfg = self.cfg
        Ht, Wt = cfg.size_hw
        if sample.image_rgb is None:
            return sample
        imgs = TVF.convert_image_dtype(sample.image_rgb, dtype=torch.float32)  # (K,C,H,W)
        masks = sample.masks.float() if sample.masks is not None else None
        K = imgs.shape[0]

        out_imgs, out_masks, out_valid, out_orig = [], [], [], []
        crop_offsets, scales, pad_offsets, flip_flags, rot_degs = [], [], [], [], []
        geometric_enabled = self._geometric_enabled()
        occlusion_enabled = self._occlusion_enabled()

        for i in range(K):
            frame = imgs[i]
            mask = masks[i] if masks is not None else None

            if cfg.crop_to_mask and mask is not None:
                pad = torch.empty(()).uniform_(*cfg.crop_padding).item() if geometric_enabled else cfg.crop_padding[0]
                y1, x1, y2, x2 = _get_mask_bbox(mask, padding=pad)
                frame = frame[:, y1:y2, x1:x2]
                mask = mask[:, y1:y2, x1:x2]
                crop = (x1, y1)
            else:
                crop = (0, 0)

            if geometric_enabled and cfg.recrop_p > 0 and torch.rand(()).item() < cfg.recrop_p:
                Hc, Wc = frame.shape[-2:]
                y1r, x1r, y2r, x2r = _sample_recrop_hw(
                    Hc, Wc, cfg.recrop_scale, cfg.recrop_ratio,
                    device=frame.device, min_size=cfg.recrop_min_size,
                )
                frame = frame[:, y1r:y2r, x1r:x2r]
                if mask is not None:
                    mask = mask[:, y1r:y2r, x1r:x2r]
                crop = (crop[0] + x1r, crop[1] + y1r)

            pre_sz = torch.tensor(frame.shape[-2:], dtype=torch.float32)

            frame_4d, s, cam_off, (vH, vW) = _scale_to_fit_and_blur_pad(frame.unsqueeze(0), Ht, Wt, "bilinear")
            frame = frame_4d.squeeze(0)
            valid = torch.zeros((1, Ht, Wt), device=frame.device, dtype=torch.uint8)
            valid[:, cam_off[1]:cam_off[1] + vH, cam_off[0]:cam_off[0] + vW] = 1
            if mask is not None:
                mask_4d, *_ = _scale_to_fit_and_zero_pad(mask.unsqueeze(0), Ht, Wt, "nearest")
                mask = mask_4d.squeeze(0)

            do_flip = geometric_enabled and cfg.flip_p > 0 and torch.rand(()).item() < cfg.flip_p
            if do_flip:
                frame = TVF.hflip(frame)
                valid = TVF.hflip(valid)
                if mask is not None:
                    mask = TVF.hflip(mask)

            ang = 0.0
            if geometric_enabled and cfg.rotate_p > 0 and torch.rand(()).item() < cfg.rotate_p:
                ang = torch.empty(()).uniform_(-cfg.rotate_deg, cfg.rotate_deg).item()
                ch_mean = frame.mean(dim=(-2, -1), keepdim=True)
                rgba = torch.cat([frame, torch.ones_like(frame[:1])], dim=0)
                rgba = TVF.rotate(rgba, angle=ang, interpolation=TVF.InterpolationMode.BILINEAR, expand=False, fill=0.0)
                frame_rot, cov = rgba[:3], rgba[3:4]
                frame = frame_rot + (1.0 - cov) * ch_mean
                if mask is not None:
                    mask = TVF.rotate(mask, angle=ang, interpolation=TVF.InterpolationMode.NEAREST, expand=False, fill=0)
                valid = TVF.rotate(valid, angle=ang, interpolation=TVF.InterpolationMode.NEAREST, expand=False, fill=0)

            if occlusion_enabled and self.occluder is not None:
                content_box = (max(0, cam_off[0]), max(0, cam_off[1]), vW, vH)
                frame, keep = self.occluder(frame, content_box=content_box)
                valid = valid * keep.to(dtype=valid.dtype)

            if self.color_jitter is not None:
                frame = self.color_jitter(frame)
            if self.extra is not None:
                frame = self.extra(frame)

            out_imgs.append(frame)
            if mask is not None:
                out_masks.append(mask)
            out_valid.append(valid)
            out_orig.append(pre_sz)
            crop_offsets.append(crop)
            scales.append(s)
            pad_offsets.append(cam_off)
            flip_flags.append(do_flip)
            rot_degs.append(ang)

        imgs_out = torch.stack(out_imgs, dim=0)
        if cfg.normalize:
            imgs_out = TVF.normalize(imgs_out, mean=list(cfg.normalize_mean), std=list(cfg.normalize_std))
        masks_out = (torch.stack(out_masks, dim=0) > 0.5).to(torch.uint8) if out_masks else None
        valid_out = (torch.stack(out_valid, dim=0) > 0).to(torch.uint8)

        cams = sample.cameras
        flip_mask = torch.tensor(flip_flags, device=cams.R.device) if any(flip_flags) else None
        _adjust_cameras_for_crop_scale_pad(cams, crop_offsets, scales, pad_offsets, (Ht, Wt), flip_mask)

        if any(a != 0.0 for a in rot_degs):
            deg_t = torch.tensor(rot_degs, device=cams.R.device, dtype=cams.R.dtype)
            Rz = _rotz_batch(deg_t).to(dtype=cams.R.dtype)
            R_new, T_new = _rotate_on_spot(cams.R, cams.T, Rz)
            cams.R, cams.T = R_new, T_new
            _rotate_principal_point_on_image_(cams, Rz)

        return SequenceSample(
            sequence_name=sample.sequence_name,
            frame_numbers=sample.frame_numbers,
            cameras=cams,
            image_rgb=imgs_out,
            masks=masks_out,
            valid_masks=valid_out,
            orig_sizes_hw=torch.stack(out_orig, dim=0),
            segmented_point_cloud=sample.segmented_point_cloud,
            object_size=sample.object_size,
            obj_center=sample.obj_center,
        )


def transform_from_cfg(cfg, training: bool = True) -> SampleTransform:
    size_hw = tuple(cfg.image_size)
    aug = getattr(cfg, "augmentation", cfg)
    cp = getattr(aug, "crop_padding", 0.2)
    cp_range = getattr(aug, "crop_padding_range", None)
    crop_padding = tuple(cp_range) if cp_range else (float(cp), float(cp))
    common_start = _cfg_int(
        aug,
        ("start_iter", "aug_start_iter", "augmentation_start_iter"),
        0,
    )
    return SampleTransform(SampleTransformCfg(
        size_hw=size_hw,
        training=training,
        flip_p=0.0 if not training else getattr(aug, "flip_p", 0.0),
        crop_to_mask=getattr(aug, "crop_to_mask", True),
        crop_padding=crop_padding,
        geometric_start_iter=0 if not training else _cfg_int(
            aug,
            ("geometric_start_iter", "geometric_aug_start_iter", "geometry_start_iter"),
            common_start,
        ),
        recrop_p=0.0 if not training else getattr(aug, "recrop_p", 0.7),
        recrop_scale=tuple(getattr(aug, "recrop_scale", (0.9, 1.0))),
        recrop_ratio=tuple(getattr(aug, "recrop_ratio", (0.9, 1.1))),
        recrop_min_size=getattr(aug, "recrop_min_size", 48),
        rotate_p=0.0 if not training else getattr(aug, "rotate_p", 0.3),
        rotate_deg=getattr(aug, "rotate_deg", 30.0),
        occlusion_start_iter=0 if not training else _cfg_int(
            aug,
            ("occlusion_start_iter", "occlusion_aug_start_iter", "patch_mask_start_iter"),
            common_start,
        ),
        patch_mask_p=0.0 if not training else getattr(aug, "patch_mask_p", 0.5),
        patch_mask_num=tuple(getattr(aug, "patch_mask_num", (1, 2))),
        patch_mask_size=tuple(getattr(aug, "patch_mask_size", (0.05, 0.3))),
        blur_p=0.0 if not training else getattr(aug, "blur_p", 0.4),
        solarize_p=0.0 if not training else getattr(aug, "solarize_p", 0.0),
        brightness=getattr(aug, "jitter_brightness", 0.4),
        contrast=getattr(aug, "jitter_contrast", 0.4),
        saturation=getattr(aug, "jitter_saturation", 0.2),
        hue=getattr(aug, "jitter_hue", 0.1),
    ))


def _cfg_int(cfg: Any, names: Tuple[str, ...], default: int) -> int:
    for name in names:
        value = getattr(cfg, name, None)
        if value is not None:
            return int(value)
    return int(default)


# ---------------------------------------------------------------------------
# Dataset wrapper applying transform per sample
# ---------------------------------------------------------------------------
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform: SampleTransform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx) -> Optional[SequenceSample]:
        sample = self.base[idx]
        if sample is None:
            return None
        return self.transform(sample)

    def set_iteration(self, iteration: int) -> None:
        self.transform.set_iteration(iteration)


# ---------------------------------------------------------------------------
# CUDA prefetcher (H2D only)
# ---------------------------------------------------------------------------
class CudaPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.sampler = getattr(loader, "sampler", None)

    def __iter__(self):
        stream = torch.cuda.Stream(device=self.device)
        for batch in self.loader:
            with torch.cuda.stream(stream):
                batch = batch.to(device=self.device, non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            yield batch

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def set_iteration(self, iteration: int) -> None:
        dataset = getattr(self.loader, "dataset", None)
        if hasattr(dataset, "set_iteration"):
            dataset.set_iteration(iteration)


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------
def _worker_init(_worker_id: int) -> None:
    # 16 workers × N intra-op threads each = thread oversubscription and cache
    # thrashing. Pin each worker to a single torch thread; the outer DataLoader
    # gives us parallelism across samples.
    torch.set_num_threads(1)


def build_loader(
    dataset,
    cfg,
    device: torch.device,
    *,
    training: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    transform: Optional[SampleTransform] = None,
) -> CudaPrefetcher:
    if transform is None:
        transform = transform_from_cfg(cfg, training=training)
    ds = TransformedDataset(dataset, transform)

    sampler = None
    shuffle = training
    if distributed and training:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False

    loader = DataLoader(
        ds,
        batch_size=1 if not training else cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        collate_fn=safe_collate,
        worker_init_fn=_worker_init if cfg.num_workers > 0 else None,
    )
    return CudaPrefetcher(loader, device)
