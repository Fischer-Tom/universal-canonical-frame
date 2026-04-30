from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from utils.camera import Camera, concat_cameras


@dataclass(slots=True)
class FrameRecord:
    sequence_name: str
    frame_number: int
    image_rgb: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    camera: Optional[Camera] = None
    orig_size_hw: Optional[torch.Tensor] = None
    segmented_point_cloud: Optional[Any] = None


@dataclass(slots=True)
class SequenceSample:
    sequence_name: str
    frame_numbers: torch.Tensor
    cameras: Camera
    image_rgb: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    valid_masks: Optional[torch.Tensor] = None
    orig_sizes_hw: Optional[torch.Tensor] = None
    segmented_point_cloud: Optional[Any] = None
    object_size: Optional[torch.Tensor] = None
    obj_center: Optional[torch.Tensor] = None

    @property
    def num_frames(self) -> int:
        return int(self.frame_numbers.numel())

    @classmethod
    def from_frames(cls, frames: List[FrameRecord]) -> "SequenceSample":
        if not frames:
            raise ValueError("SequenceSample.from_frames requires at least one frame")
        seq = frames[0].sequence_name
        for f in frames:
            if f.sequence_name != seq:
                raise ValueError(f"Mixed sequences: {seq!r} vs {f.sequence_name!r}")
            if f.camera is None or f.camera.R.dim() != 3 or f.camera.R.shape[0] != 1:
                raise ValueError("Each frame must have a camera with batch=1")

        cameras = concat_cameras([f.camera for f in frames])
        frame_numbers = torch.tensor([f.frame_number for f in frames], dtype=torch.long)
        image_rgb = _stack([f.image_rgb for f in frames])
        masks = _stack([f.mask for f in frames])
        orig_sizes_hw = _stack([f.orig_size_hw for f in frames])
        seg_pc = next(
            (f.segmented_point_cloud for f in frames if f.segmented_point_cloud is not None),
            None,
        )
        return cls(
            sequence_name=seq,
            frame_numbers=frame_numbers,
            cameras=cameras,
            image_rgb=image_rgb,
            masks=masks,
            valid_masks=None,
            orig_sizes_hw=orig_sizes_hw,
            segmented_point_cloud=seg_pc,
        )


@dataclass
class Batch:
    sequences: List[SequenceSample]
    cameras: Camera
    image_rgb: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    valid_masks: Optional[torch.Tensor] = None
    orig_sizes_hw: Optional[torch.Tensor] = None
    sequence_names: List[str] = field(default_factory=list)
    seq_lengths: List[int] = field(default_factory=list)
    seq_offsets: List[int] = field(default_factory=list)
    obj_sizes: Optional[torch.Tensor] = None
    obj_centers: Optional[torch.Tensor] = None
    segmented_point_clouds: List[Any] = field(default_factory=list)
    annotations: Optional[Any] = None

    def __len__(self) -> int:
        return sum(self.seq_lengths)

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def num_frames(self) -> int:
        return sum(self.seq_lengths)

    def to(self, device=None, non_blocking: bool = False) -> "Batch":
        def mv(t):
            return None if t is None else t.to(device=device, non_blocking=non_blocking)

        return Batch(
            sequences=self.sequences,
            cameras=self.cameras.to(device=device, non_blocking=non_blocking),
            image_rgb=mv(self.image_rgb),
            masks=mv(self.masks),
            valid_masks=mv(self.valid_masks),
            orig_sizes_hw=mv(self.orig_sizes_hw),
            sequence_names=list(self.sequence_names),
            seq_lengths=list(self.seq_lengths),
            seq_offsets=list(self.seq_offsets),
            obj_sizes=mv(self.obj_sizes),
            obj_centers=mv(self.obj_centers),
            segmented_point_clouds=[
                pc if pc is None else pc.to(device=device) for pc in self.segmented_point_clouds
            ],
            annotations=None if self.annotations is None else self.annotations.to(device=device),
        )


def _stack(tensors):
    return torch.stack(tensors) if tensors[0] is not None else None
