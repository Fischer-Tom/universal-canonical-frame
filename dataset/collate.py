from __future__ import annotations

from typing import List, Optional

import torch

from utils.camera import concat_cameras

from .schema import Batch, SequenceSample


class EmptyBatchError(RuntimeError):
    pass


def sequences_to_batch(samples: List[SequenceSample]) -> Batch:
    if not samples:
        raise EmptyBatchError("sequences_to_batch received no samples")

    seq_lengths = [s.num_frames for s in samples]
    seq_offsets: List[int] = []
    running = 0
    for length in seq_lengths:
        seq_offsets.append(running)
        running += length

    return Batch(
        sequences=samples,
        cameras=concat_cameras([s.cameras for s in samples]),
        image_rgb=_cat([s.image_rgb for s in samples]),
        masks=_cat([s.masks for s in samples]),
        valid_masks=_cat([s.valid_masks for s in samples]),
        orig_sizes_hw=_cat([s.orig_sizes_hw for s in samples]),
        sequence_names=[s.sequence_name for s in samples],
        seq_lengths=seq_lengths,
        seq_offsets=seq_offsets,
        obj_sizes=_per_frame([s.object_size for s in samples], seq_lengths),
        obj_centers=_per_frame([s.obj_center for s in samples], seq_lengths),
        segmented_point_clouds=[s.segmented_point_cloud for s in samples],
    )


def safe_collate(samples: List[Optional[SequenceSample]]) -> Batch:
    kept = [s for s in samples if s is not None]
    if not kept:
        raise EmptyBatchError(f"All {len(samples)} samples were dropped")
    return sequences_to_batch(kept)


def _cat(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    return None if any(t is None for t in tensors) else torch.cat(tensors, dim=0)


def _per_frame(
    per_seq: List[Optional[torch.Tensor]], seq_lengths: List[int]
) -> Optional[torch.Tensor]:
    if any(t is None for t in per_seq):
        return None
    parts = []
    for vec, length in zip(per_seq, seq_lengths):
        v = vec if vec.dim() == 2 else vec.unsqueeze(0)
        parts.append(v.expand(length, -1))
    return torch.cat(parts, dim=0)
