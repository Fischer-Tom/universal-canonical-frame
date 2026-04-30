from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torchvision.transforms.functional as TVF

from utils.camera import Camera

from ..collate import EmptyBatchError, safe_collate, sequences_to_batch
from ..data_loader import (
    SampleTransform,
    SampleTransformCfg,
    _rotate_on_spot,
    _rotate_principal_point_on_image_,
    _rotz_batch,
)
from ..sampling import stratified_sample
from ..schema import Batch, SequenceSample
from ..transforms import MissingPointCloudError, align_point_cloud, normalize_sequence_to_bbox


H, W = 4, 6
K = 3


@dataclass
class _FakePC:
    xyz: torch.Tensor

    def to(self, *, device=None):
        return _FakePC(xyz=self.xyz.to(device=device))


@dataclass
class _FakeAlignment:
    R: torch.Tensor
    T: torch.Tensor
    scale: float


@dataclass
class _FakeSequenceOrm:
    alignment: Any = None


def _cameras(batch: int = 1) -> Camera:
    return Camera(
        R=torch.eye(3).unsqueeze(0).expand(batch, -1, -1).clone(),
        T=torch.zeros(batch, 3),
        focal_length=torch.tensor([[1.0, 1.0]]).expand(batch, -1).clone(),
        principal_point=torch.zeros(batch, 2),
    )


def _sample(seq: str, k: int = K, with_pc: bool = True) -> SequenceSample:
    pc = _FakePC(xyz=torch.randn(64, 3)) if with_pc else None
    return SequenceSample(
        sequence_name=seq,
        frame_numbers=torch.arange(k, dtype=torch.long),
        cameras=_cameras(k),
        image_rgb=torch.zeros(k, 3, H, W, dtype=torch.uint8),
        masks=torch.zeros(k, 1, H, W, dtype=torch.uint8),
        orig_sizes_hw=torch.tensor([[H, W]] * k, dtype=torch.long),
        segmented_point_cloud=pc,
        object_size=torch.tensor([1.0, 2.0, 3.0]),
        obj_center=torch.tensor([0.1, 0.2, 0.3]),
    )


def test_collate_shapes_regression_obj_sizes():
    samples = [_sample(f"s{i}") for i in range(3)]
    batch = sequences_to_batch(samples)
    NF = sum(s.num_frames for s in samples)
    assert batch.image_rgb.shape == (NF, 3, H, W)
    assert batch.masks.shape == (NF, 1, H, W)
    assert batch.cameras.R.shape == (NF, 3, 3)
    assert batch.seq_lengths == [K, K, K]
    assert batch.seq_offsets == [0, K, 2 * K]
    assert batch.obj_sizes is not None and batch.obj_sizes.shape == (NF, 3)
    assert batch.obj_centers is not None and batch.obj_centers.shape == (NF, 3)
    assert batch.sequence_names == ["s0", "s1", "s2"]


def test_safe_collate_drops_none():
    batch = safe_collate([_sample("s0"), None, _sample("s1")])
    assert batch.num_sequences == 2


def test_safe_collate_all_bad_raises():
    with pytest.raises(EmptyBatchError):
        safe_collate([None, None])


def test_device_move_cpu_roundtrip():
    batch = sequences_to_batch([_sample("s0")])
    moved = batch.to(device="cpu")
    assert moved.cameras.R.device.type == "cpu"
    assert moved.image_rgb.dtype == torch.uint8


def test_normalize_raises_without_pointcloud():
    with pytest.raises(MissingPointCloudError):
        normalize_sequence_to_bbox(_sample("s0", with_pc=False))


def test_normalize_sets_object_fields():
    sample = _sample("s0")
    out = normalize_sequence_to_bbox(sample)
    assert out.object_size.shape == (3,)
    assert out.obj_center.shape == (3,)
    assert out.segmented_point_cloud.xyz.shape == (64, 3)


def test_align_point_cloud_before_normalize():
    pc = _FakePC(xyz=torch.tensor([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ]))
    seq = _FakeSequenceOrm(
        alignment=_FakeAlignment(
            R=torch.eye(3),
            T=torch.tensor([1.0, 0.0, 0.0]),
            scale=2.0,
        )
    )
    sample = _sample("s0")
    sample.segmented_point_cloud = align_point_cloud(pc, seq)

    out = normalize_sequence_to_bbox(sample, q_low=0.0, q_high=1.0)
    assert torch.allclose(out.object_size, torch.tensor([20.0, 0.0, 0.0]))
    assert torch.allclose(out.obj_center, torch.tensor([12.0, 0.0, 0.0]))
    assert torch.allclose(
        out.segmented_point_cloud.xyz,
        torch.tensor([
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]),
    )


def test_stratified_sample_is_deterministic():
    a = stratified_sample(30, 5, random.Random(7))
    b = stratified_sample(30, 5, random.Random(7))
    assert a == b
    assert len(a) == 5 and a == sorted(a)


def test_annotations_slot_is_typed():
    @dataclass
    class _Ann:
        nocs: torch.Tensor
        depth: torch.Tensor
        geom_mask: torch.Tensor
        obj_xyz: torch.Tensor
        obj_idx: torch.Tensor
        cameras: Any

        def to(self, device=None, non_blocking: bool = False):
            return _Ann(
                nocs=self.nocs.to(device=device),
                depth=self.depth.to(device=device),
                geom_mask=self.geom_mask.to(device=device),
                obj_xyz=self.obj_xyz.to(device=device),
                obj_idx=self.obj_idx.to(device=device),
                cameras=self.cameras,
            )

    batch = sequences_to_batch([_sample("s0")])
    NF = batch.num_frames
    batch.annotations = _Ann(
        nocs=torch.zeros(NF, 3, H, W),
        depth=torch.zeros(NF, H, W),
        geom_mask=torch.zeros(NF, H, W, dtype=torch.bool),
        obj_xyz=torch.zeros(NF, H, W, 3),
        obj_idx=torch.zeros(NF, dtype=torch.long),
        cameras=batch.cameras,
    )
    moved = batch.to(device="cpu")
    assert moved.annotations is not None
    assert moved.annotations.nocs.shape == (NF, 3, H, W)


def test_geometric_augmentation_waits_for_start_iter():
    transform = SampleTransform(SampleTransformCfg(
        size_hw=(H, W),
        crop_to_mask=False,
        geometric_start_iter=10,
        recrop_p=1.0,
        recrop_scale=(0.5, 0.5),
        recrop_ratio=(1.0, 1.0),
        recrop_min_size=2,
        rotate_p=0.0,
        patch_mask_p=0.0,
        blur_p=0.0,
        normalize=False,
        training=True,
    ))

    transform.set_iteration(9)
    before = transform(_sample("before", k=1))
    assert before.orig_sizes_hw.tolist() == [[H, W]]

    transform.set_iteration(10)
    after = transform(_sample("after", k=1))
    assert after.orig_sizes_hw.tolist() != [[H, W]]


def test_occlusion_augmentation_waits_for_start_iter():
    sample = _sample("occlusion", k=1)
    sample.image_rgb.fill_(255)
    transform = SampleTransform(SampleTransformCfg(
        size_hw=(H, W),
        crop_to_mask=False,
        geometric_start_iter=0,
        recrop_p=0.0,
        rotate_p=0.0,
        occlusion_start_iter=5,
        patch_mask_p=1.0,
        patch_mask_num=(1, 1),
        patch_mask_size=(1.0, 1.0),
        blur_p=0.0,
        normalize=False,
        training=True,
    ))

    transform.set_iteration(4)
    before = transform(sample)
    assert before.valid_masks.sum().item() == H * W

    transform.set_iteration(5)
    after = transform(sample)
    assert after.valid_masks.sum().item() < H * W


def test_camera_inplane_rotation_matches_torchvision_rotate_direction():
    h = w = 101
    center = (w - 1) / 2
    scale = (w - 1) / 2
    angle = 30.0
    u0, v0 = 70, 40

    img = torch.zeros(1, h, w)
    img[0, v0, u0] = 1.0
    img_rot = TVF.rotate(
        img,
        angle=angle,
        interpolation=TVF.InterpolationMode.NEAREST,
        expand=False,
        fill=0,
    )
    ys, xs = (img_rot[0] > 0).nonzero(as_tuple=True)
    uv_rot = torch.stack([xs.float().mean(), ys.float().mean()])

    x_ndc = (center - u0) / scale
    y_ndc = (center - v0) / scale
    pt = torch.tensor([[x_ndc, y_ndc, 1.0]])
    cam = Camera(
        R=torch.eye(3).unsqueeze(0),
        T=torch.zeros(1, 3),
        focal_length=torch.ones(1, 2),
        principal_point=torch.zeros(1, 2),
        image_size=torch.tensor([[h, w]], dtype=torch.float32),
    )
    Rz = _rotz_batch(torch.tensor([angle]))
    R, T = _rotate_on_spot(cam.R, cam.T, Rz)
    cam.R, cam.T = R, T
    ndc = cam.transform_points(pt)[0, 0]
    uv_cam = torch.tensor([center - ndc[0] * scale, center - ndc[1] * scale])

    assert torch.allclose(uv_cam, uv_rot, atol=1.0)


def test_camera_inplane_rotation_rotates_principal_point():
    cam = Camera(
        R=torch.eye(3).unsqueeze(0),
        T=torch.zeros(1, 3),
        focal_length=torch.ones(1, 2),
        principal_point=torch.tensor([[0.2, -0.1]]),
        image_size=torch.tensor([[64, 64]], dtype=torch.float32),
    )
    Rz = _rotz_batch(torch.tensor([90.0]))
    _rotate_principal_point_on_image_(cam, Rz)

    assert torch.allclose(cam.principal_point, torch.tensor([[-0.1, -0.2]]), atol=1e-6)
