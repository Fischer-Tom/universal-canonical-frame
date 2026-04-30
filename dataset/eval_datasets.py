from __future__ import annotations

import _pickle as cPickle
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from utils.camera import Camera

from .schema import SequenceSample


@dataclass
class PointCloud:
    xyz: torch.Tensor
    rgb: torch.Tensor
    faces: torch.Tensor | None = None

    def to(self, *args, **kwargs):
        return PointCloud(
            xyz=self.xyz.to(*args, **kwargs),
            rgb=self.rgb.to(*args, **kwargs),
            faces=self.faces.to(*args, **kwargs) if self.faces is not None else None,
        )


def load_off(off_file_name: str, to_torch: bool = False) -> tuple:
    with open(off_file_name) as file_handle:
        file_list = file_handle.readlines()
        n_points = int(file_list[1].split(" ")[0])
        all_strings = "".join(file_list[2 : 2 + n_points])
        verts = np.fromstring(all_strings, dtype=np.float32, sep="\n").reshape((-1, 3))
        all_strings = "".join(file_list[2 + n_points :])
        faces = np.fromstring(all_strings, dtype=np.int32, sep="\n").reshape((-1, 4))[
            :, 1:
        ]
    if to_torch:
        return torch.from_numpy(verts), torch.from_numpy(faces)
    return verts, faces


def _opencv_to_camera(R_cv, tvec, K, image_size_hw) -> Camera:
    R_cv = torch.as_tensor(R_cv, dtype=torch.float32)
    tvec = torch.as_tensor(tvec, dtype=torch.float32).reshape(3)
    K = torch.as_tensor(K, dtype=torch.float32)
    H, W = int(image_size_hw[0]), int(image_size_hw[1])

    cv_to_p3d = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32))
    R_row = R_cv.transpose(0, 1) @ cv_to_p3d
    T_row = tvec @ cv_to_p3d

    fx = K[0, 0] / (W / 2.0)
    fy = K[1, 1] / (H / 2.0)
    px = K[0, 2] / (W / 2.0) - 1.0
    py = 1.0 - K[1, 2] / (H / 2.0)
    return Camera(
        R=R_row.unsqueeze(0),
        T=T_row.unsqueeze(0),
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32),
        principal_point=torch.tensor([[px, py]], dtype=torch.float32),
        image_size=torch.tensor([[H, W]], dtype=torch.float32),
    )


def pascal3d_to_camera(
    azim, elev, theta, dist, px, py, viewport, img_h, img_w
) -> Camera:
    C = torch.tensor(
        [
            dist * math.cos(elev) * math.sin(azim),
            -dist * math.cos(elev) * math.cos(azim),
            dist * math.sin(elev),
        ],
        dtype=torch.float32,
    ).view(3, 1)

    azim2 = -azim
    elev2 = -(math.pi / 2.0 - elev)
    theta = theta + math.pi

    Rz = torch.tensor(
        [
            [math.cos(azim2), -math.sin(azim2), 0.0],
            [math.sin(azim2), math.cos(azim2), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    Rx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elev2), -math.sin(elev2)],
            [0.0, math.sin(elev2), math.cos(elev2)],
        ],
        dtype=torch.float32,
    )
    Rroll = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    R = Rroll @ Rx @ Rz
    t = (-R @ C).view(3)
    flip = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    R = flip @ R
    t = (flip @ t.view(3, 1)).view(3)
    K = torch.tensor(
        [[viewport, 0.0, px], [0.0, viewport, py], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return _opencv_to_camera(R, t, K, (img_h, img_w))


def _mesh_to_normalized_pc(verts, faces):
    if verts is None:
        size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        return None, size, center
    verts = torch.as_tensor(verts, dtype=torch.float32)
    high = torch.max(verts, dim=0).values
    low = torch.min(verts, dim=0).values
    size = high - low
    center = (high + low) * 0.5
    pc = PointCloud(xyz=verts - center[None, :], rgb=torch.ones_like(verts), faces=faces)
    pc.xyz /= torch.linalg.norm(size).clamp_min(1e-8)
    return pc, size, center


def _sequence_sample(
    *,
    sequence_name: str,
    camera: Camera,
    image_rgb: torch.Tensor,
    mask: torch.Tensor,
    point_cloud: PointCloud | None,
    object_size: torch.Tensor,
    obj_center: torch.Tensor,
) -> SequenceSample:
    H, W = image_rgb.shape[-2:]
    return SequenceSample(
        sequence_name=sequence_name,
        frame_numbers=torch.tensor([0], dtype=torch.long),
        cameras=camera,
        image_rgb=image_rgb.unsqueeze(0),
        masks=mask.unsqueeze(0),
        valid_masks=None,
        orig_sizes_hw=torch.tensor([[H, W]], dtype=torch.long),
        segmented_point_cloud=point_cloud,
        object_size=object_size.reshape(3),
        obj_center=obj_center.reshape(3),
    )


def _load_symmetries(data_root: str) -> dict[str, int]:
    symmetries: dict[str, int] = {}
    sym_path = os.path.join(data_root, "symmetry_labels.csv")
    if not os.path.exists(sym_path):
        return symmetries
    import csv

    with open(sym_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            category = clean.get("category")
            value = clean.get("symmetry_class") or clean.get("n_faces")
            if category is not None and value not in (None, ""):
                symmetries[category] = int(value)
    return symmetries


class ImageNet3D(Dataset):
    def __init__(self, data_root: str, split: str = "val", cate=None, **kwargs) -> None:
        self.data_root = data_root
        self.split = split
        categories = sorted(os.listdir(os.path.join(data_root, "annotations"))) if cate is None else cate
        self.categories = list(categories)
        self.symmetries = _load_symmetries(data_root)
        self.rotation_only = False
        self.samples: dict[str, list[str]] = {}
        split_suffix = "_imagenet_train.txt" if "train" in split else "_imagenet_val.txt"
        for category in self.categories:
            list_file = os.path.join(data_root, "lists", f"{category}{split_suffix}")
            if not os.path.exists(list_file):
                continue
            self.samples[category] = []
            with open(list_file, "r") as f:
                for line in f:
                    img_name = line.strip()
                    annot_path = os.path.join(data_root, "annotations", category, img_name + ".npz")
                    if img_name and os.path.exists(annot_path):
                        self.samples[category].append(img_name)
        self.setup_samples()

    def setup_samples(self, category_filter: Optional[list[Any]] = None):
        self.cur_samples = [
            (category, img_name)
            for category, img_names in self.samples.items()
            if category_filter is None or category in category_filter
            for img_name in img_names
        ]

    def __len__(self) -> int:
        return len(self.cur_samples)

    def __getitem__(self, idx: int) -> SequenceSample:
        category, img_name = self.cur_samples[idx]
        img_path = os.path.join(self.data_root, "images", category, img_name + ".JPEG")
        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())
        annot_path = os.path.join(self.data_root, "annotations", category, img_name + ".npz")
        annot_data = dict(np.load(annot_path, allow_pickle=True))["annotations"]
        annot = next((a for a in annot_data if "azimuth" in a), None)
        if annot is None:
            return self.__getitem__((idx + 1) % len(self))

        az = float(annot["azimuth"])
        el = float(annot["elevation"])
        ro = float(annot["theta"])
        distance = float(annot["distance"])
        px = float(annot.get("px", img_w / 2))
        py = float(annot.get("py", img_h / 2))
        viewport = float(annot.get("viewport", 3000))
        camera = pascal3d_to_camera(az, el, ro, distance, px, py, viewport, img_h, img_w)

        x1, y1, x2, y2 = [int(b) for b in annot["bbox"]]
        mask = np.zeros((1, img_h, img_w), dtype=np.uint8)
        mask[0, max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = 1
        pc, size, center = _mesh_to_normalized_pc(*self.get_cad_path(annot))
        return _sequence_sample(
            sequence_name=f"{category}/{img_name.replace('.JPEG', '')}",
            camera=camera,
            image_rgb=img_tensor,
            mask=torch.from_numpy(mask),
            point_cloud=pc,
            object_size=size,
            obj_center=center,
        )

    def get_cad_path(self, anno):
        category = anno["class"]
        cad_index = int(anno["cad_index"])
        cad_path = os.path.join(self.data_root, "CAD", category, f"{cad_index:02d}.off")
        return load_off(cad_path, to_torch=True)


class Pascal3D(Dataset):
    def __init__(self, data_root: str, split: str = "val", cate=None, **kwargs) -> None:
        self.data_root = data_root
        self.categories = list(cate) if cate is not None else [
            "aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair",
            "diningtable", "motorbike", "sofa", "train", "tvmonitor",
        ]
        self.rotation_only = False
        self.symmetries = _load_symmetries(data_root)
        self.setup_samples(self.categories)

    def setup_samples(self, categories):
        self.image_list = []
        for category in categories:
            set_list = os.path.join(self.data_root, "Image_sets", f"{category}_imagenet_val.txt")
            if not os.path.exists(set_list):
                continue
            with open(set_list, "r") as f:
                img_indices = [line.strip() for line in f.readlines()]
            self.image_list += [(category, img_idx) for img_idx in img_indices]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        try:
            category, img_idx = self.image_list[index]
            image_name = os.path.join(self.data_root, "Images", f"{category}_imagenet", f"{img_idx}.JPEG")
            anno = image_name.replace("Images", "Annotations").replace(".JPEG", ".mat")
            img = np.array(Image.open(image_name).convert("RGB"))
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1).copy())
            record = sio.loadmat(anno)["record"]
            distance, az, el, ro, px, py, bbox = self.get_annotation(record)
            img_h, img_w = record["imgsize"][0][0][0][0:2]
            camera = pascal3d_to_camera(az, el, ro, distance, px, py, 3000.0, img_h, img_w)
            x1, y1, x2, y2 = [int(b) for b in bbox]
            mask = np.zeros((1, img_h, img_w), dtype=np.uint8)
            mask[0, max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = 1
            pc, size, center = _mesh_to_normalized_pc(*self.get_cad_path(category))
            return _sequence_sample(
                sequence_name=f"{category}/{img_idx}",
                camera=camera,
                image_rgb=img_tensor,
                mask=torch.from_numpy(mask),
                point_cloud=pc,
                object_size=size,
                obj_center=center,
            )
        except Exception as e:
            print(f"Error loading index {index}: {e}")
            return self.__getitem__(0)

    def get_annotation(self, record):
        objects = record["objects"]
        viewpoint = objects[0][0]["viewpoint"][0][0]
        azimuth = viewpoint["azimuth"].item()[0]
        elevation = viewpoint["elevation"].item()[0]
        theta = viewpoint["theta"].item()[0]
        distance = viewpoint["distance"].item()[0].astype(np.float32)
        bbox = objects[0][0]["bbox"][0][0][0]
        if distance == 0:
            distance = np.array([5])
            azimuth = viewpoint["azimuth_coarse"].item()[0]
            elevation = viewpoint["elevation_coarse"].item()[0]
        pose = np.array(
            [distance, azimuth * math.pi / 180.0, elevation * math.pi / 180.0, theta * math.pi / 180.0],
            dtype=np.float32,
        ).squeeze(-1)
        return pose[0], pose[1], pose[2], pose[3], viewpoint["px"].item()[0][0], viewpoint["py"].item()[0][0], bbox

    def get_cad_path(self, category):
        try:
            return load_off(os.path.join(self.data_root, "CAD", category, "01.off"), to_torch=True)
        except Exception:
            return None, None


def quat_wxyz_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class OmniPose6D(Dataset):
    def __init__(self, data_root: str, split: str = "test", cate=None, **kwargs) -> None:
        self.data_root = data_root
        self.rotation_only = False
        self.symmetries = _load_symmetries(data_root)
        with open(os.path.join(data_root, "Meta", "real_obj_meta.json"), "r") as f:
            obj_meta = json.load(f)
            self.categories = [c["name"] for c in obj_meta["class_list"]]
        category_filter = set(cate) if cate is not None else None
        self.all_samples = []
        self.samples_by_category = {cat: [] for cat in self.categories}
        sample_dir = os.path.join(data_root, "ROPE_sampled")
        for scene_id in sorted(d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d)) and d.isdigit()):
            meta_dir = os.path.join(sample_dir, scene_id, "meta")
            if not os.path.isdir(meta_dir):
                continue
            for fname in sorted(os.listdir(meta_dir)):
                if not fname.endswith(".json"):
                    continue
                frame_id = fname.replace(".json", "")
                try:
                    with open(os.path.join(meta_dir, fname), "r") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                for obj_key, obj in meta.get("objects", {}).items():
                    if not obj.get("is_valid", True):
                        continue
                    cat = obj["meta"]["class_name"]
                    if category_filter is not None and cat not in category_filter:
                        continue
                    entry = (scene_id, frame_id, obj_key, cat)
                    self.all_samples.append(entry)
                    if cat in self.samples_by_category:
                        self.samples_by_category[cat].append(entry)

        mesh_root = os.path.join(data_root, "PAM", "object_meshes")
        self._mesh_by_category = defaultdict(list)
        if os.path.isdir(mesh_root):
            for d in os.listdir(mesh_root):
                parts = d.split("-", 1)
                if len(parts) == 2:
                    cat = parts[1].rsplit("_", 1)[0]
                    self._mesh_by_category[cat].append(os.path.join(mesh_root, d, "Aligned.obj"))
        self.setup_samples()
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    def setup_samples(self, category_filter=None):
        if category_filter is None:
            self.cur_samples = list(self.all_samples)
            self.categories = sorted(list(set([d[-1] for d in self.all_samples])))
        else:
            self.cur_samples = [s for s in self.all_samples if s[3] in category_filter]

    def __len__(self):
        return len(self.cur_samples)

    def __getitem__(self, index):
        scene_id, frame_id, obj_key, category = self.cur_samples[index]
        base_dir = os.path.join(self.data_root, "ROPE_sampled", scene_id)
        frame_id = frame_id.split("_")[0]
        rgb = np.array(Image.open(os.path.join(base_dir, "color", f"{frame_id}_color.png")).convert("RGB"))
        img_h, img_w = rgb.shape[:2]
        with open(os.path.join(base_dir, "meta", f"{frame_id}_meta.json"), "r") as f:
            meta = json.load(f)
        obj = meta["objects"][obj_key]
        R = quat_wxyz_to_rotation_matrix(obj["quaternion_wxyz"])
        T = np.array(obj["translation"], dtype=np.float32)
        intr = meta["camera"]["intrinsics"]
        scale_x = img_w / intr["width"]
        scale_y = img_h / intr["height"]
        K = np.array(
            [[intr["fx"] * scale_x, 0, intr["cx"] * scale_x], [0, intr["fy"] * scale_y, intr["cy"] * scale_y], [0, 0, 1]],
            dtype=np.float32,
        )
        camera = _opencv_to_camera(R, T, K, (img_h, img_w))
        mask_path = os.path.join(base_dir, "mask", f"{frame_id}_mask.exr")
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_raw is not None:
            mask_raw = np.round(mask_raw * 255)
            if mask_raw.ndim == 3:
                mask_raw = mask_raw[:, :, 0]
            obj_mask = (np.round(mask_raw).astype(np.int32) == obj["id"]).astype(np.uint8)
        else:
            obj_mask = np.ones((img_h, img_w), dtype=np.uint8)
        pc, size, center = _mesh_to_normalized_pc(*self.get_cad_path(category))
        object_size = torch.tensor(obj["meta"]["bbox_side_len"], dtype=torch.float32)
        return _sequence_sample(
            sequence_name=f"{category}/{scene_id}_{frame_id}_{obj_key}",
            camera=camera,
            image_rgb=torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            mask=torch.from_numpy(obj_mask[None]),
            point_cloud=pc,
            object_size=object_size,
            obj_center=center,
        )

    def get_cad_path(self, category):
        candidates = self._mesh_by_category.get(category, [])
        if not candidates:
            return None, None
        try:
            from pytorch3d.io import load_obj

            verts, faces, _ = load_obj(random.choice(candidates))
            return verts, faces.verts_idx
        except Exception:
            return None, None


class AbsOriBench(Dataset):
    DATA_ROOT = "/hnvme/OriAnyV2_Inference"

    def __init__(self, data_root: str, split: str = "objectron", cate=None, **kwargs) -> None:
        base_root = data_root or self.DATA_ROOT
        meta_paths = {
            "objectron": (
                os.path.join(base_root, "objectron", "objectron_meta.csv"),
                os.path.join(base_root, "objectron", "objectron_data", "test_crop"),
            ),
            "sunrgbd": (
                os.path.join(base_root, "SUNRGBD_test", "sunrgbd_meta.csv"),
                os.path.join(base_root, "SUNRGBD_test", "test_crop"),
            ),
            "arkitscenes": (
                os.path.join(base_root, "ARKitScenes", "arkitscenes_meta.csv"),
                os.path.join(base_root, "ARKitScenes", "test_crop"),
            ),
        }
        meta_path, image_root = meta_paths[split]
        flat_meta = os.path.join(base_root, f"{split}_meta.csv")
        if os.path.exists(flat_meta):
            meta_path = flat_meta
        flat_image_root = os.path.join(base_root, "test_crop")
        if os.path.isdir(flat_image_root):
            image_root = flat_image_root
        self.data_root = data_root
        self.meta = pd.read_csv(meta_path)
        self.image_root = image_root
        self.categories = self.meta["category_name"].unique().tolist()
        if cate is not None:
            self.categories = list(cate)
            self.meta = self.meta[self.meta["category_name"].isin(self.categories)]
        self.rotation_only = True
        self.cate_meta = None
        self.symmetries = _load_symmetries(data_root)

    def setup_samples(self, categories):
        self.cate_meta = self.meta[self.meta["category_name"].isin(categories)] if categories is not None else self.meta

    def __len__(self):
        return len(self.cate_meta) if self.cate_meta is not None else len(self.meta)

    def __getitem__(self, index):
        meta = self.cate_meta if self.cate_meta is not None else self.meta
        try:
            meta_info = meta.iloc[index]
            img_idx = meta_info["img_idx"]
            rgb = np.array(Image.open(os.path.join(self.image_root, f"{img_idx}.jpg")).convert("RGB"))
            img_h, img_w = rgb.shape[:2]
            az = torch.deg2rad(torch.tensor(meta_info["azi"], dtype=torch.float32)).item()
            el = torch.deg2rad(torch.tensor(meta_info["ele"], dtype=torch.float32)).item()
            ro = -torch.deg2rad(torch.tensor(meta_info["rot"], dtype=torch.float32)).item()
            camera = pascal3d_to_camera(az, el, ro, 5.0, img_w / 2.0, img_h / 2.0, 3000.0, img_h, img_w)
            category = meta_info["category_name"]
            pc, size, center = _mesh_to_normalized_pc(*self.get_cad_path(category))
            return _sequence_sample(
                sequence_name=f"{category}/{img_idx}",
                camera=camera,
                image_rgb=torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
                mask=torch.ones(1, img_h, img_w, dtype=torch.uint8),
                point_cloud=pc,
                object_size=size,
                obj_center=center,
            )
        except Exception as e:
            print(f"Error loading index {index}: {e}")
            return self.__getitem__(0)

    def get_cad_path(self, category):
        try:
            cat = category[:-1] if category.endswith("s") else category
            return load_off(os.path.join("/hnvme/imagenet3d/imagenet3d_v1", "CAD", cat, "01.off"), to_torch=True)
        except Exception:
            try:
                from pytorch3d.io import load_obj

                verts, faces, _ = load_obj(os.path.join("/hnvme/random_mesh", f"{category}.obj"))
                verts = verts[:, [1, 0, 2]]
                return verts, faces.verts_idx
            except Exception:
                return None, None


Objectron = AbsOriBench
Omni6DPose = OmniPose6D


CATEGORY_NAMES = ["bottle", "bowl", "camera", "can", "laptop", "mug"]


class Real275(Dataset):
    def __init__(self, data_root: str, split: str = "test", cate=None, **kwargs) -> None:
        self.data_root = data_root
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.categories = list(CATEGORY_NAMES) if cate is None else list(cate)
        self.sym_ids = {0: 0, 1: 0, 3: 0}
        self.symmetries = {"bottle": 0, "bowl": 0, "can": 0}
        self.rotation_only = False
        with open(os.path.join(data_root, "real_test_list.txt")) as f:
            self.img_list = [line.rstrip("\n") for line in f]
        self.all_samples = []
        self.samples_by_category = {cat: [] for cat in CATEGORY_NAMES}
        for scene_path in self.img_list:
            pkl_path = os.path.join(data_root, scene_path + "_label.pkl")
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                gts = cPickle.load(f)
            for inst_idx, cls_id in enumerate(gts["class_ids"]):
                cls_0 = cls_id - 1
                if 0 <= cls_0 < len(CATEGORY_NAMES):
                    cat = CATEGORY_NAMES[cls_0]
                    entry = (scene_path, inst_idx, cat)
                    self.all_samples.append(entry)
                    self.samples_by_category[cat].append(entry)
        self.setup_samples(self.categories)

    def setup_samples(self, category_filter=None):
        self.cur_samples = []
        for cat, samples in self.samples_by_category.items():
            if category_filter is None or cat in category_filter:
                self.cur_samples.extend(samples)

    def __len__(self):
        return len(self.cur_samples)

    def __getitem__(self, index):
        scene_path, inst_idx, category = self.cur_samples[index]
        full_path = os.path.join(self.data_root, scene_path)
        rgb = np.array(Image.open(full_path + "_color.png").convert("RGB"))
        img_h, img_w = rgb.shape[:2]
        with open(full_path + "_label.pkl", "rb") as f:
            gts = cPickle.load(f)
        class_id = gts["class_ids"][inst_idx] - 1
        R = gts["rotations"][inst_idx].astype(np.float32).copy()
        T = gts["translations"][inst_idx].astype(np.float32).copy().squeeze()
        if class_id in self.sym_ids:
            R = self._canonicalize_rotation(R, self.sym_ids[class_id])
        y1, x1, y2, x2 = [int(v) for v in gts["bboxes"][inst_idx]]
        mask = np.zeros((1, img_h, img_w), dtype=np.uint8)
        mask[0, max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = 1
        fx, fy, cx, cy = self.real_intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        camera = _opencv_to_camera(R, T, K, (img_h, img_w))
        pc, _size, _center = _mesh_to_normalized_pc(*self.get_cad_path(category))
        return _sequence_sample(
            sequence_name=f"{category}/{scene_path}_{inst_idx}",
            camera=camera,
            image_rgb=torch.from_numpy(rgb.transpose(2, 0, 1).copy()),
            mask=torch.from_numpy(mask),
            point_cloud=pc,
            object_size=torch.ones(3),
            obj_center=torch.zeros(3),
        )

    def _canonicalize_rotation(self, rotation, n_faces=0):
        if n_faces == 0:
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            if r_norm == 0:
                return rotation
            s_map = np.array(
                [[theta_x / r_norm, 0.0, -theta_y / r_norm], [0.0, 1.0, 0.0], [theta_y / r_norm, 0.0, theta_x / r_norm]],
                dtype=np.float32,
            )
            return rotation if np.any(np.isnan(s_map)) else rotation @ s_map
        theta_x = rotation[0, 0] + rotation[2, 2]
        theta_y = rotation[0, 2] - rotation[2, 0]
        y_angle = math.atan2(theta_y, theta_x)
        sector = 2.0 * math.pi / n_faces
        snapped = round(y_angle / sector) * sector
        c, s = math.cos(snapped), math.sin(snapped)
        R_y_inv = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
        return rotation @ R_y_inv

    def get_cad_path(self, category):
        try:
            mesh = "03.off" if category == "laptop" else "01.off"
            cad_category = "cup" if category == "mug" else category
            return load_off(os.path.join("/hnvme/imagenet3d/imagenet3d_v1", "CAD", cad_category, mesh), to_torch=True)
        except Exception:
            return None, None
