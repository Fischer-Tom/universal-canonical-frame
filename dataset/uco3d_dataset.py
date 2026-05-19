from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import sqlalchemy as sa
import torch
from torch.utils.data import Dataset

from uco3d.dataset_utils.data_types import PointCloud
from uco3d.dataset_utils.io_utils import load_image, load_mask, load_point_cloud
from uco3d.dataset_utils.orm_types import (
    UCO3DFrameAnnotation,
    UCO3DSequenceAnnotation,
)
from uco3d.dataset_utils.utils import LruCacheWithCleanup, get_dataset_root

from utils.camera import Camera

from .sampling import stratified_sample
from .schema import FrameRecord, SequenceSample
from .transforms import (
    MissingPointCloudError,
    align_frame,
    align_point_cloud,
    normalize_sequence_to_bbox,
    undistort_frame,
)

logger = logging.getLogger(__name__)


class UCO3DDataset(Dataset):
    """Sequence-level UCO3D dataset.

    Each ``__getitem__(i)`` returns a ``SequenceSample`` with K frames sampled
    stratified-by-time from sequence ``i``. Image+mask are undistorted with
    COLMAP coefficients, the camera is rewritten to rectified intrinsics, the
    sequence similarity transform is applied, and (by default) the point cloud
    bbox is used to center+scale the object.

    Bad sequences (missing video frame / point cloud / etc.) return ``None``
    when ``skip_bad=True``; use :func:`dataset.collate.safe_collate` to filter
    them out. No recursive retry.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        data_root: Optional[str] = None,
        subset_lists_file: Optional[str] = None,
        categories: Optional[List[str]] = None,
        frames_per_sequence: int = 4,
        load_images: bool = True,
        load_masks: bool = True,
        load_point_clouds: bool = True,
        apply_alignment: bool = True,
        normalize_object: bool = True,
        remove_empty_masks: bool = True,
        limit_sequences: int = 0,
        seed: int = 0,
        skip_bad: bool = True,
        undistort_alpha: float = 0.5,
        video_cache_size: int = 12,
        point_cloud_cache_size: int = 16,
        frame_orm_cache_size: int = 1024,
    ) -> None:
        self.data_root = data_root or get_dataset_root(assert_exists=True)
        meta = os.path.join(self.data_root, "metadata.sqlite")
        if not os.path.exists(meta):
            raise FileNotFoundError(f"UCO3D metadata.sqlite not found at {meta}")
        self._meta_path = meta

        if subset_lists_file is None:
            subset_lists_file = os.path.join(
                self.data_root, "set_lists", "set_lists_all-categories.sqlite"
            )
        self.subset_lists_file = subset_lists_file
        self.subsets = ["train" if "train" in split else "val"]
        self.categories = categories
        self.frames_per_sequence = frames_per_sequence
        self.load_images = load_images
        self.load_masks = load_masks
        self.load_point_clouds = load_point_clouds
        self.apply_alignment = apply_alignment
        self.normalize_object = normalize_object
        self.remove_empty_masks = remove_empty_masks
        self.limit_sequences = limit_sequences
        self.seed = seed
        self.epoch = 0
        self.skip_bad = skip_bad
        self.undistort_alpha = undistort_alpha

        self._engine: Optional[sa.engine.Engine] = None
        self._frames_by_seq: Dict[str, List[Tuple[int, float]]] = {}
        self._sequence_list: List[str] = []

        self._frame_cache: "OrderedDict[Tuple[str, int], UCO3DFrameAnnotation]" = OrderedDict()
        self._frame_cache_cap = frame_orm_cache_size
        self._seq_cache: Dict[str, UCO3DSequenceAnnotation] = {}
        self._video_cache = LruCacheWithCleanup[str, cv2.VideoCapture](
            create_fn=cv2.VideoCapture,
            cleanup_fn=lambda cap: cap.release(),
            max_size=video_cache_size,
        )
        self._pcl_cache = LruCacheWithCleanup[str, PointCloud](
            create_fn=self._load_pointcloud_nocache,
            cleanup_fn=lambda _: None,
            max_size=point_cloud_cache_size,
        )

        self._build_index()
        if not self._frames_by_seq:
            raise ValueError("UCO3D index is empty after filtering")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._sequence_list)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def sequence_names(self) -> List[str]:
        return list(self._sequence_list)

    def __getitem__(self, idx: int) -> Optional[SequenceSample]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        try:
            return self._build_sample(self._sequence_list[idx])
        except (FileNotFoundError, MissingPointCloudError, ValueError) as exc:
            if self.skip_bad:
                logger.warning("Skipping bad sequence %s: %s", self._sequence_list[idx], exc)
                return None
            raise

    def get_by_name(self, seq_name: str) -> SequenceSample:
        if seq_name not in self._frames_by_seq:
            raise KeyError(seq_name)
        return self._build_sample(seq_name)

    # ------------------------------------------------------------------
    # Sample construction
    # ------------------------------------------------------------------
    def _build_sample(self, seq_name: str) -> SequenceSample:
        positions = self._frames_by_seq[seq_name]
        seed_bytes = f"{self.seed}:{self.epoch}:{seq_name}".encode()
        rng = random.Random(int.from_bytes(hashlib.blake2b(seed_bytes, digest_size=8).digest(), "big"))
        k = max(1, min(self.frames_per_sequence, len(positions)))
        picks = stratified_sample(len(positions), k, rng, jitter=True)
        frame_numbers = [positions[i][0] for i in picks]

        seq_orm = self._fetch_seq(seq_name)
        frames = [self._build_frame(seq_name, fn, seq_orm) for fn in frame_numbers]
        sample = SequenceSample.from_frames(frames)
        point_cloud = self._load_pointcloud(seq_orm)
        if self.apply_alignment:
            point_cloud = align_point_cloud(point_cloud, seq_orm)
        sample.segmented_point_cloud = point_cloud
        if self.normalize_object:
            sample = normalize_sequence_to_bbox(sample)
        return sample

    def _build_frame(
        self, seq_name: str, frame_number: int, seq_orm: UCO3DSequenceAnnotation
    ) -> FrameRecord:
        fa = self._fetch_frame(seq_name, frame_number)

        record = FrameRecord(sequence_name=seq_name, frame_number=int(frame_number))
        camera, distortion = self._make_camera(fa)
        record.camera = camera
        record.image_rgb = self._load_image(fa, seq_orm) if self.load_images else None
        record.mask = self._load_mask(fa, seq_orm) if self.load_masks else None

        if record.image_rgb is not None:
            record = undistort_frame(record, distortion, alpha=self.undistort_alpha)
        if self.apply_alignment:
            record = align_frame(record, seq_orm)
        return record

    # ------------------------------------------------------------------
    # Camera / image / mask loading
    # ------------------------------------------------------------------
    def _make_camera(self, fa: UCO3DFrameAnnotation) -> Tuple[Camera, torch.Tensor]:
        vp = fa.viewpoint
        if vp is None:
            raise ValueError(f"Frame {fa.sequence_name}/{fa.frame_number} missing viewpoint")
        camera = Camera(
            R=torch.tensor(vp.R, dtype=torch.float)[None],
            T=torch.tensor(vp.T, dtype=torch.float)[None],
            focal_length=torch.tensor(vp.focal_length, dtype=torch.float)[None],
            principal_point=torch.tensor(vp.principal_point, dtype=torch.float)[None],
        )
        distortion = torch.tensor(vp.colmap_distortion_coeffs, dtype=torch.float)
        return camera, distortion

    def _load_image(
        self, fa: UCO3DFrameAnnotation, seq: UCO3DSequenceAnnotation
    ) -> Optional[torch.Tensor]:
        if fa.image is None:
            return None
        if seq.video is not None:
            arr = self._frame_from_video(
                os.path.join(self.data_root, seq.video.path), fa.frame_timestamp
            )
            if arr is None:
                raise FileNotFoundError(
                    f"RGB frame missing for {fa.sequence_name} @ {fa.frame_timestamp:.3f}s"
                )
        else:
            arr = load_image(os.path.join(self.data_root, fa.image.path))
            if arr.ndim == 3 and arr.shape[0] != 3:
                arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _load_mask(
        self, fa: UCO3DFrameAnnotation, seq: UCO3DSequenceAnnotation
    ) -> Optional[torch.Tensor]:
        if fa.mask is None:
            return None
        if seq.mask_video is not None:
            arr = self._frame_from_video(
                os.path.join(self.data_root, seq.mask_video.path), fa.frame_timestamp
            )
            if arr is None:
                raise FileNotFoundError(
                    f"Mask frame missing for {fa.sequence_name} @ {fa.frame_timestamp:.3f}s"
                )
            arr = arr[:1]
        else:
            arr = load_mask(os.path.join(self.data_root, fa.mask.path))
            if arr.ndim == 2:
                arr = arr[None]
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _load_pointcloud(self, seq: UCO3DSequenceAnnotation) -> Optional[PointCloud]:
        if not self.load_point_clouds:
            return None
        meta = getattr(seq, "segmented_point_cloud", None)
        path = getattr(meta, "path", None) if meta is not None else None
        if path is None:
            warnings.warn(f"Sequence {seq.sequence_name} has no segmented point cloud path")
            return None
        return deepcopy(self._pcl_cache[os.path.join(self.data_root, path)])

    def _load_pointcloud_nocache(self, full_path: str) -> PointCloud:
        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)
        return load_point_cloud(full_path)

    def _frame_from_video(self, video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(video_path)
        cap = self._video_cache[video_path]
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp_sec)) * 1000.0)
        ok, bgr = cap.read()
        if not ok:
            return None
        rgb = bgr[..., ::-1]
        return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    # ------------------------------------------------------------------
    # ORM caches
    # ------------------------------------------------------------------
    def _fetch_seq(self, name: str) -> UCO3DSequenceAnnotation:
        hit = self._seq_cache.get(name)
        if hit is not None:
            return hit
        with sa.orm.Session(self._sql_engine()) as ses:
            obj = ses.scalars(
                sa.select(UCO3DSequenceAnnotation).where(
                    UCO3DSequenceAnnotation.sequence_name == name
                )
            ).one()
        self._seq_cache[name] = obj
        return obj

    def _fetch_frame(self, seq: str, frame: int) -> UCO3DFrameAnnotation:
        key = (seq, int(frame))
        if key in self._frame_cache:
            self._frame_cache.move_to_end(key)
            return self._frame_cache[key]
        with sa.orm.Session(self._sql_engine()) as ses:
            obj = ses.scalars(
                sa.select(UCO3DFrameAnnotation).where(
                    UCO3DFrameAnnotation.sequence_name == seq,
                    UCO3DFrameAnnotation.frame_number == int(frame),
                )
            ).one()
        self._frame_cache[key] = obj
        if len(self._frame_cache) > self._frame_cache_cap:
            self._frame_cache.popitem(last=False)
        return obj

    def _sql_engine(self) -> sa.engine.Engine:
        if self._engine is None:
            path = quote(self._meta_path)
            self._engine = sa.create_engine(
                f"sqlite:///file:{path}?mode=ro&uri=true",
                echo=False,
                hide_parameters=True,
            )
        return self._engine

    # ------------------------------------------------------------------
    # Index build (subset lists + filters)
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        pairs = self._read_subset_pairs()
        if not pairs:
            raise ValueError("No (sequence, frame) pairs found in subset lists")

        fa = UCO3DFrameAnnotation
        meta = sa.MetaData()
        PairTable = sa.Table(
            "__pairs",
            meta,
            sa.Column("sequence_name", sa.String, primary_key=True),
            sa.Column("frame_number", sa.Integer, primary_key=True),
            prefixes=["TEMP"],
        )

        with self._sql_engine().begin() as con:
            PairTable.create(con)
            con.execute(
                PairTable.insert(),
                [{"sequence_name": s, "frame_number": int(f)} for s, f in pairs],
            )
            stmt = sa.select(
                fa.sequence_name,
                fa.frame_number,
                sa.func.coalesce(fa.frame_timestamp, 0.0),
            ).select_from(
                PairTable.join(
                    fa,
                    sa.and_(
                        fa.sequence_name == PairTable.c.sequence_name,
                        fa.frame_number == PairTable.c.frame_number,
                    ),
                )
            )
            if self.categories or self.limit_sequences > 0:
                stmt = stmt.join(
                    self._sequence_filter_subquery(),
                    sa.text("seq_filter.sequence_name = frame_annotations.sequence_name"),
                )
            if self.remove_empty_masks:
                stmt = stmt.where(sa.or_(fa._mask_mass.is_(None), fa._mask_mass != 0))
            rows = list(con.execute(stmt).fetchall())

        grouped: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for seq, frame, ts in rows:
            grouped[str(seq)].append((int(frame), float(ts or 0.0)))
        for seq in grouped:
            grouped[seq].sort(key=lambda p: (p[1], p[0]))
        self._frames_by_seq = dict(grouped)
        self._sequence_list = sorted(self._frames_by_seq.keys())

    def _sequence_filter_subquery(self):
        conds = []
        if self.categories:
            conds.append(UCO3DSequenceAnnotation.category.in_(self.categories))
        sq = sa.select(UCO3DSequenceAnnotation.sequence_name.label("sequence_name"))
        if conds:
            sq = sq.where(*conds)
        if self.limit_sequences > 0:
            sq = sq.order_by(sa.text("ROWID")).limit(self.limit_sequences)
        return sq.subquery().alias("seq_filter")

    def _read_subset_pairs(self) -> List[Tuple[str, int]]:
        path = self.subset_lists_file
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"subset_lists_file not found: {path}")

        pairs: List[Tuple[str, int]] = []
        if path.lower().endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            for subset in self.subsets:
                for row in data.get(subset, []):
                    pairs.append((row[0], int(row[1])))
        else:
            eng = sa.create_engine(f"sqlite:///{path}")
            mdata = sa.MetaData()
            table = sa.Table("set_lists", mdata, autoload_with=eng)
            with eng.connect() as con:
                rows = con.execute(
                    sa.select(table.c.sequence_name, table.c.frame_number).where(
                        table.c.subset.in_(self.subsets)
                    )
                ).fetchall()
            pairs = [(str(s), int(f)) for s, f in rows]
        return pairs
