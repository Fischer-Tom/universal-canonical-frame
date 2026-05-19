import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import healpy as hp
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor, nn

_RANSAC_WAHBA_ITERS = 100
_RANSAC_WAHBA_SAMPLE_SIZE = 4
_RANSAC_WAHBA_INLIER_COS = math.cos(math.radians(25.0))
_RANSAC_WAHBA_EPS = 1e-6


@dataclass
class AlignmentResult:
    R_frame: torch.Tensor
    R_obj: torch.Tensor
    obj_id: torch.Tensor
    v: torch.Tensor
    Rv: torch.Tensor
    Lf_student: torch.Tensor
    b_sel: torch.Tensor
    sched: dict
    snap_ang_deg: torch.Tensor
    alignment_angle: torch.Tensor
    ransac_inlier_ratio: torch.Tensor
    G: int
    K: int


def _cosine(t, start, end):
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def _phase_value(t, vA, vB, vC):
    if t <= 0.3:
        x = 0.0 if not isinstance(vA, tuple) else (t / 0.3)
        return vA if not isinstance(vA, tuple) else _cosine(x, vA[0], vA[1])
    elif t <= 0.8:
        x = (t - 0.3) / 0.5
        return vB if not isinstance(vB, tuple) else _cosine(x, vB[0], vB[1])
    else:
        x = (t - 0.8) / 0.2
        return vC if not isinstance(vC, tuple) else _cosine(x, vC[0], vC[1])


def build_schedule(step: int, total_steps: int):
    t = min(max(step / max(1, total_steps - 1), 0.0), 1.0)
    T_f = _phase_value(t, (0.20, 0.17), (0.17, 0.13), (0.13, 0.10))
    return dict(
        T_f=T_f,
        t=t,
        discretize=True,
        teacher="top",
        cons_thresh=0.25,
    )


class Criterion(nn.Module):
    V: Tensor
    F: Tensor

    def __init__(
        self,
        V: torch.Tensor,
        F: torch.Tensor,
        total_iters: int = None,
        representation: str = "cube",
        gravity: bool = False,
        mask_loss_weight: float = 1.0,
        discretize: bool = True,
        force_identity_alignment: bool = False,
    ):
        super().__init__()
        self.register_buffer("V", V)
        self.register_buffer("F", F)
        # V is static; precompute its L2-normalized form once (non-persistent
        # so it's not saved in checkpoints — reconstructed from V on load).
        self.register_buffer("Vn", self.l2_normalize(V, dim=1), persistent=False)
        self.total_iters = total_iters
        self.c_iter = 0
        self.mask_loss_weight = float(mask_loss_weight)
        self.force_identity_alignment = bool(force_identity_alignment)

        self.discretize = (
            discretize
            and (representation == "cube")
            and not self.force_identity_alignment
        )
        if gravity and not self.discretize and not self.force_identity_alignment:
            raise ValueError(
                "gravity=True is only supported with representation='cube'"
            )
        if self.discretize:
            Rset = self._valid_rotations_cube(torch.device("cpu"), torch.float32)
            if gravity:
                y_axis = torch.tensor([0.0, 1.0, 0.0])
                mask = torch.tensor(
                    [
                        bool(torch.allclose(torch.abs(R[:, 1]), y_axis, atol=1e-6))
                        for R in Rset
                    ]
                )
                Rset = Rset[mask]
                assert Rset.shape[0] == 8, (
                    f"Expected 8 y-up rotations, got {Rset.shape[0]}"
                )
            self.register_buffer("Rset", Rset)

    @staticmethod
    def l2_normalize(x, dim):
        return F.normalize(x, p=2, dim=dim, eps=1e-12)

    @torch.no_grad()
    def _find_closest_vertex(self, v3d, k=3, single=True):
        m3d_hat = self.Vn
        v3d_hat = self.l2_normalize(v3d, dim=1)
        sim = einsum(m3d_hat, v3d_hat, "v i, b i -> v b")
        if single:
            idx = torch.argmax(sim, dim=0)
            return idx.unsqueeze(1), None
        idx = torch.topk(sim, k=k, dim=0).indices.permute(1, 0)
        nbrs = m3d_hat[idx]
        bary_coords = self.invdist_weights(v3d_hat, nbrs)
        return idx, bary_coords

    def invdist_weights(self, p: torch.Tensor, nbrs: torch.Tensor, eps: float = 1e-6):
        sims = (p.unsqueeze(1) * nbrs).sum(dim=-1)
        d = 2.0 * (1.0 - sims).clamp_min(0.0)
        inv = 1.0 / (d + eps)
        return inv / inv.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def matching(self, logits):
        _, v_idx = logits.max(dim=1)
        return self.V[v_idx]

    def _valid_rotations_cube(self, device, dtype):
        Rs = []
        seen = set()
        eye = torch.eye(3, device=device, dtype=dtype)

        for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            P = eye[:, perm]
            for signs in (
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (-1, 1, 1),
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (-1, -1, -1),
            ):
                S = torch.diag(torch.tensor(signs, device=device, dtype=dtype))
                R = P @ S
                if int(round(torch.det(R).item())) == 1:
                    key = tuple((R.cpu().numpy() * 100).round().astype(int).flatten())
                    if key not in seen:
                        seen.add(key)
                        Rs.append(R)

        R = torch.stack(Rs, dim=0)
        assert R.shape[0] == 24
        return R

    @torch.no_grad()
    def _snap_deg(self, R_snap: torch.Tensor, R_cont: torch.Tensor):
        R_delta = R_snap.transpose(-1, -2) @ R_cont
        tr = R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2]
        cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
        return torch.acos(cos) * (180.0 / math.pi)

    @torch.no_grad()
    def _wahba_batched(self, H: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Row-convention Procrustes: R = V @ diag(corr) @ U.T."""
        G = H.shape[0]
        if G == 0:
            return torch.empty_like(H)

        original_dtype = H.dtype

        with torch.amp.autocast(device_type="cuda", enabled=False):
            H_f32 = H.float()
            U, S, VT = torch.linalg.svd(H_f32, full_matrices=False)
            V = VT.transpose(-1, -2)
            det = torch.det(V @ U.transpose(-1, -2))
            sgn = det < 0.0
            corr = torch.ones((G, 3), device=H.device, dtype=torch.float32)
            corr[:, -1] = torch.where(sgn, -1.0, 1.0)
            Rf = V @ torch.diag_embed(corr) @ U.transpose(-1, -2)

        if (~valid_mask).any():
            n_bad = int((~valid_mask).sum().item())
            eye3 = torch.eye(3, device=H.device, dtype=torch.float32).expand(
                n_bad, 3, 3
            )
            Rf[~valid_mask] = eye3

        return Rf.to(original_dtype)

    @staticmethod
    @torch.no_grad()
    def _wahba_matrix_from_points(
        v: torch.Tensor,
        v_match: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        outer = torch.einsum("ni,nj->nij", v, v_match)
        return (weights[:, None, None] * outer).sum(dim=0)

    @torch.no_grad()
    def _wahba_ransac(
        self,
        v: torch.Tensor,
        v_match: torch.Tensor,
        weights: torch.Tensor,
        obj_id: torch.Tensor,
        G: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Robust per-object Wahba via RANSAC + inlier refit.

        RANSAC is internal to the teacher construction: it suppresses stray
        high-confidence pixels before the existing discrete snap chooses the
        final canonical-frame rotation.
        """
        device = v.device
        if G == 0:
            return (
                torch.empty((0, 3, 3), device=device, dtype=torch.float32),
                torch.tensor(0.0, device=device),
            )

        R_out = torch.eye(3, device=device, dtype=torch.float32).expand(G, 3, 3).clone()
        inlier_ratio = torch.zeros(G, device=device, dtype=torch.float32)
        has_inlier_metric = torch.zeros(G, device=device, dtype=torch.bool)
        v_f32 = v.float()
        v_match_f32 = v_match.float()
        weights_f32 = weights.float().clamp_min(0.0)

        for g in range(G):
            mask = obj_id == g
            if not bool(mask.any()):
                continue

            vg = v_f32[mask]
            mg = v_match_f32[mask]
            wg = weights_f32[mask]
            valid_global = wg.sum() > _RANSAC_WAHBA_EPS
            H_global = self._wahba_matrix_from_points(vg, mg, wg).unsqueeze(0)
            R_global = self._wahba_batched(
                H_global,
                valid_mask=valid_global.view(1),
            )[0].float()

            positive = wg > _RANSAC_WAHBA_EPS
            n_positive = int(positive.sum().item())
            if n_positive == 0:
                R_out[g] = R_global
                continue

            vp = vg[positive]
            mp = mg[positive]
            wp = wg[positive]
            R_final = R_global

            if n_positive >= _RANSAC_WAHBA_SAMPLE_SIZE:
                prob = wp / wp.sum().clamp_min(_RANSAC_WAHBA_EPS)

                sample_idx = torch.stack(
                    [
                        torch.multinomial(
                            prob,
                            _RANSAC_WAHBA_SAMPLE_SIZE,
                            replacement=False,
                        )
                        for _ in range(_RANSAC_WAHBA_ITERS)
                    ],
                    dim=0,
                )
                vs = vp[sample_idx]
                ms = mp[sample_idx]
                ws = wp[sample_idx]
                H_samples = (
                    ws[:, :, None, None] * torch.einsum("ksi,ksj->ksij", vs, ms)
                ).sum(dim=1)
                R_candidates = self._wahba_batched(
                    H_samples,
                    valid_mask=ws.sum(dim=1) > _RANSAC_WAHBA_EPS,
                ).float()

                pred = torch.matmul(vp.unsqueeze(0), R_candidates.transpose(-1, -2))
                cos = (pred * mp.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
                inliers = cos >= _RANSAC_WAHBA_INLIER_COS
                consensus = (inliers.float() * wp.unsqueeze(0)).sum(dim=1)
                mean_cos = (cos * wp.unsqueeze(0)).sum(dim=1) / wp.sum().clamp_min(
                    _RANSAC_WAHBA_EPS
                )
                best = (consensus + 1e-3 * mean_cos).argmax()
                best_inliers = inliers[best]

                if int(best_inliers.sum().item()) >= _RANSAC_WAHBA_SAMPLE_SIZE:
                    H_inlier = self._wahba_matrix_from_points(
                        vp[best_inliers],
                        mp[best_inliers],
                        wp[best_inliers],
                    ).unsqueeze(0)
                    R_final = self._wahba_batched(
                        H_inlier,
                        valid_mask=torch.ones(1, device=device, dtype=torch.bool),
                    )[0].float()

            R_out[g] = R_final
            pred_final = vp @ R_final.transpose(0, 1)
            final_inliers = (pred_final * mp).sum(dim=-1).clamp(
                -1.0, 1.0
            ) >= _RANSAC_WAHBA_INLIER_COS
            inlier_ratio[g] = (final_inliers.float() * wp).sum() / wp.sum().clamp_min(
                _RANSAC_WAHBA_EPS
            )
            has_inlier_metric[g] = True

        if bool(has_inlier_metric.any()):
            mean_inlier_ratio = inlier_ratio[has_inlier_metric].mean()
        else:
            mean_inlier_ratio = torch.tensor(0.0, device=device)
        return R_out, mean_inlier_ratio

    def _discretize_wahba(self, R_cont: torch.Tensor):
        scores = torch.einsum("kij,gij->gk", self.Rset, R_cont)
        idx = scores.argmax(dim=1)
        R_best = self.Rset[idx]
        return R_best, self._snap_deg(R_best, R_cont)

    @torch.no_grad()
    def _teacher_distribution_bary(self, Rv_world, b_sel, k=3, eps=1e-12):
        device = Rv_world.device
        dtype = Rv_world.dtype
        N = Rv_world.shape[0]
        M = self.V.shape[0]

        idx, bary = self._find_closest_vertex(Rv_world, k=k, single=False)
        bary = bary / (bary.sum(dim=1, keepdim=True) + eps)
        q = torch.zeros(N, M, device=device, dtype=dtype)
        q.scatter_add_(1, idx, bary.to(dtype))
        return q / (q.sum(dim=1, keepdim=True) + eps)

    @torch.no_grad()
    def _teacher_distribution_top(self, Rv_world, b_sel, eps=1e-12):
        device = Rv_world.device
        dtype = Rv_world.dtype
        N = Rv_world.shape[0]
        M = self.V.shape[0]

        idx, _ = self._find_closest_vertex(Rv_world, single=True)
        q = torch.zeros(N, M, device=device, dtype=dtype)
        q.scatter_add_(1, idx, torch.ones_like(idx).to(dtype))
        return q

    def align(
        self,
        logits: torch.Tensor,
        yx_sel: torch.Tensor,
        v_sel: torch.Tensor,
        b_sel: torch.Tensor,
        n_views: int,
        resolution: List[int],
        *,
        img2obj: Optional[torch.Tensor] = None,
        n_objects: Optional[int] = None,
    ) -> Optional[AlignmentResult]:
        device = logits.device
        N = yx_sel.shape[0]
        if N == 0:
            return None

        assert self.total_iters is not None
        sched = build_schedule(self.c_iter, self.total_iters)

        Un = self.Vn.to(dtype=logits.dtype, device=device)
        v = self.l2_normalize(v_sel.to(device=device, dtype=logits.dtype), dim=1)

        if img2obj is None:
            obj_id = torch.div(b_sel, n_views, rounding_mode="floor")
            frame_obj_id = None
        else:
            frame_obj_id = img2obj.to(device=device, dtype=torch.long)
            obj_id = frame_obj_id[b_sel]
        obj_id = obj_id.to(torch.long)
        G = n_objects if n_objects is not None else int(obj_id.max().item()) + 1

        B, M, HW = logits.shape
        H, W = resolution
        assert H * W == HW, f"Mismatch: H*W={H * W}, but logits last dim is {HW}"

        y = yx_sel[:, 0].long().clamp(0, H - 1)
        x = yx_sel[:, 1].long().clamp(0, W - 1)
        pix_idx = y * W + x

        Lf_student = logits[b_sel.long(), :, pix_idx] / float(sched["T_f"])
        p_student = torch.softmax(Lf_student, dim=1)
        v_match = self.l2_normalize(p_student @ Un, dim=1)

        eps = 1e-8
        logp = torch.log(p_student.clamp_min(eps))
        Hn = -(p_student * logp).sum(dim=1)
        Hn_norm = Hn / math.log(M)
        gamma = 2.0
        w = (1.0 - Hn_norm).clamp(0.0, 1.0).pow(gamma)

        K = int(frame_obj_id.shape[0]) if frame_obj_id is not None else n_views * G
        snap_ang_deg = torch.tensor(0.0, device=device)
        if self.force_identity_alignment:
            R_obj = torch.eye(3, device=device, dtype=torch.float32).expand(
                G, 3, 3
            )
            ransac_inlier_ratio = torch.tensor(0.0, device=device)
        else:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                R_obj, ransac_inlier_ratio = self._wahba_ransac(
                    v, v_match, w, obj_id.long(), G
                )

                if self.discretize:
                    R_obj_snap, ang_deg = self._discretize_wahba(R_obj)
                    snap_ang_deg = ang_deg.mean()
                    if sched["discretize"]:
                        R_obj = R_obj_snap

        if frame_obj_id is None:
            frame_obj_id = torch.arange(G, device=device).repeat_interleave(n_views)
        R_frame = R_obj.to(dtype=logits.dtype)[frame_obj_id]
        R_pix = R_frame[b_sel.long()]
        Rv = torch.bmm(v.unsqueeze(1), R_pix.transpose(-2, -1)).squeeze(1)
        with torch.no_grad():
            alignment_angle = (Rv.detach() * v_match.detach()).sum(dim=1).mean()

        return AlignmentResult(
            R_frame=R_frame,
            R_obj=R_obj.to(dtype=logits.dtype),
            obj_id=obj_id,
            v=v,
            Rv=Rv,
            Lf_student=Lf_student,
            b_sel=b_sel,
            sched=sched,
            snap_ang_deg=snap_ang_deg,
            alignment_angle=alignment_angle.detach(),
            ransac_inlier_ratio=ransac_inlier_ratio.detach(),
            G=G,
            K=K,
        )

    def compute_loss(
        self,
        alignment: AlignmentResult,
        mask_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        valid_region: Optional[torch.Tensor] = None,
        Lf_student: Optional[torch.Tensor] = None,
        Rv: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if Lf_student is None:
            Lf_student = alignment.Lf_student
        if Rv is None:
            Rv = alignment.Rv

        obj_id = alignment.obj_id
        b_sel = alignment.b_sel
        sched = alignment.sched
        device = Lf_student.device

        if sched["teacher"] == "top":
            q = self._teacher_distribution_top(Rv, b_sel)
        else:
            q = self._teacher_distribution_bary(Rv, b_sel, k=3)

        logp = torch.log_softmax(Lf_student, dim=1).clamp(min=-100)
        loss_kd_pp = -(q * logp).sum(dim=1)

        K = int(b_sel.max().item()) + 1
        sums = torch.zeros(K, device=loss_kd_pp.device, dtype=loss_kd_pp.dtype)
        counts = torch.zeros(K, device=loss_kd_pp.device, dtype=loss_kd_pp.dtype)
        ones = torch.ones_like(loss_kd_pp)
        sums.index_add_(0, b_sel, loss_kd_pp)
        counts.index_add_(0, b_sel, ones)
        nonempty = counts > 0
        loss_kd = (sums[nonempty] / counts[nonempty]).mean()

        entropy = -(torch.exp(logp) * logp).sum(dim=1)
        loss_entropy = entropy.mean()

        with torch.no_grad():
            Un = self.Vn.to(dtype=Lf_student.dtype, device=device)
            p_student = torch.softmax(Lf_student, dim=1)
            v_match = self.l2_normalize(p_student @ Un, dim=1)
            ratio = loss_entropy / (loss_kd_pp.mean() + 1e-8)
            alignment_cos = (Rv * v_match).sum(dim=1).mean()

        loss_mask = self._edge_mask_loss(
            mask_logits, gt_mask, valid_region=valid_region
        )
        return {
            "loss_mesh": loss_kd,
            "loss_entropy": loss_entropy.detach(),
            "loss_mask": loss_mask,
            "entropy_ratio": ratio.detach(),
            "alignment_cos": alignment_cos,
            "loss_total": loss_kd + self.mask_loss_weight * loss_mask,
            "R_obj": alignment.R_obj.detach(),
            "R_frame": alignment.R_frame.detach(),
            "obj_id_per_pixel": obj_id,
            "snap_ang_deg": alignment.snap_ang_deg.detach(),
            "alignment_angle": alignment_cos.detach(),
            "ransac_inlier_ratio": alignment.ransac_inlier_ratio.detach(),
        }

    def forward(
        self,
        logits: torch.Tensor,
        mask_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        yx_sel: torch.Tensor,
        v_sel: torch.Tensor,
        b_sel: torch.Tensor,
        n_views: int,
        resolution: List[int],
        *,
        valid_region: Optional[torch.Tensor] = None,
        img2obj: Optional[torch.Tensor] = None,
        n_objects: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        device = logits.device
        if yx_sel.shape[0] == 0:
            dummy_loss = (logits * 0).sum()
            loss_mask = self._edge_mask_loss(
                mask_logits, gt_mask, valid_region=valid_region
            )
            return {
                "loss_mesh": dummy_loss,
                "loss_mask": loss_mask,
                "loss_entropy": torch.tensor(0.0, device=device),
                "entropy_ratio": torch.tensor(0.0, device=device),
                "alignment_cos": torch.tensor(0.0, device=device),
                "loss_total": dummy_loss + self.mask_loss_weight * loss_mask,
                "R_obj": torch.empty((0, 3, 3), device=device),
                "obj_id_per_pixel": torch.empty((0,), dtype=torch.long, device=device),
                "snap_ang_deg": torch.tensor(0.0, device=device),
                "alignment_angle": torch.tensor(0.0, device=device),
                "ransac_inlier_ratio": torch.tensor(0.0, device=device),
            }

        alignment = self.align(
            logits=logits,
            yx_sel=yx_sel,
            v_sel=v_sel,
            b_sel=b_sel,
            n_views=n_views,
            resolution=resolution,
            img2obj=img2obj,
            n_objects=n_objects,
        )
        return self.compute_loss(
            alignment, mask_logits, gt_mask, valid_region=valid_region
        )

    def _get_mask_edges(self, mask: torch.Tensor, edge_width: int = 2) -> torch.Tensor:
        """
        Extract edge pixels from a binary mask using morphological operations.

        Args:
            mask: Binary mask of shape (B, H, W) or (B, 1, H, W)
            edge_width: Width of the edge band in pixels

        Returns:
            Edge mask of same shape as input, with 1s at edge pixels
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Use max pooling for dilation, -max(-x) for erosion
        kernel_size = 2 * edge_width + 1
        padding = edge_width

        # Dilate: expand the mask outward
        dilated = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=padding)
        # Erode: shrink the mask inward
        eroded = -F.max_pool2d(-mask.float(), kernel_size, stride=1, padding=padding)

        # Edge = dilated - eroded (band around the boundary)
        edges = (dilated - eroded).clamp(0, 1)
        # Zero out edges at image boundaries (artifact from padding in pooling)
        # When mask=1 at image boundary, erosion shrinks it due to zero-padding,
        # creating spurious edges that cause the model to predict masks near black borders
        H, W = edges.shape[-2:]
        boundary_mask = torch.ones_like(edges)
        boundary_mask[..., :edge_width, :] = 0  # top
        boundary_mask[..., -edge_width:, :] = 0  # bottom
        boundary_mask[..., :, :edge_width] = 0  # left
        boundary_mask[..., :, -edge_width:] = 0  # right
        edges = edges * boundary_mask
        return edges.squeeze(1)  # (B, H, W)

    def _edge_mask_loss(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        edge_width: int = 2,
        interior_weight: float = 0.1,
        valid_region: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute mask loss focused on edges (sparse computation when interior_weight=0).

        Args:
            pred_mask: Predicted mask logits (B, 1, H, W)
            gt_mask: Ground truth binary mask (B, H, W)
            edge_width: Width of edge band in pixels
            interior_weight: Weight for non-edge pixels (0 = sparse edge-only)
            valid_region: (B, H, W) or (B, 1, H, W) bool/uint8 mask. Loss is only
                            computed where valid_region=1 (excludes padding & occlusions).

        Returns:
            Weighted mask loss (BCE + Dice on edges)
        """
        # Get edge regions from GT mask
        edge_mask = self._get_mask_edges(gt_mask, edge_width=edge_width)  # (B, H, W)
        edge_pixels = edge_mask > 0.5

        # Restrict to valid region (exclude padding and synthetic occlusions)
        if valid_region is not None:
            if valid_region.dim() == 4:
                valid_region = valid_region.squeeze(1)
            valid_bool = valid_region.bool()
            edge_pixels = edge_pixels & valid_bool

        pred_flat = pred_mask.squeeze(1)  # (B, H, W)
        gt_flat = gt_mask.float()

        if interior_weight > 0:
            # Weighted: compute full BCE then weight, masked to valid region
            bce = F.binary_cross_entropy_with_logits(
                pred_flat, gt_flat, reduction="none"
            )
            weights = torch.where(edge_pixels, 1.0, interior_weight)
            if valid_region is not None:
                weights = weights * valid_bool.float()
            weighted_bce = (bce * weights).sum() / weights.sum().clamp(min=1)
        else:
            # Sparse: only compute BCE on edge pixels
            if edge_pixels.sum() > 0:
                pred_edge = pred_flat[edge_pixels]  # (N_edge,)
                gt_edge = gt_flat[edge_pixels]  # (N_edge,)
                weighted_bce = F.binary_cross_entropy_with_logits(pred_edge, gt_edge)
            else:
                weighted_bce = pred_flat.new_tensor(0.0)

        # Sparse Dice on edge pixels only
        if edge_pixels.sum() > 0:
            pred_sigmoid = torch.sigmoid(pred_flat)
            pred_edge_vals = pred_sigmoid[edge_pixels]  # (N_edge,)
            gt_edge_vals = gt_flat[edge_pixels]  # (N_edge,)
            # Dice per batch: need to track batch membership
            # Simplified: compute global dice on edge pixels
            intersection = (pred_edge_vals * gt_edge_vals).sum()
            union = pred_edge_vals.sum() + gt_edge_vals.sum()
            dice = 1 - (2 * intersection + 1) / (union + 1)
        else:
            dice = pred_flat.new_tensor(0.0)

        return weighted_bce + dice


def healpix_sphere_mesh(nside: int, nest: bool = False, round_decimals: int = 14):
    npix = hp.nside2npix(nside)
    verts = []
    vindex = {}

    def vid_of(v):
        key = tuple(np.round(v, round_decimals))
        if key in vindex:
            return vindex[key]
        k = len(verts)
        vindex[key] = k
        verts.append(v / np.linalg.norm(v))
        return k

    faces = []
    for p in range(npix):
        b = hp.boundaries(nside, p, step=1, nest=nest)
        quad = b.T
        ids = [vid_of(quad[i]) for i in range(4)]
        faces.append([ids[0], ids[1], ids[2]])
        faces.append([ids[0], ids[2], ids[3]])

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)

    v0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    n = np.cross(v0, v1)
    centroids = verts[faces].mean(axis=1)
    inward = np.einsum("ij,ij->i", n, centroids) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]

    bbox = np.max(verts, axis=0) - np.min(verts, axis=0)
    verts *= 1 / np.linalg.norm(bbox)

    return (
        verts.astype(np.float32),
        faces.astype(np.float32),
    )


def healpix_cube_mesh(subdivisions: int, round_decimals: int = 14):
    n = subdivisions
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    uu, vv = np.meshgrid(u, v)
    uu = uu - 0.5
    vv = vv - 0.5

    verts = []
    vindex = {}

    def vid_of(vertex):
        key = tuple(np.round(vertex, round_decimals))
        if key in vindex:
            return vindex[key]
        k = len(verts)
        vindex[key] = k
        verts.append(vertex)
        return k

    faces = []
    face_defs = [
        (2, +0.5, 0, 1),
        (2, -0.5, 0, 1),
        (1, +0.5, 0, 2),
        (1, -0.5, 0, 2),
        (0, +0.5, 1, 2),
        (0, -0.5, 1, 2),
    ]
    for const_axis, const_val, u_axis, v_axis in face_defs:
        face_vids = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                vertex = np.zeros(3)
                vertex[const_axis] = const_val
                vertex[u_axis] = uu[i, j]
                vertex[v_axis] = vv[i, j]
                face_vids[i, j] = vid_of(vertex)

        for i in range(n - 1):
            for j in range(n - 1):
                v00 = face_vids[i, j]
                v01 = face_vids[i, j + 1]
                v10 = face_vids[i + 1, j]
                v11 = face_vids[i + 1, j + 1]
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)

    v0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    n = np.cross(v0, v1)
    centroids = verts[faces].mean(axis=1)
    inward = np.einsum("ij,ij->i", n, centroids) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]

    bbox = np.max(verts, axis=0) - np.min(verts, axis=0)
    verts *= 1 / np.linalg.norm(bbox)

    return verts.astype(np.float32), faces.astype(np.float32)
