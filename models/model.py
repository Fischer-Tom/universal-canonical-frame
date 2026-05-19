import torch
import torch.nn.functional as F
from dataset.schema import Batch
from tools.annotate import AnnotationResult, Annotator
from torch import nn

from .criterion import Criterion
from .dino import DINO
from .heads import MaskHead, MeshCorrespondenceHead
from .mesh_decoder import MeshDecoder

AUG_CONSISTENCY_TEMP = 0.1
AUG_CONSISTENCY_MAX_PIXELS = 16384
AUG_CONSISTENCY_MIN_CONFIDENCE = 0.0


def _gather_valid_feats(feats_bchw: torch.Tensor, valid_mask_bhw: torch.Tensor):
    """Gather features at valid-mask pixels.

    Returns:
      feats_sel: (M, C)
      b_sel:     (M,)
      yx_sel:    (M, 2)
      mask_flat: (B*H*W,)
    """
    if valid_mask_bhw.dim() > 3:
        valid_mask_bhw = valid_mask_bhw.squeeze(1)
    B, C, H, W = feats_bchw.shape
    idx = valid_mask_bhw.nonzero(as_tuple=False)
    if idx.numel() == 0:
        empty_feats = feats_bchw.new_zeros((0, C))
        empty_long = torch.empty(0, dtype=torch.long, device=feats_bchw.device)
        return (
            empty_feats,
            empty_long,
            empty_long.view(0, 2),
            valid_mask_bhw.reshape(-1),
        )

    b_sel = idx[:, 0]
    y_sel = idx[:, 1]
    x_sel = idx[:, 2]
    feats_sel = feats_bchw[b_sel, :, y_sel, x_sel]
    yx_sel = torch.stack([y_sel, x_sel], dim=-1).to(torch.long)
    return feats_sel, b_sel, yx_sel, valid_mask_bhw.reshape(-1)


class GpuPhotometricAug(nn.Module):
    NOISE_STD = 0.03
    COLOR_SCALE = 0.10
    COLOR_BIAS = 0.05
    PATCH_P = 0.25
    PATCH_NUM = (1, 2)
    PATCH_SIZE = (0.05, 0.20)

    def __init__(self):
        super().__init__()
        self.noise_std = self.NOISE_STD
        self.color_scale = self.COLOR_SCALE
        self.color_bias = self.COLOR_BIAS
        self.patch_p = self.PATCH_P
        self.patch_num = self.PATCH_NUM
        self.patch_size = self.PATCH_SIZE
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _patch_dropout(self, x: torch.Tensor):
        B, _C, H, W = x.shape
        keep = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
        if self.patch_p <= 0:
            return x, keep

        out = x.clone()
        base = float(min(H, W))
        for b in range(B):
            if torch.rand((), device=x.device).item() >= self.patch_p:
                continue
            n_patches = int(
                torch.randint(
                    self.patch_num[0],
                    self.patch_num[1] + 1,
                    (1,),
                    device=x.device,
                ).item()
            )
            for _ in range(n_patches):
                h_frac = (
                    torch.empty((), device=x.device).uniform_(*self.patch_size).item()
                )
                h_frac = (
                    torch.empty((), device=x.device).uniform_(*self.patch_size).item()
                )
                w_frac = (
                    torch.empty((), device=x.device).uniform_(*self.patch_size).item()
                )
                ph = max(1, min(H, int(round(base * h_frac))))
                pw = max(1, min(W, int(round(base * w_frac))))
                y = int(torch.randint(0, H - ph + 1, (1,), device=x.device).item())
                x0 = int(torch.randint(0, W - pw + 1, (1,), device=x.device).item())
                out[b, :, y : y + ph, x0 : x0 + pw] = 0.0
                keep[b, :, y : y + ph, x0 : x0 + pw] = 0.0
        return out, keep

    def forward(self, x: torch.Tensor):
        out = x
        if self.color_scale > 0 or self.color_bias > 0:
            rgb = (out * self.std.to(out) + self.mean.to(out)).clamp(0, 1)
            B = rgb.shape[0]
            if self.color_scale > 0:
                scale = (
                    1.0
                    + (
                        torch.rand((B, 3, 1, 1), device=rgb.device, dtype=rgb.dtype)
                        * 2.0
                        - 1.0
                    )
                    * self.color_scale
                )
                rgb = rgb * scale
            if self.color_bias > 0:
                bias = (
                    torch.rand((B, 3, 1, 1), device=rgb.device, dtype=rgb.dtype) * 2.0
                    - 1.0
                ) * self.color_bias
                rgb = rgb + bias
            out = (rgb.clamp(0, 1) - self.mean.to(out)) / self.std.to(out)

        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std

        return self._patch_dropout(out)


def _resize_mask(mask: torch.Tensor, size_hw):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if tuple(mask.shape[-2:]) == tuple(size_hw):
        return mask.squeeze(1).bool()
    return F.interpolate(mask.float(), size=size_hw, mode="nearest").squeeze(1).bool()


class Model(nn.Module):
    def __init__(
        self,
        cfg,
        V: torch.Tensor,
        F: torch.Tensor,
        total_iters: int = None,
        annotator: Annotator | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.annotator = annotator
        self.backbone = DINO(512, cfg.model, adapt=cfg.model.get("adapt", True))
        self.mesh_decoder = MeshDecoder(
            V, n_blocks=6, n_heads=8, d_model=512, dim_feedforward=2048
        )
        self.correspondence_head = MeshCorrespondenceHead(d_model=512, d_desc=512)
        self.mask_head = MaskHead(d_model=512)
        pca_init = bool(cfg.representation.get("pca_init", True))
        discretize = bool(cfg.representation.get("discretize", True)) and pca_init
        self.criterion = Criterion(
            V,
            F,
            total_iters=total_iters,
            representation=cfg.representation.mesh_type,
            gravity=cfg.representation.get("gravity", False),
            mask_loss_weight=cfg.get("loss", {}).get("mask_weight", 10.0),
            discretize=discretize,
            force_identity_alignment=cfg.representation.get(
                "force_identity_alignment", False
            ),
        )
        self.n_views_per_seq = int(cfg.dataset.frames_per_sequence)
        self.rerender_without_discretization = bool(
            cfg.representation.get("rerender_without_discretization", True)
        )
        self.force_identity_alignment = bool(
            cfg.representation.get("force_identity_alignment", False)
        )
        self.register_buffer("V", V)
        self.register_buffer("F", F)
        loss_cfg = cfg.get("loss", {})
        self.aug_consistency_weight = float(loss_cfg.get("aug_consistency_weight", 0.0))
        self.aug_consistency_temp = AUG_CONSISTENCY_TEMP
        self.aug_consistency_max_pixels = AUG_CONSISTENCY_MAX_PIXELS
        self.aug_consistency_min_conf = AUG_CONSISTENCY_MIN_CONFIDENCE
        self.photometric_aug = GpuPhotometricAug()

    def predict(self, img: torch.Tensor):
        feats = self.backbone(img)
        mesh_descriptors = self.mesh_decoder(feats)
        logits = self.correspondence_head(mesh_descriptors, feats)
        mask_logits = self.mask_head(feats)
        return feats, logits, mask_logits

    def _correspondence_consistency_loss(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        valid_mask: torch.Tensor,
        keep_mask_b: torch.Tensor,
        feat_hw,
    ):
        B, V, HW = logits_a.shape
        valid = _resize_mask(valid_mask, feat_hw)
        keep = _resize_mask(keep_mask_b, feat_hw)
        valid = (valid & keep).reshape(B * HW)
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return logits_a.sum() * 0.0, logits_a.new_tensor(0.0)

        max_pixels = self.aug_consistency_max_pixels
        if max_pixels > 0 and idx.numel() > max_pixels:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_pixels]
            idx = idx[perm]

        logits_a_flat = logits_a.transpose(1, 2).reshape(B * HW, V)[idx].float()
        logits_b_flat = logits_b.transpose(1, 2).reshape(B * HW, V)[idx].float()
        temp = max(self.aug_consistency_temp, 1e-6)
        z_a = logits_a_flat / temp
        z_b = logits_b_flat / temp

        if self.aug_consistency_min_conf > 0:
            with torch.no_grad():
                conf_a = torch.softmax(z_a, dim=-1).amax(dim=-1)
                conf_b = torch.softmax(z_b, dim=-1).amax(dim=-1)
                keep_conf = (conf_a >= self.aug_consistency_min_conf) & (
                    conf_b >= self.aug_consistency_min_conf
                )
            if not keep_conf.any():
                return logits_a.sum() * 0.0, logits_a.new_tensor(0.0)
            z_a = z_a[keep_conf]
            z_b = z_b[keep_conf]

        logp_a = F.log_softmax(z_a, dim=-1)
        logp_b = F.log_softmax(z_b, dim=-1)
        p_a = logp_a.exp()
        p_b = logp_b.exp()
        if not logits_b.requires_grad:
            return F.kl_div(
                logp_a, p_b.detach(), reduction="batchmean"
            ), logits_a.new_tensor(float(z_a.shape[0]))
        loss_ab = F.kl_div(logp_b, p_a.detach(), reduction="batchmean")
        loss_ba = F.kl_div(logp_a, p_b.detach(), reduction="batchmean")
        return 0.5 * (loss_ab + loss_ba), logits_a.new_tensor(float(z_a.shape[0]))

    def forward(self, batch: Batch, annotations: AnnotationResult):
        img = batch.image_rgb
        geom_mask = annotations.geom_mask
        render_mask = annotations.render_mask
        valid_mask = annotations.valid_mask
        xyz = annotations.obj_xyz

        feats, logits, mask_logits = self.predict(img)

        self._last_logits = logits.detach()
        self._last_img = img.detach()
        self._last_geom_mask = geom_mask.detach()
        self._last_feat_hw = tuple(feats.shape[-2:])

        _, b_sel, yx_sel, mask_flat = _gather_valid_feats(feats, geom_mask.long())
        xyz_sel = xyz.reshape(-1, 3)[mask_flat.bool()]

        use_rerender = (
            self.training
            and self.rerender_without_discretization
            and not self.criterion.discretize
            and not self.force_identity_alignment
        )
        if use_rerender:
            if self.annotator is None:
                raise ValueError(
                    "Rerendering without criterion discretization requires an "
                    "Annotator. Pass annotator to Model(...)."
                )
            loss_dict, annotations_for_loss, geom_mask_for_vis = (
                self._loss_with_rerendered_annotations(
                    batch=batch,
                    annotations=annotations,
                    feats=feats,
                    logits=logits,
                    mask_logits=mask_logits,
                    yx_sel=yx_sel,
                    xyz_sel=xyz_sel,
                    b_sel=b_sel,
                )
            )
        else:
            loss_dict = self.criterion(
                logits=logits,
                mask_logits=mask_logits,
                gt_mask=render_mask,
                valid_region=valid_mask,
                yx_sel=yx_sel,
                v_sel=xyz_sel,
                b_sel=b_sel,
                n_views=self.n_views_per_seq,
                resolution=list(feats.shape[-2:]),
                img2obj=annotations.obj_idx,
                n_objects=batch.num_sequences,
            )
            annotations_for_loss = annotations
            geom_mask_for_vis = geom_mask

        aug_scale = self.aug_consistency_weight if self.training else 0.0
        if aug_scale > 0:
            img_aug, keep_aug = self.photometric_aug(img)
            with torch.no_grad():
                _, logits_aug, _ = self.predict(img_aug)
            loss_aug, n_aug = self._correspondence_consistency_loss(
                logits,
                logits_aug,
                geom_mask_for_vis,
                keep_aug,
                tuple(feats.shape[-2:]),
            )
            aug_weight = logits.new_tensor(aug_scale)
            loss_dict["loss_aug_consistency"] = loss_aug
            loss_dict["aug_consistency_weight"] = aug_weight
            loss_dict["aug_consistency_pixels"] = n_aug.detach()
            loss_dict["loss_total"] = loss_dict["loss_total"] + aug_weight * loss_aug
        else:
            loss_dict["loss_aug_consistency"] = logits.sum() * 0.0
            loss_dict["aug_consistency_weight"] = logits.new_tensor(0.0)
            loss_dict["aug_consistency_pixels"] = logits.new_tensor(0.0)

        batch.annotations = annotations_for_loss
        self._last_geom_mask = geom_mask_for_vis.detach()
        self.criterion.c_iter += 1
        return loss_dict

    def _loss_with_rerendered_annotations(
        self,
        *,
        batch: Batch,
        annotations: AnnotationResult,
        feats: torch.Tensor,
        logits: torch.Tensor,
        mask_logits: torch.Tensor,
        yx_sel: torch.Tensor,
        xyz_sel: torch.Tensor,
        b_sel: torch.Tensor,
    ):
        alignment = self.criterion.align(
            logits=logits,
            yx_sel=yx_sel,
            v_sel=xyz_sel,
            b_sel=b_sel,
            n_views=self.n_views_per_seq,
            resolution=list(feats.shape[-2:]),
            img2obj=annotations.obj_idx,
            n_objects=batch.num_sequences,
        )
        if alignment is None:
            loss_dict = self.criterion(
                logits=logits,
                mask_logits=mask_logits,
                gt_mask=annotations.render_mask,
                valid_region=annotations.valid_mask,
                yx_sel=yx_sel,
                v_sel=xyz_sel,
                b_sel=b_sel,
                n_views=self.n_views_per_seq,
                resolution=list(feats.shape[-2:]),
                img2obj=annotations.obj_idx,
                n_objects=batch.num_sequences,
            )
            return loss_dict, annotations, annotations.geom_mask

        device = logits.device
        frame_indices = torch.arange(len(batch.cameras), device=device)
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type=device.type, enabled=False),
        ):
            rerendered = self.annotator.rerender_aligned(
                batch,
                alignment.R_frame.float(),
                obj_id_per_frame=frame_indices,
                base_cameras=annotations.cameras,
                deform=False,
            )

        _, b_new, yx_new, mask_flat_new = _gather_valid_feats(
            feats, rerendered.geom_mask.long()
        )
        xyz_new = rerendered.obj_xyz.reshape(-1, 3)[mask_flat_new.bool()]
        if yx_new.shape[0] == 0:
            loss_dict = self.criterion(
                logits=logits,
                mask_logits=mask_logits,
                gt_mask=rerendered.render_mask,
                valid_region=rerendered.valid_mask,
                yx_sel=yx_new,
                v_sel=xyz_new,
                b_sel=b_new,
                n_views=self.n_views_per_seq,
                resolution=list(feats.shape[-2:]),
                img2obj=rerendered.obj_idx,
                n_objects=batch.num_sequences,
            )
            return loss_dict, rerendered, rerendered.geom_mask

        H, W = feats.shape[-2:]
        y = yx_new[:, 0].long().clamp(0, H - 1)
        x = yx_new[:, 1].long().clamp(0, W - 1)
        pix_idx = y * W + x
        Lf_student = logits[b_new.long(), :, pix_idx] / float(alignment.sched["T_f"])

        v_new = self.criterion.l2_normalize(
            xyz_new.to(device=device, dtype=logits.dtype), dim=1
        )

        alignment.v = v_new
        alignment.Rv = v_new
        alignment.Lf_student = Lf_student
        alignment.b_sel = b_new
        alignment.obj_id = rerendered.obj_idx.to(device=device, dtype=torch.long)[
            b_new.long()
        ]

        loss_dict = self.criterion.compute_loss(
            alignment,
            mask_logits,
            gt_mask=rerendered.render_mask,
            valid_region=rerendered.valid_mask,
        )
        loss_dict["obj_xyz_pre_align"] = annotations.obj_xyz.detach()
        loss_dict["obj_xyz_post_align"] = rerendered.obj_xyz.detach()
        return loss_dict, rerendered, rerendered.geom_mask

    @torch.no_grad()
    def infer(
        self,
        batch: Batch,
        annotations: AnnotationResult | None = None,
        *,
        use_gt_mask: bool = False,
    ) -> dict:
        """Return the legacy evaluation correspondence contract.

        The pose evaluator expects sparse pixel-to-canonical-vertex matches:
        selected matched vertices (`m_v3d`), selected pixel indices (`yx_sel`),
        and selected frame indices (`b_sel`).
        """
        annotations = annotations if annotations is not None else batch.annotations
        if annotations is None:
            raise ValueError("Model.infer requires annotations or batch.annotations")

        img = batch.image_rgb
        feats, logits, mask_logits = self.predict(img)

        matched_vertices = self.criterion.matching(logits)
        if use_gt_mask or not bool(self.cfg.model.get("learn_mask", True)):
            pred_mask_binary = annotations.geom_mask.bool()
        else:
            pred_mask_binary = torch.sigmoid(mask_logits).squeeze(1) > 0.5
            valid_mask = annotations.valid_mask
            if valid_mask.dim() == 4:
                valid_mask = valid_mask.squeeze(1)
            if valid_mask.shape[-2:] != pred_mask_binary.shape[-2:]:
                valid_mask = torch.nn.functional.interpolate(
                    valid_mask.float().unsqueeze(1),
                    size=pred_mask_binary.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
            pred_mask_binary = pred_mask_binary & valid_mask.bool()

        _, b_sel, yx_sel, mask_flat = _gather_valid_feats(
            feats, pred_mask_binary.long()
        )
        m_v3d = matched_vertices.reshape(-1, 3)[mask_flat.bool()]

        return {
            "m_v3d": m_v3d,
            "match_dict": {
                "matches": m_v3d,
                "b_sel": b_sel,
                "yx_sel": yx_sel,
            },
            "mask": pred_mask_binary.float(),
            "pred_mask": pred_mask_binary.float(),
            "b_sel": b_sel,
            "yx_sel": yx_sel,
            "logits": logits,
            "mask_logits": mask_logits,
            "feat_hw": tuple(feats.shape[-2:]),
        }
