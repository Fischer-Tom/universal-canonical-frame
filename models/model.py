import torch
from torch import nn

from dataset.schema import Batch
from tools.annotate import AnnotationResult

from .criterion import Criterion
from .dino import DINO
from .heads import MaskHead, MeshCorrespondenceHead
from .mesh_decoder import MeshDecoder


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


class Model(nn.Module):
    def __init__(self, cfg, V: torch.Tensor, F: torch.Tensor, total_iters: int = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = DINO(256, cfg.model, adapt=cfg.model.get("adapt", True))
        self.mesh_decoder = MeshDecoder(
            V, n_blocks=4, n_heads=8, d_model=256, dim_feedforward=1024
        )
        self.correspondence_head = MeshCorrespondenceHead(d_model=256, d_desc=256)
        self.mask_head = MaskHead(d_model=256)
        self.criterion = Criterion(
            V,
            F,
            total_iters=total_iters,
            representation=cfg.representation.mesh_type,
            gravity=cfg.representation.get("gravity", False),
            mask_loss_weight=cfg.get("loss", {}).get("mask_weight", 10.0),
        )
        self.n_views_per_seq = int(cfg.dataset.frames_per_sequence)
        self.register_buffer("V", V)
        self.register_buffer("F", F)

    def forward(self, batch: Batch, annotations: AnnotationResult):
        img = batch.image_rgb
        geom_mask = annotations.geom_mask
        render_mask = annotations.render_mask
        valid_mask = annotations.valid_mask
        xyz = annotations.obj_xyz

        feats = self.backbone(img)
        mesh_descriptors = self.mesh_decoder(feats)
        logits = self.correspondence_head(mesh_descriptors, feats)
        mask_logits = self.mask_head(feats)

        self._last_logits = logits.detach()
        self._last_img = img.detach()
        self._last_geom_mask = geom_mask.detach()
        self._last_feat_hw = tuple(feats.shape[-2:])

        _, b_sel, yx_sel, mask_flat = _gather_valid_feats(feats, geom_mask.long())
        xyz_sel = xyz.reshape(-1, 3)[mask_flat.bool()]

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
        self.criterion.c_iter += 1
        return loss_dict

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
        feats = self.backbone(img)
        mesh_descriptors = self.mesh_decoder(feats)
        logits = self.correspondence_head(mesh_descriptors, feats)
        mask_logits = self.mask_head(feats)

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
