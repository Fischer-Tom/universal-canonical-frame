import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MeshCorrespondenceHead(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_desc: int = 128,
        mask_topk: int = 8,
        mask_hidden: int = 32,
    ):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_desc)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(d_model, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
        )
        self.mask_topk = mask_topk
        self.mask_head = nn.Sequential(
            nn.Linear(2, mask_hidden),
            nn.GELU(),
            nn.Linear(mask_hidden, 1),
        )

    def forward(self, hs: Tensor, feats: Tensor):
        # hs: (num_queries, B, d_model)  feats: (B, d_model, H, W)
        # returns sim: (B, Q, H*W), mask_logits: (B, 1, H, W)
        q = hs.permute(1, 0, 2)
        q_desc = self.query_proj(q)

        k_desc = self.feat_proj(feats)
        B, D, H, W = k_desc.shape
        k_desc = k_desc.flatten(2).transpose(1, 2)

        q_desc = F.normalize(q_desc, dim=-1)
        k_desc = F.normalize(k_desc, dim=-1)

        sim = torch.matmul(q_desc, k_desc.transpose(1, 2))  # (B, Q, H*W)

        # Mask logit derived from the correspondence similarity itself: a
        # pixel that matches some vertex well is foreground. Uses pooled
        # stats (not the raw per-vertex sim) so param count doesn't scale
        # with the number of mesh vertices.
        k = min(self.mask_topk, sim.shape[1])
        topk_sim = sim.topk(k, dim=1).values  # (B, k, H*W)
        stats = torch.stack(
            [topk_sim[:, 0, :], topk_sim.mean(dim=1)], dim=-1
        )  # (B, H*W, 2)
        mask_logits = self.mask_head(stats).squeeze(-1).view(B, 1, H, W)

        return sim, mask_logits



class PoseHead(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        final.bias.data[0] = 1.0

    def forward(self, mesh_descriptors: Tensor):
        # mesh_descriptors: (num_vertices, B, C) 
        # NOTE: How is spatial information preserved here
        pooled = mesh_descriptors.mean(dim=0)
        quat = self.head(pooled)
        return F.normalize(quat, dim=-1, eps=1e-8)
