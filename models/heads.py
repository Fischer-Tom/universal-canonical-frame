import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MeshCorrespondenceHead(nn.Module):
    def __init__(self, d_model: int = 256, d_desc: int = 128):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_desc)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(d_model, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
        )

    def forward(self, hs: Tensor, feats: Tensor):
        # hs: (num_queries, B, d_model)  feats: (B, d_model, H, W)
        q = hs.permute(1, 0, 2)
        q_desc = self.query_proj(q)

        k_desc = self.feat_proj(feats)
        B, D, H, W = k_desc.shape
        k_desc = k_desc.flatten(2).transpose(1, 2)

        q_desc = F.normalize(q_desc, dim=-1)
        k_desc = F.normalize(k_desc, dim=-1)

        return torch.matmul(q_desc, k_desc.transpose(1, 2))  # (B, Q, H*W)


class MaskHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1),
            nn.GroupNorm(16, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(16, d_model // 4),
            nn.Conv2d(d_model // 4, 1, 1),
        )

    def forward(self, feats):
        return self.head(feats)
