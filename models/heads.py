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
        # mesh_descriptors: (num_vertices, B, C) from MeshDecoder.
        pooled = mesh_descriptors.mean(dim=0)
        quat = self.head(pooled)
        return F.normalize(quat, dim=-1, eps=1e-8)
