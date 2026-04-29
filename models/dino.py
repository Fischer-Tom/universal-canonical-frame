import math
import os.path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn


class LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r=16, alpha=32, dropout=0.05, target=("q", "v")):
        super().__init__()
        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        assert self.out_features == 3 * self.in_features, "Expect fused qkv with out=3*in"
        self.dim = self.in_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        for p in self.qkv.parameters():
            p.requires_grad = False

        def make_pair():
            A = nn.Linear(self.dim, r, bias=False)
            B = nn.Linear(r, self.dim, bias=False)
            nn.init.kaiming_uniform_(A.weight, a=5**0.5)
            nn.init.zeros_(B.weight)
            return A, B

        self.target = set(target)
        self.A_q, self.B_q = make_pair() if "q" in self.target else (None, None)
        self.A_k, self.B_k = make_pair() if "k" in self.target else (None, None)
        self.A_v, self.B_v = make_pair() if "v" in self.target else (None, None)

    def forward(self, x):
        base = self.qkv(x)
        C = self.dim
        q, k, v = base[..., :C], base[..., C : 2 * C], base[..., 2 * C : 3 * C]

        if self.A_q is not None:
            dq = self.B_q(self.lora_dropout(self.A_q(x))) * self.scaling
            q = q + dq.to(q.dtype)
        if self.A_k is not None:
            dk = self.B_k(self.lora_dropout(self.A_k(x))) * self.scaling
            k = k + dk.to(k.dtype)
        if self.A_v is not None:
            dv = self.B_v(self.lora_dropout(self.A_v(x))) * self.scaling
            v = v + dv.to(v.dtype)

        return torch.cat([q, k, v], dim=-1)


class FFNAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64, s: float = 0.1, p: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.down = nn.Linear(dim, bottleneck_dim, bias=True)
        self.up = nn.Linear(bottleneck_dim, dim, bias=True)
        self.drop = nn.Dropout(p)
        self.s = s

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.down(self.ln(x))
        z = F.gelu(z)
        z = self.drop(z)
        z = self.up(z)
        return self.s * z


class MLPWithAdapter(nn.Module):
    def __init__(self, mlp: nn.Module, dim: int, bottleneck_dim=64, s=0.1, p=0.0):
        super().__init__()
        self.mlp = mlp
        self.adapter = FFNAdapter(dim, bottleneck_dim=bottleneck_dim, s=s, p=p)
        for p0 in self.mlp.parameters():
            p0.requires_grad = False

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        self.pool_scales = pool_scales
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        ppm_outs = [x] + [
            F.interpolate(stage(x), size=x.shape[2:], mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(ppm_outs, dim=1))


class UPerNetDecoderWithAux(nn.Module):
    def __init__(self, in_channels, ppm_channels=512, fpn_channels=512, dropout=0.1, aux_in_index=2):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=False),
            )
            for ch in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=False),
            )
            for _ in in_channels[:-1]
        ])
        self.ppm = PPM(in_channels[-1], fpn_channels, pool_scales=(1, 2, 3, 6), dropout=dropout)
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * fpn_channels, fpn_channels, 3, padding=1, bias=False),
        )
        self.aux_in_index = aux_in_index

    def forward(self, feats):
        assert len(feats) == 4, "Expecting [P2, P3, P4, P5] features"

        laterals = [l_conv(feats[i]) for i, l_conv in enumerate(self.lateral_convs)]
        top = self.ppm(feats[-1])
        laterals.append(top)

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )
            laterals[i - 1] = laterals[i - 1] + up

        fpn_outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        fpn_outs.append(laterals[-1])

        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False
            )

        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


class Feature2Pyramid(nn.Module):
    def __init__(self, embed_dim, rescales=(4, 2, 1, 0.5)):
        super().__init__()
        self.rescales = rescales
        self.ops = nn.ModuleList()
        for r in rescales:
            if r == 4:
                self.ops.append(nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2, bias=False),
                    nn.GroupNorm(32, embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ))
            elif r == 2:
                self.ops.append(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            elif r == 1:
                self.ops.append(nn.Identity())
            elif r == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise KeyError(f"Invalid rescale factor: {r}")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        return tuple(self.ops[i](f) for i, f in enumerate(inputs))


class DINO(nn.Module):
    def __init__(
        self,
        out_ch: int,
        cfg: DictConfig,
        out_indices=(6, 14, 18, 23),
        adapt: bool = False,
    ):
        super().__init__()
        weights_path = cfg.remote_weights if os.path.exists(cfg.remote_weights) else cfg.local_weights
        repo_path = cfg.remote_repo_dir if os.path.exists(cfg.remote_repo_dir) else cfg.local_repo_dir
        self.backbone = torch.hub.load(repo_path, cfg.model, source="local", pretrained=False)
        self.backbone.load_state_dict(torch.load(weights_path))
        

        self.finetune = cfg.get("finetune_backbone", False)
        if not self.finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.neck = Feature2Pyramid(embed_dim=self.backbone.embed_dim)
        self.decoder = UPerNetDecoderWithAux(
            in_channels=[self.backbone.embed_dim] * 4,
            ppm_channels=out_ch,
            fpn_channels=out_ch,
        )
        self.patch_size = 16
        depth = len(self.backbone.blocks)
        req = list(out_indices)
        if any(i >= depth for i in req):
            self.out_indices = [round((k + 1) * depth / 4) - 1 for k in range(4)]
        else:
            self.out_indices = req
        self.adapt = adapt
        self.use_lora = cfg.get("use_lora", False)
        self.lora_frozen = False
        if (self.use_lora or self.adapt) and not self.finetune:
            inject_adapters_and_lora_into_vit(
                self.backbone,
                use_lora=self.use_lora,
                use_adapter=self.adapt,
                lora_r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                lora_targets=("q", "v"),
                adapter_bottleneck=cfg.bottleneck_dim,
                adapter_scale=cfg.get("adapter_scale", 0.1),
                adapter_dropout=cfg.get("adapter_dropout", 0.0),
                target_blocks=cfg.get("adapt_target_blocks", None),
            )

    def forward(self, x):
        lora_active = self.use_lora and not self.lora_frozen
        ctx = torch.enable_grad() if (self.finetune or self.adapt or lora_active) else torch.no_grad()
        with ctx:
            feat_maps = self.backbone.get_intermediate_layers(
                x, n=self.out_indices, reshape=True, norm=True, return_class_token=False
            )
        feat_maps = list(feat_maps)
        pyramid_feats = self.neck(tuple(feat_maps))
        return self.decoder(pyramid_feats)

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            if self.lora_frozen:
                self.backbone.eval()
            else:
                set_train_mode_for_trainable_modules(self.backbone, mode=mode)


def set_train_mode_for_trainable_modules(module: nn.Module, mode: bool = True):
    module.eval()
    for m in module.modules():
        if any(p.requires_grad for p in m.parameters(recurse=False)):
            m.train(mode)


def inject_adapters_and_lora_into_vit(
    vit,
    use_lora: bool,
    use_adapter: bool,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_targets=("q", "v"),
    adapter_bottleneck=64,
    adapter_scale=0.1,
    adapter_dropout=0.0,
    target_blocks=None,
):
    vit.requires_grad_(False)
    blocks = vit.blocks
    if target_blocks is None:
        target_blocks = range(len(blocks))

    for i in target_blocks:
        blk = blocks[i]
        if use_lora:
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = LoRAQKV(
                    attn.qkv, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target=lora_targets
                )
            else:
                raise ValueError(f"Block {i}: attn.qkv is not a fused nn.Linear; need custom mapping.")
        if use_adapter:
            if hasattr(blk.norm1, "normalized_shape"):
                dim = blk.norm1.normalized_shape[0]
            else:
                dim = blk.norm1.weight.shape[0]
            blk.mlp = MLPWithAdapter(
                blk.mlp,
                dim=dim,
                bottleneck_dim=adapter_bottleneck,
                s=adapter_scale,
                p=adapter_dropout,
            )
    return vit
