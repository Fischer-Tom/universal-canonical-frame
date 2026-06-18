import warnings

import torch
from omegaconf import DictConfig
from torch import nn

from models.dino import (
    Feature2Pyramid,
    LoRAQKV,
    MLPWithAdapter,
    UPerNetDecoderWithAux,
    inject_adapters_and_lora_into_vit,
    set_train_mode_for_trainable_modules,
)
from models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from models.layers.vision_transformer import DinoVisionTransformer


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(
            patch_size=patch_size, embed_dim=embed_dim
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.empty(1, 2, num_register_tokens, embed_dim)
        )
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(
                    f"register_attention_block_indices contains invalid block index {idx}"
                )
            self.inter_frame_attention_types[idx] = "register"

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(
            self.camera_token, batch_size, num_frames
        )
        register_token = slice_expand_and_flatten(
            self.register_token, batch_size, num_frames
        )

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(
                H=patch_grid_size[0], W=patch_grid_size[1]
            )
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](
            camera_and_register_tokens, None
        )
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(
    token_tensor: torch.Tensor, batch_size: int, num_frames: int
) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(
        batch_size, 1, *token_tensor.shape[2:]
    )
    other_frame_tokens = token_tensor[:, 1:].expand(
        batch_size, num_frames - 1, *token_tensor.shape[2:]
    )
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError(
                "Aggregator did not cache the final layer, which VGGTOmega needs."
            )

        predictions = {
            "camera_and_register_tokens": final_tokens[
                :, :, :patch_token_start
            ].contiguous(),
        }

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (
        ("aggregator.patch_embed", aggregator.patch_embed),
        ("aggregator", aggregator),
    ):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )


def inject_adapters_and_lora_into_vggt_aggregator(
    aggregator: Aggregator,
    *,
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
    aggregator.requires_grad_(False)
    depth = aggregator.depth
    if target_blocks is None:
        target_blocks = range(depth)

    for i in target_blocks:
        if i < 0 or i >= depth:
            raise ValueError(f"VGGT target block index out of range: {i}")
        for stream_name, blocks in (
            ("frame_blocks", aggregator.frame_blocks),
            ("inter_frame_blocks", aggregator.inter_frame_blocks),
        ):
            block = blocks[i]
            if use_lora:
                attn = block.attn
                if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                    attn.qkv = LoRAQKV(
                        attn.qkv,
                        r=lora_r,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                        target=lora_targets,
                    )
                else:
                    raise ValueError(
                        f"{stream_name}[{i}]: attn.qkv is not a fused nn.Linear"
                    )
            if use_adapter:
                if hasattr(block.norm1, "normalized_shape"):
                    dim = block.norm1.normalized_shape[0]
                else:
                    dim = block.norm1.weight.shape[0]
                block.mlp = MLPWithAdapter(
                    block.mlp,
                    dim=dim,
                    bottleneck_dim=adapter_bottleneck,
                    s=adapter_scale,
                    p=adapter_dropout,
                )
    return aggregator


class VGGTExtractor(nn.Module):
    """Single-frame frozen VGGT aggregator + project-and-upsampling neck.

    VGGT's aggregator is video-shaped, but this extractor deliberately feeds one
    frame per sample to make the experiment comparable to the DINO backbone.
    `vggt_feature_source="patch_embed"` uses only the DINO-style ViT inside
    VGGT and skips the heavy frame/inter-frame aggregator blocks. The full
    aggregator path remains available with `vggt_feature_source="aggregator"`.
    """

    def __init__(
        self,
        out_ch: int,
        cfg: DictConfig,
        out_indices=None,
        adapt: bool = False,
    ):
        super().__init__()
        weights_path = cfg.vggt.remote_weights
        self.backbone = VGGTOmega()
        self.backbone.load_state_dict(
            torch.load(weights_path, map_location="cpu"), strict=False
        )

        self.finetune = cfg.get("finetune_backbone", False)
        if not self.finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.patch_size = self.backbone.patch_size
        self.embed_dim = self.backbone.embed_dim
        self.feature_source = str(cfg.get("vggt_feature_source", "patch_embed"))
        cached_layers = sorted(self.backbone.aggregator.cached_layer_indices)
        if len(cached_layers) != 4:
            raise ValueError(
                f"VGGTExtractor expects 4 cached layers for the FPN, got {cached_layers}"
            )
        self.cached_layers = tuple(cached_layers)
        if self.feature_source == "patch_embed":
            self.input_dim = self.embed_dim
            self.out_indices = tuple(self.cached_layers)
        elif self.feature_source == "aggregator":
            self.input_dim = self.embed_dim * 2
            self.out_indices = tuple(self.cached_layers)
        else:
            raise ValueError(
                "vggt_feature_source must be 'patch_embed' or 'aggregator', "
                f"got {self.feature_source!r}"
            )
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(self.input_dim, out_ch, kernel_size=1) for _ in self.out_indices]
        )
        self.neck = Feature2Pyramid(embed_dim=out_ch)
        self.decoder = UPerNetDecoderWithAux(
            in_channels=[out_ch] * 4,
            ppm_channels=out_ch,
            fpn_channels=out_ch,
        )
        self.adapt = bool(adapt or cfg.get("adapt", False))
        self.use_lora = bool(cfg.get("use_lora", False))
        self.lora_target = str(cfg.get("vggt_lora_target", "patch_embed"))
        self.lora_frozen = False
        if (self.use_lora or self.adapt) and not self.finetune:
            if self.lora_target == "patch_embed":
                inject_adapters_and_lora_into_vit(
                    self.backbone.aggregator.patch_embed,
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
            elif self.lora_target == "aggregator":
                inject_adapters_and_lora_into_vggt_aggregator(
                    self.backbone.aggregator,
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
            else:
                raise ValueError(
                    "vggt_lora_target must be 'patch_embed' or 'aggregator', "
                    f"got {self.lora_target!r}"
                )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"VGGTExtractor expects (B,3,H,W), got {x.shape}")

        B, _C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"VGGTExtractor expects H/W divisible by patch size {self.patch_size}, "
                f"got {(H, W)}"
            )

        lora_active = self.use_lora and not self.lora_frozen
        ctx = (
            torch.enable_grad()
            if (self.finetune or self.adapt or lora_active)
            else torch.no_grad()
        )
        if self.feature_source == "patch_embed":
            with ctx:
                feat_maps = list(
                    self.backbone.aggregator.patch_embed.get_intermediate_layers(
                        x,
                        n=self.out_indices,
                        reshape=True,
                        norm=True,
                        return_class_token=False,
                    )
                )
            feat_maps = [proj(fmap) for proj, fmap in zip(self.input_proj, feat_maps)]
            pyramid_feats = self.neck(tuple(feat_maps))
            return self.decoder(pyramid_feats)

        images = x.unsqueeze(1)
        with ctx:
            outputs, patch_token_start = self.backbone.aggregator(images)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        feat_maps = []
        for proj, cached in zip(self.input_proj, [o for o in outputs if o is not None]):
            patches = cached[:, 0, patch_token_start:, :]
            if patches.shape[1] != grid_h * grid_w:
                raise ValueError(
                    f"VGGT patch-token count mismatch: got {patches.shape[1]}, "
                    f"expected {grid_h * grid_w}"
                )
            fmap = patches.transpose(1, 2).reshape(B, self.input_dim, grid_h, grid_w)
            feat_maps.append(proj(fmap))

        pyramid_feats = self.neck(tuple(feat_maps))
        return self.decoder(pyramid_feats)

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            if self.lora_frozen:
                self.backbone.eval()
            else:
                set_train_mode_for_trainable_modules(self.backbone, mode=mode)
