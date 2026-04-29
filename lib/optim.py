import math

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, SequentialLR


def construct_optimizer(model: torch.nn.Module, total_iters, cfg: DictConfig):
    backbone_lr = getattr(cfg, "backbone_lr", cfg.lr)
    params = split_parameters(model, cfg.weight_decay, cfg.lr, backbone_lr)

    if cfg.name == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    elif cfg.name == "Adam":
        optimizer = torch.optim.Adam(params, lr=cfg.lr)
    elif cfg.name == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg.lr)
    else:
        raise NotImplementedError("Optimizer not implemented")

    backbone_eta_min = getattr(cfg, "backbone_eta_min", cfg.eta_min)

    def make_lr_lambda(eta_min_ratio):
        def lr_lambda(step):
            if step >= total_iters - cfg.warmup_iters:
                return eta_min_ratio
            progress = step / (total_iters - cfg.warmup_iters)
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        return lr_lambda

    lr_lambdas = []
    for group in params:
        if group.get("is_backbone", False):
            lr_lambdas.append(make_lr_lambda(backbone_eta_min / backbone_lr))
        else:
            lr_lambdas.append(make_lr_lambda(cfg.eta_min / cfg.lr))

    cosine_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)
    if cfg.warmup_iters > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / cfg.warmup_iters)
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.warmup_iters],
        )
    else:
        lr_scheduler = cosine_scheduler

    return optimizer, lr_scheduler


def freeze_lora(model):
    for name, param in model.named_parameters():
        if any(k in name for k in ("A_q", "B_q", "A_k", "B_k", "A_v", "B_v")):
            param.requires_grad_(False)
    if hasattr(model, "backbone") and hasattr(model.backbone, "lora_frozen"):
        model.backbone.lora_frozen = True
        model.backbone.backbone.eval()


def split_parameters(model: torch.nn.Module, wd: float, lr: float, backbone_lr: float):
    decay = set()
    no_decay = set()
    backbone_params = set()
    whitelist_weight_modules = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.MultiheadAttention, nn.Parameter)
    blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if not p.requires_grad:
                continue

            if fpn.startswith("backbone.backbone."):
                backbone_params.add(fpn)

            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn in ("cls_token", "storage_tokens", "mask_token", "logit_scale", "mask_pos") or pn.endswith(".gamma"):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, (
        "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
    )

    backbone_decay = decay & backbone_params
    backbone_no_decay = no_decay & backbone_params
    other_decay = decay - backbone_params
    other_no_decay = no_decay - backbone_params

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(backbone_decay)], "weight_decay": wd, "lr": backbone_lr, "is_backbone": True},
        {"params": [param_dict[pn] for pn in sorted(backbone_no_decay)], "weight_decay": 0.0, "lr": backbone_lr, "is_backbone": True},
        {"params": [param_dict[pn] for pn in sorted(other_decay)], "weight_decay": wd, "lr": lr, "is_backbone": False},
        {"params": [param_dict[pn] for pn in sorted(other_no_decay)], "weight_decay": 0.0, "lr": lr, "is_backbone": False},
    ]
    optim_groups = [g for g in optim_groups if len(g["params"]) > 0]
    return optim_groups
