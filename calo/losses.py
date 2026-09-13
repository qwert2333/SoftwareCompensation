"""Shared regression objectives."""

import torch
import torch.nn.functional as F


def relative_residual(pred, target, denom_min=1.0):
    denom = torch.clamp(target, min=float(denom_min))
    return (pred - target) / denom


def loss_weights(target, cfg):
    if not cfg:
        return torch.ones_like(target)
    kind = cfg.get("kind", "exp")
    if kind == "exp":
        alpha = float(cfg.get("alpha", 0.3))
        e0 = float(cfg.get("e0", 3.0))
        return 1.0 + alpha * torch.exp(-target / e0)
    raise ValueError(f"Unknown loss weight kind: {kind}")


def sigma_from_target(target, cfg=None):
    cfg = cfg or {}
    if cfg.get("kind", "resolution") != "resolution":
        raise ValueError(f"Unknown sigma kind: {cfg.get('kind')}")
    stochastic = float(cfg.get("stochastic", 0.45))
    constant = float(cfg.get("constant", 0.03))
    noise = float(cfg.get("noise", 0.0))
    sigma_min = float(cfg.get("sigma_min", 0.05))
    target = torch.clamp(target, min=1e-6)
    sigma2 = (stochastic * torch.sqrt(target)) ** 2 + (constant * target) ** 2 + noise ** 2
    return torch.clamp(torch.sqrt(sigma2), min=sigma_min)


def compute_loss(pred, target, weight_cfg=None, denom_min=1.0,
                 loss_name="smooth_l1_relative", sigma_cfg=None):
    if loss_name in {"smooth_l1_relative", "l1_relative", "relative_mse"}:
        resid = relative_residual(pred, target, denom_min=denom_min)
    elif loss_name in {"smooth_l1_standardized", "l1_standardized"}:
        resid = (pred - target) / sigma_from_target(target, sigma_cfg)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    if loss_name.startswith("smooth_l1"):
        base = F.smooth_l1_loss(resid, torch.zeros_like(resid), reduction="none")
    elif loss_name == "relative_mse":
        base = 0.5 * resid.pow(2)
    else:
        base = torch.abs(resid)
    if weight_cfg:
        base = base * loss_weights(target, weight_cfg)
    return torch.mean(base)
