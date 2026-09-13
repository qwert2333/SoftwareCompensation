"""Early-fusion 3D CNN for tile-HCAL scintillation/Cherenkov readout."""

import torch
import torch.nn as nn


def _group_count(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct3d(channels, channels),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class DualReadoutHCALNet(nn.Module):
    def __init__(self, in_channels=2, aux_dim=11, long_dim=64, residual_scale=10.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        # Keep the existing early-fusion S/C input, but make the trunk more
        # stable for small batches with GroupNorm and shallow residual blocks.
        self.encoder = nn.Sequential(
            ConvNormAct3d(in_channels, 16),
            ResidualBlock3d(16),
            ConvNormAct3d(16, 32, stride=(2, 1, 2)),
            ResidualBlock3d(32),
            ConvNormAct3d(32, 64),
            ResidualBlock3d(64),
        )
        self.global_head = nn.Sequential(
            ConvNormAct3d(64, 64),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.longitudinal_head = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, long_dim),
            nn.SiLU(inplace=True),
        )
        self.aux_head = nn.Sequential(nn.Linear(aux_dim, 32), nn.SiLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Linear(64 + long_dim + 32, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self._init_residual_head()

    def _init_residual_head(self):
        final = self.fusion[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, voxel, aux):
        encoded = self.encoder(voxel)
        global_features = self.global_head(encoded)
        longitudinal_profile = encoded.mean(dim=(2, 4))
        longitudinal_features = self.longitudinal_head(longitudinal_profile)
        aux_features = self.aux_head(aux)
        raw_residual = self.fusion(
            torch.cat([global_features, longitudinal_features, aux_features], dim=1)
        ).squeeze(1)
        return self.residual_scale * torch.tanh(raw_residual)


def residual_base_from_batch(batch, mode, device=None):
    """Select the physics baseline used by the scaled residual head."""
    if mode == "s_only":
        key = "s_total"
    elif mode == "sc_full":
        key = "dr_reco"
    else:
        raise ValueError(f"Unknown dual-readout mode: {mode}")
    base = batch[key]
    if device is not None:
        base = base.to(device, non_blocking=True)
    return base


def energy_from_scaled_residual(residual, base, eps=0.7):
    """Convert network residual r to E_pred = E_base + sqrt(max(E_base, eps)) * r."""
    scale = torch.sqrt(torch.clamp(base, min=float(eps)))
    return base + scale * residual
