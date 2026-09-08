"""Early-fusion 3D CNN for tile-HCAL scintillation/Cherenkov readout."""

import torch
import torch.nn as nn


class DualReadoutHCALNet(nn.Module):
    def __init__(self, in_channels=2, aux_dim=11, long_dim=64):
        super().__init__()
        # Downsample only the two transverse axes. All 100 longitudinal
        # sampling layers reach the longitudinal head.
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=(2, 1, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.global_head = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.longitudinal_head = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, long_dim),
            nn.ReLU(inplace=True),
        )
        self.aux_head = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(
            nn.Linear(64 + long_dim + 32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, voxel, aux):
        encoded = self.encoder(voxel)
        global_features = self.global_head(encoded)
        longitudinal_profile = encoded.mean(dim=(2, 4))
        longitudinal_features = self.longitudinal_head(longitudinal_profile)
        aux_features = self.aux_head(aux)
        return self.fusion(
            torch.cat([global_features, longitudinal_features, aux_features], dim=1)
        ).squeeze(1)
