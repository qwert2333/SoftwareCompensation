import torch
import torch.nn as nn
import torch.nn.functional as F


class LongitudinalHead(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, prof: torch.Tensor) -> torch.Tensor:
        x = self.net(prof)
        x = x.mean(dim=2)
        return self.proj(x)


class TimeAwareFEMNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        aux_dim: int = 2,
        use_longitudinal_profile: bool = False,
        long_dim: int = 64,
    ):
        super().__init__()
        self.use_longitudinal_profile = bool(use_longitudinal_profile)
        self.aux_dim = int(aux_dim)

        self.ecal_branch = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.hcal_branch = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        if self.use_longitudinal_profile:
            self.long_head = LongitudinalHead(in_channels=64, out_dim=long_dim, hidden=64)
        if self.aux_dim > 0:
            self.aux_fc = nn.Sequential(nn.Linear(self.aux_dim, 32), nn.ReLU())

        fusion_in = 64 + (32 if self.aux_dim > 0 else 0) + (long_dim if self.use_longitudinal_profile else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.out = nn.Linear(32, 1)

    def forward(self, ecal, hcal, aux):
        e = self.ecal_branch(ecal)
        h = self.hcal_branch(hcal)
        h = F.interpolate(h, size=e.shape[2:], mode="trilinear", align_corners=False)
        shared_in = torch.cat([e, h], dim=1)
        parts = [self.shared(shared_in)]

        if self.use_longitudinal_profile:
            prof = shared_in.mean(dim=(2, 4))
            parts.append(self.long_head(prof))

        if self.aux_dim > 0:
            parts.append(self.aux_fc(aux))
        y = self.fusion(torch.cat(parts, dim=1))
        return self.out(y).squeeze(1)
