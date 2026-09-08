"""Dual-readout cell aggregation and reconstructed event features.

Only the central transverse crop is used. All scalar features are recomputed
from that crop so cells outside it cannot leak information into the model.
"""

from dataclasses import dataclass

import numpy as np


CHANNEL_BASE = 1_000_000_000
AUX_NAMES = (
    "log1p_S_total_GeV",
    "log1p_C_total_GeV",
    "log1p_n_active_S",
    "log1p_n_active_C",
    "C_over_S",
    "S_minus_C_over_sum",
    "h_over_e_S_reco",
    "h_over_e_C_reco",
    "dualreadout_chi_reco",
    "signed_log1p_E_DR_reco_GeV",
    "E_DR_over_S",
)


@dataclass(frozen=True)
class DualReadoutGeometry:
    nx: int = 60
    ny: int = 60
    nz: int = 100
    crop_nx: int = 30
    crop_ny: int = 30

    @property
    def cell_base(self) -> int:
        base = 10
        while base <= max(self.nx, self.ny, self.nz):
            base *= 10
        return base

    @property
    def crop_bounds(self):
        if self.crop_nx > self.nx or self.crop_ny > self.ny:
            raise ValueError("Central crop exceeds the detector grid")
        x0 = (self.nx - self.crop_nx) // 2
        y0 = (self.ny - self.crop_ny) // 2
        return x0, x0 + self.crop_nx, y0, y0 + self.crop_ny


@dataclass(frozen=True)
class DualReadoutCalibration:
    # Electron EM calibrations for ZnWO4_5_Quartz_5_Steel_10.
    s_gev_per_count: float = 0.0232917 / 1000.0
    c_gev_per_count: float = 0.100991 / 1000.0
    # h/e(E)=intercept+slope*E from pion_hovere_energy_fit.csv.
    h_s_intercept: float = 0.699321
    h_s_slope_per_gev: float = -0.000170119
    h_c_intercept: float = 0.404076
    h_c_slope_per_gev: float = -0.000260572
    max_reco_energy_gev: float = 200.0


def _decode_physical_ids(physical_ids, geo):
    base = geo.cell_base
    iz = physical_ids // (base * base) - 1
    rest = physical_ids % (base * base)
    iy = rest // base - 1
    ix = rest % base - 1
    valid = (
        (ix >= 0) & (ix < geo.nx)
        & (iy >= 0) & (iy < geo.ny)
        & (iz >= 0) & (iz < geo.nz)
    )
    return ix, iy, iz, valid


def _fill_channel(cell_ids, counts, channel_id, scale, geo, out):
    channel = cell_ids // CHANNEL_BASE
    selected = (channel == channel_id) & (counts > 0)
    if not np.any(selected):
        return
    physical = (cell_ids[selected] % CHANNEL_BASE).astype(np.int64, copy=False)
    values = counts[selected].astype(np.float64, copy=False) * float(scale)
    ix, iy, iz, valid = _decode_physical_ids(physical, geo)
    if not np.all(valid):
        raise RuntimeError("Positive S/C signal has an invalid physical CellID")

    x0, x1, y0, y1 = geo.crop_bounds
    inside = (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
    # Layout is [x transverse, z longitudinal, y transverse].
    np.add.at(out, (ix[inside] - x0, iz[inside], iy[inside] - y0), values[inside])


def reconstruct_standard_dr(s_total, c_total, calibration, n_iter=6):
    """Evaluate h/e(E), chi(E), and DR using only reconstructed S and C."""
    energy = float(np.clip(s_total, 0.0, calibration.max_reco_energy_gev))
    h_s = h_c = chi = 0.0
    e_dr = energy
    for _ in range(int(n_iter)):
        h_s = calibration.h_s_intercept + calibration.h_s_slope_per_gev * energy
        h_c = calibration.h_c_intercept + calibration.h_c_slope_per_gev * energy
        chi = (1.0 - h_s) / max(1.0 - h_c, 1.0e-8)
        denom = 1.0 - chi
        if abs(denom) < 1.0e-8:
            raise RuntimeError("Invalid reconstructed dual-readout denominator")
        e_dr = (float(s_total) - chi * float(c_total)) / denom
        energy = float(np.clip(e_dr, 0.0, calibration.max_reco_energy_gev))
    return float(h_s), float(h_c), float(chi), float(e_dr)


def build_aux(mode, s_voxel, c_voxel, calibration):
    s_total = float(np.sum(s_voxel, dtype=np.float64))
    n_s = int(np.count_nonzero(s_voxel))
    aux = np.zeros(len(AUX_NAMES), dtype=np.float32)
    aux[0] = np.log1p(max(s_total, 0.0))
    aux[2] = np.log1p(n_s)
    diagnostics = {"S_total_GeV": s_total, "n_active_S": n_s}

    if mode == "s_only":
        diagnostics.update({"C_total_GeV": 0.0, "n_active_C": 0, "E_DR_reco_GeV": 0.0})
        return aux, diagnostics
    if mode != "sc_full":
        raise ValueError(f"Unknown dual-readout mode: {mode}")

    c_total = float(np.sum(c_voxel, dtype=np.float64))
    n_c = int(np.count_nonzero(c_voxel))
    h_s, h_c, chi, e_dr = reconstruct_standard_dr(s_total, c_total, calibration)
    eps = 1.0e-8
    aux[1] = np.log1p(max(c_total, 0.0))
    aux[3] = np.log1p(n_c)
    aux[4] = c_total / max(s_total, eps)
    aux[5] = (s_total - c_total) / max(s_total + c_total, eps)
    aux[6:9] = (h_s, h_c, chi)
    aux[9] = np.sign(e_dr) * np.log1p(abs(e_dr))
    aux[10] = e_dr / max(s_total, eps)
    diagnostics.update({
        "C_total_GeV": c_total,
        "n_active_C": n_c,
        "h_over_e_S_reco": h_s,
        "h_over_e_C_reco": h_c,
        "chi_reco": chi,
        "E_DR_reco_GeV": e_dr,
    })
    return aux, diagnostics


def build_event_input(cell_ids, n_scint, n_cherenkov, mode, geo, calibration,
                      cell_signal_scale_gev=1.0e-3):
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    n_scint = np.asarray(n_scint, dtype=np.int64)
    if cell_ids.size != n_scint.size:
        raise RuntimeError("vecCellID and vecNScint lengths differ")

    s_voxel = np.zeros((geo.crop_nx, geo.nz, geo.crop_ny), dtype=np.float32)
    c_voxel = np.zeros_like(s_voxel)
    _fill_channel(cell_ids, n_scint, 1, calibration.s_gev_per_count, geo, s_voxel)

    if mode == "sc_full":
        if n_cherenkov is None:
            raise RuntimeError("sc_full requires vecNChren")
        n_cherenkov = np.asarray(n_cherenkov, dtype=np.int64)
        if cell_ids.size != n_cherenkov.size:
            raise RuntimeError("vecCellID and vecNChren lengths differ")
        _fill_channel(cell_ids, n_cherenkov, 2, calibration.c_gev_per_count, geo, c_voxel)
    elif mode != "s_only":
        raise ValueError(f"Unknown dual-readout mode: {mode}")

    aux, diagnostics = build_aux(mode, s_voxel, c_voxel, calibration)
    stacked = np.stack([s_voxel, c_voxel])
    stacked = np.log1p(stacked / float(cell_signal_scale_gev)).astype(np.float32)
    return stacked, aux, diagnostics


def split_name(event_id, energy_gev, seed, train_fraction=0.70, val_fraction=0.15):
    """Stable energy-stratified event split shared by both ablations."""
    value = (
        int(event_id) * 0x9E3779B1
        + int(round(float(energy_gev) * 1000.0)) * 0x85EBCA6B
        + int(seed)
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    u = value / float(2**32)
    if u < train_fraction:
        return "train"
    if u < train_fraction + val_fraction:
        return "val"
    return "test"
