#!/usr/bin/env python3
"""CALICE-style cell-level local software compensation for ZnWO4 samples.

This is an independent, non-ML reconstruction chain based on the local
software-compensation prescription in arXiv:1705.10363.  It aggregates
simulation steps into readout cells, derives ten local-density bins using only
the calibration split, fits the nine energy-dependent weight parameters, and
evaluates the result on an independent application split.

Only reconstructed scintillation information is used by the compensation:
  * vecCellID identifies the readout cell;
  * vecNScint is summed per scintillation cell;
  * the electron S calibration converts counts to EM-scale GeV.

vecEdep, Cherenkov information, truthFem, and the beam-energy label are not
used when applying the fitted correction to an event.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.optimize import least_squares
from scipy.stats import norm


ENERGIES = (5, 10, 20, 30, 40, 50)
N_DENSITY_BINS = 10
ENERGY_SCALE_GEV = 50.0

# Runtime-selectable naming lets closely related reconstruction chains reuse
# the validated fitting, metric and plotting code without copying it. The
# defaults below preserve the original S-hit pipeline exactly.
BASELINE_ENERGY_COLUMN = "S_EM_GeV"
BASELINE_METHOD = "S_EM_unweighted"
BASELINE_LABEL = "S EM-scale"
HIT_ENERGY_LABEL = "S hit energy"
CLOSURE_SYMBOL = r"E_S"
CORRECTED_METHOD = "CALICE_local_SC"
CORRECTED_LABEL = "CALICE local SC"
ACTIVE_CELL_COLUMN = "n_active_S_cells"
BIN_SUM_COLUMN = "sum_binned_hit_energy_GeV"
IMPROVEMENT_COLUMN = "resolution_improvement_vs_S_percent"


@dataclass(frozen=True)
class Geometry:
    nx: int
    ny: int
    nz: int
    transverse_x_mm: float
    transverse_y_mm: float
    sampling_depth_mm: float

    @property
    def cell_volume_cm3(self) -> float:
        return (
            self.transverse_x_mm
            * self.transverse_y_mm
            * self.sampling_depth_mm
            / 1000.0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CALICE-style hit-level local software compensation."
    )
    parser.add_argument("--geometry", default="ZnWO4_5_Quartz_5_Steel_10")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--calibration-table", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=170510363)
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Maximum events per energy; negative means all events.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse intermediate/cell_hits.root if it already exists.",
    )
    return parser.parse_args()


def discover_samples(input_dir: Path) -> dict[int, Path]:
    samples: dict[int, Path] = {}
    pattern = re.compile(r"_pi-_([0-9]+)GeV\.root$")
    for path in sorted(input_dir.glob("*.root")):
        match = pattern.search(path.name)
        if match:
            samples[int(match.group(1))] = path
    missing = sorted(set(ENERGIES) - set(samples))
    if missing:
        raise RuntimeError(f"Missing pion samples at energies: {missing}")
    return {energy: samples[energy] for energy in ENERGIES}


def parse_geometry(sample_name: str) -> Geometry:
    transverse = re.search(
        r"Voxel([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)mm", sample_name
    )
    longitudinal = re.search(
        r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)-"
        r"([0-9]+(?:\.[0-9]+)?)mm_N",
        sample_name,
    )
    counts = re.search(r"_N([0-9]+)x([0-9]+)x([0-9]+)_", sample_name)
    if not transverse or not longitudinal or not counts:
        raise RuntimeError(f"Cannot parse geometry from sample name: {sample_name}")
    return Geometry(
        nx=int(counts.group(1)),
        ny=int(counts.group(2)),
        nz=int(counts.group(3)),
        transverse_x_mm=float(transverse.group(1)),
        transverse_y_mm=float(transverse.group(2)),
        sampling_depth_mm=sum(float(longitudinal.group(i)) for i in (1, 2, 3)),
    )


def read_s_calibration(path: Path) -> float:
    table = pd.read_csv(path)
    rows = table.loc[table["channel"].astype(str).str.upper() == "S"]
    if rows.empty:
        raise RuntimeError(f"No S calibration row in {path}")
    value_column = next(
        (
            name
            for name in (
                "calibration_MeV_per_readout",
                "calibration_MeV_per_count",
            )
            if name in rows.columns
        ),
        None,
    )
    if value_column is None:
        raise RuntimeError(
            f"No calibration_MeV_per_readout column in {path}; "
            f"available columns are {list(rows.columns)}"
        )
    values = rows[value_column].to_numpy(dtype=float)
    if not np.allclose(values, values[0], rtol=0.0, atol=5e-7):
        raise RuntimeError(f"Inconsistent S calibration constants in {path}")
    return float(values[0]) / 1000.0


def cell_base(geometry: Geometry) -> int:
    base = 10
    while base <= max(geometry.nx, geometry.ny, geometry.nz):
        base *= 10
    return base


def split_code(event_id: int, energy_gev: int, seed: int) -> int:
    """Stable 50/50 split: 0 calibration, 1 application."""
    value = (
        (int(event_id) * 0x9E3779B1)
        + (int(energy_gev) * 0x85EBCA6B)
        + int(seed)
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    return int(value & 1)


def aggregate_scintillation_cells(
    cell_ids: np.ndarray,
    n_scint: np.ndarray,
    geometry: Geometry,
) -> tuple[np.ndarray, int, int]:
    """Return positive S counts per unique scintillation readout cell."""
    if cell_ids.size != n_scint.size:
        raise RuntimeError("vecCellID and vecNScint have different lengths")
    base = cell_base(geometry)
    layer_offset = base**3
    layer = cell_ids // layer_offset
    voxel_copy = cell_ids - layer * layer_offset
    iz = voxel_copy // (base * base) - 1
    rest = voxel_copy % (base * base)
    iy = rest // base - 1
    ix = rest % base - 1

    positive_s = n_scint > 0
    invalid_positive = positive_s & (
        ((layer != 1) & (layer != 2))
        | (ix < 0)
        | (ix >= geometry.nx)
        | (iy < 0)
        | (iy >= geometry.ny)
        | (iz < 0)
        | (iz >= geometry.nz)
    )
    if np.any(invalid_positive):
        bad = np.flatnonzero(invalid_positive)[:5]
        raise RuntimeError(f"Positive S signal has invalid CellID at indices {bad.tolist()}")

    selected = positive_s & (layer == 1)
    ids = cell_ids[selected].astype(np.int64, copy=False)
    counts = n_scint[selected].astype(np.float64, copy=False)
    if ids.size == 0:
        return np.empty(0, dtype=np.float64), 0, int(np.sum(n_scint))

    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    counts = counts[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    per_cell = np.add.reduceat(counts, starts)
    return per_cell, int(np.sum(counts)), int(np.sum(n_scint))


def export_cell_cache(
    samples: dict[int, Path],
    geometry: Geometry,
    s_gev_per_count: float,
    cache_path: Path,
    validation_path: Path,
    split_seed: int,
    max_events: int,
) -> None:
    event_ids: list[int] = []
    energy_values: list[float] = []
    split_values: list[int] = []
    scalar_counts: list[int] = []
    vector_counts: list[int] = []
    s_calib_values: list[float] = []
    cell_hit_energies: list[np.ndarray] = []
    cell_hit_densities: list[np.ndarray] = []
    validations: list[dict[str, object]] = []

    required = [
        "eventID",
        "counter_Scintillation_ScintLayer",
        "vecCellID",
        "vecNScint",
    ]
    volume_scale = 1000.0 / geometry.cell_volume_cm3

    for energy, sample_path in samples.items():
        with uproot.open(sample_path) as root_file:
            if "eventTree" not in root_file:
                raise RuntimeError(f"eventTree missing from {sample_path}")
            tree = root_file["eventTree"]
            missing = [branch for branch in required if branch not in tree.keys()]
            if missing:
                raise RuntimeError(f"Missing branches {missing} in {sample_path}")
            n_total = int(tree.num_entries)
            entry_stop = n_total if max_events < 0 else min(n_total, max_events)
            n_calibration = 0
            n_application = 0
            max_abs_delta = 0
            n_empty = 0
            n_events_with_offlayer_s = 0
            offlayer_s_count = 0

            for arrays in tree.iterate(
                required,
                entry_stop=entry_stop,
                step_size="80 MB",
                library="ak",
            ):
                for idx in range(len(arrays["eventID"])):
                    event_id = int(arrays["eventID"][idx])
                    scalar_s = int(arrays["counter_Scintillation_ScintLayer"][idx])
                    ids = ak.to_numpy(arrays["vecCellID"][idx]).astype(np.int64)
                    counts = ak.to_numpy(arrays["vecNScint"][idx]).astype(np.int64)
                    per_cell_counts, vector_layer_s, vector_all_s = (
                        aggregate_scintillation_cells(ids, counts, geometry)
                    )
                    if vector_layer_s != scalar_s:
                        raise RuntimeError(
                            f"S scalar/vector mismatch at {energy} GeV eventID={event_id}: "
                            f"scalar={scalar_s}, layer1-vector={vector_layer_s}, "
                            f"all-vector={vector_all_s}"
                        )
                    max_abs_delta = max(max_abs_delta, abs(vector_layer_s - scalar_s))
                    event_offlayer_s = vector_all_s - vector_layer_s
                    offlayer_s_count += event_offlayer_s
                    n_events_with_offlayer_s += int(event_offlayer_s > 0)
                    hit_energy = per_cell_counts * s_gev_per_count
                    hit_density = hit_energy * volume_scale
                    if hit_energy.size == 0:
                        n_empty += 1
                    split = split_code(event_id, energy, split_seed)
                    n_calibration += int(split == 0)
                    n_application += int(split == 1)

                    event_ids.append(event_id)
                    energy_values.append(float(energy))
                    split_values.append(split)
                    scalar_counts.append(scalar_s)
                    vector_counts.append(vector_layer_s)
                    s_calib_values.append(scalar_s * s_gev_per_count)
                    cell_hit_energies.append(hit_energy.astype(np.float32))
                    cell_hit_densities.append(hit_density.astype(np.float32))

            validations.append(
                {
                    "energy_GeV": energy,
                    "source_file": str(sample_path),
                    "source_entries": n_total,
                    "exported_entries": entry_stop,
                    "calibration_entries": n_calibration,
                    "application_entries": n_application,
                    "empty_cell_events": n_empty,
                    "events_with_offlayer_S_signal": n_events_with_offlayer_s,
                    "offlayer_S_count_ignored": offlayer_s_count,
                    "max_abs_scalar_vector_count_delta": max_abs_delta,
                    "status": "pass",
                }
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(cache_path) as output:
        output["events"] = {
            "eventID": np.asarray(event_ids, dtype=np.int32),
            "energy_GeV": np.asarray(energy_values, dtype=np.float32),
            "split": np.asarray(split_values, dtype=np.int8),
            "S_count_scalar": np.asarray(scalar_counts, dtype=np.int32),
            "S_count_vector": np.asarray(vector_counts, dtype=np.int32),
            "S_EM_GeV": np.asarray(s_calib_values, dtype=np.float32),
            "hit_energy_GeV": ak.Array(cell_hit_energies),
            "hit_density_GeV_per_1000cm3": ak.Array(cell_hit_densities),
        }
    pd.DataFrame(validations).to_csv(validation_path, index=False)


def load_cache(path: Path) -> dict[str, object]:
    with uproot.open(path) as root_file:
        tree = root_file["events"]
        arrays = tree.arrays(library="ak")
    return {
        "eventID": ak.to_numpy(arrays["eventID"]).astype(np.int32),
        "energy_GeV": ak.to_numpy(arrays["energy_GeV"]).astype(np.float64),
        "split": ak.to_numpy(arrays["split"]).astype(np.int8),
        "S_EM_GeV": ak.to_numpy(arrays["S_EM_GeV"]).astype(np.float64),
        "hit_energy": arrays["hit_energy_GeV"],
        "hit_density": arrays["hit_density_GeV_per_1000cm3"],
    }


def weighted_quantile_edges(
    values: np.ndarray, weights: np.ndarray, n_bins: int
) -> np.ndarray:
    if values.size == 0 or not np.all(values > 0):
        raise RuntimeError("Density values must be non-empty and positive")
    unique, inverse = np.unique(values, return_inverse=True)
    unique_weights = np.bincount(inverse, weights=weights, minlength=unique.size)
    cumulative = np.cumsum(unique_weights)
    total = cumulative[-1]
    boundaries: list[float] = [0.0]
    for fraction in np.arange(1, n_bins) / n_bins:
        idx = int(np.searchsorted(cumulative, fraction * total, side="left"))
        if idx >= unique.size - 1:
            idx = unique.size - 2
        boundaries.append(0.5 * (unique[idx] + unique[idx + 1]))
    boundaries.append(float("inf"))
    edges = np.asarray(boundaries, dtype=float)
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(
            "Could not construct ten distinct density bins; signal quantisation is too coarse"
        )
    return edges


def derive_density_binning(
    cache: dict[str, object], n_bins: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    split = np.asarray(cache["split"])
    energies = np.asarray(cache["energy_GeV"])
    density_jagged = cache["hit_density"]
    energy_jagged = cache["hit_energy"]

    density_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []
    quantile_weights: list[np.ndarray] = []
    for idx in np.flatnonzero(split == 0):
        density = ak.to_numpy(density_jagged[idx]).astype(np.float64)
        hit_energy = ak.to_numpy(energy_jagged[idx]).astype(np.float64)
        positive = (density > 0) & (hit_energy > 0)
        density_parts.append(density[positive])
        energy_parts.append(hit_energy[positive])
        quantile_weights.append(hit_energy[positive] / energies[idx])
    densities = np.concatenate(density_parts)
    hit_energies = np.concatenate(energy_parts)
    weights = np.concatenate(quantile_weights)
    edges = weighted_quantile_edges(densities, weights, n_bins)
    bin_index = np.digitize(densities, edges[1:-1], right=False)

    representatives = np.zeros(n_bins, dtype=float)
    rows: list[dict[str, object]] = []
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        if not np.any(mask):
            raise RuntimeError(f"Density bin {bin_id} is empty")
        representatives[bin_id] = np.average(densities[mask], weights=hit_energies[mask])
        rows.append(
            {
                "density_bin": bin_id,
                "lower_GeV_per_1000cm3": edges[bin_id],
                "upper_GeV_per_1000cm3": edges[bin_id + 1],
                "representative_GeV_per_1000cm3": representatives[bin_id],
                "n_hits_calibration": int(np.count_nonzero(mask)),
                "sum_hit_energy_GeV_calibration": float(np.sum(hit_energies[mask])),
                "normalized_energy_share": float(np.sum(weights[mask]) / np.sum(weights)),
            }
        )
    return edges, representatives, pd.DataFrame(rows)


def build_event_bin_table(
    cache: dict[str, object], edges: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    event_ids = np.asarray(cache["eventID"])
    energies = np.asarray(cache["energy_GeV"])
    split = np.asarray(cache["split"])
    baseline_energy = np.asarray(cache[BASELINE_ENERGY_COLUMN])
    density_jagged = cache["hit_density"]
    energy_jagged = cache["hit_energy"]
    n_events = event_ids.size
    bin_sums = np.zeros((n_events, len(edges) - 1), dtype=float)
    n_cells = np.zeros(n_events, dtype=int)

    for idx in range(n_events):
        density = ak.to_numpy(density_jagged[idx]).astype(np.float64)
        hit_energy = ak.to_numpy(energy_jagged[idx]).astype(np.float64)
        bin_index = np.digitize(density, edges[1:-1], right=False)
        bin_sums[idx] = np.bincount(
            bin_index, weights=hit_energy, minlength=bin_sums.shape[1]
        )
        n_cells[idx] = hit_energy.size

    vector_sum = np.sum(bin_sums, axis=1)
    if not np.allclose(vector_sum, baseline_energy, rtol=2e-6, atol=2e-6):
        bad = np.flatnonzero(~np.isclose(vector_sum, s_em, rtol=2e-6, atol=2e-6))[:5]
        raise RuntimeError(
            f"Binned hit energy does not close to {BASELINE_ENERGY_COLUMN}: rows {bad}"
        )

    table = pd.DataFrame(
        {
            "eventID": event_ids,
            "energy_GeV": energies,
            "split": np.where(split == 0, "calibration", "application"),
            ACTIVE_CELL_COLUMN: n_cells,
            BASELINE_ENERGY_COLUMN: baseline_energy,
            BIN_SUM_COLUMN: vector_sum,
        }
    )
    for bin_id in range(bin_sums.shape[1]):
        table[f"Ebin_{bin_id}_GeV"] = bin_sums[:, bin_id]
    return table, bin_sums


def canonical_parameters(theta: np.ndarray) -> dict[str, float]:
    return {
        "p10": float(theta[0]),
        "p11": float(theta[1] / ENERGY_SCALE_GEV),
        "p12": float(theta[2] / ENERGY_SCALE_GEV**2),
        "p20": float(theta[3]),
        "p21": float(theta[4] / ENERGY_SCALE_GEV),
        "p22": float(theta[5] / ENERGY_SCALE_GEV**2),
        "p30": float(theta[6]),
        "p31": float(theta[7]),
        "p32": float(theta[8] / ENERGY_SCALE_GEV),
    }


def weight_components(theta: np.ndarray, e_sum: np.ndarray) -> tuple[np.ndarray, ...]:
    x = np.asarray(e_sum, dtype=float) / ENERGY_SCALE_GEV
    p1 = theta[0] + theta[1] * x + theta[2] * x * x
    p2 = theta[3] + theta[4] * x + theta[5] * x * x
    exponent = np.clip(theta[8] * x, -60.0, 60.0)
    p3 = theta[6] / (theta[7] + np.exp(exponent))
    return p1, p2, p3


def density_weights(
    theta: np.ndarray, e_sum: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    p1, p2, p3 = weight_components(theta, e_sum)
    return p1[..., None] * np.exp(
        np.clip(p2[..., None] * np.asarray(rho), -60.0, 60.0)
    ) + p3[..., None]


def fit_weight_parameterization(
    event_table: pd.DataFrame,
    bin_sums: np.ndarray,
    representatives: np.ndarray,
) -> tuple[np.ndarray, object, np.ndarray]:
    calibration = event_table["split"].to_numpy() == "calibration"
    beam = event_table.loc[calibration, "energy_GeV"].to_numpy(dtype=float)
    e_sum = event_table.loc[calibration, BASELINE_ENERGY_COLUMN].to_numpy(dtype=float)
    bins = bin_sums[calibration]
    unique_energies, counts = np.unique(beam, return_counts=True)
    count_map = dict(zip(unique_energies, counts))
    equal_energy_factor = np.asarray(
        [math.sqrt(len(beam) / (len(unique_energies) * count_map[e])) for e in beam]
    )

    global_scale = float(np.sum(beam) / np.sum(e_sum))
    initial = np.array(
        [0.5, 0.0, 0.0, -0.15, 0.0, 0.0, max(0.2, 2 * (global_scale - 0.25)), 1.0, 0.0],
        dtype=float,
    )
    lower = np.array([0.0, -5, -5, -10, -10, -10, 0.001, 0.001, -10])
    upper = np.array([5.0, 5, 5, -1e-5, 10, 10, 10.0, 10.0, 10])
    energy_grid = np.linspace(max(0.1, e_sum.min()), e_sum.max(), 25)

    def residual(theta: np.ndarray) -> np.ndarray:
        weights = density_weights(theta, e_sum, representatives)
        reconstructed = np.sum(bins * weights, axis=1)
        statistical = (reconstructed - beam) / (0.5 * np.sqrt(beam))
        statistical *= equal_energy_factor

        p1, p2, p3 = weight_components(theta, energy_grid)
        grid_weights = density_weights(theta, energy_grid, representatives)
        penalties = np.concatenate(
            [
                30.0 * np.maximum(0.01 - p1, 0.0),
                30.0 * np.maximum(p2 + 1e-5, 0.0),
                30.0 * np.maximum(0.01 - p3, 0.0),
                10.0 * np.maximum(0.05 - grid_weights.ravel(), 0.0),
                10.0 * np.maximum(grid_weights.ravel() - 5.0, 0.0),
            ]
        )
        return np.concatenate([statistical, penalties])

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=4000,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        verbose=0,
    )
    if not result.success:
        raise RuntimeError(f"Weight fit failed: {result.message}")

    parameter_error = np.full(result.x.size, np.nan)
    try:
        n_data = len(beam)
        jacobian = result.jac[:n_data]
        covariance = np.linalg.inv(jacobian.T @ jacobian)
        dof = max(1, n_data - result.x.size)
        variance = 2.0 * result.cost / dof
        parameter_error = np.sqrt(np.diag(covariance) * variance)
    except np.linalg.LinAlgError:
        pass
    return result.x, result, parameter_error


def apply_weights(
    theta: np.ndarray,
    event_table: pd.DataFrame,
    bin_sums: np.ndarray,
    representatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    e_sum = event_table[BASELINE_ENERGY_COLUMN].to_numpy(dtype=float)
    weights = density_weights(theta, e_sum, representatives)
    reconstructed = np.sum(bin_sums * weights, axis=1)
    return reconstructed, weights


def fit_per_energy_diagnostics(
    global_theta: np.ndarray,
    event_table: pd.DataFrame,
    bin_sums: np.ndarray,
    representatives: np.ndarray,
) -> pd.DataFrame:
    """Independent three-parameter fits used only as plotting diagnostics.

    These fits do not enter the final reconstruction.  They provide honest
    per-energy points against which the global nine-parameter energy
    parameterization can be inspected.
    """
    rows: list[dict[str, object]] = []
    for energy in ENERGIES:
        mask = (
            (event_table["split"].to_numpy() == "calibration")
            & (event_table["energy_GeV"].to_numpy(dtype=float) == energy)
        )
        bins = bin_sums[mask]
        e_sum = event_table.loc[mask, BASELINE_ENERGY_COLUMN].to_numpy(dtype=float)
        mean_input = float(np.mean(e_sum))
        initial_components = weight_components(global_theta, np.array([mean_input]))
        initial = np.array([component[0] for component in initial_components])
        local_lower = np.array([0.0, -10.0, 0.0])
        local_upper = np.array([5.0, -1e-7, 5.0])
        initial = np.clip(initial, local_lower + 1.0e-9, local_upper - 1.0e-9)

        def residual(parameters: np.ndarray) -> np.ndarray:
            p1, p2, p3 = parameters
            weights = p1 * np.exp(
                np.clip(p2 * representatives, -60.0, 60.0)
            ) + p3
            reconstructed = bins @ weights
            return (reconstructed - energy) / (0.5 * math.sqrt(energy))

        fit = least_squares(
            residual,
            initial,
            bounds=(local_lower, local_upper),
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )
        errors = np.full(3, np.nan)
        try:
            covariance = np.linalg.inv(fit.jac.T @ fit.jac)
            variance = np.sum(fit.fun**2) / max(1, bins.shape[0] - 3)
            errors = np.sqrt(np.diag(covariance) * variance)
        except np.linalg.LinAlgError:
            pass
        rows.append(
            {
                "beam_energy_GeV": energy,
                "mean_unweighted_input_energy_GeV": mean_input,
                "p1_diagnostic": float(fit.x[0]),
                "p1_error": float(errors[0]),
                "p2_diagnostic": float(fit.x[1]),
                "p2_error": float(errors[1]),
                "p3_diagnostic": float(fit.x[2]),
                "p3_error": float(errors[2]),
                "chi2_like": float(np.sum(fit.fun**2)),
                "ndf": int(max(0, bins.shape[0] - 3)),
                "fit_success": bool(fit.success),
                "role": "diagnostic only; not used for final E_LSC",
            }
        )
    return pd.DataFrame(rows)


def fit_two_stage_parameterization(
    diagnostics: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit the prescribed p_i(E) forms directly to per-energy p_i points.

    This is the default auditable calibration: the energy dependence is fitted
    to independently extracted per-energy parameters. The direct event-level
    nine-parameter minimization is retained as a reference.
    """
    energy = diagnostics["mean_unweighted_input_energy_GeV"].to_numpy(dtype=float)
    x = energy / ENERGY_SCALE_GEV
    design = np.column_stack([np.ones_like(x), x, x * x])

    coefficients: list[np.ndarray] = []
    coefficient_errors: list[np.ndarray] = []
    chi2_parts: list[float] = []
    constraint_grid = np.linspace(max(0.0, 0.5 * x.min()), 1.2 * x.max(), 40)
    constraint_design = np.column_stack(
        [np.ones_like(constraint_grid), constraint_grid, constraint_grid**2]
    )
    for component_id, (value_column, error_column) in enumerate(
        (
            ("p1_diagnostic", "p1_error"),
            ("p2_diagnostic", "p2_error"),
        )
    ):
        values = diagnostics[value_column].to_numpy(dtype=float)
        errors = diagnostics[error_column].to_numpy(dtype=float)
        weighted_design = design / errors[:, None]
        weighted_values = values / errors
        initial, _, _, _ = np.linalg.lstsq(
            weighted_design, weighted_values, rcond=None
        )

        def component_residual(parameters: np.ndarray) -> np.ndarray:
            statistical = (design @ parameters - values) / errors
            grid_values = constraint_design @ parameters
            if component_id == 0:
                constraint = 50.0 * np.maximum(0.01 - grid_values, 0.0)
            else:
                constraint = 50.0 * np.maximum(grid_values + 1e-5, 0.0)
            return np.concatenate([statistical, constraint])

        component_fit = least_squares(
            component_residual,
            initial,
            x_scale="jac",
            max_nfev=5000,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
        )
        covariance = np.linalg.inv(component_fit.jac.T @ component_fit.jac)
        coefficients.append(component_fit.x)
        coefficient_errors.append(np.sqrt(np.diag(covariance)))
        chi2_parts.append(
            float(np.sum(((design @ component_fit.x - values) / errors) ** 2))
        )

    p3_values = diagnostics["p3_diagnostic"].to_numpy(dtype=float)
    p3_errors = diagnostics["p3_error"].to_numpy(dtype=float)

    def p3_residual(parameters: np.ndarray) -> np.ndarray:
        prediction = parameters[0] / (
            parameters[1] + np.exp(np.clip(parameters[2] * x, -60.0, 60.0))
        )
        return (prediction - p3_values) / p3_errors

    p3_fit = least_squares(
        p3_residual,
        np.array([1.0, 0.2, -0.1]),
        bounds=(np.array([1e-5, 1e-5, -10.0]), np.array([10.0, 10.0, 10.0])),
        x_scale="jac",
        max_nfev=10000,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    p3_covariance = np.linalg.inv(p3_fit.jac.T @ p3_fit.jac)
    p3_parameter_errors = np.sqrt(np.diag(p3_covariance))

    theta = np.concatenate([coefficients[0], coefficients[1], p3_fit.x])
    errors = np.concatenate(
        [coefficient_errors[0], coefficient_errors[1], p3_parameter_errors]
    )
    chi2_total = float(sum(chi2_parts) + np.sum(p3_fit.fun**2))
    summary = {
        "strategy": "two_stage_p_i_fit",
        "success": bool(p3_fit.success),
        "chi2": chi2_total,
        "ndf": int(3 * len(diagnostics) - len(theta)),
        "p1_chi2": chi2_parts[0],
        "p2_chi2": chi2_parts[1],
        "p3_chi2": float(np.sum(p3_fit.fun**2)),
        "role": "primary two-stage parameterization",
    }
    return theta, errors, summary


def shortest_interval(values: np.ndarray, fraction: float = 0.9) -> np.ndarray:
    ordered = np.sort(np.asarray(values, dtype=float))
    count = max(2, int(math.ceil(fraction * ordered.size)))
    widths = ordered[count - 1 :] - ordered[: ordered.size - count + 1]
    start = int(np.argmin(widths))
    return ordered[start : start + count]


def scintglass_reference_resolution(
    values: np.ndarray,
) -> tuple[float, float, float]:
    """Return MPV, sigma68 and resolution using the S-only ROOT convention.

    This reproduces ``ComputeResolution`` in
    ``RunScintGlassSOnlyEdepScintLayer.cpp``: the MPV is the centre of the
    maximum bin of an 80-bin histogram spanning the full sample with a 5%
    margin, and sigma68 is half the width of the narrowest interval containing
    round(0.68*N) entries.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    ordered = np.sort(values)
    minimum = float(ordered[0])
    maximum = float(ordered[-1])
    span = max(1.0e-9, maximum - minimum)
    histogram, edges = np.histogram(
        values,
        bins=80,
        range=(minimum - 0.05 * span, maximum + 0.05 * span),
    )
    maximum_bin = int(np.argmax(histogram))
    mpv = 0.5 * float(edges[maximum_bin] + edges[maximum_bin + 1])

    # C++ std::round is half-away-from-zero. N is positive here.
    count = max(1, int(math.floor(0.68 * ordered.size + 0.5)))
    widths = ordered[count - 1 :] - ordered[: ordered.size - count + 1]
    start = int(np.argmin(widths))
    sigma68 = 0.5 * float(ordered[start + count - 1] - ordered[start])
    resolution = sigma68 / mpv if mpv > 0.0 else float("nan")
    return mpv, sigma68, resolution


def distribution_metrics(values: np.ndarray, beam_energy: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mpv, sigma68, resolution68 = scintglass_reference_resolution(values)
    central = shortest_interval(values, 0.9)
    fit_mean, fit_sigma = norm.fit(central)
    mean90 = float(np.mean(central))
    rms90 = float(np.std(central, ddof=0))
    return {
        "entries": int(values.size),
        "mean_GeV": float(np.mean(values)),
        "std_GeV": float(np.std(values, ddof=0)),
        "median_GeV": float(np.median(values)),
        "mpv_GeV": mpv,
        "sigma68_GeV": sigma68,
        "response_mpv": float(mpv / beam_energy),
        "resolution_sigma68_over_mpv": resolution68,
        "resolution_error": float(
            resolution68 / math.sqrt(max(1.0, 2.0 * values.size))
        ),
        "mpv_error_GeV": float(sigma68 / math.sqrt(max(1.0, values.size))),
        "gauss90_mean_GeV": float(fit_mean),
        "gauss90_sigma_GeV": float(fit_sigma),
        "response_gauss90": float(fit_mean / beam_energy),
        "resolution_gauss90": float(fit_sigma / fit_mean),
        "resolution_gauss90_error": float(
            (fit_sigma / fit_mean) / math.sqrt(max(1.0, 2.0 * (central.size - 1)))
        ),
        "mean90_GeV": mean90,
        "rms90_GeV": rms90,
        "rms90_over_mean90": float(rms90 / mean90),
    }


def build_metrics(event_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("calibration", "application"):
        for energy in ENERGIES:
            subset = event_table.loc[
                (event_table["split"] == split)
                & (event_table["energy_GeV"] == energy)
            ]
            for method, column in (
                (BASELINE_METHOD, BASELINE_ENERGY_COLUMN),
                (CORRECTED_METHOD, "E_LSC_GeV"),
            ):
                row: dict[str, object] = {
                    "split": split,
                    "energy_GeV": energy,
                    "method": method,
                }
                row.update(distribution_metrics(subset[column].to_numpy(), energy))
                rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics[IMPROVEMENT_COLUMN] = np.nan
    for split in ("calibration", "application"):
        for energy in ENERGIES:
            mask = (metrics["split"] == split) & (metrics["energy_GeV"] == energy)
            baseline = float(
                metrics.loc[
                    mask & (metrics["method"] == BASELINE_METHOD),
                    "resolution_sigma68_over_mpv",
                ].iloc[0]
            )
            local_mask = mask & (metrics["method"] == CORRECTED_METHOD)
            local = float(
                metrics.loc[local_mask, "resolution_sigma68_over_mpv"].iloc[0]
            )
            metrics.loc[local_mask, IMPROVEMENT_COLUMN] = (
                100.0 * (baseline - local) / baseline
            )
    return metrics


def fit_resolution_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = metrics.loc[metrics["split"] == "application"]
    for method in (BASELINE_METHOD, CORRECTED_METHOD):
        part = selected.loc[selected["method"] == method].sort_values("energy_GeV")
        energy = part["energy_GeV"].to_numpy(dtype=float)
        resolution = part["resolution_sigma68_over_mpv"].to_numpy(dtype=float)
        error = part["resolution_error"].to_numpy(dtype=float)

        def residual(parameters: np.ndarray) -> np.ndarray:
            predicted = np.sqrt(parameters[0] ** 2 / energy + parameters[1] ** 2)
            return (predicted - resolution) / np.maximum(error, 1e-5)

        fit = least_squares(residual, np.array([0.5, 0.02]), bounds=(0.0, 5.0))
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
        parameter_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
        # At a=0 or b=0 this squared-parameter model has zero derivative and
        # the corresponding symmetric error is not identifiable.
        parameter_errors = np.where(fit.x > 1.0e-8, parameter_errors, np.nan)
        rows.append(
            {
                "split": "application",
                "method": method,
                "stochastic_term_fraction_sqrtGeV": float(fit.x[0]),
                "stochastic_term_error_fraction_sqrtGeV": float(parameter_errors[0]),
                "constant_term_fraction": float(fit.x[1]),
                "constant_term_error_fraction": float(parameter_errors[1]),
                "chi2": float(np.sum(fit.fun**2)),
                "ndf": int(max(0, energy.size - 2)),
                "formula": "sigma/E = sqrt(a^2/E + b^2)",
            }
        )
    return pd.DataFrame(rows)


def write_hit_histograms(
    cache: dict[str, object],
    output_csv: Path,
    plots_dir: Path,
    edges: np.ndarray,
    cell_volume_cm3: float,
) -> None:
    all_positive = ak.to_numpy(ak.flatten(cache["hit_density"]))
    all_positive = all_positive[all_positive > 0]
    hist_edges = np.geomspace(all_positive.min(), all_positive.max(), 101)
    rows: list[dict[str, object]] = []
    energy_values = np.asarray(cache["energy_GeV"])
    split_values = np.asarray(cache["split"])
    density_jagged = cache["hit_density"]
    hit_energy_jagged = cache["hit_energy"]

    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    for energy in ENERGIES:
        application_idx = np.flatnonzero((energy_values == energy) & (split_values == 1))
        densities = np.concatenate(
            [ak.to_numpy(density_jagged[idx]) for idx in application_idx]
        )
        counts, _ = np.histogram(densities, bins=hist_edges)
        centers = np.sqrt(hist_edges[:-1] * hist_edges[1:])
        ax.step(centers, counts / max(1, counts.sum()), where="mid", label=f"{energy} GeV")

        fig_one, ax_one = plt.subplots(figsize=(7.8, 5.4))
        ax_one.hist(densities, bins=hist_edges, histtype="step", color="black")
        for edge in edges[1:-1]:
            ax_one.axvline(edge, color="tab:blue", alpha=0.35, linewidth=0.8)
        ax_one.set_xscale("log")
        ax_one.set_yscale("log")
        ax_one.set_xlabel(r"Cell hit density $\rho$ [GeV / 1000 cm$^3$]")
        ax_one.set_ylabel("Cells / bin")
        ax_one.set_title(f"Application sample, $\pi^-$ {energy} GeV")
        fig_one.tight_layout()
        fig_one.savefig(plots_dir / f"hit_density_distribution_{energy}GeV.png", dpi=180)
        plt.close(fig_one)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cell hit density $\rho$ [GeV / 1000 cm$^3$]")
    ax.set_ylabel("Normalized cells / bin")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "hit_density_distributions_all_energies.png", dpi=180)
    plt.close(fig)

    # CALICE Fig. 2-style diagnostic for the independent 30 GeV sample.
    indices_30 = np.flatnonzero((energy_values == 30) & (split_values == 1))
    density_30 = np.concatenate([ak.to_numpy(density_jagged[idx]) for idx in indices_30])
    positive_30 = density_30[density_30 > 0]
    display_max = max(edges[-2] * 1.25, float(np.quantile(positive_30, 0.9995)))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(edges) - 1))
    fig_bins, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    linear_edges = np.linspace(0.0, display_max, 220)
    axes[0].hist(positive_30, bins=linear_edges, histtype="step", color="black")
    log_edges = np.geomspace(positive_30.min(), display_max, 170)
    axes[1].hist(positive_30, bins=log_edges, histtype="step", color="black")
    for axis in axes:
        for bin_id in range(len(edges) - 1):
            lower = max(edges[bin_id], positive_30.min() if axis is axes[1] else 0.0)
            upper = min(edges[bin_id + 1], display_max)
            if upper > lower:
                axis.axvspan(lower, upper, color=colors[bin_id], alpha=0.22)
        for edge in edges[1:-1]:
            if edge <= display_max:
                axis.axvline(edge, color="white", linewidth=0.7, alpha=0.9)
        axis.set_yscale("log")
        axis.set_ylim(bottom=0.8)
        axis.set_xlabel(r"Cell hit density $\rho$ [GeV / 1000 cm$^3$]")
        axis.set_ylabel("Cells / bin")
    axes[0].set_xlim(0.0, display_max)
    axes[0].set_title("Paper-style linear density axis")
    axes[1].set_xscale("log")
    axes[1].set_xlim(positive_30.min(), display_max)
    axes[1].set_title("Log density axis: low-density bins visible")
    mev_per_density_unit = cell_volume_cm3
    secondary = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda rho: rho * mev_per_density_unit,
            lambda hit_mev: hit_mev / mev_per_density_unit,
        ),
    )
    secondary.set_xlabel(f"Equivalent {HIT_ENERGY_LABEL} [MeV]")
    fig_bins.suptitle(
        r"30 GeV $\pi^-$ application sample: CALICE local-density bins"
    )
    fig_bins.tight_layout()
    fig_bins.savefig(
        plots_dir / "fig2_style_30GeV_hit_density_binning.png", dpi=200
    )
    plt.close(fig_bins)

    for split_code_value, split_name in ((0, "calibration"), (1, "application")):
        for energy in ENERGIES:
            indices = np.flatnonzero(
                (energy_values == energy) & (split_values == split_code_value)
            )
            densities = np.concatenate([ak.to_numpy(density_jagged[idx]) for idx in indices])
            hit_energy = np.concatenate(
                [ak.to_numpy(hit_energy_jagged[idx]) for idx in indices]
            )
            counts, _ = np.histogram(densities, bins=hist_edges)
            energy_sum, _ = np.histogram(densities, bins=hist_edges, weights=hit_energy)
            for bin_id in range(hist_edges.size - 1):
                rows.append(
                    {
                        "split": split_name,
                        "energy_GeV": energy,
                        "hist_bin": bin_id,
                        "lower_GeV_per_1000cm3": hist_edges[bin_id],
                        "upper_GeV_per_1000cm3": hist_edges[bin_id + 1],
                        "n_cells": int(counts[bin_id]),
                        "sum_hit_energy_GeV": float(energy_sum[bin_id]),
                    }
                )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def write_weight_outputs(
    theta: np.ndarray,
    parameter_error: np.ndarray,
    direct_global_theta: np.ndarray,
    event_table: pd.DataFrame,
    representatives: np.ndarray,
    edges: np.ndarray,
    per_energy_diagnostics: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
) -> None:
    canonical = canonical_parameters(theta)
    canonical_errors = canonical_parameters(parameter_error)
    parameter_rows = []
    definitions = {
        "p10": "p1(E) constant",
        "p11": "p1(E) linear coefficient [1/GeV]",
        "p12": "p1(E) quadratic coefficient [1/GeV^2]",
        "p20": "p2(E) constant",
        "p21": "p2(E) linear coefficient [1/GeV]",
        "p22": "p2(E) quadratic coefficient [1/GeV^2]",
        "p30": "p3(E) numerator",
        "p31": "p3(E) denominator constant",
        "p32": "p3(E) exponent coefficient [1/GeV]",
    }
    for name, value in canonical.items():
        parameter_rows.append(
            {
                "parameter": name,
                "value": value,
                "fit_error_approx": canonical_errors[name],
                "definition": definitions[name],
            }
        )
    pd.DataFrame(parameter_rows).to_csv(
        tables_dir / "weight_parameterization.csv", index=False
    )

    weight_rows: list[dict[str, object]] = []
    fig_density, ax_density = plt.subplots(figsize=(8.2, 5.7))
    for energy in ENERGIES:
        subset = event_table.loc[
            (event_table["split"] == "calibration")
            & (event_table["energy_GeV"] == energy)
        ]
        input_energy = float(subset[BASELINE_ENERGY_COLUMN].mean())
        weights = density_weights(theta, np.array([input_energy]), representatives)[0]
        p1, p2, p3 = weight_components(theta, np.array([input_energy]))
        for bin_id, weight in enumerate(weights):
            weight_rows.append(
                {
                    "beam_energy_GeV": energy,
                    "mean_unweighted_input_energy_GeV": input_energy,
                    "density_bin": bin_id,
                    "lower_GeV_per_1000cm3": edges[bin_id],
                    "upper_GeV_per_1000cm3": edges[bin_id + 1],
                    "representative_GeV_per_1000cm3": representatives[bin_id],
                    "weight": float(weight),
                    "p1": float(p1[0]),
                    "p2": float(p2[0]),
                    "p3": float(p3[0]),
                }
            )
        rho_curve = np.geomspace(
            max(1e-6, edges[1] / 3.0), representatives[-1] * 2.0, 250
        )
        curve = density_weights(theta, np.array([input_energy]), rho_curve)[0]
        ax_density.plot(rho_curve, curve, label=f"{energy} GeV")

    weights_table = pd.DataFrame(weight_rows)
    weights_table.to_csv(tables_dir / "hit_density_weights.csv", index=False)
    ax_density.set_xscale("log")
    ax_density.set_xlabel(r"Cell hit density $\rho$ [GeV / 1000 cm$^3$]")
    ax_density.set_ylabel(r"Weight $\omega(\rho,E_{sum})$")
    ax_density.legend(ncol=2)
    ax_density.grid(alpha=0.2)
    fig_density.tight_layout()
    fig_density.savefig(plots_dir / "weights_vs_hit_density.png", dpi=180)
    plt.close(fig_density)

    fig_energy, ax_energy = plt.subplots(figsize=(8.2, 5.7))
    e_grid = np.linspace(
        max(0.1, event_table[BASELINE_ENERGY_COLUMN].min()),
        event_table[BASELINE_ENERGY_COLUMN].max(),
        250,
    )
    selected_bins = sorted(set([0, 1, 3, 5, 7, 9]))
    for bin_id in selected_bins:
        curve = density_weights(theta, e_grid, np.array([representatives[bin_id]]))[:, 0]
        ax_energy.plot(e_grid, curve, label=f"density bin {bin_id}")
    ax_energy.set_xlabel(r"Unweighted $E_{sum}$ [GeV]")
    ax_energy.set_ylabel(r"Weight $\omega$")
    ax_energy.legend(ncol=2)
    ax_energy.grid(alpha=0.2)
    fig_energy.tight_layout()
    fig_energy.savefig(plots_dir / "weight_energy_parameterization.png", dpi=180)
    plt.close(fig_energy)

    # One panel per beam energy: independent three-parameter diagnostic points
    # versus the final global energy-dependent parameterization.
    fig_panels, axes = plt.subplots(2, 3, figsize=(14.0, 8.2), sharex=True, sharey=True)
    for axis, energy in zip(axes.flat, ENERGIES):
        diagnostic = per_energy_diagnostics.loc[
            per_energy_diagnostics["beam_energy_GeV"] == energy
        ].iloc[0]
        input_energy = float(diagnostic["mean_unweighted_input_energy_GeV"])
        local_weights = (
            diagnostic["p1_diagnostic"]
            * np.exp(diagnostic["p2_diagnostic"] * representatives)
            + diagnostic["p3_diagnostic"]
        )
        global_weights = density_weights(
            theta, np.array([input_energy]), representatives
        )[0]
        rho_curve = np.geomspace(
            max(1e-5, representatives[0] / 2.0), representatives[-1] * 1.8, 300
        )
        global_curve = density_weights(
            theta, np.array([input_energy]), rho_curve
        )[0]
        direct_global_curve = density_weights(
            direct_global_theta, np.array([input_energy]), rho_curve
        )[0]
        axis.scatter(
            representatives,
            local_weights,
            s=29,
            color="black",
            zorder=3,
            label="per-energy diagnostic fit",
        )
        axis.scatter(
            representatives,
            global_weights,
            s=30,
            facecolors="white",
            edgecolors="tab:blue",
            zorder=4,
            label="global fit at bin centers",
        )
        axis.plot(
            rho_curve,
            global_curve,
            color="tab:blue",
            linewidth=1.7,
            label="two-stage primary",
        )
        axis.plot(
            rho_curve,
            direct_global_curve,
            color="tab:gray",
            linestyle="--",
            linewidth=1.5,
            label="direct-global reference",
        )
        axis.set_xscale("log")
        axis.set_title(f"{energy} GeV; mean $E_{{sum}}$={input_energy:.2f} GeV")
        axis.grid(alpha=0.18)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\rho$ [GeV / 1000 cm$^3$]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Weight $\omega$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig_panels.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig_panels.suptitle("Hit-density weights: two-stage primary and direct-global reference", y=0.96)
    fig_panels.tight_layout(rect=(0, 0, 1, 0.92))
    fig_panels.savefig(
        plots_dir / "weights_vs_hit_density_parameterization_by_energy.png", dpi=190
    )
    plt.close(fig_panels)

    # Energy dependence of p1, p2 and p3. Markers are independent per-energy
    # diagnostic fits; curves are the global nine-parameter fit used in E_LSC.
    fig_parameters, axes_parameters = plt.subplots(1, 3, figsize=(14.2, 4.6))
    e_grid = np.linspace(
        max(0.1, event_table[BASELINE_ENERGY_COLUMN].min()),
        event_table[BASELINE_ENERGY_COLUMN].max(),
        350,
    )
    two_stage_components = weight_components(theta, e_grid)
    direct_global_components = weight_components(direct_global_theta, e_grid)
    x_points = per_energy_diagnostics["mean_unweighted_input_energy_GeV"].to_numpy()
    parameter_specs = (
        ("p1", "p1_diagnostic", "p1_error", direct_global_components[0], two_stage_components[0]),
        ("p2", "p2_diagnostic", "p2_error", direct_global_components[1], two_stage_components[1]),
        ("p3", "p3_diagnostic", "p3_error", direct_global_components[2], two_stage_components[2]),
    )
    for axis, (label, value_col, error_col, direct_curve, corrected_curve) in zip(
        axes_parameters, parameter_specs
    ):
        values = per_energy_diagnostics[value_col].to_numpy(dtype=float)
        errors = per_energy_diagnostics[error_col].to_numpy(dtype=float)
        finite_reasonable = np.isfinite(errors) & (errors < 2.0 * max(1e-6, np.ptp(values)))
        axis.plot(
            e_grid,
            direct_curve,
            color="tab:gray",
            linestyle="--",
            label="direct-global reference",
        )
        axis.plot(
            e_grid,
            corrected_curve,
            color="tab:blue",
            label="two-stage primary",
        )
        axis.scatter(x_points, values, color="black", zorder=3, label="per-energy fit")
        if np.any(finite_reasonable):
            axis.errorbar(
                x_points[finite_reasonable],
                values[finite_reasonable],
                yerr=errors[finite_reasonable],
                fmt="none",
                color="black",
                capsize=2,
            )
        axis.set_xlabel(r"Unweighted $E_{sum}$ [GeV]")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes_parameters[0].legend()
    fig_parameters.suptitle(
        "Energy dependence of CALICE weight parameters: corrected two-stage fit"
    )
    fig_parameters.tight_layout()
    fig_parameters.savefig(
        plots_dir / "energy_dependent_weight_parameters.png", dpi=190
    )
    plt.close(fig_parameters)


def write_energy_distribution_plots(event_table: pd.DataFrame, plots_dir: Path) -> None:
    for energy in ENERGIES:
        subset = event_table.loc[
            (event_table["split"] == "application")
            & (event_table["energy_GeV"] == energy)
        ]
        combined = np.concatenate(
            [
                subset[BASELINE_ENERGY_COLUMN].to_numpy(),
                subset["E_LSC_GeV"].to_numpy(),
            ]
        )
        low, high = np.quantile(combined, [0.003, 0.997])
        bins = np.linspace(low, high, 65)
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        ax.hist(
            subset[BASELINE_ENERGY_COLUMN],
            bins=bins,
            histtype="step",
            linewidth=1.7,
            label=f"{BASELINE_LABEL}, unweighted",
        )
        ax.hist(
            subset["E_LSC_GeV"],
            bins=bins,
            histtype="step",
            linewidth=1.7,
            label=CORRECTED_LABEL,
        )
        ax.axvline(energy, color="black", linestyle="--", linewidth=1.0, label="beam energy")
        ax.set_xlabel("Reconstructed energy [GeV]")
        ax.set_ylabel("Events / bin")
        ax.set_title(f"Independent application sample, $\pi^-$ {energy} GeV")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"reweighted_energy_distribution_{energy}GeV.png", dpi=180)
        plt.close(fig)


def write_summary_plots(
    metrics: pd.DataFrame, resolution_fit: pd.DataFrame, plots_dir: Path
) -> None:
    selected = metrics.loc[metrics["split"] == "application"]
    colors = {BASELINE_METHOD: "tab:gray", CORRECTED_METHOD: "tab:blue"}
    labels = {BASELINE_METHOD: BASELINE_LABEL, CORRECTED_METHOD: CORRECTED_LABEL}

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for method in colors:
        part = selected.loc[selected["method"] == method].sort_values("energy_GeV")
        ax.errorbar(
            part["energy_GeV"],
            100.0 * part["resolution_sigma68_over_mpv"],
            yerr=100.0 * part["resolution_error"],
            marker="o",
            capsize=3,
            color=colors[method],
            label=labels[method],
        )
        fit = resolution_fit.loc[resolution_fit["method"] == method].iloc[0]
        e_grid = np.linspace(min(ENERGIES), max(ENERGIES), 300)
        curve = np.sqrt(
            fit["stochastic_term_fraction_sqrtGeV"] ** 2 / e_grid
            + fit["constant_term_fraction"] ** 2
        )
        ax.plot(e_grid, 100.0 * curve, color=colors[method], alpha=0.65)
    ax.set_xlabel("Beam energy [GeV]")
    ax.set_ylabel(r"$\sigma_{68}/\mathrm{MPV}$ [%]")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "reweighted_energy_resolution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for method in colors:
        part = selected.loc[selected["method"] == method].sort_values("energy_GeV")
        ax.plot(
            part["energy_GeV"],
            part["response_mpv"],
            marker="o",
            color=colors[method],
            label=labels[method],
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Beam energy [GeV]")
    ax.set_ylabel("Reconstructed response")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "reweighted_energy_linearity.png", dpi=180)
    plt.close(fig)


def write_closure_plots(
    event_table: pd.DataFrame,
    metrics: pd.DataFrame,
    plots_dir: Path,
) -> None:
    # Numerical closure of cell hits -> density-bin sums.
    delta_kev = 1.0e6 * (
        event_table[BIN_SUM_COLUMN].to_numpy(dtype=float)
        - event_table[BASELINE_ENERGY_COLUMN].to_numpy(dtype=float)
    )
    fig_numeric, axes_numeric = plt.subplots(1, 2, figsize=(12.2, 4.8))
    axes_numeric[0].hist(delta_kev, bins=100, histtype="step", color="tab:purple")
    axes_numeric[0].set_xlabel(rf"$\sum_b E_b-{CLOSURE_SYMBOL}$ [keV]")
    axes_numeric[0].set_ylabel("Events / bin")
    axes_numeric[0].set_title("Density-bin energy closure")
    axes_numeric[1].scatter(
        event_table[BASELINE_ENERGY_COLUMN],
        delta_kev,
        s=3,
        alpha=0.18,
        color="tab:purple",
    )
    axes_numeric[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes_numeric[1].set_xlabel(f"Event {BASELINE_LABEL} [GeV]")
    axes_numeric[1].set_ylabel(rf"$\sum_b E_b-{CLOSURE_SYMBOL}$ [keV]")
    axes_numeric[1].set_title(f"max |closure| = {np.max(np.abs(delta_kev)):.3f} keV")
    for axis in axes_numeric:
        axis.grid(alpha=0.18)
    fig_numeric.tight_layout()
    fig_numeric.savefig(plots_dir / "bin_energy_closure.png", dpi=190)
    plt.close(fig_numeric)

    # Physics response closure on the independent application split.
    application = metrics.loc[metrics["split"] == "application"]
    fig_response, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.06},
    )
    method_styles = {
        BASELINE_METHOD: ("tab:gray", BASELINE_LABEL),
        CORRECTED_METHOD: ("tab:blue", CORRECTED_LABEL),
    }
    for method, (color, label) in method_styles.items():
        part = application.loc[application["method"] == method].sort_values("energy_GeV")
        beam = part["energy_GeV"].to_numpy(dtype=float)
        mean = part["mpv_GeV"].to_numpy(dtype=float)
        mean_error = part["mpv_error_GeV"].to_numpy(dtype=float)
        ax_top.errorbar(
            beam,
            mean,
            yerr=mean_error,
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
        residual = 100.0 * (mean - beam) / beam
        residual_error = 100.0 * mean_error / beam
        ax_bottom.errorbar(
            beam,
            residual,
            yerr=residual_error,
            marker="o",
            capsize=3,
            color=color,
        )
    beam_line = np.linspace(min(ENERGIES), max(ENERGIES), 250)
    ax_top.plot(beam_line, beam_line, color="black", linestyle="--", label="ideal closure")
    ax_top.set_ylabel("Reconstructed MPV [GeV]")
    ax_top.legend()
    ax_top.grid(alpha=0.2)
    ax_bottom.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_bottom.set_xlabel("Beam energy [GeV]")
    ax_bottom.set_ylabel(r"$(E_{rec}-E_{beam})/E_{beam}$ [%]")
    ax_bottom.grid(alpha=0.2)
    fig_response.suptitle("Independent application-sample energy closure")
    fig_response.subplots_adjust(top=0.92, hspace=0.06)
    fig_response.savefig(plots_dir / "energy_response_closure.png", dpi=190)
    plt.close(fig_response)


def write_resolution_fit_plot(
    metrics: pd.DataFrame,
    resolution_fit: pd.DataFrame,
    plots_dir: Path,
) -> None:
    application = metrics.loc[metrics["split"] == "application"]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    styles = {
        BASELINE_METHOD: ("tab:gray", BASELINE_LABEL),
        CORRECTED_METHOD: ("tab:blue", CORRECTED_LABEL),
    }
    e_grid = np.linspace(min(ENERGIES), max(ENERGIES), 400)
    for method, (color, label) in styles.items():
        part = application.loc[application["method"] == method].sort_values("energy_GeV")
        fit = resolution_fit.loc[resolution_fit["method"] == method].iloc[0]
        a = float(fit["stochastic_term_fraction_sqrtGeV"])
        b = float(fit["constant_term_fraction"])
        curve = np.sqrt(a * a / e_grid + b * b)
        fit_label = (
            f"{label} fit: a={100*a:.2f}%$\\sqrt{{GeV}}$, "
            f"b={100*b:.2f}%, $\\chi^2$/ndf={fit['chi2']:.1f}/{int(fit['ndf'])}"
        )
        ax.errorbar(
            part["energy_GeV"],
            100.0 * part["resolution_sigma68_over_mpv"],
            yerr=100.0 * part["resolution_error"],
            fmt="o",
            capsize=3,
            color=color,
            label=f"{label} points",
        )
        ax.plot(e_grid, 100.0 * curve, linestyle="--", color=color, label=fit_label)
    ax.set_xlabel("Beam energy [GeV]")
    ax.set_ylabel(r"$\sigma_{68}/\mathrm{MPV}$ [%]")
    ax.set_title(
        r"Resolution fit: $\sigma_{68}/\mathrm{MPV}=a/\sqrt{E}\oplus b$",
        pad=12,
    )
    ax.legend(fontsize=8.6)
    ax.grid(alpha=0.2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(plots_dir / "final_resolution_parameterization_fit.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(__file__).resolve().parent
    input_dir = args.input_dir or (run_dir / args.geometry)
    calibration_table = args.calibration_table or (
        run_dir
        / "full_dualreadout"
        / args.geometry
        / "tables"
        / "electron_em_calibration.csv"
    )
    output_dir = args.output_dir or (run_dir / "calice_local_sc" / args.geometry)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    intermediate_dir = output_dir / "intermediate"
    for directory in (tables_dir, plots_dir, intermediate_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(input_dir)
    geometry = parse_geometry(next(iter(samples.values())).name)
    s_gev_per_count = read_s_calibration(calibration_table)
    cache_path = intermediate_dir / "cell_hits.root"
    validation_path = tables_dir / "export_validation.csv"
    if not (args.reuse_cache and cache_path.exists()):
        export_cell_cache(
            samples,
            geometry,
            s_gev_per_count,
            cache_path,
            validation_path,
            args.split_seed,
            args.max_events,
        )

    cache = load_cache(cache_path)
    edges, representatives, binning_table = derive_density_binning(
        cache, N_DENSITY_BINS
    )
    binning_table["lower_hit_energy_GeV"] = (
        binning_table["lower_GeV_per_1000cm3"] * geometry.cell_volume_cm3 / 1000.0
    )
    binning_table["upper_hit_energy_GeV"] = (
        binning_table["upper_GeV_per_1000cm3"] * geometry.cell_volume_cm3 / 1000.0
    )
    binning_table.to_csv(tables_dir / "hit_density_binning.csv", index=False)
    write_hit_histograms(
        cache,
        tables_dir / "hit_density_histograms.csv",
        plots_dir,
        edges,
        geometry.cell_volume_cm3,
    )

    event_table, bin_sums = build_event_bin_table(cache, edges)
    event_table.to_csv(intermediate_dir / "event_density_bin_sums.csv", index=False)
    direct_global_theta, direct_global_fit, direct_global_errors = fit_weight_parameterization(
        event_table, bin_sums, representatives
    )
    per_energy_diagnostics = fit_per_energy_diagnostics(
        direct_global_theta, event_table, bin_sums, representatives
    )
    per_energy_diagnostics.to_csv(
        tables_dir / "per_energy_weight_diagnostic_fit.csv", index=False
    )
    two_stage_theta, two_stage_errors, two_stage_summary = (
        fit_two_stage_parameterization(per_energy_diagnostics)
    )

    two_stage_canonical = canonical_parameters(two_stage_theta)
    two_stage_canonical_errors = canonical_parameters(two_stage_errors)
    pd.DataFrame(
        [
            {
                "parameter": name,
                "value": value,
                "fit_error_approx": two_stage_canonical_errors[name],
                "strategy": "two_stage_fit_to_per_energy_p_i",
                "role": "primary; used for E_LSC_GeV",
            }
            for name, value in two_stage_canonical.items()
        ]
    ).to_csv(tables_dir / "weight_parameterization_two_stage.csv", index=False)

    direct_global_canonical = canonical_parameters(direct_global_theta)
    direct_global_canonical_errors = canonical_parameters(direct_global_errors)
    pd.DataFrame(
        [
            {
                "parameter": name,
                "value": value,
                "fit_error_approx": direct_global_canonical_errors[name],
                "strategy": "direct_global_event_fit",
                "role": "reference; not used for primary E_LSC",
            }
            for name, value in direct_global_canonical.items()
        ]
    ).to_csv(
        tables_dir / "weight_parameterization_direct_global_reference.csv",
        index=False,
    )

    reconstructed_primary, _ = apply_weights(
        two_stage_theta, event_table, bin_sums, representatives
    )
    reconstructed_direct_global, _ = apply_weights(
        direct_global_theta, event_table, bin_sums, representatives
    )
    event_table["E_LSC_GeV"] = reconstructed_primary
    event_table["E_LSC_two_stage_GeV"] = reconstructed_primary
    event_table["E_LSC_two_stage_candidate_GeV"] = reconstructed_primary
    event_table["E_LSC_direct_global_reference_GeV"] = reconstructed_direct_global
    event_table.to_csv(tables_dir / "event_reconstructed_energies.csv", index=False)

    write_weight_outputs(
        two_stage_theta,
        two_stage_errors,
        direct_global_theta,
        event_table,
        representatives,
        edges,
        per_energy_diagnostics,
        tables_dir,
        plots_dir,
    )
    write_energy_distribution_plots(event_table, plots_dir)
    metrics = build_metrics(event_table)
    metrics["fit_strategy"] = "two_stage_fit_to_per_energy_p_i"
    metrics["role"] = "primary"
    metrics.to_csv(tables_dir / "reweighted_energy_metrics.csv", index=False)
    metrics.to_csv(
        tables_dir / "two_stage_primary_energy_metrics.csv", index=False
    )
    # Backward-compatible filename from the development phase. Its role column
    # now clearly records that two-stage has been promoted to primary.
    metrics.to_csv(
        tables_dir / "two_stage_candidate_energy_metrics.csv", index=False
    )
    resolution_fit = fit_resolution_curve(metrics)
    resolution_fit.to_csv(tables_dir / "reweighted_resolution_fit.csv", index=False)
    resolution_fit.to_csv(
        tables_dir / "two_stage_primary_resolution_fit.csv", index=False
    )
    resolution_fit.to_csv(
        tables_dir / "two_stage_candidate_resolution_fit.csv", index=False
    )

    direct_reference_table = event_table.copy()
    direct_reference_table["E_LSC_GeV"] = direct_reference_table[
        "E_LSC_direct_global_reference_GeV"
    ]
    direct_reference_metrics = build_metrics(direct_reference_table)
    direct_reference_metrics["fit_strategy"] = "direct_global_event_fit"
    direct_reference_metrics["role"] = "reference"
    direct_reference_metrics.to_csv(
        tables_dir / "direct_global_reference_energy_metrics.csv", index=False
    )
    fit_resolution_curve(direct_reference_metrics).to_csv(
        tables_dir / "direct_global_reference_resolution_fit.csv", index=False
    )
    write_summary_plots(metrics, resolution_fit, plots_dir)
    write_closure_plots(event_table, metrics, plots_dir)
    write_resolution_fit_plot(metrics, resolution_fit, plots_dir)

    diagnostic = pd.DataFrame(
        [
            {
                "strategy": "direct_global_event_fit",
                "success": bool(direct_global_fit.success),
                "status": int(direct_global_fit.status),
                "message": "reference only; " + str(direct_global_fit.message),
                "cost_half_chi2_like": float(direct_global_fit.cost),
                "optimality": float(direct_global_fit.optimality),
                "n_function_evaluations": int(direct_global_fit.nfev),
                "n_calibration_events": int(np.count_nonzero(np.asarray(cache["split"]) == 0)),
                "n_application_events": int(np.count_nonzero(np.asarray(cache["split"]) == 1)),
            },
            {
                "strategy": two_stage_summary["strategy"],
                "success": two_stage_summary["success"],
                "status": "n/a",
                "message": two_stage_summary["role"],
                "cost_half_chi2_like": 0.5 * two_stage_summary["chi2"],
                "optimality": np.nan,
                "n_function_evaluations": np.nan,
                "n_calibration_events": int(np.count_nonzero(np.asarray(cache["split"]) == 0)),
                "n_application_events": int(np.count_nonzero(np.asarray(cache["split"]) == 1)),
                "chi2": two_stage_summary["chi2"],
                "ndf": two_stage_summary["ndf"],
            },
        ]
    )
    diagnostic.to_csv(tables_dir / "weight_fit_diagnostics.csv", index=False)

    config = {
        "method": "CALICE cell-level local software compensation",
        "reference": "https://arxiv.org/abs/1705.10363",
        "geometry": args.geometry,
        "input_dir": str(input_dir.resolve()),
        "calibration_table": str(calibration_table.resolve()),
        "S_GeV_per_count": s_gev_per_count,
        "cell_dimensions_mm": [
            geometry.transverse_x_mm,
            geometry.transverse_y_mm,
            geometry.sampling_depth_mm,
        ],
        "cell_volume_cm3": geometry.cell_volume_cm3,
        "density_definition": "rho = hit_energy_GeV / (cell_volume_cm3 / 1000)",
        "n_density_bins": N_DENSITY_BINS,
        "binning": "calibration-only weighted quantiles; weight=hit_energy/Ebeam",
        "split_seed": args.split_seed,
        "split_definition": "stable hash; 0=calibration, 1=application",
        "max_events_per_energy": args.max_events,
        "application_inputs": ["aggregated vecNScint per CellID", "unweighted S_EM_GeV"],
        "excluded_inputs": ["vecEdep", "vecNChren", "truthFem", "beam energy"],
        "weight_formula": "omega(rho,E)=p1(E)*exp(p2(E)*rho)+p3(E)",
        "p1_formula": "p10+p11*E+p12*E^2",
        "p2_formula": "p20+p21*E+p22*E^2",
        "p3_formula": "p30/(p31+exp(p32*E))",
        "loss": "sum_events ((Ebeam-E_LSC)/(0.5*sqrt(Ebeam)))^2 with equal energy-point weight",
        "primary_fit_strategy": "per-energy three-parameter fits followed by two-stage p_i(E) fit",
        "reference_fit_strategy": "direct global event-level nine-parameter fit",
        "primary_output": "tables/reweighted_energy_metrics.csv",
        "reference_output": "tables/direct_global_reference_energy_metrics.csv",
        "primary_evaluation_split": "application",
        "primary_resolution_definition": (
            "sigma68/MPV following RunScintGlassSOnlyEdepScintLayer.cpp: "
            "sigma68 is half the narrowest interval containing round(0.68*N) "
            "events; MPV is the maximum-bin centre of an 80-bin histogram "
            "over [min-0.05*span,max+0.05*span]"
        ),
        "primary_resolution_error": "(sigma68/MPV)/sqrt(2*N)",
        "resolution_curve": "sqrt(a^2/E_GeV+b^2)",
        "legacy_resolution_diagnostics": (
            "gauss90_mean_GeV, gauss90_sigma_GeV, response_gauss90, "
            "resolution_gauss90 and RMS90/Mean90 are retained but are not primary"
        ),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    application = metrics.loc[
        (metrics["split"] == "application")
        & (metrics["method"] == CORRECTED_METHOD),
        [
            "energy_GeV",
            "response_mpv",
            "resolution_sigma68_over_mpv",
            IMPROVEMENT_COLUMN,
        ],
    ]
    print(f"Wrote CALICE local-SC results to {output_dir}")
    print(application.to_string(index=False))


if __name__ == "__main__":
    main()
