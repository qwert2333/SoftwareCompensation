#!/usr/bin/env python3
"""Export dual-readout calibration and leakage-safe CNN baseline diagnostics.

The electron calibration and pion h/e(E) parameterization are read from the
standard dual-readout analysis.  Continuous Train is used for input statistics
and bias characterization; continuous Valid is used only for closure checks.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calo.dual_readout import DualReadoutCalibration


BRANCHES = (
    "eventID",
    "MCtruth_energy",
    "counter_Scintillation_ScintLayer",
    "counter_Cerenkov_CherenkovLayer",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("outputs_dual_readout/standard_dual_readout/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_dual_readout/cnn_inputs/Sapphire_5-5-5mm_N30x30x100"),
    )
    return parser.parse_args()


def unique_channel_value(table, channel, column):
    values = table.loc[table["channel"].str.upper() == channel, column].to_numpy(float)
    if not values.size or not np.allclose(values, values[0], rtol=0.0, atol=5e-7):
        raise RuntimeError(f"Missing or inconsistent {channel} {column}")
    return float(values[0])


def load_calibration(reference_dir):
    calibration_path = reference_dir / "tables" / "electron_em_calibration.csv"
    hovere_path = reference_dir / "tables" / "pion_hovere_energy_fit.csv"
    em = pd.read_csv(calibration_path)
    he = pd.read_csv(hovere_path)
    calibration = DualReadoutCalibration(
        s_gev_per_count=unique_channel_value(em, "S", "calibration_MeV_per_readout") / 1000.0,
        c_gev_per_count=unique_channel_value(em, "C", "calibration_MeV_per_readout") / 1000.0,
        h_s_intercept=unique_channel_value(he, "S", "intercept"),
        h_s_slope_per_gev=unique_channel_value(he, "S", "slope_per_GeV"),
        h_c_intercept=unique_channel_value(he, "C", "intercept"),
        h_c_slope_per_gev=unique_channel_value(he, "C", "slope_per_GeV"),
    )
    return calibration, calibration_path.resolve(), hovere_path.resolve()


def read_sample(path, calibration):
    with uproot.open(path) as source:
        arrays = source["eventTree"].arrays(BRANCHES, library="np")
    truth = arrays["MCtruth_energy"].astype(np.float64) / 1000.0
    s_energy = arrays["counter_Scintillation_ScintLayer"].astype(np.float64) * calibration.s_gev_per_count
    c_energy = arrays["counter_Cerenkov_CherenkovLayer"].astype(np.float64) * calibration.c_gev_per_count
    dr_energy = reconstruct_dr_vectorized(s_energy, c_energy, calibration)
    return pd.DataFrame({
        "event_id": arrays["eventID"].astype(np.int64),
        "true_energy_GeV": truth,
        "S_EM_GeV": s_energy,
        "C_EM_GeV": c_energy,
        "standard_DR_self_consistent_GeV": dr_energy,
    })


def reconstruct_dr_vectorized(s_energy, c_energy, calibration, n_iter=6):
    energy = np.clip(s_energy, 0.0, calibration.max_reco_energy_gev)
    result = energy.copy()
    for _ in range(n_iter):
        h_s = calibration.h_s_intercept + calibration.h_s_slope_per_gev * energy
        h_c = calibration.h_c_intercept + calibration.h_c_slope_per_gev * energy
        chi = (1.0 - h_s) / np.maximum(1.0 - h_c, 1.0e-8)
        denominator = 1.0 - chi
        if np.any(np.abs(denominator) < 1.0e-8):
            raise RuntimeError("Invalid dual-readout denominator")
        result = (s_energy - chi * c_energy) / denominator
        energy = np.clip(result, 0.0, calibration.max_reco_energy_gev)
    return result


def sigma68(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size < 2:
        return float("nan")
    count = max(2, int(round(0.68 * values.size)))
    return 0.5 * float(np.min(values[count - 1:] - values[: values.size - count + 1]))


def metric_row(split, bin_lo, bin_hi, method, truth, reconstructed):
    ratio = reconstructed / truth
    width = sigma68(reconstructed - truth)
    mean_reco = float(np.mean(reconstructed))
    mean_truth = float(np.mean(truth))
    return {
        "split": split,
        "energy_bin_low_GeV": bin_lo,
        "energy_bin_high_GeV": bin_hi,
        "method": method,
        "N": int(truth.size),
        "mean_true_energy_GeV": mean_truth,
        "mean_reco_energy_GeV": mean_reco,
        "mean_response": float(np.mean(ratio)),
        "mean_relative_bias": float(np.mean(ratio - 1.0)),
        "median_relative_bias": float(np.median(ratio - 1.0)),
        "sigma68_residual_GeV": width,
        "sigma68_residual_over_mean_true": width / mean_truth,
    }


def binned_metrics(frame, split):
    rows = []
    bins = np.arange(5.0, 65.0, 5.0)
    methods = ("S_EM_GeV", "C_EM_GeV", "standard_DR_self_consistent_GeV")
    energy = frame["true_energy_GeV"].to_numpy(float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        selected = (energy >= lo) & ((energy < hi) if hi < bins[-1] else (energy <= hi))
        if not np.any(selected):
            continue
        truth = energy[selected]
        for method in methods:
            rows.append(metric_row(
                split, lo, hi, method, truth, frame.loc[selected, method].to_numpy(float)
            ))
    return rows


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q50": float(np.quantile(values, 0.50)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration, calibration_path, hovere_path = load_calibration(args.reference_dir)

    train_path = next(args.input_dir.glob("*_pi-_5-60GeV_Train.root"))
    valid_path = next(args.input_dir.glob("*_pi-_5-60GeV_Valid.root"))
    train = read_sample(train_path, calibration)
    valid = read_sample(valid_path, calibration)

    metrics = binned_metrics(train, "continuous_train")
    metrics.extend(binned_metrics(valid, "continuous_valid_closure"))
    pd.DataFrame(metrics).to_csv(args.output_dir / "baseline_bias_by_energy_bin.csv", index=False)

    normalization = {
        "source": "continuous Train sample only",
        "n_events": int(len(train)),
        "features": {
            name: describe(train[name].to_numpy(float))
            for name in (
                "true_energy_GeV",
                "S_EM_GeV",
                "C_EM_GeV",
                "standard_DR_self_consistent_GeV",
            )
        },
        "derived_auxiliary_transforms": {
            "log1p_S_total_GeV": describe(np.log1p(np.maximum(train["S_EM_GeV"], 0.0))),
            "log1p_C_total_GeV": describe(np.log1p(np.maximum(train["C_EM_GeV"], 0.0))),
            "C_over_S": describe(train["C_EM_GeV"] / np.maximum(train["S_EM_GeV"], 1.0e-8)),
            "S_minus_C_over_sum": describe(
                (train["S_EM_GeV"] - train["C_EM_GeV"])
                / np.maximum(train["S_EM_GeV"] + train["C_EM_GeV"], 1.0e-8)
            ),
        },
    }
    with (args.output_dir / "training_input_statistics.json").open("w") as stream:
        json.dump(normalization, stream, indent=2)

    config = {
        "geometry_label": "Sapphire_5-5-5mm_N30x30x100",
        "geometry": {"nx": 30, "ny": 30, "nz": 100, "crop_nx": 30, "crop_ny": 30, "crop_nz": 100},
        "crop_policy": "x=[0,30), y=[0,30), z=[0,100); this equals the full generated grid",
        "electron_calibration": {
            "S_GeV_per_count": calibration.s_gev_per_count,
            "C_GeV_per_count": calibration.c_gev_per_count,
            "source_table": str(calibration_path),
            "fit": "through-origin fit of mean readout versus electron MC truth at 5,10,20,30,40,50 GeV",
        },
        "pion_h_over_e": {
            "S_intercept": calibration.h_s_intercept,
            "S_slope_per_GeV": calibration.h_s_slope_per_gev,
            "C_intercept": calibration.h_c_intercept,
            "C_slope_per_GeV": calibration.h_c_slope_per_gev,
            "source_table": str(hovere_path),
        },
        "standard_DR": {
            "chi": "(1-h_over_e_S)/(1-h_over_e_C)",
            "energy": "(S-chi*C)/(1-chi)",
            "cnn_aux_policy": "six iterations; evaluate h/e at reconstructed energy only",
            "truth_policy": "MC truth is target/diagnostic only and never enters CNN inputs",
        },
        "bias_policy": {
            "characterization_sample": str(train_path.resolve()),
            "closure_sample": str(valid_path.resolve()),
            "table": "baseline_bias_by_energy_bin.csv",
            "warning": "Do not derive corrections or normalization from the validation rows.",
        },
        "cell_signal_scale_GeV_for_log1p_voxels": 0.001,
    }
    with (args.output_dir / "cnn_calibration_and_bias_config.json").open("w") as stream:
        json.dump(config, stream, indent=2)

    summary = pd.DataFrame([
        metric_row(
            split,
            float(frame["true_energy_GeV"].min()),
            float(frame["true_energy_GeV"].max()),
            method,
            frame["true_energy_GeV"].to_numpy(float),
            frame[method].to_numpy(float),
        )
        for split, frame in (("continuous_train", train), ("continuous_valid_closure", valid))
        for method in ("S_EM_GeV", "C_EM_GeV", "standard_DR_self_consistent_GeV")
    ])
    summary.to_csv(args.output_dir / "baseline_bias_global_summary.csv", index=False)
    print(f"Wrote CNN calibration and bias inputs to {args.output_dir}")
    print(summary[["split", "method", "N", "mean_relative_bias", "sigma68_residual_over_mean_true"]].to_string(index=False))


if __name__ == "__main__":
    main()
