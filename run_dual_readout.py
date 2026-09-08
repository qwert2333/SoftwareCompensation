#!/usr/bin/env python3
"""Train and evaluate the controlled S-only versus full-S+C comparison."""

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import asdict

from calo.dr_evaluate import evaluate_one
from calo.dr_experiment_grid import EXPERIMENTS
from calo.dr_train import train_one
from calo.dual_readout import AUX_NAMES, DualReadoutCalibration
from calo.seed import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="outputs_dual_readout")
    parser.add_argument("--experiment", choices=("s_only", "sc_full", "all"), default="all")
    parser.add_argument("--max-events-per-file", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.input_dir, "*_pi-_*.root")))
    if not files:
        raise RuntimeError(f"No pion ROOT files found in {args.input_dir}")
    expected_energies = {5, 10, 20, 30, 40, 50}
    found_energies = set()
    for path in files:
        name = os.path.basename(path)
        if "ScintZnWO4_Quartz_AbsSTAINLESS-STEEL_5-5-10mm_N60x60x100" not in name:
            raise RuntimeError(
                "The built-in geometry/calibration is only valid for "
                f"ZnWO4_5_Quartz_5_Steel_10; incompatible file: {name}"
            )
        match = re.search(r"_pi-_([0-9]+)GeV\.root$", name)
        if match:
            found_energies.add(int(match.group(1)))
    if found_energies != expected_energies:
        raise RuntimeError(
            f"Expected pion energies {sorted(expected_energies)}, got {sorted(found_energies)}"
        )
    selected = EXPERIMENTS if args.experiment == "all" else [
        exp for exp in EXPERIMENTS if exp["mode"] == args.experiment
    ]
    seed = 20260908
    all_metrics = []
    for base in selected:
        exp = dict(base)
        if args.epochs is not None:
            exp["epochs"] = args.epochs
        if args.batch_size is not None:
            exp["batch_size"] = args.batch_size
        if args.num_workers is not None:
            exp["num_workers"] = args.num_workers
        exp["device"] = args.device
        set_global_seed(seed)
        out_dir = os.path.join(args.output_dir, exp["name"])
        os.makedirs(out_dir, exist_ok=True)
        config = {
            **exp,
            "seed": seed,
            "input_files": files,
            "aux_names": AUX_NAMES,
            "calibration": asdict(DualReadoutCalibration()),
            "crop_policy": "fixed central 30x30 transverse cells; all 100 z layers; outside discarded",
            "truth_policy": "MCtruth_energy is target only; no truth or filename energy enters features",
            "standard_dr_policy": "h/e and chi evaluated self-consistently from reconstructed crop S/C",
        }
        with open(os.path.join(out_dir, "run_config.json"), "w") as handle:
            json.dump(config, handle, indent=2)
        train_one(exp, files, out_dir, seed, args.max_events_per_file)
        all_metrics.extend(evaluate_one(exp, files, out_dir, args.max_events_per_file))

    if len(selected) == 2:
        combined_path = os.path.join(args.output_dir, "comparison_metrics.csv")
        with open(combined_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
            writer.writeheader()
            writer.writerows(all_metrics)
        ml = {(row["mode"], row["true_energy_GeV"]): row for row in all_metrics
              if row["method"] == "ML"}
        comparison = []
        comparison_energies = sorted(
            energy for mode, energy in ml
            if mode == "s_only" and ("sc_full", energy) in ml
        )
        for energy in comparison_energies:
            s_row = ml[("s_only", float(energy))]
            sc_row = ml[("sc_full", float(energy))]
            s_resolution = s_row["resolution_sigma68_over_mpv"]
            sc_resolution = sc_row["resolution_sigma68_over_mpv"]
            improvement = float("nan")
            if math.isfinite(s_resolution) and abs(s_resolution) > 1.0e-12:
                improvement = 100.0 * (s_resolution - sc_resolution) / s_resolution
            comparison.append({
                "true_energy_GeV": energy,
                "s_only_resolution": s_resolution,
                "sc_full_resolution": sc_resolution,
                "sc_improvement_percent": improvement,
                "s_only_mean_relative_bias": s_row["mean_relative_bias"],
                "sc_full_mean_relative_bias": sc_row["mean_relative_bias"],
            })
        with open(os.path.join(args.output_dir, "sc_vs_s_only.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)


if __name__ == "__main__":
    main()
