"""Evaluation and directly comparable metrics for dual-readout ablations."""

import csv
import json
import math
import os

import numpy as np
import torch

from .device import select_device
from .dr_model import DualReadoutHCALNet
from .dr_train import make_loader
from .dual_readout import AUX_NAMES, DualReadoutCalibration, DualReadoutGeometry


def sigma68(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size < 2:
        return float("nan")
    n = max(2, int(round(0.68 * values.size)))
    return 0.5 * float(np.min(values[n - 1:] - values[:values.size - n + 1]))


def energy_metrics(values, true_energy):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"N": int(values.size), "mean_GeV": float("nan"),
                "mpv_GeV": float("nan"), "sigma68_GeV": float("nan"),
                "resolution_sigma68_over_mpv": float("nan"),
                "mean_relative_bias": float("nan")}
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = 0.05 * max(hi - lo, 1.0e-6)
    hist, edges = np.histogram(values, bins=80, range=(lo - margin, hi + margin))
    peak = int(np.argmax(hist))
    mpv = 0.5 * float(edges[peak] + edges[peak + 1])
    width = sigma68(values)
    return {
        "N": int(values.size),
        "mean_GeV": float(np.mean(values)),
        "mpv_GeV": mpv,
        "sigma68_GeV": width,
        "resolution_sigma68_over_mpv": width / max(abs(mpv), 1.0e-9),
        "mean_relative_bias": (float(np.mean(values)) - true_energy) / true_energy,
    }


def evaluate_one(exp, files, out_dir, max_events_per_file=-1):
    device = select_device(exp.get("device", "auto"))
    geo = DualReadoutGeometry(**exp["geometry"])
    calibration = DualReadoutCalibration()
    model = DualReadoutHCALNet(aux_dim=len(AUX_NAMES)).to(device)
    checkpoint = torch.load(
        os.path.join(out_dir, "checkpoints", "best.pth"), map_location=device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = make_loader(files, exp, "test", geo, calibration, device, max_events_per_file)

    columns = {name: [] for name in (
        "event_id", "true_energy_GeV", "pred_energy_GeV",
        "S_crop_GeV", "C_crop_GeV", "E_DR_reco_crop_GeV",
    )}
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["voxel"].to(device, non_blocking=True),
                batch["aux"].to(device, non_blocking=True),
            ).cpu().numpy()
            values = {
                "event_id": batch["event_id"].numpy(),
                "true_energy_GeV": batch["energy_true"].numpy(),
                "pred_energy_GeV": prediction,
                "S_crop_GeV": batch["s_total"].numpy(),
                "C_crop_GeV": batch["c_total"].numpy(),
                "E_DR_reco_crop_GeV": batch["dr_reco"].numpy(),
            }
            for name in columns:
                columns[name].append(values[name])
    if not columns["event_id"]:
        raise RuntimeError("No test events were read")
    columns = {name: np.concatenate(parts) for name, parts in columns.items()}
    order = np.lexsort((columns["event_id"], columns["true_energy_GeV"]))
    columns = {name: values[order] for name, values in columns.items()}

    prediction_path = os.path.join(out_dir, "test_predictions.csv")
    with open(prediction_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(columns[name] for name in columns)))

    methods = {"ML": columns["pred_energy_GeV"], "S_crop": columns["S_crop_GeV"]}
    if exp["mode"] == "sc_full":
        methods.update({
            "C_crop": columns["C_crop_GeV"],
            "standard_DR_reco_crop": columns["E_DR_reco_crop_GeV"],
        })
    rows = []
    for energy in sorted(np.unique(columns["true_energy_GeV"])):
        selected = columns["true_energy_GeV"] == energy
        for method, values in methods.items():
            rows.append({
                "experiment": exp["name"], "mode": exp["mode"],
                "method": method, "true_energy_GeV": float(energy),
                **energy_metrics(values[selected], float(energy)),
            })
    metrics_path = os.path.join(out_dir, "test_metrics.csv")
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as handle:
        json.dump(rows, handle, indent=2)
    return rows
