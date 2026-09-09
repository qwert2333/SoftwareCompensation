"""Resolution comparisons for the dual-readout software-compensation study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


FIT_FORMULA = "sigma/E = sqrt(a^2/E_GeV + b^2)"


def _infer_calice_dir(run_dir: Path) -> Path:
    config_path = run_dir / "dr_j1_s_only" / "run_config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    calibration_table = Path(config["calibration_table"]).resolve()
    # <DualReadout_MLSC>/full_dualreadout/<sample>/tables/<table>.csv
    dual_readout_root = calibration_table.parents[3]
    return dual_readout_root / "calice_local_sc" / config["baseline"]


def _current_rows(
    metrics: pd.DataFrame,
    comparison: str,
    mode: str,
    source_method: str,
    method: str,
    label: str,
) -> list[dict[str, object]]:
    selected = metrics.loc[
        (metrics["mode"] == mode) & (metrics["method"] == source_method)
    ]
    if selected.empty:
        raise RuntimeError(f"No current metrics for mode={mode}, method={source_method}")
    rows = []
    for row in selected.sort_values("true_energy_GeV").itertuples(index=False):
        resolution = float(row.resolution_sigma68_over_mpv)
        entries = int(row.N)
        rows.append({
            "comparison": comparison,
            "method": method,
            "label": label,
            "energy_GeV": float(row.true_energy_GeV),
            "entries": entries,
            "resolution_fraction": resolution,
            "resolution_error_fraction": resolution / math.sqrt(2.0 * entries),
            "resolution_percent": 100.0 * resolution,
            "resolution_error_percent": 100.0 * resolution / math.sqrt(2.0 * entries),
            "source_scope": "current 30x30x100 crop; common 15% test split",
            "source_kind": "current_test",
        })
    return rows


def _calice_rows(metrics_path: Path) -> list[dict[str, object]]:
    metrics = pd.read_csv(metrics_path)
    selected = metrics.loc[
        (metrics["split"] == "application")
        & (metrics["method"] == "CALICE_local_SC")
    ]
    if selected.empty:
        raise RuntimeError(f"No application CALICE_local_SC rows in {metrics_path}")
    rows = []
    for row in selected.sort_values("energy_GeV").itertuples(index=False):
        rows.append({
            "comparison": "s_only",
            "method": "calice_local_sc_reference",
            "label": "CALICE local SC reference (full grid)",
            "energy_GeV": float(row.energy_GeV),
            "entries": int(row.entries),
            "resolution_fraction": float(row.resolution_sigma68_over_mpv),
            "resolution_error_fraction": float(row.resolution_error),
            "resolution_percent": 100.0 * float(row.resolution_sigma68_over_mpv),
            "resolution_error_percent": 100.0 * float(row.resolution_error),
            "source_scope": "external full 60x60x120 grid; independent 50% application split",
            "source_kind": "external_reference",
        })
    return rows


def build_resolution_points(run_dir: Path, calice_dir: Path) -> pd.DataFrame:
    current_path = run_dir / "comparison_metrics.csv"
    calice_path = calice_dir / "tables" / "reweighted_energy_metrics.csv"
    current = pd.read_csv(current_path)
    rows = []
    rows.extend(_current_rows(
        current, "s_only", "s_only", "S_crop", "raw_s_crop", "Raw S (crop)"
    ))
    rows.extend(_calice_rows(calice_path))
    rows.extend(_current_rows(
        current, "s_only", "s_only", "ML", "s_only_model", "S-only CNN (crop)"
    ))
    rows.extend(_current_rows(
        current, "dual_readout", "sc_full", "standard_DR_reco_crop",
        "standard_dr_crop", "Standard dual readout (crop)"
    ))
    rows.extend(_current_rows(
        current, "dual_readout", "sc_full", "ML", "sc_full_model", "S+C CNN (crop)"
    ))
    result = pd.DataFrame(rows)
    expected_energies = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    for method, group in result.groupby("method"):
        energies = group["energy_GeV"].tolist()
        if energies != expected_energies:
            raise RuntimeError(f"Unexpected energy points for {method}: {energies}")
    result["resolution_definition"] = "sigma68/MPV"
    return result


def fit_resolution_curves(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparison, method, label, scope, source_kind), part in points.groupby(
        ["comparison", "method", "label", "source_scope", "source_kind"], sort=False
    ):
        part = part.sort_values("energy_GeV")
        energy = part["energy_GeV"].to_numpy(dtype=float)
        resolution = part["resolution_fraction"].to_numpy(dtype=float)
        error = part["resolution_error_fraction"].to_numpy(dtype=float)

        def residual(parameters: np.ndarray) -> np.ndarray:
            prediction = np.sqrt(parameters[0] ** 2 / energy + parameters[1] ** 2)
            return (prediction - resolution) / np.maximum(error, 1.0e-5)

        fit = least_squares(residual, np.array([0.3, 0.03]), bounds=(0.0, 5.0))
        covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
        parameter_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
        parameter_errors = np.where(fit.x > 1.0e-8, parameter_errors, np.nan)
        chi2 = float(np.sum(fit.fun**2))
        ndf = int(max(0, energy.size - 2))
        at_boundary = bool(np.any(fit.x <= 1.0e-8))
        chi2_ndf = chi2 / ndf if ndf else float("nan")
        fit_quality = "ok"
        if at_boundary:
            fit_quality = "parameter_at_boundary"
        elif chi2_ndf > 5.0:
            fit_quality = "poor_chi2"
        rows.append({
            "comparison": comparison,
            "method": method,
            "label": label,
            "source_scope": scope,
            "source_kind": source_kind,
            "n_points": int(energy.size),
            "stochastic_term_fraction_sqrtGeV": float(fit.x[0]),
            "stochastic_term_error_fraction_sqrtGeV": float(parameter_errors[0]),
            "constant_term_fraction": float(fit.x[1]),
            "constant_term_error_fraction": float(parameter_errors[1]),
            "stochastic_term_percent_sqrtGeV": 100.0 * float(fit.x[0]),
            "stochastic_term_error_percent_sqrtGeV": 100.0 * float(parameter_errors[0]),
            "constant_term_percent": 100.0 * float(fit.x[1]),
            "constant_term_error_percent": 100.0 * float(parameter_errors[1]),
            "chi2": chi2,
            "ndf": ndf,
            "chi2_over_ndf": chi2_ndf,
            "fit_quality": fit_quality,
            "formula": FIT_FORMULA,
        })
    return pd.DataFrame(rows)


def _plot_comparison(
    points: pd.DataFrame,
    fits: pd.DataFrame,
    comparison: str,
    title: str,
    output_stem: Path,
) -> None:
    selected = points.loc[points["comparison"] == comparison]
    fit_selected = fits.loc[fits["comparison"] == comparison].set_index("method")
    figure, axis = plt.subplots(figsize=(7.4, 5.3))
    colors = plt.get_cmap("tab10")
    for index, (method, part) in enumerate(selected.groupby("method", sort=False)):
        part = part.sort_values("energy_GeV")
        fit = fit_selected.loc[method]
        color = colors(index)
        a = float(fit["stochastic_term_percent_sqrtGeV"])
        b = float(fit["constant_term_percent"])
        quality_suffix = ""
        if fit["fit_quality"] != "ok":
            quality_text = {
                "poor_chi2": "poor fit",
                "parameter_at_boundary": "invalid fit: a at boundary",
            }.get(fit["fit_quality"], str(fit["fit_quality"]))
            quality_suffix = f", {quality_text}"
        legend_label = f"{part['label'].iloc[0]}: a={a:.2f}%, b={b:.2f}%{quality_suffix}"
        axis.errorbar(
            part["energy_GeV"], part["resolution_percent"],
            yerr=part["resolution_error_percent"], marker="o", linestyle="none",
            capsize=2.5, color=color, label=legend_label,
        )
        energy_grid = np.linspace(5.0, 50.0, 300)
        curve = np.sqrt(a**2 / energy_grid + b**2)
        axis.plot(energy_grid, curve, color=color, linewidth=1.5)
    axis.set(
        xlabel="Pion energy [GeV]",
        ylabel=r"Energy resolution $\sigma_{68}/\mathrm{MPV}$ [%]",
        title=title,
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    axis.set_xlim(3.0, 51.8)
    axis.set_ylim(bottom=0.0)
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".png"), dpi=180)
    figure.savefig(output_stem.with_suffix(".pdf"))
    plt.close(figure)


def run_performance_validation(
    run_dir: Path,
    calice_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = run_dir.resolve()
    calice_dir = (calice_dir or _infer_calice_dir(run_dir)).resolve()
    output_dir = (output_dir or (run_dir / "performance_validation")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points = build_resolution_points(run_dir, calice_dir)
    fits = fit_resolution_curves(points)
    points.to_csv(output_dir / "energy_resolution_points.csv", index=False)
    fits.to_csv(output_dir / "energy_resolution_fits.csv", index=False)
    with (output_dir / "validation_sources.json").open("w") as handle:
        json.dump({
            "current_metrics": str((run_dir / "comparison_metrics.csv").resolve()),
            "calice_local_sc_metrics": str(
                (calice_dir / "tables" / "reweighted_energy_metrics.csv").resolve()
            ),
            "resolution_definition": "sigma68/MPV",
            "resolution_error_definition": "resolution/sqrt(2*N)",
            "fit_formula": FIT_FORMULA,
            "comparability_note": (
                "CALICE local-SC is an external full-grid application-split reference. "
                "All other methods use the current 30x30x100 common test split."
            ),
        }, handle, indent=2)
    _plot_comparison(
        points, fits, "s_only", "S-only software compensation",
        output_dir / "s_only_resolution_comparison",
    )
    _plot_comparison(
        points, fits, "dual_readout", "Dual-readout software compensation",
        output_dir / "dual_readout_resolution_comparison",
    )
    return points, fits
