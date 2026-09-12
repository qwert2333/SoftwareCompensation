# Dual-readout calibration and local-SC outputs

Generated from the `N30x30x100` Sapphire 5-5-5 mm electron and pion ROOT
samples in `data/`.

## Standard dual readout

Directory: `standard_dual_readout/data/`

- Electron EM calibration uses the fixed 5, 10, 20, 30, 40 and 50 GeV
  electron samples.
- Pion `h/e` and `h/e(E)` use the fixed pion samples at the same six energies.
- The standard reference uses
  `chi=(1-h/e_S)/(1-h/e_C)` and `E_DR=(S-chi*C)/(1-chi)`.
- The fixed-pion resolution is an in-calibration-sample reference because the
  same pion samples determine `h/e(E)`; it is not a leakage-free ML test.
- Compact reconstructed pion ROOT ntuples are under `ntuples/`.

Primary tables:

- `tables/electron_em_calibration.csv`
- `tables/pion_hovere_fit.csv`
- `tables/pion_hovere_energy_fit.csv`
- `tables/pion_dualreadout_resolution.csv`
- `tables/pion_dualreadout_resolution_fit.csv`

## CALICE-style local software compensation

Directory: `calice_local_sc/Sapphire_5-5-5mm_N30x30x100/`

- Only aggregated S-channel cell signal is used.
- Each fixed-energy pion sample is deterministically divided 50/50 into
  calibration and application events using `eventID`, energy and split seed.
- C signal, `vecEdep`, truth `f_EM` and beam energy are excluded from the
  application reconstruction.
- `tables/reweighted_energy_metrics.csv` and
  `tables/reweighted_resolution_fit.csv` are the primary performance outputs.
- `intermediate/cell_hits.root` is a reusable 191 MB cell cache.

The two-stage local-SC fit converged, but its application resolution-curve fit
has a large chi2/ndf.  Treat the energy-dependent closure as a diagnostic and
inspect the per-energy responses rather than relying only on the global `a,b`
fit.

## CNN inputs and diagnostics

Directory: `cnn_inputs/Sapphire_5-5-5mm_N30x30x100/`

- `cnn_calibration_and_bias_config.json`: S/C calibration, `h/e(E)`, chi and
  self-consistent reconstructed-energy policy used by the CNN auxiliaries.
- `training_input_statistics.json`: feature/auxiliary summary statistics from
  the continuous 60k Train sample only.
- `baseline_bias_by_energy_bin.csv`: 5 GeV-bin S, C and self-consistent standard
  DR response/bias for Train and independent Valid closure.
- `baseline_bias_global_summary.csv`: compact whole-range summary.

Do not derive corrections or normalization from rows labeled
`continuous_valid_closure`.  MC truth is used only for targets and diagnostics;
the CNN auxiliary standard-DR estimate evaluates `h/e` iteratively at its own
reconstructed energy.
