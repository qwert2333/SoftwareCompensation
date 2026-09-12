# CALICE cell-level local software compensation

This is a traditional, non-ML software-compensation chain for
`ZnWO4_5_Quartz_5_Steel_10`. It follows the cell-level method in
[H. L. Tran et al., Eur. Phys. J. C 77 (2017) 698](https://arxiv.org/abs/1705.10363),
not the cluster-level method in CALICE CAN-021.

## Reconstruction definition

Simulation steps with the same scintillation-layer `vecCellID` are first
merged into one calorimeter cell. The electron S calibration converts the
summed `vecNScint` count to EM-scale hit energy:

```text
E_hit [GeV] = sum_cell(vecNScint) * S_EM_calibration [GeV/count]
```

The local density includes the active layer, absorber and passive layer in the
cell volume, as in the reference:

```text
rho = E_hit [GeV] / (V_cell / 1000 cm^3)
V_cell = 30 mm * 30 mm * (5 + 5 + 10) mm = 18 cm^3
```

Ten density bins are derived from calibration events only. Their boundaries
are weighted quantiles chosen so that each bin contributes approximately the
same normalized hit energy. The correction is

```text
E_LSC = sum_density_bins E_bin * omega(rho_bin, E_sum)
omega(rho,E) = p1(E) * exp(p2(E) * rho) + p3(E)
p1(E) = p10 + p11 E + p12 E^2
p2(E) = p20 + p21 E + p22 E^2
p3(E) = p30 / (p31 + exp(p32 E))
```

`E_sum` is the event's unweighted S energy. Beam energy is used only as the
target in the calibration half of the sample; application uses neither beam
energy nor truth information. `vecEdep`, C, `truthFem`, and filename energy
labels are excluded from application inputs.

## Run

From `/Users/fangyi/WorkingArea/DualReadout_MLSC`:

```bash
MPLCONFIGDIR=/tmp/calice-lsc-matplotlib \
conda run -n ml4hep python run_calice_local_sc.py
```

A fast interface test can limit the number of events in each ROOT file:

```bash
MPLCONFIGDIR=/tmp/calice-lsc-matplotlib \
conda run -n ml4hep python run_calice_local_sc.py \
  --max-events 50 --output-dir /tmp/calice_lsc_smoke
```

The deterministic split is approximately 50% calibration and 50%
application at each energy. Only the independent application split is used
for the quoted performance.

## Inspectable products

Results are written below
`calice_local_sc/ZnWO4_5_Quartz_5_Steel_10/`.

- `run_config.json`: geometry, calibration, formulae, split and exclusions.
- `intermediate/cell_hits.root`: aggregated per-cell energies and densities.
- `tables/export_validation.csv`: scalar/vector closure and event counts.
- `tables/hit_density_histograms.csv`: full hit-density histograms.
- `tables/hit_density_binning.csv`: ten bin edges, representatives and energy
  shares.
- `intermediate/event_density_bin_sums.csv`: event energy contribution in
  every density bin before fitting.
- `tables/weight_parameterization.csv`: the default two-stage nine-parameter
  result; `weight_parameterization_two_stage.csv` is an explicit alias.
- `tables/weight_parameterization_direct_global_reference.csv`: the retained
  direct-global event-fit reference.
- `tables/hit_density_weights.csv`: weight value in every density bin and
  beam-energy sample.
- `tables/event_reconstructed_energies.csv`: unweighted and reweighted energy
  for every event. `E_LSC_GeV` is the default two-stage result;
  `E_LSC_direct_global_reference_GeV` preserves the reference result.
- `tables/reweighted_energy_metrics.csv`: primary MPV response and
  `sigma68/MPV` resolution at every energy. The former central-90% Gaussian
  and RMS90/Mean90 quantities remain as explicitly named diagnostic columns.
- `tables/reweighted_resolution_fit.csv`: stochastic and constant terms.
- `plots/hit_density_distribution_*GeV.png`: hit density and selected bins.
- `plots/weights_vs_hit_density.png` and
  `plots/weight_energy_parameterization.png`: fitted weights.
- `plots/reweighted_energy_distribution_*GeV.png`: independent application
  energy spectra before and after compensation.
- `plots/reweighted_energy_resolution.png`: final resolution curve.
- `plots/fig2_style_30GeV_hit_density_binning.png`: paper Fig. 2-style 30 GeV
  density distribution with all ten bin ranges.
- `plots/weights_vs_hit_density_parameterization_by_energy.png`: one panel per
  energy comparing auxiliary per-energy fits with the global parameterization.
- `plots/energy_dependent_weight_parameters.png`: `p1(E)`, `p2(E)` and `p3(E)`
  with independent per-energy diagnostic points.
- `plots/bin_energy_closure.png` and `plots/energy_response_closure.png`:
  numerical bin-sum closure and application-sample physics closure.
- `plots/final_resolution_parameterization_fit.png`: resolution points and the
  fitted `a/sqrt(E) combined with b` curves.
- `tables/per_energy_weight_diagnostic_fit.csv`: auxiliary three-parameter
  fits used only for validation plots; they do not enter the final energy.

The primary resolution convention is identical to `ComputeResolution` in
`RunScintGlassSOnlyEdepScintLayer.cpp`. The MPV is the centre of the maximum
bin in an 80-bin histogram spanning the full sample range with a 5% margin.
`sigma68` is half the width of the narrowest interval containing
`round(0.68*N)` events. The reported resolution and error are

```text
resolution = sigma68 / MPV
resolution_error = resolution / sqrt(2*N)
```

The six energy points are fitted with
`sqrt(a^2/E_GeV + b^2)`. The central-90% Gaussian and RMS90/Mean90 values are
retained only as auxiliary diagnostics and are not used in the default plots,
improvement numbers or resolution fit.

## Validated full-sample result

The reference run used all 2000 events at each of 5, 10, 20, 30, 40 and
50 GeV, with 6008 calibration and 5992 independent application events. The
application-sample `sigma68/MPV` results are:

| Beam energy [GeV] | LSC response | S resolution [%] | LSC resolution [%] | Improvement [%] |
|---:|---:|---:|---:|---:|
| 5  | 0.9940 | 12.548 | 10.255 | 18.27 |
| 10 | 1.0014 | 9.513 | 7.050 | 25.89 |
| 20 | 1.0068 | 7.540 | 4.706 | 37.58 |
| 30 | 1.0093 | 7.116 | 3.810 | 46.46 |
| 40 | 1.0163 | 6.400 | 3.370 | 47.35 |
| 50 | 1.0058 | 6.599 | 3.133 | 52.52 |

All ten density bins contribute 9.986%--10.018% of the normalized calibration
hit energy. The fitted weights are positive and monotonically decreasing with
density at every reference energy, spanning approximately 0.77--1.89. Binned
hit energies close to the event S energy within `2.4e-6 GeV`.

Seven off-layer scintillation counts were found in seven of the 12000 events.
They occur in the Quartz/C layer and are excluded because the S calibration is
defined for `counter_Scintillation_ScintLayer`. Their counts and affected event
numbers are recorded per energy in `tables/export_validation.csv`; no event is
removed. Re-running the fit from `cell_hits.root` reproduced the binning,
parameter and metric CSV files byte-for-byte.

The bin borders are not equal-width or equal-hit-count bins. They are weighted
quantiles derived from calibration events only. Each cell contributes
`E_hit/E_beam` to the quantile calculation. With approximately equal event
counts at each beam energy, this prevents the high-energy samples from
dominating and gives every bin approximately 10% of the normalized visible
energy. For the 18 cm3 cell volume, the ten hit-energy ranges are 0--4.53,
4.53--14.38, 14.38--26.42, 26.42--49.34, 49.34--85.52, 85.52--139.60,
139.60--230.41, 230.41--412.93, 412.93--846.51 and above 846.51 MeV.

## Default two-stage p_i fit and direct-global reference

The default result uses an explicit two-stage fit. First, independent
three-parameter weight curves are fitted at each of the six energies. Second,
`p1(E)` and `p2(E)` are fitted to those points with their errors, and the
prescribed saturating form is fitted to the six `p3(E)` points. Positivity of
`p1`, negativity of `p2`, and finite positive weights are enforced over the
application range. This makes the displayed energy-dependent parameter curves
close the per-energy diagnostic points by construction.

The original direct-global fit, which minimizes event-energy residuals over all
nine parameters at once, is retained as a reference. Because the three
parameters of an individual exponential-plus-constant curve are correlated,
its energy-dependent curves need not pass through the independently fitted
parameter points even when the resulting weight curves are similar.

The two-stage result is now the pipeline default: it fills `E_LSC_GeV`,
`weight_parameterization.csv`, `reweighted_energy_metrics.csv`, all closure
plots and the final resolution fit. It gives a parameter-fit chi2/ndf of
5.80/9. Its application MPV responses range from 0.9940 to 1.0163;
`sigma68/MPV` resolutions are 10.255%, 7.050%, 4.706%, 3.810%, 3.370% and
3.133% from 5 to 50 GeV. The fitted resolution is
`21.71%/sqrt(E/GeV) combined with 0.00%`, with chi2/ndf 13.69/4.

For comparisons, the former result remains in
`E_LSC_direct_global_reference_GeV`,
`tables/weight_parameterization_direct_global_reference.csv`,
`tables/direct_global_reference_energy_metrics.csv`, and
`tables/direct_global_reference_resolution_fit.csv`. The legacy
`E_LSC_two_stage_candidate_GeV` and `two_stage_candidate_*` names are retained
as compatibility aliases of the two-stage primary result.
