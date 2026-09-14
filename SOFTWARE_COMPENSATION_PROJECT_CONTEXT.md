# Software Compensation Project Context

> Updated: 2026-09-14
> Purpose: handoff reference for another agent working on the dual-readout tile-HCAL software-compensation study.
> Canonical code checkout: `/Users/fangyi/WorkingArea/SoftwareCompensation`

## 1. Project goal

The active goal is to adapt an existing calorimeter energy-regression project to the dual-readout tile HCAL samples produced in `/Users/fangyi/WorkingArea/DualReadout_MLSC`, starting from the `j1` no-timing baseline.

The first controlled physics comparison is:

- `s_only`: use scintillation information only and expose no Cherenkov-derived information to the model.
- `sc_full`: use both scintillation and Cherenkov cell signals, plus standard dual-readout reconstructed quantities as auxiliary inputs.

The primary observable is pion energy resolution. The required comparisons are:

- S-only: raw cropped S, CALICE-style local software compensation, and the S-only ML model.
- Dual readout: standard dual-readout reconstruction and the S+C ML model.
- Report resolution at each energy and fit
  \[
  \frac{\sigma_E}{E}=\sqrt{\frac{a^2}{E/\mathrm{GeV}}+b^2},
  \]
  where `a` is the stochastic term and `b` is the constant term.

This document distinguishes current verified implementation facts from interpretations and future recommendations. Existing results are diagnostic and must not yet be presented as final physics performance.

## 2. Workspace and repository state

### Code checkout

- Path: `/Users/fangyi/WorkingArea/SoftwareCompensation`
- Current branch: `feature/dual-readout-s-vs-sc`
- Remote branch exists: `origin/feature/dual-readout-s-vs-sc`
- The S-only and S+C implementations currently coexist as two experiments on one feature branch; there are not separate development branches for them.

Relevant commits, newest last:

- `54e95ec` — controlled dual-readout S versus S+C training pipeline
- `8f337cc` — Apple MPS support
- `2de087a` — dynamic Sapphire baseline configuration
- `21f5522` — per-epoch training progress output
- `a972efa` — performance validation and resolution-fit scripts
- `ab61ce4` — residual dual-readout CNN workflow, HTCondor submission, and compact diagnostic summaries

### Artifact policy

- `data/` is untracked and occupies approximately 11 GB. It contains the ROOT
  samples described in Section 11 and must not be added to Git.
- `outputs_dual_readout/` is ignored. The current baseline products occupy about
  230 MB, dominated by the reusable local-SC cell cache, and remain local.
- Git tracks the analysis scripts, this context document, the two calibration
  tables required by CNN input construction, and compact summary/configuration
  products under `outputs_dual_readout/`. Generated ROOT ntuples, cell caches,
  per-event tables and plots remain local.

## 3. Two layers of code in this checkout

### Original fixed-direction timing-ablation project

The original code is a five-particle ECAL+HCAL energy-regression study using HDF5 shards. It trains on fixed-direction `pi+` and evaluates independently on `pi+`, `K-`, `KL0`, neutrons, and electrons.

Important files:

- `calo/model.py`
- `calo/dataset.py`
- `calo/features.py`
- `calo/voxel.py`
- `calo/experiment_grid.py`
- `calo/train.py`
- `calo/evaluate.py`

The eight original jobs include no-timing, binned timing, propagation-time, oracle-timing, time-only, and shuffled-time ablations. This legacy path is separate from the active dual-readout ROOT pipeline.

### Active dual-readout extension

Important files:

- `run_dual_readout.py` — main training/evaluation entry point
- `calo/dual_readout.py` — CellID decoding, S/C voxel construction, calibration and auxiliary features
- `calo/dr_dataset.py` — streaming ROOT dataset
- `calo/dr_model.py` — dual-readout 3D CNN
- `calo/dr_train.py` — training, resume and checkpoint logic
- `calo/dr_evaluate.py` — per-event predictions and per-energy metrics
- `calo/dr_experiment_grid.py` — controlled `s_only` and `sc_full` configurations
- `calo/dr_performance.py` — comparison tables, fits and plots
- `validate_dual_readout_performance.py` — validation entry point
- `README_DUAL_READOUT.md` — user-facing usage notes
- `tests/test_dual_readout.py` — current unit tests

## 4. Raw hit and voxel semantics

The ROOT input uses variable-length event arrays:

- `vecCellID`
- `vecNScint`
- `vecNChren`
- `eventID`
- `MCtruth_energy`

Channel decoding follows:

```text
channel          = cellID // 1_000_000_000
physical_cell_id = cellID %  1_000_000_000
```

The current implementation treats channel 1 as scintillation and channel 2 as Cherenkov. Physical coordinates are decoded from the remaining integer ID.

A raw hit/vector entry is not necessarily in one-to-one correspondence with a final voxel:

- multiple entries with the same physical cell are summed with `numpy.add.at`;
- S and C belong to different readout channels but are placed at the corresponding physical cell position;
- cells outside the accepted grid are discarded;
- most voxels are empty.

Therefore the ML tensor is a detector-grid aggregation of hit-level signals, not a list that preserves one token per original entry.

## 5. Geometry and crop policy

The completed first training used the old Sapphire files with full geometry:

```text
60 x 60 x 120
```

Only the following region was retained:

```text
x = [15, 45)
y = [15, 45)
z = [0, 100)
```

The model input shape is:

```text
[batch, 2, 30, 100, 30]
```

The spatial layout after the channel dimension is `[x, z, y]`. The last 20 longitudinal layers and all transverse cells outside the central `30 x 30` window are discarded.

All S/C totals, active-cell counts and derived auxiliary quantities are recomputed after this crop. Information from discarded cells is not allowed to re-enter through the auxiliary branch.

The newly generated files in `data/` already use `N30x30x100`; they are not merely old `N60x60x120` files with an analysis-time crop. For this production the accepted CNN grid is the complete generated grid:

```text
x = [0, 30), y = [0, 30), z = [0, 100)
```

No cells exist outside that region, so the full-grid scalar counters and the
voxel sums refer to the same geometrical acceptance.

## 6. Current dual-readout model

`calo/dr_model.py` defines `DualReadoutHCALNet`.

### Voxel encoder

- Input channels: S and C in a single early-fusion tensor; there is deliberately no S/C dual-branch split.
- The trunk uses shallow residual 3D blocks with `GroupNorm` and `SiLU` activations, chosen to be more stable than BatchNorm for the current small batch size.
- Only the two transverse axes are downsampled with stride `(2, 1, 2)`; the full 100-bin longitudinal direction reaches the downstream branches.
- The encoded volume has 64 channels after the current GroupNorm residual trunk.

### Global branch

- Additional normalized 3D convolution at 64 channels.
- Adaptive global average pooling.
- Produces a 64-dimensional event representation.

### Longitudinal shower-profile branch

- The encoded volume is averaged over both transverse axes.
- The resulting learned longitudinal feature sequence is processed by two normalized 1D convolutions and global pooling.
- It produces a 64-dimensional vector.

This branch is manually specified as an architectural prior: the developer explicitly chooses to preserve and summarize longitudinal development. Its convolution weights and extracted representation are learned from data. It is therefore neither a fully hand-calculated shower variable nor a structure discovered automatically by an unconstrained network.

### Auxiliary branch, fusion and residual output

- Eleven scalar inputs are mapped to 32 learned features.
- Global, longitudinal and auxiliary features are concatenated.
- The final MLP outputs a bounded scaled residual `r = residual_scale * tanh(raw_r)`.
- The final linear layer is zero-initialized, so a new model starts from no correction: `r = 0` and `E_pred = E_base`.

The reconstructed energy is now:

```text
E_pred = E_base + sqrt(max(E_base, base_eps)) * r
```

The baseline energy is selected by mode:

```text
s_only  -> calibrated cropped S total
sc_full -> self-consistent cropped standard dual-readout energy E_DR
```

This parameterization is intended to let the CNN learn a stochastic-scale correction on top of a physically meaningful baseline rather than freely regressing absolute energy from scratch.

## 7. Auxiliary features

The common 11-position interface is:

1. `log1p_S_total_GeV`
2. `log1p_C_total_GeV`
3. `log1p_n_active_S`
4. `log1p_n_active_C`
5. `C_over_S`
6. `S_minus_C_over_sum`
7. `h_over_e_S_reco`
8. `h_over_e_C_reco`
9. `dualreadout_chi_reco`
10. `signed_log1p_E_DR_reco_GeV`
11. `E_DR_over_S`

These are manually engineered deterministic summaries. The auxiliary MLP learns how to use them, but the formulas themselves are not learned.

The motivation is to expose known global dual-readout relations directly, while the voxel encoder learns local shower morphology. This can improve data efficiency but may also cause the model to rely strongly on the standard reconstruction. An eventual ablation should compare voxel-only, global S/C-only, standard-DR-only and combined models.

### Strict `s_only` behavior

In `s_only`:

- the C voxel is exactly zero;
- every C-derived and dual-readout auxiliary position is exactly zero;
- only S total and S active-cell count remain nonzero.

This prevents hidden C-channel leakage while preserving an identical tensor and model interface between the two experiments.

### Standard dual-readout calculation

The implemented equations are:

\[
\chi(E)=\frac{1-(h/e)_S(E)}{1-(h/e)_C(E)},
\qquad
E_\mathrm{DR}=\frac{S-\chi C}{1-\chi}.
\]

The code starts from reconstructed cropped S and iterates the energy-dependent `h/e` lines and `E_DR` to self-consistency. Truth energy and filename energy are not used as input features. `MCtruth_energy` is used only as the regression target.

Calibration constants are loaded from external tables and recorded in each `run_config.json`; the sample name and constants are not hardcoded in the training entry point.

## 8. Oracle timing and original auxiliary features

Oracle timing belongs only to the original ECAL+HCAL timing-ablation path, not to the current dual-readout `s_only`/`sc_full` model.

The original oracle vector contains manually calculated event-level and per-layer timing summaries, including energy-weighted time mean/RMS, time quantiles, early/middle/late fractions, earliest time, and per-layer time RMS. It has dimension:

\[
11 + 5(30+48)=401.
\]

These features were introduced deliberately to test whether information exists in detailed timing summaries. They are an upper-bound/diagnostic feature construction rather than evidence that a practical end-to-end model has learned the same quantities. The original `j8` shuffles times within detector/event as a control.

## 9. Training setup and device support

Current dual-readout defaults:

```text
epochs          = 40
batch_size       = 8
learning_rate    = 1e-3
optimizer        = Adam
loss             = report-style relative MSE, 0.5 * ((E_pred - E_true) / max(E_true, 0.7 GeV))^2
denom_min        = 0.7 GeV
model output     = bounded scaled residual
base_eps         = 0.7 GeV
residual_scale   = 10.0
num_workers      = 2
train            = explicit *_5-60GeV_Train.root files
validation       = explicit *_5-60GeV_Valid.root files
test             = fixed-energy pion ROOT files only
data order seed  = 170510363
global seed      = 20260908
voxel transform  = log1p(signal_GeV / 1e-3 GeV)
```

The generated ROOT files reuse sample-local `eventID` values. Event identity is
therefore the tuple `(source filename, ROOT entry index, eventID)`. File roles
are explicit rather than derived from an event hash. Prediction files retain
all identity components and reject duplicate composite IDs.

Device selection priority for `--device auto` is:

```text
CUDA -> MPS -> CPU
```

Explicit `--device mps` fails if MPS is unavailable instead of silently falling back. CUDA uses automatic mixed precision and gradient scaling; MPS runs FP32 in the current implementation.

Training resumes automatically from `checkpoints/last.pth`. Every epoch is saved as `epoch_NNN.pth`; `best.pth` and `last.pth` are also maintained. Each epoch prints train/validation loss, event counts, best validation loss, learning rate, device, elapsed time and checkpoint name.

The earlier local runs used MPS. The HTCondor production run in Section 14 used CUDA on a Tesla V100 GPU.

## 10. Historical training and validation results

Output root:

```text
/Users/fangyi/WorkingArea/SoftwareCompensation/outputs_dual_readout/Sapphire_5-5-5mm
```

These results belong to the earlier `N60x60x120` production cropped to
`N30x30x100`. They are retained as history and must not be mixed with the new
baseline analysis in Sections 12--13. Both experiments completed 40 epochs.

| Experiment | Best epoch | Best validation loss | Last validation loss |
|---|---:|---:|---:|
| `dr_j1_s_only` | 9 | 0.08685 | 0.29389 |
| `dr_j1_sc_full` | 28 | 0.08428 | 0.12785 |

The large degradation after the best epoch, especially for `s_only`, indicates overfitting or unstable optimization. Evaluation correctly uses `best.pth`, not the final epoch.

### Per-energy model resolution

Resolution is currently defined as `sigma68 / MPV`.

| Energy [GeV] | S-only CNN | S+C CNN | Relative S+C improvement |
|---:|---:|---:|---:|
| 5 | 0.95% | 1.30% | -36.1% |
| 10 | 15.18% | 11.39% | 24.9% |
| 20 | 9.14% | 7.06% | 22.7% |
| 30 | 8.09% | 6.24% | 22.9% |
| 40 | 8.18% | 6.22% | 24.0% |
| 50 | 8.79% | 6.78% | 22.9% |

At 10--50 GeV, the S+C network is numerically better than the S-only network by approximately 23--25%. The 5 GeV point is anomalously narrow for both models, while the 10 GeV point is much broader. This non-smooth behavior is not compatible with an ordinary calorimeter resolution curve.

### Resolution fits

| Method | `a` [% sqrt(GeV)] | `b` [%] | Fit status |
|---|---:|---:|---|
| Raw S, cropped | 21.95 | 8.82 | poor chi-square |
| CALICE local SC, full-grid external reference | 25.89 | 1.47 | poor chi-square |
| Standard DR, cropped | 24.49 | 7.23 | poor chi-square |
| S-only CNN, cropped | approximately 0 | 1.36 | parameter at boundary; invalid fit |
| S+C CNN, cropped | approximately 0 | 2.10 | parameter at boundary; invalid fit |

The CNN fit values must not be interpreted as measured stochastic and constant terms. The fits fail because the six discrete training energies led to a strongly non-physical energy dependence.

### Interpretation of the current result

The most likely failure mode is regression toward the six discrete training energies. The model can partially classify which generated energy point an event belongs to and collapse predictions around that value, producing an artificially tiny width at 5 GeV and irregular resolution elsewhere.

Consequences:

- the current S+C versus S-only improvement is useful as a pipeline diagnostic, not yet a final physics result;
- the fitted stochastic/constant terms from the CNN are invalid;
- a continuous-energy training sample and independent fixed-energy evaluation samples are required;
- response linearity, bias and interpolation must be checked before interpreting resolution.

### Comparability limitation for CALICE local SC

The CALICE local-SC curve currently comes from an external full `60x60x120` grid and an independent 50% application split. Raw S, standard DR and both CNN curves use the current cropped `30x30x100` region and common 15% test split.

Therefore CALICE local SC is explicitly labelled as a reference, not a strictly apples-to-apples comparison. A final study should rerun local SC on the same new geometry and event split.

Generated validation products are under:

```text
outputs_dual_readout/Sapphire_5-5-5mm/performance_validation/
```

They include point tables, fit tables, source provenance JSON, and PNG/PDF comparison plots.

## 11. Newly generated samples currently in `data/`

All inspected new files contain the required branches `eventID`, `MCtruth_energy`, `vecCellID`, `vecNScint`, and `vecNChren`.

### Continuous pion samples

| Role suggested by filename | Events | Energy range | Distinct truth energies |
|---|---:|---:|---:|
| Train | 60,000 | 5--60 GeV | 60,000 |
| Validation | 10,000 | 5--60 GeV | 10,000 |

Every inspected event in these files has a distinct truth energy, which is appropriate for removing the discrete-energy shortcut.

### Fixed-energy pion evaluation samples

There are 5,000 events at each of:

```text
5, 10, 15, 20, 25, 30, 40, 50 GeV
```

### Fixed-energy electron samples

There are 5,000 events at each of:

```text
5, 10, 20, 30, 40, 50 GeV
```

### Integration status

The fixed calibration samples are integrated into the standard dual-readout and local-SC workflows. The CNN runner is also integrated with explicit file roles:

- `*_pi-_5-60GeV_Train.root` is used only for training;
- `*_pi-_5-60GeV_Valid.root` is used only for validation/checkpoint selection;
- fixed-energy pion files are used only for test evaluation;
- 15 and 25 GeV pion files are accepted as fixed-energy evaluation points and are not absorbed into calibration fits.

Each run records its data manifest, event identity protocol and calibration provenance in `run_config.json`. Prediction files retain `(source_file, entry_index, eventID)` so repeated sample-local `eventID` values cannot collide.

## 12. New-production baseline workflow and calibration

The following scripts are now the canonical new-production baseline tools:

- `scripts/RunFullDualReadout.cpp` -- standard S/C calibration, pion `h/e`,
  standard dual-readout reconstruction, resolution fits and compact ntuples.
- `scripts/run_calice_local_sc.py` -- CALICE-style cell-density local software
  compensation using S only.
- `scripts/export_cnn_baseline_inputs.py` -- frozen calibration JSON, Train-only
  input statistics and Train/Valid bias diagnostics for the CNN.

### Reproduction commands

Run from `/Users/fangyi/WorkingArea/SoftwareCompensation`:

```bash
root -l -b -q 'scripts/RunFullDualReadout.cpp'

conda run -n ml4hep python scripts/run_calice_local_sc.py \
  --geometry Sapphire_5-5-5mm_N30x30x100 \
  --input-dir data \
  --calibration-table outputs_dual_readout/standard_dual_readout/data/tables/electron_em_calibration.csv \
  --output-dir outputs_dual_readout/calice_local_sc/Sapphire_5-5-5mm_N30x30x100

conda run -n ml4hep python scripts/export_cnn_baseline_inputs.py
```

The copied C++ macro was changed in two essential ways:

1. input/output paths now point to this checkout and `outputs_dual_readout`;
2. energy parsing requires an exact numeric token and accepts only the common
   fixed calibration grid `{5,10,20,30,40,50}`. Therefore the extra 15/25 GeV
   pion samples and `5-60GeV_Train/Valid` files cannot be accidentally absorbed
   into the standard calibration fit.

### Frozen calibration

Electron calibration is a through-origin fit over the six fixed electron
energies:

```text
S = 0.0946402 MeV/count
C = 0.0381864 MeV/count
```

The fixed pion samples give:

```text
h/e_S(E) = 0.693453 - 0.00149546 E/GeV
h/e_C(E) = 0.352674 - 0.000731648 E/GeV
```

Primary source tables:

```text
outputs_dual_readout/standard_dual_readout/data/tables/electron_em_calibration.csv
outputs_dual_readout/standard_dual_readout/data/tables/pion_hovere_fit.csv
outputs_dual_readout/standard_dual_readout/data/tables/pion_hovere_energy_fit.csv
```

The standard reference evaluates `h/e(E)` at the nominal fixed sample energy.
The CNN auxiliary implementation cannot use nominal or truth energy and instead
performs six self-consistency iterations starting from reconstructed S. This
difference is intentional and must be preserved in comparisons.

## 13. New-production baseline results

### Standard dual readout

| Energy [GeV] | MPV response | Resolution `sigma68/MPV` |
|---:|---:|---:|
| 5  | 0.9809 | 17.53% |
| 10 | 0.9656 | 12.60% |
| 20 | 1.0050 | 10.23% |
| 30 | 1.0121 | 9.03% |
| 40 | 1.0093 | 8.90% |
| 50 | 1.0184 | 9.17% |

The fitted resolution is:

```text
DR: a = 34.04% sqrt(GeV), b = 7.13%, chi2/ndf = 87.67/4
```

This is an in-calibration-sample reference: the same fixed pion samples are
used to determine `h/e(E)` and quote the fixed-energy performance. It is not a
leakage-free ML evaluation.

### CALICE-style local SC

Each fixed-energy pion file is deterministically divided using `eventID`,
energy and seed into 15,009 calibration and 14,991 application events. Only
aggregated channel-1 `vecNScint` cell signal and unweighted S energy enter the
application reconstruction. C, `vecEdep`, truth `f_EM` and beam energy are
excluded.

| Energy [GeV] | Raw S resolution | Local-SC resolution | Improvement vs S |
|---:|---:|---:|---:|
| 5  | 16.54% | 15.71% | 5.00% |
| 10 | 12.13% | 9.67%  | 20.28% |
| 20 | 10.64% | 8.56%  | 19.55% |
| 30 | 10.09% | 7.11%  | 29.57% |
| 40 | 10.26% | 5.99%  | 41.61% |
| 50 | 10.32% | 5.17%  | 49.90% |

Application-split fits are:

```text
Raw S:   a = 28.71% sqrt(GeV), b = 8.94%, chi2/ndf = 51.58/4
Local SC: a = 32.95% sqrt(GeV), b = 2.82%, chi2/ndf = 145.31/4
```

The local-SC optimizer converged, but the large resolution-fit chi-square and
non-unity response, especially at 5, 20 and 30 GeV, show imperfect closure.
Report the per-energy points and response together with `a,b`; do not summarize
this result using the fitted terms alone.

### CNN calibration, scale and bias products

Stored under:

```text
outputs_dual_readout/cnn_inputs/Sapphire_5-5-5mm_N30x30x100/
```

- `cnn_calibration_and_bias_config.json` records geometry, S/C calibration,
  `h/e(E)`, chi/DR formula, self-consistency policy and provenance.
- `training_input_statistics.json` contains statistics derived only from the
  60,000-event continuous Train sample.
- `baseline_bias_by_energy_bin.csv` contains 5 GeV-bin S, C and self-consistent
  DR response/bias for Train and independent Valid closure.
- `baseline_bias_global_summary.csv` is a compact whole-range diagnostic.

Whole-range diagnostics are:

| Split | Method | Mean relative bias | `sigma68(residual)/mean(Etrue)` |
|---|---|---:|---:|
| Train 60k | S | -22.07% | 12.15% |
| Train 60k | C | -37.88% | 21.02% |
| Train 60k | self-consistent DR | -4.38% | 9.64% |
| Valid 10k | S | -22.04% | 11.93% |
| Valid 10k | C | -37.77% | 20.81% |
| Valid 10k | self-consistent DR | -4.42% | 9.64% |

These whole-range residual numbers mix the 5--60 GeV spectrum and are not
single-energy calorimeter resolutions. Never derive normalization or a bias
correction from the rows labelled `continuous_valid_closure`.

## 14. 2026-09-13 HTCondor residual-CNN production

The newest production run is:

```text
outputs_dual_readout/production/condor_321958
```

It was submitted on 2026-09-13 at 11:45 CERN time and ran commit:

```text
ab61ce42b684f07308d0b1992e524dba192cb37e
```

The job used the residual-output GroupNorm CNN described in Section 6 and the report-style relative-MSE loss described in Section 9. It ran both `s_only` and `sc_full` on CUDA using a Tesla V100-PCIE-32GB and completed normally on 2026-09-13 at 23:01 CERN time. The Condor return value was 0 and stderr was empty.

Training losses are not directly comparable to the older `condor_317536` run because the loss changed from relative L1 to relative MSE. Within the new run, validation behavior is substantially more stable than the old direct-energy model:

| Experiment | Best epoch | Best validation loss | Final train loss | Final validation loss |
|---|---:|---:|---:|---:|
| `dr_j1_s_only` | 38 | 0.007363 | 0.006480 | 0.007806 |
| `dr_j1_sc_full` | 35 | 0.007079 | 0.006741 | 0.007354 |

### Per-energy CNN results for `condor_321958`

Resolution remains defined as `sigma68 / MPV`.

| Energy [GeV] | S-only CNN | S+C CNN | Relative S+C improvement | S-only mean bias | S+C mean bias |
|---:|---:|---:|---:|---:|---:|
| 5  | 7.11% | 7.93% | -11.42% | 5.17% | 4.70% |
| 10 | 6.74% | 7.04% | -4.44% | 0.17% | -0.46% |
| 15 | 6.15% | 6.17% | -0.34% | -0.70% | -0.87% |
| 20 | 5.72% | 5.69% | 0.46% | -1.64% | -1.95% |
| 25 | 5.67% | 5.54% | 2.27% | -1.71% | -2.23% |
| 30 | 5.33% | 5.24% | 1.65% | -1.87% | -2.63% |
| 40 | 4.98% | 4.90% | 1.59% | -2.86% | -3.49% |
| 50 | 4.40% | 4.27% | 3.01% | -4.57% | -5.08% |

Compared with the older direct-energy `condor_317536` model, the residual CNN has a more physical resolution scaling but slightly worse average absolute resolution:

| Version | Mode | Average resolution | Average absolute mean bias |
|---|---|---:|---:|
| `condor_317536` direct energy | `s_only` | 5.67% | 2.51% |
| `condor_317536` direct energy | `sc_full` | 5.56% | 2.69% |
| `condor_321958` residual | `s_only` | 5.76% | 2.34% |
| `condor_321958` residual | `sc_full` | 5.85% | 2.68% |

Weighted fits to `sqrt(a^2/E + b^2)` improved in shape relative to the old direct-energy fit, but are still not high-quality fits:

| Version | Mode | `a` [% sqrt(GeV)] | `b` [%] | chi2/ndf |
|---|---|---:|---:|---:|
| `condor_317536` direct energy | `s_only` | 7.41 | 5.24 | 106.9 |
| `condor_317536` direct energy | `sc_full` | 4.69 | 5.31 | 137.7 |
| `condor_321958` residual | `s_only` | 14.05 | 4.51 | 59.2 |
| `condor_321958` residual | `sc_full` | 16.84 | 4.05 | 41.0 |

Against the same-geometry baselines, the new residual `sc_full` CNN remains much better in absolute resolution than standard dual readout and CALICE-style local SC:

| Method | Average resolution | Average absolute mean bias |
|---|---:|---:|
| residual CNN `sc_full` | 5.85% | 2.68% |
| standard dual readout | 11.25% | 3.70% |
| CALICE local SC | 8.70% | 3.38% |

Interpretation: the residual parameterization and GroupNorm residual trunk fixed part of the unphysical flat-resolution behavior and made the validation curve calmer, but the S+C model no longer clearly beats S-only at low energy. At 20--50 GeV it gives small positive gains over S-only; at 5--15 GeV it is slightly worse. The next tuning target is therefore not more architecture complexity, but low-energy response control, learning-rate scheduling or early stopping, and ablations of the DR auxiliary variables.

## 15. Recommended next implementation steps

Priority order:

1. Add learning-rate scheduling or early stopping to avoid training far past the best validation epoch.
2. Tune low-energy behavior of the residual CNN, especially the 5--15 GeV region where `sc_full` is currently worse than `s_only`.
3. Run controlled ablations for voxel information, global auxiliary information, standard-DR auxiliary variables and longitudinal branch contributions.
4. Compare alternate residual scales and bounded/unbounded residual heads.
5. Repeat final results over multiple random seeds.
6. Report response linearity and mean bias together with resolution.
7. Fit `a` and `b` only after the resolution points show a physically plausible smooth energy dependence and fit quality is acceptable.
8. Estimate statistical uncertainties with bootstrap or toy resampling rather than relying only on `resolution/sqrt(2N)`.

## 16. Validation and safety rules for future agents

- Do not interpret a tiny width around a generated discrete energy as genuine detector resolution.
- Do not use truth energy or filename energy as an input feature.
- Do not calculate calibration, `h/e`, `chi`, thresholds or correction parameters from the evaluation sample.
- Keep `s_only` free of all C-derived information, including global summaries.
- Keep S/C comparison splits and event selections identical.
- Recompute global features after any crop or selection so discarded regions cannot leak through auxiliaries.
- Do not compare the earlier external `N60x60x120` local-SC curve with the new
  `N30x30x100` baseline; use the Section 13 application-split result.
- Preserve the new ROOT files as untracked data; do not commit them.
- Treat a smoke test as software validation only, not as physics validation.
- Check the working tree before editing; preserve user data and unrelated changes.

## 17. Current test status

The `ml4hep` environment does not currently contain `pytest`, but the test suite is compatible with Python `unittest`.

Verified command:

```bash
source /eos/home-f/faguo/ML/setup_env.sh
cd /eos/home-f/faguo/ML/SoftwareCompensation
python -m unittest tests.test_dual_readout
```

Status on 2026-09-14: all ten tests passed in the `ml4hep` environment.

The tests cover:

- crop behavior and signal aggregation;
- strict removal of C/DR information in `s_only`;
- explicit file-role discovery and composite event identity;
- explicit CPU selection;
- scaled-residual base selection and prediction formula;
- report-style relative-MSE loss;
- zero-initialized residual model output;
- recovery of known synthetic stochastic/constant terms by the fitter.

The Matplotlib cache may fall back to a temporary directory because the default user cache is not writable in a restricted agent environment; this warning did not cause a test failure.

## 18. Suggested reading order

For a new agent, read in this order:

1. `README_DUAL_READOUT.md`
2. `run_dual_readout.py`
3. `calo/dual_readout.py`
4. `calo/dr_dataset.py`
5. `calo/dr_model.py`
6. `calo/dr_train.py`
7. `calo/dr_evaluate.py`
8. `calo/dr_performance.py`
9. `scripts/RunFullDualReadout.cpp`
10. `scripts/run_calice_local_sc.py`
11. `scripts/export_cnn_baseline_inputs.py`
12. `outputs_dual_readout/README.md`
13. the primary tables and JSON files listed in Sections 12--13
14. inspect the new `data/*.root` schemas and event IDs before changing the loader
