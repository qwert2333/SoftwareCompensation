# Dual-readout tile-HCAL S-only versus S+C comparison

This branch adds two controlled experiments for dual-readout tile-HCAL ROOT
samples. The current baseline is `Sapphire_5-5-5mm`:

- `dr_j1_s_only`: calibrated scintillation cells only. The C voxel and every
  C/dual-readout auxiliary position are exactly zero.
- `dr_j1_sc_full`: calibrated S and C cells enter the first 3D convolution as
  separate channels. Reconstructed S/C totals, hit counts, h/e values, chi,
  and the standard dual-readout estimate are also supplied as auxiliaries.

Both use the same `[2,30,100,30]` input shape and the same 11-dimensional
auxiliary interface. For the current `N30x30x100` production this is the full
generated grid: `x,y=[0,30)` and `z=[0,100)`. All totals and derived
auxiliaries are recomputed from the selected cells.

The standard dual-readout auxiliaries do not look up the ROOT filename energy
or `MCtruth_energy`. Starting from cropped reconstructed S, the code iterates
the configured h/e(E) lines and the resulting DR estimate to self-consistency.
The geometry is inferred from the ROOT filenames. Electron EM calibration and
h/e coefficients are loaded at runtime from the matching
`full_dualreadout/<sample>/tables/` directory and recorded in every
`run_config.json`; no sample name, grid depth, or calibration value is fixed in
the training entry point.

The data roles are determined strictly from filenames:

- `*_pi-_5-60GeV_Train.root`: training only;
- `*_pi-_5-60GeV_Valid.root`: validation/checkpoint selection only;
- `*_pi-_<energy>GeV.root`: post-training evaluation only.

The required evaluation energies are 5, 10, 20, 30, 40 and 50 GeV. Additional
fixed-energy files such as 15 and 25 GeV are included as held-out checks.

Run a small CPU interface test:

```bash
source ../setup_env.sh
python run_dual_readout.py \
  --input-dir /eos/user/f/faguo/DualReadoutTileCal/outputs_ML/condor/work \
  --reference-dir outputs_dual_readout/standard_dual_readout/data \
  --experiment all --epochs 1 --max-events-per-file 50 \
  --batch-size 2 --num-workers 0 --device cpu
```

`--no-capture-output` lets each completed epoch appear in the terminal
immediately. Each line reports train/validation loss and event counts, the
best validation loss so far, learning rate, device, elapsed time, and saved
checkpoint. A `*` after `best_val` marks a new best epoch.

Device selection defaults to `--device auto`, with priority CUDA, MPS, then
CPU. Use `--device mps` to require Apple GPU execution; the run fails instead
of silently falling back if MPS is unavailable. Automatic mixed precision and
gradient scaling remain CUDA-only, while MPS uses FP32.

Run the full comparison by omitting the smoke-test overrides. Outputs are
written below `outputs_dual_readout/<sample>/<experiment>/`, including
configuration, all epoch checkpoints, loss curves, test-event predictions,
and per-energy `sigma68/MPV`, response, and bias metrics.

With the default `--experiment all`, the top-level output also contains
`comparison_metrics.csv` and `sc_vs_s_only.csv`. The latter reports the
per-energy resolution change from enabling all C information.

## HTCondor GPU training

`submit_dual_readout_gpu.sh` checks that the Git worktree is clean, creates
the log directory and submits `condor/dual_readout_gpu.sub`. The job requests
one GPU, four CPU cores and 24 GB of memory with the `testmatch` flavour. It
uses the shared EOS data directly, runs both experiments sequentially, and
evaluates each best checkpoint on all fixed-energy pion files.

Each cluster writes to
`outputs_dual_readout/production/condor_<ClusterId>/work/`. A `COMPLETED`
file is created only after both training and evaluation finish successfully.

Submit from the repository root with:

```bash
./submit_dual_readout_gpu.sh
```

Build the post-training resolution comparison and the
`sqrt(a^2/E + b^2)` stochastic/constant-term fits with:

```bash
conda run --no-capture-output -n ml4hep \
  python validate_dual_readout_performance.py \
  --run-dir outputs_dual_readout/Sapphire_5-5-5mm
```

The S-only plot compares raw cropped S, the cropped S-only CNN, and the
existing CALICE local-SC application result. The CALICE result is retained as
an explicitly labelled full-grid external reference; the other curves use the
common `30x30x100` test split. The dual-readout plot compares the standard DR
calculation and the S+C CNN on that common cropped test split. Point tables,
fit parameters, fit quality, plots, and exact source paths are written below
`performance_validation/`.
