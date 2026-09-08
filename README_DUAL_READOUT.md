# Dual-readout tile-HCAL S-only versus S+C comparison

This branch adds two controlled experiments for dual-readout tile-HCAL ROOT
samples. The current baseline is `Sapphire_5-5-5mm`:

- `dr_j1_s_only`: calibrated scintillation cells only. The C voxel and every
  C/dual-readout auxiliary position are exactly zero.
- `dr_j1_sc_full`: calibrated S and C cells enter the first 3D convolution as
  separate channels. Reconstructed S/C totals, hit counts, h/e values, chi,
  and the standard dual-readout estimate are also supplied as auxiliaries.

Both use the same `[2,30,100,30]` input shape and the same 11-dimensional
auxiliary interface. Transverse indices `x,y=[15,45)` and longitudinal indices
`z=[0,100)` are retained. Thus the last 20 Sapphire layers are discarded. All
totals and derived auxiliaries are recomputed after the crop, so discarded
cells cannot leak through scalars.

The standard dual-readout auxiliaries do not look up the ROOT filename energy
or `MCtruth_energy`. Starting from cropped reconstructed S, the code iterates
the configured h/e(E) lines and the resulting DR estimate to self-consistency.
The geometry is inferred from the ROOT filenames. Electron EM calibration and
h/e coefficients are loaded at runtime from the matching
`full_dualreadout/<sample>/tables/` directory and recorded in every
`run_config.json`; no sample name, grid depth, or calibration value is fixed in
the training entry point.

Run a small interface test:

```bash
conda run -n ml4hep python run_dual_readout.py \
  --input-dir /Users/fangyi/WorkingArea/DualReadout_MLSC/Sapphire_5-5-5mm \
  --experiment all --epochs 1 --max-events-per-file 50 \
  --batch-size 2 --num-workers 0
```

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
