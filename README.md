# Fixed-direction five-particle timing ablations

This is the fixed-direction counterpart of `project_time_ipstart_isotropic_ablation`. It keeps the same eight controlled jobs, layer-ID geometry, corrected-time definitions, relative L1 objective (`denom_min=0.7`), threshold sources, and per-epoch checkpoints.

Training uses only pi+ shards from `Train_Samples/fixed_direction/pi+/shards_fixed_y`. Evaluation is performed independently on pi+, K-, KL0, neutron, and electron samples. All particles share the same 26 nominal momentum points; particle-specific nominal energies are computed with `E=sqrt(p^2+m^2)`. Evaluation aborts unless every particle has exactly 15,000 events at every point.

Results are written to `prediction_results/<particle>/`, plus `prediction_results/five_particle_summary.json` for direct generalization comparisons. Evaluation uses one worker per HDF5 file to avoid duplicating the large variable-length hit arrays in memory. Every epoch is retained as `checkpoints/epoch_NNN.pth`; `best.pth` and `last.pth` are also saved.

Submit with:

```bash
bash submit_grid.sh
```
