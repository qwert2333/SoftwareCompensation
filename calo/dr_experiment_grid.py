"""Controlled first dual-readout comparison."""

COMMON = dict(
    loss_name="l1_relative",
    denom_min=0.7,
    epochs=40,
    lr=1.0e-3,
    batch_size=8,
    num_workers=2,
    device="auto",
    split_seed=170510363,
    cell_signal_scale_gev=1.0e-3,
)


def _experiment(name, mode, note):
    return dict(COMMON, name=name, mode=mode, note=note)


EXPERIMENTS = [
    _experiment(
        "dr_j1_s_only",
        "s_only",
        "Central 30x30x100 S-only input; C voxel and every C/DR auxiliary are zero.",
    ),
    _experiment(
        "dr_j1_sc_full",
        "sc_full",
        "Central 30x30x100 early-fusion S+C input with reconstructed DR auxiliaries.",
    ),
]
