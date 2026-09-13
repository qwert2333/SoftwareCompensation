"""Controlled first dual-readout comparison."""

COMMON = dict(
    loss_name="relative_mse",
    denom_min=0.7,
    model_output="scaled_residual",
    model_version="groupnorm-residual-v1",
    base_eps=0.7,
    residual_scale=10.0,
    epochs=40,
    lr=1.0e-3,
    batch_size=8,
    num_workers=2,
    device="auto",
    data_order_seed=170510363,
    event_identity_version="explicit-file-roles-v3-groupnorm-scaled-residual",
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
