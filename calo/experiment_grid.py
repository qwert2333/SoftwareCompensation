"""The eight controlled ablations used by this project."""

GEO = dict(ecal_cell_size_xz=5.0, ecal_windowsize_xz=200.0,
           hcal_cell_size_xz=30.0, hcal_windowsize_xz=1200.0)
BASE = dict(
    geo=GEO, y_mode="layerid", loss_name="l1_relative", denom_min=0.7,
    time_source="corrected", time_resolution_ns=0.0, epochs=40, early_stopping=False, lr=1.0e-3,
    aux_mode="energy_nhits", include_energy_channels=True,
    include_hitcount=True, use_longitudinal_profile=True,
    time_mode="no_time", include_propagation=False,
    oracle_features=False, shuffle_times=False,
    threshold_mode="fixed_train_eventwise_median",
)

def _exp(tag, note, **updates):
    cfg = dict(BASE)
    cfg.update(name=tag, note=note, **updates)
    return cfg

EXPERIMENTS = [
    _exp("j1_notime", "Energy/nhits baseline; no timing input."),
    _exp("j2_t5fixed", "Five corrected-time bins; common train thresholds.", time_mode="bins5"),
    _exp("j3_t5fixed_prop", "Job 2 plus propagation-time voxel channel.", time_mode="bins5", include_propagation=True),
    _exp("j4_t5edep_prop", "Energy-dependent five bins plus propagation.", time_mode="bins5", include_propagation=True, threshold_mode="energy_dependent_eval_eventwise_median"),
    _exp("j5_notime_oracle", "Job 1 plus event and per-layer oracle timing features.", oracle_features=True),
    _exp("j6_t5fixed_prop_oracle", "Job 3 plus oracle timing features.", time_mode="bins5", include_propagation=True, oracle_features=True),
    _exp("j7_timeonly", "Only five corrected-time fractions and propagation time.", time_mode="bins5", include_propagation=True, include_energy_channels=False, include_hitcount=False, aux_mode="none", use_longitudinal_profile=False),
    _exp("j8_oracle_shuf", "Job 5 with times shuffled within detector/event.", oracle_features=True, shuffle_times=True),
]

def make_run_name(exp):
    return exp["name"]
