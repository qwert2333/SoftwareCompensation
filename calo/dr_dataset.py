"""Streaming ROOT dataset for central-crop dual-readout training."""

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import IterableDataset, get_worker_info

from .dual_readout import build_event_input, split_name


class DualReadoutIterable(IterableDataset):
    def __init__(self, root_files, mode, split, geo, calibration, split_seed,
                 shuffle=True, max_events_per_file=-1, cell_signal_scale_gev=1.0e-3):
        super().__init__()
        self.files = sorted(str(path) for path in root_files)
        self.mode = mode
        self.split = split
        self.geo = geo
        self.calibration = calibration
        self.split_seed = int(split_seed)
        self.shuffle = bool(shuffle)
        self.max_events_per_file = int(max_events_per_file)
        self.cell_signal_scale_gev = float(cell_signal_scale_gev)
        self._iteration = 0

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        self._iteration += 1
        rng = np.random.default_rng(self.split_seed + 1009 * worker_id + 9176 * self._iteration)
        files = list(self.files)
        if self.shuffle:
            rng.shuffle(files)
        files = files[worker_id::n_workers]

        branches = ["eventID", "MCtruth_energy", "vecCellID", "vecNScint"]
        if self.mode == "sc_full":
            branches.append("vecNChren")

        for path in files:
            with uproot.open(path) as root_file:
                tree = root_file["eventTree"]
                missing = [name for name in branches if name not in tree]
                if missing:
                    raise RuntimeError(f"Missing branches {missing} in {path}")
                entry_stop = tree.num_entries
                if self.max_events_per_file >= 0:
                    entry_stop = min(entry_stop, self.max_events_per_file)
                for arrays in tree.iterate(
                    branches, entry_stop=entry_stop, step_size="64 MB", library="ak"
                ):
                    indices = np.arange(len(arrays["eventID"]))
                    if self.shuffle:
                        rng.shuffle(indices)
                    for idx in indices:
                        event_id = int(arrays["eventID"][idx])
                        target = float(arrays["MCtruth_energy"][idx]) / 1000.0
                        if split_name(event_id, target, self.split_seed) != self.split:
                            continue
                        ids = ak.to_numpy(arrays["vecCellID"][idx])
                        n_s = ak.to_numpy(arrays["vecNScint"][idx])
                        n_c = None
                        if self.mode == "sc_full":
                            n_c = ak.to_numpy(arrays["vecNChren"][idx])
                        voxel, aux, diagnostics = build_event_input(
                            ids, n_s, n_c, self.mode, self.geo, self.calibration,
                            self.cell_signal_scale_gev,
                        )
                        yield {
                            "voxel": torch.from_numpy(voxel),
                            "aux": torch.from_numpy(aux),
                            "energy_true": torch.tensor(target, dtype=torch.float32),
                            "event_id": torch.tensor(event_id, dtype=torch.int64),
                            "s_total": torch.tensor(diagnostics["S_total_GeV"], dtype=torch.float32),
                            "c_total": torch.tensor(diagnostics["C_total_GeV"], dtype=torch.float32),
                            "dr_reco": torch.tensor(diagnostics["E_DR_reco_GeV"], dtype=torch.float32),
                        }
