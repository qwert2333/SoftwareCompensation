"""Streaming ROOT dataset for central-crop dual-readout training."""

import os

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import IterableDataset, get_worker_info

from .dual_readout import build_event_input, event_uid


class DualReadoutIterable(IterableDataset):
    def __init__(self, root_files, mode, geo, calibration, shuffle_seed,
                 shuffle=True, max_events_per_file=-1, cell_signal_scale_gev=1.0e-3):
        super().__init__()
        self.files = sorted(str(path) for path in root_files)
        source_names = [os.path.basename(path) for path in self.files]
        if len(source_names) != len(set(source_names)):
            raise ValueError("ROOT input filenames must be unique for stable event identities")
        self.mode = mode
        self.geo = geo
        self.calibration = calibration
        self.shuffle_seed = int(shuffle_seed)
        self.shuffle = bool(shuffle)
        self.max_events_per_file = int(max_events_per_file)
        self.cell_signal_scale_gev = float(cell_signal_scale_gev)
        self._iteration = 0

    def __iter__(self):
        worker = get_worker_info()
        worker_id, n_workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        self._iteration += 1
        rng = np.random.default_rng(
            self.shuffle_seed + 1009 * worker_id + 9176 * self._iteration
        )
        files = list(self.files)
        if self.shuffle:
            rng.shuffle(files)

        branches = ["eventID", "MCtruth_energy", "vecCellID", "vecNScint"]
        if self.mode == "sc_full":
            branches.append("vecNChren")

        for path in files:
            source_file = os.path.basename(path)
            with uproot.open(path) as root_file:
                tree = root_file["eventTree"]
                missing = [name for name in branches if name not in tree]
                if missing:
                    raise RuntimeError(f"Missing branches {missing} in {path}")
                total_entries = tree.num_entries
                if self.max_events_per_file >= 0:
                    total_entries = min(total_entries, self.max_events_per_file)
                entry_start = total_entries * worker_id // n_workers
                entry_stop = total_entries * (worker_id + 1) // n_workers
                if entry_start == entry_stop:
                    continue
                entry_offset = entry_start
                for arrays in tree.iterate(
                    branches,
                    entry_start=entry_start,
                    entry_stop=entry_stop,
                    step_size="64 MB",
                    library="ak",
                ):
                    entry_indices = np.arange(entry_offset, entry_offset + len(arrays["eventID"]))
                    entry_offset += len(arrays["eventID"])
                    indices = np.arange(len(arrays["eventID"]))
                    if self.shuffle:
                        rng.shuffle(indices)
                    for idx in indices:
                        event_id = int(arrays["eventID"][idx])
                        entry_index = int(entry_indices[idx])
                        target = float(arrays["MCtruth_energy"][idx]) / 1000.0
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
                            "event_uid": event_uid(source_file, entry_index, event_id),
                            "source_file": source_file,
                            "entry_index": torch.tensor(entry_index, dtype=torch.int64),
                            "event_id": torch.tensor(event_id, dtype=torch.int64),
                            "s_total": torch.tensor(diagnostics["S_total_GeV"], dtype=torch.float32),
                            "c_total": torch.tensor(diagnostics["C_total_GeV"], dtype=torch.float32),
                            "dr_reco": torch.tensor(diagnostics["E_DR_reco_GeV"], dtype=torch.float32),
                        }
