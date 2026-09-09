import unittest

import numpy as np
import pandas as pd
import torch

from calo.device import select_device
from calo.dr_performance import fit_resolution_curves
from calo.dual_readout import (
    CHANNEL_BASE,
    DualReadoutCalibration,
    DualReadoutGeometry,
    build_event_input,
    split_name,
)


def encoded_cell(channel, ix, iy, iz, base=1000):
    physical = (iz + 1) * base * base + (iy + 1) * base + (ix + 1)
    return channel * CHANNEL_BASE + physical


class DualReadoutInputTest(unittest.TestCase):
    def setUp(self):
        self.geo = DualReadoutGeometry(nx=60, ny=60, nz=120)
        self.calibration = DualReadoutCalibration(
            s_gev_per_count=0.181099 / 1000.0,
            c_gev_per_count=0.0382009 / 1000.0,
            h_s_intercept=0.766375,
            h_s_slope_per_gev=-0.000985188,
            h_c_intercept=0.366592,
            h_c_slope_per_gev=-0.000484105,
        )

    def test_fixed_central_crop_discards_outer_cells(self):
        ids = np.array([
            encoded_cell(1, 15, 15, 0),
            encoded_cell(1, 15, 15, 0),
            encoded_cell(1, 14, 15, 0),
            encoded_cell(1, 15, 15, 100),
            encoded_cell(2, 15, 15, 0),
        ])
        n_s = np.array([100, 20, 999, 999, 0])
        n_c = np.array([0, 0, 0, 0, 50])
        voxel, _, diagnostics = build_event_input(
            ids, n_s, n_c, "sc_full", self.geo, self.calibration
        )
        self.assertEqual(voxel.shape, (2, 30, 100, 30))
        self.assertAlmostEqual(
            diagnostics["S_total_GeV"], 120 * self.calibration.s_gev_per_count, places=8
        )
        self.assertAlmostEqual(
            diagnostics["C_total_GeV"], 50 * self.calibration.c_gev_per_count, places=8
        )

    def test_s_only_has_no_c_or_dr_information(self):
        ids = np.array([
            encoded_cell(1, 15, 15, 0),
            encoded_cell(2, 15, 15, 0),
        ])
        n_s = np.array([100, 0])
        voxel, aux, diagnostics = build_event_input(
            ids, n_s, None, "s_only", self.geo, self.calibration
        )
        self.assertEqual(np.count_nonzero(voxel[1]), 0)
        self.assertTrue(np.all(aux[[1, 3, 4, 5, 6, 7, 8, 9, 10]] == 0))
        self.assertEqual(diagnostics["C_total_GeV"], 0.0)
        self.assertEqual(diagnostics["E_DR_reco_GeV"], 0.0)

    def test_split_is_deterministic(self):
        first = split_name(123, 30.0, 170510363)
        self.assertEqual(first, split_name(123, 30.0, 170510363))
        self.assertIn(first, {"train", "val", "test"})

    def test_explicit_cpu_device(self):
        self.assertEqual(select_device("cpu"), torch.device("cpu"))


class DualReadoutPerformanceTest(unittest.TestCase):
    def test_resolution_fit_recovers_stochastic_and_constant_terms(self):
        energy = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        stochastic, constant = 0.30, 0.04
        resolution = np.sqrt(stochastic**2 / energy + constant**2)
        points = pd.DataFrame({
            "comparison": "test",
            "method": "known_curve",
            "label": "Known curve",
            "source_scope": "unit test",
            "source_kind": "synthetic",
            "energy_GeV": energy,
            "resolution_fraction": resolution,
            "resolution_error_fraction": np.full(energy.size, 0.001),
        })
        result = fit_resolution_curves(points).iloc[0]
        self.assertAlmostEqual(
            result["stochastic_term_fraction_sqrtGeV"], stochastic, places=7
        )
        self.assertAlmostEqual(result["constant_term_fraction"], constant, places=7)
        self.assertAlmostEqual(result["chi2"], 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
