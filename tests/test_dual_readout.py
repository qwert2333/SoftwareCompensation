import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from calo.device import select_device
from calo.dr_model import (
    DualReadoutHCALNet,
    energy_from_scaled_residual,
    residual_base_from_batch,
)
from calo.dr_performance import fit_resolution_curves
from calo.losses import compute_loss
from calo.dual_readout import (
    CHANNEL_BASE,
    DualReadoutCalibration,
    DualReadoutGeometry,
    build_event_input,
    event_uid,
)
from run_dual_readout import discover_file_roles


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

    def test_event_uid_disambiguates_repeated_root_event_ids(self):
        first = event_uid("sample.root", 123, 17)
        self.assertEqual(first, event_uid("/different/path/sample.root", 123, 17))
        self.assertNotEqual(first, event_uid("sample.root", 5123, 17))
        self.assertNotEqual(first, event_uid("other.root", 123, 17))

    def test_explicit_file_roles(self):
        prefix = "Detector_N30x30x100_pi-_"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / f"{prefix}5-60GeV_Train.root"
            validation = root / f"{prefix}5-60GeV_Valid.root"
            train.touch()
            validation.touch()
            for energy in (5, 10, 15, 20, 25, 30, 40, 50):
                (root / f"{prefix}{energy}GeV.root").touch()

            train_files, validation_files, test_files = discover_file_roles(root)

        self.assertEqual(train_files, [str(train)])
        self.assertEqual(validation_files, [str(validation)])
        self.assertEqual(len(test_files), 8)
        self.assertFalse(any("Train" in path or "Valid" in path for path in test_files))

    def test_explicit_cpu_device(self):
        self.assertEqual(select_device("cpu"), torch.device("cpu"))

    def test_scaled_residual_uses_expected_energy_base(self):
        batch = {
            "s_total": torch.tensor([4.0, 9.0]),
            "dr_reco": torch.tensor([5.0, 10.0]),
        }
        self.assertTrue(torch.equal(
            residual_base_from_batch(batch, "s_only"), batch["s_total"]
        ))
        self.assertTrue(torch.equal(
            residual_base_from_batch(batch, "sc_full"), batch["dr_reco"]
        ))

    def test_scaled_residual_prediction_formula(self):
        residual = torch.tensor([1.0, -2.0])
        base = torch.tensor([4.0, 0.1])
        prediction = energy_from_scaled_residual(residual, base, eps=0.25)
        expected = torch.tensor([6.0, -0.9])
        self.assertTrue(torch.allclose(prediction, expected))

    def test_residual_model_initializes_to_no_correction(self):
        model = DualReadoutHCALNet(residual_scale=10.0)
        voxel = torch.randn(3, 2, 8, 12, 8)
        aux = torch.randn(3, 11)
        residual = model(voxel, aux)
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))

    def test_relative_mse_matches_report_loss(self):
        pred = torch.tensor([11.0, 18.0])
        target = torch.tensor([10.0, 20.0])
        loss = compute_loss(pred, target, denom_min=0.7, loss_name="relative_mse")
        expected = 0.5 * torch.mean(torch.tensor([0.1, -0.1]).pow(2))
        self.assertAlmostEqual(float(loss), float(expected), places=7)


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
