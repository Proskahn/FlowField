import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from flowfield.case import generate_case
from flowfield.evaluation import Evaluator, run_case
from flowfield.metrics import extract_metrics, read_surface_history
from flowfield.model import ChannelConfig, decode_design, reference_pressure_drop
from flowfield.runtime import Runtime, SimulationError


class ModelTests(unittest.TestCase):
    def test_square_reference_and_scaling(self):
        c = ChannelConfig()
        # Known square-duct resistance coefficient: dp = 28.45415 mu L Q / a^4.
        coefficient = reference_pressure_drop(c) * c.width_m**4 / (
            c.density_kg_m3 * c.kinematic_viscosity_m2_s * c.length_m * c.flow_rate_m3_s)
        self.assertAlmostEqual(coefficient, 28.45415, places=4)
        self.assertAlmostEqual(reference_pressure_drop(replace(c, flow_rate_m3_s=2e-8)),
                               2 * reference_pressure_drop(c))
        self.assertAlmostEqual(reference_pressure_drop(replace(c, width_m=0.002, height_m=0.002)),
                               reference_pressure_drop(c) / 16)

    def test_rectangular_reference_is_symmetric(self):
        self.assertAlmostEqual(reference_pressure_drop(ChannelConfig(width_m=0.0005, height_m=0.002)),
                               reference_pressure_drop(ChannelConfig(width_m=0.002, height_m=0.0005)))

    def test_decode_bounds(self):
        self.assertEqual(decode_design([0, 1]).width_m, 0.0005)
        self.assertEqual(decode_design([0, 1]).height_m, 0.002)
        self.assertAlmostEqual(decode_design([1/3, 1/3]).width_m, 0.001)
        for x in ([], [0], [0, 0, 0], [-0.1, 0], [0, 1.1], [math.nan, 0], [math.inf, 0], [True, 0]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                decode_design(x)

    def test_invalid_config(self):
        for kw in ({"width_m": 0}, {"height_m": math.nan}, {"cells": (1, 8, 8)},
                   {"cells": (10.5, 8, 8)}, {"cells": (2000, 2000, 2000)},
                   {"max_iterations": 0}, {"max_iterations": 1.5},
                   {"residual_tolerance": 0}, {"density_kg_m3": -1},
                   {"flow_rate_m3_s": 1}, {"length_m": 0.001}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                ChannelConfig(**kw)


class CaseTests(unittest.TestCase):
    def test_generation_is_complete_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            case = generate_case(Path(temp) / "case with spaces", ChannelConfig(width_m=0.002))
            self.assertIn("(50 8 8)", (case / "system/blockMeshDict").read_text())
            self.assertIn("(0.005 0 0)", (case / "0/U").read_text())
            manifest = json.loads((case / "case.json").read_text())
            self.assertEqual(len(manifest["input_sha256"]), 8)
            with self.assertRaises(FileExistsError):
                generate_case(case, ChannelConfig())
            self.assertEqual(manifest, json.loads((case / "case.json").read_text()))

    def test_edited_input_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as temp:
            case = generate_case(Path(temp) / "case", ChannelConfig())
            (case / "0/U").write_text("modified")
            with self.assertRaisesRegex(ValueError, "input changed"):
                run_case(case)

    def test_failure_record_and_no_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            case = generate_case(Path(temp) / "case", ChannelConfig())
            with patch.object(Runtime, "resolve", side_effect=SimulationError("unavailable")):
                with self.assertRaises(SimulationError):
                    run_case(case)
            self.assertEqual(json.loads((case / "result.json").read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                run_case(case)

    def test_failure_penalty_is_explicit_and_bounds_still_raise(self):
        with patch("flowfield.evaluation.evaluate", side_effect=SimulationError("failed")):
            with self.assertRaises(SimulationError):
                Evaluator()([0, 0])
            evaluator = Evaluator(failure_penalty=1e12)
            self.assertEqual(evaluator([0, 0]), 1e12)
            self.assertEqual(evaluator.last_result["status"], "penalized")
            with self.assertRaises(ValueError):
                evaluator([2, 0])


def fixture(case, config):
    """Synthetic tables for parser tests only, never used for CFD smoke results."""
    (case / "log.checkMesh").write_text("Mesh OK.\n")
    (case / "log.simpleFoam").write_text("Time = 10\nSIMPLE solution converged in 10 iterations\nEnd\n")
    dp = reference_pressure_drop(config) / config.density_kg_m3
    for name, column, value in (("p_inlet", "areaAverage(p)", dp),
        ("p_outlet", "areaAverage(p)", 0), ("q_inlet", "sum(phi)", -config.flow_rate_m3_s),
        ("q_outlet", "sum(phi)", config.flow_rate_m3_s)):
        folder = case / "postProcessing" / name / "0"
        folder.mkdir(parents=True)
        (folder / "surfaceFieldValue.dat").write_text(f"# Area : 1e-6\n# Time {column}\n10 {value:.16e}\n")


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        self.config = ChannelConfig()
        fixture(self.case, self.config)

    def test_density_conversion_conservation_and_reference(self):
        metrics = extract_metrics(self.case, self.config)
        self.assertAlmostEqual(metrics["pressure_drop_pa"], reference_pressure_drop(self.config))
        self.assertEqual(metrics["mass_balance_relative_error"], 0)
        self.assertTrue(metrics["reference_check_passed"])

    def test_unconverged_is_rejected_even_with_output(self):
        (self.case / "log.simpleFoam").write_text("Time = 3000\nEnd\n")
        with self.assertRaisesRegex(SimulationError, "residualControl"):
            extract_metrics(self.case, self.config)

    def test_mesh_failure(self):
        (self.case / "log.checkMesh").write_text("Failed 1 mesh checks.\n")
        with self.assertRaisesRegex(SimulationError, "valid mesh"):
            extract_metrics(self.case, self.config)

    def test_flux_sign_and_magnitude_failures(self):
        path = self.case / "postProcessing/q_outlet/0/surfaceFieldValue.dat"
        for flow in (-1e-8, 1.1e-8, 0):
            path.write_text(f"# Time sum(phi)\n10 {flow}\n")
            with self.subTest(flow=flow), self.assertRaises(SimulationError):
                extract_metrics(self.case, self.config)

    def test_stale_samples_and_restarts_are_rejected(self):
        path = self.case / "postProcessing/p_outlet/0/surfaceFieldValue.dat"
        path.write_text("# Time areaAverage(p)\n9 0\n")
        with self.assertRaisesRegex(SimulationError, "misaligned"):
            extract_metrics(self.case, self.config)
        new = self.case / "postProcessing/p_outlet/9"
        new.mkdir()
        (new / "surfaceFieldValue.dat").write_text("# Time areaAverage(p)\n10 0\n")
        with self.assertRaisesRegex(SimulationError, "one fresh-run"):
            extract_metrics(self.case, self.config)

    def test_parser_rejects_malformed_tables(self):
        path = self.case / "bad.dat"
        for body in ("", "# Time sum(p)\n10 1\n", "# Time areaAverage(p)\n10 nan\n",
                     "10 1\n", "# Time areaAverage(p)\n10 1\n9 1\n",
                     "# Time areaAverage(p)\n10 (1 2 3)\n"):
            path.write_text(body)
            with self.subTest(body=body), self.assertRaises(SimulationError):
                read_surface_history(path, "areaAverage(p)")


class RuntimeTests(unittest.TestCase):
    def test_native_version_guard(self):
        with patch.dict(os.environ, {"WM_PROJECT_VERSION": "v2412"}):
            with self.assertRaises(SimulationError):
                Runtime("native").resolve()

    def test_timeout_cleans_up_only_own_container(self):
        with tempfile.TemporaryDirectory() as temp:
            def fake_run(argv, **kwargs):
                if argv[:2] == ["docker", "run"]:
                    raise subprocess.TimeoutExpired(argv, 1)
                return subprocess.CompletedProcess(argv, 0)
            with patch("flowfield.runtime.subprocess.run", side_effect=fake_run) as run:
                with self.assertRaises(SimulationError):
                    Runtime("docker", timeout_s=1).run(temp, "simpleFoam", info={"backend": "docker", "image_id": "sha256:test"})
            launch = run.call_args_list[0].args[0]
            cleanup = run.call_args_list[1].args[0]
            self.assertEqual(cleanup, ["docker", "rm", "-f", launch[launch.index("--name") + 1]])
            self.assertIn("--pull", launch)


if __name__ == "__main__":
    unittest.main()
