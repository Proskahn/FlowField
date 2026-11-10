import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowfield import Evaluator, FlowFieldProblem, P000Problem, get_problem, list_problems, register_problem
from flowfield.experiments import random_search
from flowfield.runtime import SimulationError


class ToyProblem(FlowFieldProblem):
    """Cheap independent test problem; not a CFD benchmark or registered problem."""
    def __init__(self, name="FrameworkToy"):
        super().__init__(name, 3, [-2, 3, 10], [-1, 5, 20])
        self.calls = 0

    def decode(self, x):
        return dict(zip(("a", "b", "c"), self.validate_design(x)))

    def evaluate_raw(self, x):
        geometry = self.decode(x)
        self.calls += 1
        raw = sum(geometry.values())
        constraint = geometry["b"] - 4
        return {"objective": raw + 100 * max(0, constraint), "objectives": {"sum": raw},
                "constraints": {"b_limit": constraint}, "metadata": {"geometry": geometry},
                "status": "ok"}


class FrameworkTests(unittest.TestCase):
    def test_contract_is_abstract_and_scalar_call_preserves_raw_result(self):
        with self.assertRaises(TypeError):
            FlowFieldProblem("unfinished", 1, [0], [1])
        problem = ToyProblem()
        self.assertEqual(problem([-1, 5, 10]), 114.)
        self.assertEqual(problem.calls, 1)
        self.assertEqual(problem.last_result["objectives"], {"sum": 14.})
        self.assertEqual(problem.last_result["constraints"], {"b_limit": 1.})

    def test_invalid_design_does_not_evaluate_and_clears_last_result(self):
        problem = ToyProblem()
        problem([-1, 3, 10])
        for x in ([], [-1, 3], [-1, 3, 10, 1], [-3, 3, 10], [-1, math.inf, 10],
                  [-1, math.nan, 10], [-1, "3", 10], [-1, True, 10]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                problem(x)
            self.assertIsNone(problem.last_result)
        self.assertEqual(problem.calls, 1)

    def test_invalid_bounds_and_nonfinite_objective(self):
        for bounds in (([], [1]), ([0], [0]), ([math.nan], [1]), ([0], [math.inf])):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                FlowFieldProblem.__init__(ToyProblem(), "bad", 1, *bounds)
        problem = ToyProblem()
        with patch.object(problem, "evaluate_raw", return_value={"objective": math.nan}):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                problem([-1, 3, 10])

    def test_registry_and_legacy_api(self):
        self.assertIn("P000", list_problems())
        self.assertIsInstance(get_problem("P000"), P000Problem)
        self.assertIsInstance(Evaluator(), FlowFieldProblem)
        self.assertIs(Evaluator, P000Problem)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            get_problem("not-a-problem")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_problem("P000", ToyProblem)
        with patch.dict("flowfield.registry._FACTORIES", clear=False):
            register_problem("test-toy", ToyProblem)
            self.assertIsInstance(get_problem("test-toy"), ToyProblem)
            register_problem("bad-factory", lambda: object())
            with self.assertRaises(TypeError):
                get_problem("bad-factory")

    def test_source_root_import_and_cli(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, "-c", "from flowfield import FlowFieldProblem,get_problem; "
                   "assert isinstance(get_problem(),FlowFieldProblem); "
                   "from flowfield.base import FlowFieldProblem as B; assert B is FlowFieldProblem"]
        subprocess.run(command, cwd=root, check=True, capture_output=True)
        result = subprocess.run([sys.executable, "-m", "flowfield", "list"], cwd=root,
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["problems"], ["P000", "P001", "P002", "P003", "P004", "P004-hydraulic"])


class P000InterfaceTests(unittest.TestCase):
    def test_decode_is_pure_and_uses_physical_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "uncreated"
            problem = P000Problem(output)
            self.assertEqual(problem.name, "P000")
            self.assertEqual(problem.dimension, 2)
            self.assertEqual(problem.lower_bounds, [0., 0.])
            self.assertEqual(problem.upper_bounds, [1., 1.])
            self.assertEqual(problem.decode([0, 1]), {"width_m": .0005, "height_m": .002, "length_m": .05})
            self.assertFalse(output.exists())

    def test_raw_contract_is_persisted_and_scalar_calls_backend_once(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "case"
            case.mkdir()
            cfd = {"status": "ok", "objective": 14.0, "case_directory": str(case),
                   "metrics": {"pressure_drop_pa": 14.0}, "runtime": {"backend": "test"}}
            problem = P000Problem(temp)
            with patch("flowfield.evaluation.evaluate", return_value=cfd) as backend:
                self.assertEqual(problem([1/3, 1/3]), 14.0)
            backend.assert_called_once()
            raw = problem.last_result
            self.assertEqual(raw["objectives"], {"pressure_drop_pa": 14.0})
            self.assertEqual(raw["constraints"], {})
            self.assertTrue(raw["metadata"]["valid_cfd"])
            self.assertAlmostEqual(raw["metadata"]["geometry"]["width_m"], .001)
            self.assertEqual(json.loads((case / "evaluation.json").read_text()), json.loads(json.dumps(raw)))

    def test_direct_raw_evaluation_and_penalty_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            problem = P000Problem(temp, failure_penalty=1e12)
            error = SimulationError("solver failed", case_directory=temp)
            with patch("flowfield.evaluation.evaluate", side_effect=error):
                raw = problem.evaluate_raw([.5, .5])
            self.assertEqual(raw["objective"], 1e12)
            self.assertEqual(raw["status"], "penalized")
            self.assertEqual(raw["objectives"], {})
            self.assertEqual(raw["constraints"], {})
            self.assertFalse(raw["metadata"]["valid_cfd"])
            self.assertEqual(raw["metadata"]["case_directory"], temp)
            self.assertTrue((Path(temp) / "evaluation.json").is_file())


class ExperimentTests(unittest.TestCase):
    def test_seeded_run_uses_declared_bounds_preserves_raw_and_never_double_evaluates(self):
        with tempfile.TemporaryDirectory() as temp:
            sequences = []
            for index in range(2):
                problem = ToyProblem()
                output = Path(temp) / str(index)
                summary = random_search(problem, budget=4, seed=42, output=output, with_ioh=False)
                records = [json.loads(line) for line in (output / "evaluations.jsonl").read_text().splitlines()]
                self.assertEqual(problem.calls, 4)
                self.assertEqual(summary["evaluations"], 4)
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["best_objective"], min(r["objective"] for r in records))
                for record in records:
                    self.assertEqual(record["x"], problem.validate_design(record["x"]))
                    self.assertIn("constraints", record["result"])
                sequences.append([r["x"] for r in records])
            self.assertEqual(*sequences)
            with self.assertRaises(FileExistsError):
                random_search(ToyProblem(), output=output, with_ioh=False)

    def test_failure_writes_partial_summary_and_retains_completed_evaluations(self):
        with tempfile.TemporaryDirectory() as temp:
            problem = ToyProblem()
            raw = problem.evaluate_raw([-1, 3, 10])
            output = Path(temp) / "failed"
            with patch.object(problem, "evaluate_raw", side_effect=[raw, SimulationError("solver failed")]):
                with self.assertRaises(SimulationError):
                    random_search(problem, budget=3, output=output, with_ioh=False)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["evaluations"], 1)
            self.assertIn("solver failed", summary["error"])
            self.assertEqual(len((output / "evaluations.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
