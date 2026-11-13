"""Real IOH library tests with a cheap stand-in objective, explicitly not CFD."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from flowfield.benchmarking.ioh import analyzer_logger, make_ioh_problem
from flowfield.problem import AbstractFlowFieldProblem


class CountingObjective(AbstractFlowFieldProblem):
    def __init__(self, name="AdapterUnitTest", dimension=2, lower=None, upper=None):
        super().__init__(name, dimension, lower or [0.] * dimension, upper or [1.] * dimension)
        self.calls = []

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        x = self.decode(x)
        self.calls.append(list(x))
        return {"objective": float(sum(x)), "objectives": {"sum": float(sum(x))},
                "constraints": {}, "metadata": {}, "status": "ok"}


@unittest.skipUnless(importlib.util.find_spec("ioh"), "optional ioh dependency is not installed")
class IOHTests(unittest.TestCase):
    def test_real_native_logger_tracks_every_evaluation_and_reset(self):
        evaluator = CountingObjective()
        problem = make_ioh_problem(evaluator)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "ioh"
            with analyzer_logger(problem, output, algorithm_name="adapter-unit-test"):
                self.assertAlmostEqual(problem([0.2, 0.3]), 0.5)
                self.assertAlmostEqual(problem([0.8, 0.8]), 1.6)
                self.assertEqual(problem.state.evaluations, 2)
                problem.reset()
                self.assertAlmostEqual(problem([0, 0]), 0)
                self.assertEqual(problem.state.evaluations, 1)
            self.assertEqual(len(evaluator.calls), 3)
            metadata = list(output.glob("IOHprofiler*.json"))
            self.assertEqual(len(metadata), 1)
            info = json.loads(metadata[0].read_text())
            self.assertEqual(info["function_name"], evaluator.name)
            self.assertFalse(info["maximization"])
            runs = info["scenarios"][0]["runs"]
            self.assertEqual([run["evals"] for run in runs], [2, 1])
            data = list(output.rglob("*.dat"))
            self.assertTrue(data)
            rows = [line for line in data[0].read_text().splitlines() if line and line[0].isdigit()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(data[0].read_text().splitlines()[0].split(), ["evaluations", "raw_y", "x0", "x1"])
            with self.assertRaises(FileExistsError):
                with analyzer_logger(problem, output):
                    pass

    def test_generic_dimensions_and_heterogeneous_bounds(self):
        evaluator = CountingObjective("MixedBounds", 3, [-2., 4., 100.], [0., 6., 120.])
        problem = make_ioh_problem(evaluator)
        self.assertEqual(problem.meta_data.n_variables, 3)
        self.assertEqual(list(problem.bounds.lb), evaluator.lower_bounds)
        self.assertEqual(list(problem.bounds.ub), evaluator.upper_bounds)
        self.assertEqual(problem([-1., 5., 110.]), 114.)
        with self.assertRaises(ValueError):
            problem([-1., 0., 110.])
        self.assertEqual(len(evaluator.calls), 1)


if __name__ == "__main__":
    unittest.main()
