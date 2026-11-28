"""Compile and exercise the actual charge Newton helper, without OpenFOAM."""
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class P011NewtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('C++ compiler unavailable')
        try:
            eigen = Path(distribution('cmeel-eigen').locate_file('cmeel.prefix/include/eigen3'))
        except PackageNotFoundError as exc:
            raise unittest.SkipTest('Optional cmeel-eigen headers unavailable') from exc
        if not (eigen/'Eigen/Sparse').is_file():
            raise unittest.SkipTest('Optional Eigen headers incomplete')
        root = Path(__file__).resolve().parents[1]
        cls.temporary = tempfile.TemporaryDirectory(prefix='flowfield-p011-newton-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.executable = Path(cls.temporary.name)/'p011-charge-newton'
        source = root/'tests/openfoam/p011_charge_newton.cpp'
        solver = root/'src/flowfield/problems/level1_small_continuous/p011/solver'
        built = subprocess.run([compiler, '-std=c++14', '-O2', '-Wall', '-Wextra',
                                '-I', str(eigen), '-I', str(solver), str(source),
                                '-o', str(cls.executable)],
                               capture_output=True, text=True, timeout=120)
        if built.returncode:
            raise AssertionError('Charge Newton regression did not compile:\n'+built.stdout+built.stderr)

    def scenario(self, name):
        result = subprocess.run([str(self.executable), name], capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn(name+': passed', result.stdout)

    def test_roundoff_plateau_retains_an_already_qualified_iterate(self):
        self.scenario('cancellation')

    def test_unqualified_plateau_and_exact_acceptance_boundary_are_rejected(self):
        for name in ('unacceptable', 'boundary'):
            with self.subTest(scenario=name):
                self.scenario(name)

    def test_well_conditioned_nonlinear_equations_reach_the_preferred_accuracy(self):
        self.scenario('nonlinear')

    def test_nonfinite_iterates_residuals_corrections_and_trials_are_rejected(self):
        for name in ('iterate', 'residual', 'correction', 'trial'):
            with self.subTest(scenario=name):
                self.scenario(name)

    def test_invalid_current_normalization_is_rejected(self):
        self.scenario('scale')

    def test_butler_volmer_preserves_small_overpotential_currents_and_derivative(self):
        self.scenario('bv')


if __name__ == '__main__':
    unittest.main()
