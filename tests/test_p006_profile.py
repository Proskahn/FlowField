import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'profile_p006', Path(__file__).resolve().parents[1]/'scripts/profile_p006.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_wall_time_is_not_sum_of_rank_times_or_inclusive_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case/'case.json').write_text(json.dumps(dict(cells=100, config=dict(processes=2))))
            (case/'solver').mkdir()
            (case/'solver/FlowNetwork.H').write_text('test')
            (case/'log.flowfieldP006Foam').write_text(
                'Iteration = 1\n'
                '[1] PHASE_TIMING iteration=1 rank=1 total_s=10 assembly_s=2 linear_s=8 factor_s=4 linear_iterations=5\n'
                '[0] PHASE_TIMING iteration=1 rank=0 total_s=9 assembly_s=1 linear_s=8 factor_s=5 linear_iterations=5\n'
                'ExecutionTime = 8 s  ClockTime = 12 s\nEnd\n')
            result = profile.summarize(case)
            self.assertEqual(result['phase_seconds']['total_s'], 10)
            self.assertEqual(result['phase_seconds']['factor_s'], 4)
            self.assertEqual(result['counts']['linear_iterations'], 5)
            self.assertEqual(result['outer_wall_s'], 12)
            self.assertTrue(result['completed'])
            self.assertIsNone(result['solver_converged'])

    def test_missing_rank_is_reported_instead_of_underestimating_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case/'case.json').write_text(json.dumps(dict(cells=100, config=dict(processes=2))))
            (case/'log.flowfieldP006Foam').write_text(
                '[0] PHASE_TIMING iteration=3 rank=0 total_s=2\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete timing at step 3'):
                profile.summarize(case)


if __name__ == '__main__':
    unittest.main()
