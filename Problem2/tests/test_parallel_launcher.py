"""Prevent restarting a decomposed case with the wrong process count or mode."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RestartGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / 'Problem2'
        (self.project / 'scripts').mkdir(parents=True)
        self.case = self.project / 'cases' / 'example'
        self.case.mkdir(parents=True)
        self.launcher = self.project / 'scripts' / 'run_case.sh'
        shutil.copy2(ROOT / 'scripts/run_case.sh', self.launcher)
        self.sentinel = self.case / 'keep-result'
        self.sentinel.write_text('previous result\n')

    def reject(self, arguments, message):
        before = sorted(str(p.relative_to(self.case)) for p in self.case.rglob('*'))
        result = subprocess.run(['bash', str(self.launcher), str(self.case), *arguments],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(self.sentinel.read_text(), 'previous result\n')
        self.assertEqual(before, sorted(str(p.relative_to(self.case)) for p in self.case.rglob('*')))

    def decompose(self):
        for i in range(2):
            (self.case / f'processor{i}').mkdir()

    def test_wrong_resume_count_preserves_results(self):
        self.decompose()
        self.reject(['resume', '16'], 'requires 2 processes')

    def test_second_decomposition_preserves_results(self):
        self.decompose()
        self.reject(['parallel', '2'], 'Existing decomposition')

    def test_serial_on_parallel_case_preserves_results(self):
        self.decompose()
        self.reject(['run'], 'Parallel case')

    def test_resume_cannot_silently_run_serial(self):
        self.reject(['resume', '16'], 'No processor directories')

    def test_missing_partition_is_rejected(self):
        (self.case / 'processor0').mkdir()
        (self.case / 'processor2').mkdir()
        self.reject(['resume'], 'Incomplete processor directory sequence')

    def test_invalid_process_count(self):
        for count in ('0', '-2', '1.5', 'sixteen'):
            with self.subTest(count=count):
                self.reject(['parallel', count], 'positive integer')

    def test_unknown_action_is_rejected(self):
        self.reject(['parallem', '16'], 'Unknown action')

    def checkpoint(self, rank, iteration):
        directory = self.case / f'processor{rank}' / str(iteration)
        directory.mkdir(parents=True)
        for field in ('U', 'p', 'T', 'Cgas', 'phiS', 'phiE', 'material'):
            (directory / field).write_text('saved field\n')
        return directory

    def test_different_rank_iterations_are_rejected(self):
        self.checkpoint(0, 100)
        self.checkpoint(1, 200)
        self.reject(['resume', '2'], 'Processor checkpoints disagree')

    def test_partial_checkpoint_is_rejected(self):
        self.checkpoint(0, 100)
        (self.checkpoint(1, 100) / 'T').unlink()
        self.reject(['resume', '2'], 'Incomplete checkpoint')

    def test_generation_preserves_unreconstructed_parallel_results(self):
        # The only result can live under processor0 after an interrupted run.
        case = self.project / 'generated' / 'singleChannel'
        (case / 'processor0' / '100').mkdir(parents=True)
        saved = case / 'processor0' / '100' / 'T'
        saved.write_text('saved field\n')
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_cases.py'),
                                 '--geometry', 'single', '--mesh', 'coarse', '--output', str(case.parent)],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('contains results', result.stderr)
        self.assertEqual(saved.read_text(), 'saved field\n')
        self.assertFalse((case / '0').exists())


if __name__ == '__main__':
    unittest.main()
