import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/package_view.py'
spec = importlib.util.spec_from_file_location('package_view', SCRIPT)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.case = self.base / 'sample'
        (self.case / 'constant/polyMesh').mkdir(parents=True)
        for name in package.MESH_FILES:
            (self.case / 'constant/polyMesh' / name).write_text('original mesh fixture\n')
        for rank in range(2):
            root = self.case / f'processor{rank}'
            (root / 'constant/polyMesh').mkdir(parents=True)
            (root / '1').mkdir()
            for name in package.MESH_FILES:
                (root / 'constant/polyMesh' / name).write_text('mesh fixture\n')
            for name in ('alpha.water', 'U', 'p', 'phi'):
                (root / '1' / name).write_text('field fixture\n')
        (self.case / 'system').mkdir()
        (self.case / 'system/decomposeParDict').write_text('numberOfSubdomains 2;\n')
        (self.case / 'log.interFoam').write_text('large log placeholder\n')

    def test_missing_mesh_rejected_before_export(self):
        (self.case / 'processor1/constant/polyMesh/points').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete case'):
            package.collect(self.case, ['1'], ['alpha.water'])

    def test_missing_last_processor_rejected(self):
        import shutil
        shutil.rmtree(self.case / 'processor1')
        with self.assertRaisesRegex(ValueError, 'numberOfSubdomains'):
            package.collect(self.case, ['1'], ['alpha.water'])

    def test_archive_selection_and_existing_output_protection(self):
        before = {f.relative_to(self.case): f.read_bytes()
                  for f in self.case.rglob('*') if f.is_file()}
        output = self.base / 'view.tar.gz'
        command = [sys.executable, str(SCRIPT), str(self.case), '--times', '1',
                   '--fields', 'alpha.water', '--output', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tarfile.open(output) as archive:
            names = set(archive.getnames())
            self.assertIn('sample-view/constant/polyMesh/points', names)
            self.assertIn('sample-view/processor1/1/alpha.water', names)
            self.assertNotIn('sample-view/processor1/1/phi', names)
            self.assertNotIn('sample-view/log.interFoam', names)
        archive_bytes = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), archive_bytes)
        after = {f.relative_to(self.case): f.read_bytes()
                 for f in self.case.rglob('*') if f.is_file()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
