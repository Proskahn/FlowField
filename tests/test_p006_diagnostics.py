import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from flowfield import CaseEvaluator
from flowfield.problems.level1_small_continuous.p006 import ManifoldConfig,P006Problem
from flowfield.problems.level1_small_continuous.p006.runtime import _show_progress
from flowfield.problems.level1_small_continuous.p006.warm_start import FIELDS

spec=importlib.util.spec_from_file_location('diagnose_p006',Path(__file__).resolve().parents[1]/'scripts/diagnose_p006.py')
diagnostics=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticTests(unittest.TestCase):
    def test_prepare_retains_completed_coupling_and_only_changes_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=P006Problem(config=ManifoldConfig(mesh='screening',field_tolerance=1e-4))
            source=p.write_case(p.initial_design(),root/'source')
            manifest=json.loads((source/'case.json').read_text());saved=source/'10';saved.mkdir()
            for name in FIELDS:shutil.copy2(source/'0'/name,saved/name)
            post=source/'postProcessing/p006';post.mkdir(parents=True)
            (post/'status.json').write_text(json.dumps(dict(converged=False,iteration=10,
                cells=manifest['cells'],processes=2,momentum_fraction=1.,thermal_active=True)))
            (post/'history.csv').write_text('iteration,momentum_fraction,thermal_active,applied_voltage_V\n10,1,1,2.2\n')
            (source/'log.flowfieldP006Foam').write_text('Iteration = 10\nEnd\n')
            (source/'result.json').write_text(json.dumps(dict(status='failed',input_sha256=manifest['input_sha256'])))
            original={str(f.relative_to(source)):f.read_bytes() for f in source.rglob('*') if f.is_file()}
            result=diagnostics.prepare(source,root/'diagnostic')
            case=Path(result['case_directory']);new=json.loads((case/'case.json').read_text())
            self.assertEqual(new['config'],dict(manifest['config'],iterations=200))
            self.assertEqual(new['design_geometry'],manifest['design_geometry'])
            self.assertFalse(new['initial_state']['qualification_inherited'])
            self.assertTrue(new['initial_state']['coupling']['thermal_active'])
            self.assertFalse((case/'result.json').exists())
            CaseEvaluator(p.case_definition)._load_case(case)
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                diagnostics.main(['prepare',str(source),str(root/'serial'),'--iterations','200','--processes','1'])
            self.assertEqual(json.loads(output.getvalue())['processes'],1)
            serial=json.loads((root/'serial/case.json').read_text())
            self.assertEqual(serial['config'],dict(manifest['config'],iterations=200,processes=1))
            self.assertEqual(serial['initial_state']['source_field_sha256'],new['initial_state']['source_field_sha256'])
            self.assertEqual(serial['design_geometry'],new['design_geometry'])
            self.assertTrue(serial['initial_state']['coupling']['thermal_active'])
            self.assertFalse(serial['initial_state']['qualification_inherited'])
            CaseEvaluator(p.case_definition)._load_case(root/'serial')
            self.assertEqual(original,{str(f.relative_to(source)):f.read_bytes() for f in source.rglob('*') if f.is_file()})
            with self.assertRaisesRegex(ValueError,'outside the source'):
                diagnostics.prepare(source,source/'nested')
            with self.assertRaises(FileExistsError):diagnostics.prepare(source,case)
            (source/'result.json').unlink()
            with self.assertRaisesRegex(ValueError,'finished run'):
                diagnostics.prepare(source,root/'incomplete')
            self.assertFalse((root/'incomplete').exists())

    def history_case(self,root):
        config=ManifoldConfig(field_tolerance=1e-4)
        (root/'case.json').write_text(json.dumps(dict(problem='P006',config=config.to_dict(),config_sha256=config.fingerprint)))
        post=root/'postProcessing/p006';post.mkdir(parents=True)
        rows=[]
        for i in range(1,9):
            row=dict(iteration=i,momentum_fraction=1.,thermal_active=int(i>=5),applied_voltage_V=2.2,
                charge_residual=1e-9,energy_balance=1e-6,liquid_balance=1e-5,gas_balance=1e-5,
                local_liquid_balance=2e-4,local_gas_balance=1e-4,liquid_momentum_residual=1e-8,
                gas_momentum_residual=1e-7,bounded_cells=0)
            row.update({f'field_change_{name}':1e-6 for name in diagnostics.FIELDS})
            row['field_change_Ug']=4e-4 if i<=6 else 5e-5
            row['field_change_sg']=1e-2 if i<5 else 1e-6
            row['field_change']=max(row[f'field_change_{name}'] for name in diagnostics.FIELDS)
            rows.append(row)
        self.write_history(post/'history.csv',rows)
        return rows,post/'history.csv'

    @staticmethod
    def write_history(path,rows):
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def test_report_separates_stages_and_counts_persistent_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.history_case(root)
            (root/'result.json').write_text('{"status":"failed"}')
            report=diagnostics.summarize(root)
            self.assertEqual(report['window']['first'],5)
            self.assertEqual(report['window']['samples'],4)
            self.assertEqual(report['components']['Ug']['dominant_samples'],4)
            self.assertEqual(report['components']['Ug']['samples_at_or_above_limit'],2)
            self.assertEqual(report['components']['sg']['samples_at_or_above_limit'],0)
            self.assertEqual(report['field_change']['longest_streak_below_limit'],2)
            self.assertEqual(report['execution_status'],'failed')
            self.assertIn('not a convergence certificate',report['assessment'])
            self.assertEqual(diagnostics.summarize(root,last=2)['window']['first'],7)

    def test_report_rejects_inconsistent_nonfinite_or_missing_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,path=self.history_case(root)
            rows[-1]['field_change']=1e-10;self.write_history(path,rows)
            with self.assertRaisesRegex(ValueError,'disagree'):diagnostics.summarize(root)
            rows[-1]['field_change']=float('nan');self.write_history(path,rows)
            with self.assertRaisesRegex(ValueError,'nonfinite'):diagnostics.summarize(root)
            path.write_text('iteration,field_change\n1,0.1\n')
            with self.assertRaisesRegex(ValueError,'updated solver'):diagnostics.summarize(root)
            with self.assertRaisesRegex(ValueError,'positive'):diagnostics.summarize(root,last=0)

    def test_acceleration_uses_outer_steps_and_the_same_final_stage_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.history_case(root)
            (root/'log.flowfieldP006Foam').write_text(
                'Iteration = 1\nAnderson block residual 100: norm=1e3; history=0; step=0.01\n'
                'Iteration = 5\nAnderson block residual 200: norm=0.01; history=40; step=1\n'
                'Iteration = 6\nAnderson block residual 201: norm=0.02; history=0; step=0.2\n'
                'Iteration = 7\nAnderson block residual 202: norm=0.015; history=1; step=1\n'
                'Iteration = 8\nAnderson block residual 203: norm=0.01; history=2; step=0.5\n')
            report=diagnostics.summarize(root);a=report['acceleration']
            self.assertEqual(report['processes'],2)
            self.assertEqual(a['samples'],4)
            self.assertEqual((a['first_norm'],a['last_norm']),(.01,.01))
            self.assertEqual(a['minimum_step'],.2)
            self.assertEqual(a['limited_steps'],2)
            self.assertEqual(a['empty_history_samples'],1)
            self.assertEqual(a['maximum_history'],40)
            self.assertEqual(diagnostics.summarize(root,last=2)['acceleration']['samples'],2)
            (root/'log.flowfieldP006Foam').write_text(
                'Iteration = 8\nAnderson block residual 1: norm=nan; history=1; step=1\n')
            with self.assertRaisesRegex(ValueError,'Invalid Anderson'):diagnostics.summarize(root)

    def test_live_partial_csv_row_is_not_treated_as_a_complete_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,path=self.history_case(root)
            with path.open('a') as stream:stream.write('9,1,')
            self.assertEqual(diagnostics.summarize(root)['window']['last'],8)
            (root/'result.json').write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError,'CSV columns'):diagnostics.summarize(root)

    def test_console_shows_components_at_the_same_steps_as_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'log.flowfieldP006Foam'
            log.write_text(''.join(f'Iteration = {i}\nP006: change={i}\nP006 changes: Ug={i}; dominant=Ug\n'
                                   for i in (1,2,10)))
            stop=threading.Event();stop.set();output=io.StringIO()
            with contextlib.redirect_stdout(output):_show_progress(log,stop)
            text=output.getvalue()
            self.assertIn('P006 changes: Ug=1;',text)
            self.assertIn('P006 changes: Ug=10;',text)
            self.assertNotIn('P006 changes: Ug=2;',text)


if __name__=='__main__':unittest.main()
