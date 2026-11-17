"""Regression checks for false passes, read-only inspection and suite execution."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.registry import create_problem, list_problems
from flowfield.runtime import Runtime
from flowfield.stability.audit import inspect_case, inspect_paths
from flowfield.stability.cli import main
from flowfield.stability.compare import compare
from flowfield.stability.diagnostics import history, scan_log, trends
from flowfield.stability.fields import final_fields
from flowfield.stability.profiles import limits, stricter
from flowfield.stability.report import overall, write_reports
from flowfield.stability.runner import make_plan, run_suite


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def log(self, text):
        p = self.root/'log'; p.write_text(text)
        return scan_log(p)

    def test_fpe_trapping_banner_is_not_a_crash(self):
        d = self.log('trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\nTime = 1\nEnd\n')
        self.assertFalse(d['fatal']); self.assertTrue(d['ended'])
        self.assertTrue(self.log('Floating point exception (core dumped)\n')['fatal'])
        self.assertTrue(self.log('--> FOAM FATAL ERROR:\n')['fatal'])

    def test_bad_residuals_time_and_courant_cannot_hide_behind_end(self):
        for text in ('Solving for Ux, Initial residual = nan, Final residual = 0, No Iterations 1',
                     'Courant Number mean: 0 max: inf', 'deltaT = 0', 'Time = nan',
                     'Time = 2\nTime = 1'):
            with self.subTest(text=text):
                self.assertTrue(self.log(text+'\nEnd\n')['errors'])
        self.assertFalse(self.log('Time = 1\nEnd\nTime = 2\n')['ended'])

    def test_log_numeric_diagnostics(self):
        d = self.log('Time = 1e-5\ndeltaT = 1e-6\nCourant Number mean: 0.1 max: 0.3\n'
            'Interface Courant Number mean: 0.01 max: 0.2\n'
            'DILUPBiCGStab: Solving for Ux, Initial residual = 0.5, Final residual = 1e-8, No Iterations 2\nEnd\n')
        self.assertEqual(d['last_step'],1e-5)
        self.assertEqual(d['residuals']['Ux']['last_final'],1e-8)
        self.assertEqual(d['courant']['courant number']['maximum'],.3)

    def test_partial_live_csv_is_distinct_from_corrupt_completed_history(self):
        p=self.root/'history.csv';p.write_text('iteration,error\n1,0.1\n2,')
        d=history(p,unfinished=True)
        self.assertEqual(d['samples'],1);self.assertFalse(d['errors']);self.assertTrue(d['partial_last_row_ignored'])
        self.assertTrue(history(p)['errors'])
        for text in ('iteration,error\n1,0.1\n2,nan\n', 'iteration,iteration\n1,1\n',
                     'iteration,error\n1,0\n3,0\n', 'time_s,value\n.1,0\n.1,0\n'):
            p.write_text(text);self.assertTrue(history(p,unfinished=True)['errors'])

    def test_history_memory_is_bounded_but_extrema_cover_all_samples(self):
        p=self.root/'history.csv';p.write_text('time_s,value\n'+''.join(f'{i},{1000 if i==0 else i}\n' for i in range(300)))
        d=history(p);self.assertEqual(len(d['recent']),250)
        self.assertEqual(d['ranges']['value']['max'],1000)

    def test_trends_do_not_cross_p006_continuation_stages(self):
        rows=[dict(field_change=1,momentum_fraction=0,thermal_active=0,energy_balance=1) for _ in range(35)]
        rows += [dict(field_change=2,momentum_fraction=.01,thermal_active=0,energy_balance=1) for _ in range(5)]
        self.assertEqual(trends(rows,{'field_change':1e-5}),{})
        rows=[dict(field_change=2,momentum_fraction=.01,thermal_active=0,energy_balance=1) for _ in range(40)]
        d=trends(rows,{'field_change':1e-5,'energy_balance':1e-3})
        self.assertEqual(d['field_change']['indicator'],'stalled');self.assertNotIn('energy_balance',d)

    def field_case(self, name='P000'):
        d=self.root/'fields';(d/'1').mkdir(parents=True)
        def field(n, cls, value):
            (d/'1'/n).write_text(f'FoamFile {{ format ascii; class {cls}; }}\ninternalField uniform {value};\nboundaryField {{}}\n')
        field('p','volScalarField','0');field('U','volVectorField','(1 0 0)')
        return d, create_problem(name).config

    def test_field_finiteness_bounds_counts_and_missing_outputs(self):
        d,c=self.field_case()
        self.assertEqual(final_fields(d,'P000',c,1,2)['status'],'passed')
        p=d/'1'/'p';p.write_text('FoamFile { format ascii; class volScalarField; }\ninternalField nonuniform List<scalar> 2 (0);')
        self.assertEqual(final_fields(d,'P000',c,1,2)['status'],'failed')
        p.write_text('FoamFile { format ascii; class volScalarField; }\ninternalField uniform 0; boundaryField { outlet { value uniform nan; }}')
        self.assertEqual(final_fields(d,'P000',c,1,2)['status'],'failed')
        p.write_text('FoamFile { format ascii; class volScalarField; }\ninternalField uniform 0; boundaryField { outlet { value uniform 1e999; }}')
        self.assertEqual(final_fields(d,'P000',c,1,2)['status'],'failed')
        p.unlink();self.assertEqual(final_fields(d,'P000',c,1,2)['status'],'failed')

    def test_unreadable_or_oversized_fields_are_not_a_pass(self):
        d,c=self.field_case()
        p=d/'1'/'p';p.write_bytes(b'FoamFile { format binary; class volScalarField; }\n'+b'\x00'*40)
        self.assertEqual(final_fields(d,'P000',c,1,2)['status'],'inconclusive')
        self.assertEqual(final_fields(d,'P000',c,1,2,20)['status'],'inconclusive')
        self.assertEqual(final_fields(d,'P000',c,2,2)['status'],'failed')

    def test_alpha_internal_bounds(self):
        d,c=self.field_case('P005')
        (d/'1'/'alpha.water').write_text('FoamFile { format ascii; class volScalarField; }\ninternalField uniform 1.1;')
        a=final_fields(d,'P005',c,1,2)
        self.assertEqual(a['status'],'failed');self.assertTrue(any('Outside physical bounds' in e for e in a['errors']))

    def p006_live(self, iterations=3000):
        p=create_problem('P006',config={'iterations':iterations});case=p.write_case(p.initial_design(),self.root/'p006')
        (case/'log.checkMesh').write_text('cells: 13104\nMesh OK.\nEnd\n')
        (case/'log.flowfieldP006Foam').write_text('Iteration = 5\n')
        row=dict(iteration=1,**dict.fromkeys(limits('P006',p.config),0.), bounded_cells=0,
                 applied_voltage_V=2.2,momentum_fraction=.000256,thermal_active=0)
        post=case/'postProcessing/p006';post.mkdir(parents=True)
        with (post/'history.csv').open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(row));w.writeheader()
            for i in range(1,6):w.writerow(dict(row,iteration=i))
        return case

    def test_p006_initialization_is_not_qualified_even_with_zero_residuals(self):
        case=self.p006_live();before={p:p.read_bytes() for p in case.rglob('*') if p.is_file()}
        a=inspect_case(case)
        self.assertEqual(a['status'],'initializing');self.assertFalse(a['qualified'])
        self.assertTrue(any(c['name']=='full_physical_model' and c['status']=='pending' for c in a['checks']))
        self.assertEqual(before,{p:p.read_bytes() for p in case.rglob('*') if p.is_file()})

    def test_interruption_is_not_proof_of_divergence(self):
        case=self.p006_live();m=json.loads((case/'case.json').read_text())
        (case/'result.json').write_text(json.dumps(dict(m,status='failed',error='Interrupted by user')))
        a=inspect_case(case);self.assertEqual(a['status'],'interrupted');self.assertFalse(a['qualified'])

    def test_completed_iteration_budget_exit_remains_unqualified_and_is_explained(self):
        case=self.p006_live(iterations=5)
        (case/'log.flowfieldP006Foam').write_text('Iteration = 5\nEnd\n')
        (case/'postProcessing/p006/status.json').write_text(json.dumps(dict(converged=False,iteration=5)))
        a=inspect_case(case)
        self.assertEqual(a['status'],'failed');self.assertFalse(a['qualified'])
        self.assertEqual(a['stop_reason'],'iteration_limit')
        self.assertTrue(any(c['name']=='iteration_budget' for c in a['checks']))

    def test_cached_success_and_modified_inputs_cannot_pass(self):
        p=create_problem('P000');case=p.write_case(p.initial_design(),self.root/'p000')
        manifest=json.loads((case/'case.json').read_text())
        (case/'result.json').write_text(json.dumps(dict(manifest,status='ok',objective=0)))
        self.assertFalse(inspect_case(case)['qualified'])
        (case/'0/U').write_text('corrupted')
        self.assertEqual(inspect_case(case)['status'],'incompatible')

    def test_malformed_result_and_invalid_requested_path_report_errors(self):
        case=self.p006_live();(case/'result.json').write_text('[]')
        self.assertEqual(inspect_case(case)['status'],'failed')
        with self.assertRaisesRegex(ValueError,'not a directory'):inspect_paths([case,self.root/'absent'])

    def test_unreadable_case_evidence_does_not_abort_the_suite(self):
        case=self.p006_live()
        with patch('flowfield.stability.audit.scan_log',side_effect=PermissionError('Cannot read log')):
            a=inspect_case(case)
        self.assertEqual(a['status'],'inconclusive');self.assertFalse(a['qualified'])

    def test_p002_and_p003_inspections_disable_audit_writes(self):
        for name,module in (('P002','flowfield.problems.level2_medium_parametric.p002.metrics'),
                            ('P003','flowfield.problems.level4_high_dimensional_topology.p003.metrics')):
            case=self.root/name;case.mkdir()
            (case/'case.json').write_text(json.dumps(dict(problem=name,config={})))
            log='log.flowfield'+name+'Foam';(case/log).write_text('End\n')
            with patch('flowfield.stability.audit.CaseEvaluator._load_case'), \
                 patch(module+'.extract_metrics',side_effect=ValueError('No output')) as extract:
                inspect_case(case)
            self.assertEqual(extract.call_args.kwargs,{'write_report':False})

    def test_all_plan_covers_cfd_and_explicit_surrogate_exclusion(self):
        p=make_plan(list_problems(),studies=['repeat','mesh','numerics','parallel'])
        self.assertEqual(p['case_count'],46)
        d={e['problem']:e for e in p['problems']}
        self.assertIn('not_applicable',d['P004'])
        self.assertEqual(len(d['P006']['jobs']),6)
        self.assertEqual(d['P000']['unsupported_studies'][0]['study'],'parallel')
        self.assertEqual(d['P006']['jobs'][-1]['config']['processes'],1)
        for kw in (dict(configs={'P002':{}}),dict(processes=2),dict(studies=['unknown'])):
            with self.assertRaises(ValueError):make_plan(['P000'],**kw)

    def test_numerics_studies_preserve_physics(self):
        for name in list_problems():
            if name=='P004':continue
            a=create_problem(name).config;b=stricter(name,a)
            for key in ('cell_voltage','inlet_flow_ml_min','oxygen_velocity_m_s','end_time_s','density_kg_m3','length_m'):
                if hasattr(a,key):self.assertEqual(getattr(a,key),getattr(b,key))
        a=create_problem('P005').config;b=stricter('P005',a)
        self.assertEqual(b.max_delta_t_s,a.max_delta_t_s/2)
        self.assertEqual(b.monitor_interval_s,a.monitor_interval_s/2)

    def test_comparisons_reject_penalties_and_censored_events(self):
        a=dict(qualified=False,extracted=dict(objective=1e6))
        self.assertEqual(compare('P000','repeat',a,a)['status'],'inconclusive')
        b=dict(qualified=True,extracted=dict(objective=.4,metrics=dict(right_censored=True)))
        self.assertEqual(compare('P005','numerics',b,b)['status'],'inconclusive')
        x=dict(qualified=True,extracted=dict(objective=1e6,metrics={'pressure_drop_pa':10}))
        y=dict(qualified=True,extracted=dict(objective=-1e6,metrics={'pressure_drop_pa':10.1}))
        self.assertEqual(compare('P000','repeat',x,y)['status'],'passed')
        y['extracted']['metrics']['pressure_drop_pa']=12
        self.assertEqual(compare('P000','mesh',x,y)['status'],'failed')

    def test_unresolved_detachment_brackets_are_inconclusive(self):
        def result(t,b):return dict(qualified=True,extracted=dict(metrics=dict(right_censored=False,detached=True,detachment_time_s=t,detachment_bracket_s=b)))
        a,b=result(.1,[.08,.1]),result(.09,[.07,.09])
        self.assertEqual(compare('P005','numerics',a,b,rtol=.001)['status'],'inconclusive')

    def test_reports_are_escaped_and_nonfinite_json_is_rejected(self):
        report=dict(status='failed',cases=[dict(problem='P000',audit=dict(status='failed',error='<script>alert(1)</script>'))])
        write_reports(self.root,report)
        text=(self.root/'report.html').read_text();self.assertNotIn('<script>',text);self.assertIn('&lt;script&gt;',text)
        report['bad']=float('nan')
        with self.assertRaises(ValueError):write_reports(self.root,report)
        self.assertEqual(overall([dict(status='initializing')]),'inconclusive')
        self.assertEqual(overall([dict(status='blocked')]),'failed')

    def test_dry_run_creates_no_files_and_never_calls_cfd(self):
        out=self.root/'dry'
        with patch('flowfield.stability.cli.run_suite') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['run','--problem','all','--output',str(out),'--dry-run']),0)
        run.assert_not_called();self.assertFalse(out.exists())

    def test_inspection_refuses_reports_inside_a_case(self):
        p=create_problem('P000');case=p.write_case(p.initial_design(),self.root/'p000')
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['inspect','--case',str(case),'--output',str(case/'audit')])
        self.assertEqual(code,2);self.assertFalse((case/'audit').exists())

    def test_suite_continues_after_preflight_failure_and_resumes_without_rerun(self):
        plan=make_plan(['P000','P004-hydraulic']);out=self.root/'suite'
        def preflight(p):
            if p.name=='P000':raise RuntimeError('Missing solver')
        def run(p,case):
            (case/'.execution-started').write_text('started\n')
            (case/'result.json').write_text('{}')
        def audit(case,**kwargs):return dict(case_directory=str(case),status='passed',qualified=True,extracted=dict(metrics={'pressure_drop_pa':1}))
        with patch('flowfield.stability.runner.preflight',side_effect=preflight), \
             patch('flowfield.problem.CFDProblem.run_case',autospec=True,side_effect=run) as solver, \
             patch('flowfield.stability.runner.inspect_case',side_effect=audit), contextlib.redirect_stdout(io.StringIO()):
            r=run_suite(plan,out)
            self.assertEqual(r['status'],'failed');self.assertEqual(solver.call_count,1)
            self.assertEqual([c['audit']['status'] for c in r['cases']],['blocked','passed'])
            r=run_suite(plan,out,resume=True);self.assertEqual(solver.call_count,1)
            with self.assertRaisesRegex(ValueError,'identical plan'):run_suite(plan,out,resume=True,rtol=.1)

    def test_interrupted_suite_keeps_report_and_never_runs_optimization(self):
        plan=make_plan(['P000']);out=self.root/'suite'
        with patch('flowfield.stability.runner.preflight',side_effect=KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):run_suite(plan,out)
        self.assertEqual(json.loads((out/'report.json').read_text())['status'],'interrupted')
        self.assertFalse(json.loads((out/'report.json').read_text())['coverage']['optimization_run'])

    def test_concurrent_resume_is_refused(self):
        import fcntl
        out=self.root/'suite';out.mkdir()
        with (out/'.checker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'already running'):
                run_suite(make_plan(['P000']),out,resume=True)

    def test_transient_interior_violation_is_not_lost_from_recent_window(self):
        p=create_problem('P005');case=p.write_case(p.initial_design(),self.root/'vof')
        post=case/'postProcessing';post.mkdir()
        (case/'log.interFoam').write_text('Time = 0.001\n')
        (post/'detachment.csv').write_text('time_s,min_alpha_water,max_alpha_water,max_gas_balance_relative,max_continuity_relative\n'+
            ''.join(f'{i/1e6},0,{1.1 if i==1 else 1},0,0\n' for i in range(300)))
        a=inspect_case(case)
        self.assertEqual(a['status'],'failed')
        self.assertTrue(any(c['name']=='transient_phase_bounds' and c['status']=='failed' for c in a['checks']))


if __name__=='__main__':
    unittest.main()
