"""P009 boundaries, paper audit and acceptance failures; no synthetic CFD claims."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield.evaluation import CaseEvaluator
from flowfield.registry import create_problem
from flowfield.runtime import Runtime, SimulationError
from flowfield.problems.level1_small_continuous.p009 import ChenConfig, P009Problem
from flowfield.problems.level1_small_continuous.p009.config import BOUNDS, physical_geometry
from flowfield.problems.level1_small_continuous.p009.metrics import extract_metrics
from flowfield.problems.level1_small_continuous.p009.reference import TABLE5, printed_equation
from flowfield.problems.level1_small_continuous.p009.rsm import fit, predict, optimize_ga, reference_audit
from flowfield.problems.level1_small_continuous.p009.cli import main


class P009Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_units_domain_and_distinct_product_baseline(self):
        p=create_problem('P009')
        lo=p.decode([0,0,0]);hi=p.decode([1,1,1])
        self.assertAlmostEqual(lo['flow_m3_s'],5e-6)
        self.assertAlmostEqual(lo['inlet_speed_m_s'],5/6)
        self.assertEqual((lo['angle_deg'],hi['thickness_mm']),(70,2.24))
        baseline=physical_geometry(92.5,.66,2.44,p.config)
        self.assertFalse(baseline['within_optimization_bounds'])
        with self.assertRaises(ValueError):p.decode(baseline['normalized_design'])
        with self.assertRaises(ValueError):physical_geometry(92.5,.16,1,p.config)
        for kwargs in ({'porosity':1}, {'permeability_m2':float('nan')}, {'inertial_coefficient_m_inv':-1},
                       {'mesh':'invalid'}, {'max_iterations':True}, {'balance_tolerance':.1}):
            with self.assertRaises(ValueError):ChenConfig(**kwargs)

    def test_case_pins_sources_and_uses_paper_drag_factors(self):
        p=P009Problem();case=p.write_case(p.initial_design(),self.root/'case')
        spec=p.case_definition
        CaseEvaluator(spec)._load_case(case)
        options=(case/'constant/fvOptions').read_text()
        self.assertIn('256000000',options)
        self.assertIn('f (0 0 0)',options)
        self.assertIn('zone nickel',(case/'system/controlDict').read_text())
        self.assertIn('simulationType laminar',(case/'constant/turbulenceProperties').read_text())
        with self.assertRaises(FileExistsError):p.write_case(p.initial_design(),case)
        with (case/'mesher.py').open('a') as stream:stream.write('\n# edited\n')
        with self.assertRaisesRegex(ValueError,'Case input changed'):CaseEvaluator(spec)._load_case(case)

    def test_reference_audit_identifies_printed_equation_discrepancy(self):
        r=reference_audit()
        self.assertEqual((r['table_rows'],r['table_refit']['unique_designs']),(20,13))
        self.assertAlmostEqual(r['table_refit']['r2'],.9690048401571508)
        self.assertAlmostEqual(printed_equation((70,.3,1.92)),.037975)
        self.assertAlmostEqual(r['printed_equation']['literal_physical_variable_prediction'],3.03408165)
        self.assertEqual(r['printed_equation']['bounded_grid_best']['thickness_mm'],1.6)
        self.assertTrue(r['reported']['baseline_outside_table4'])
        self.assertFalse(r['paper_validated'])

    def test_quadratic_fit_ga_and_domain_guards(self):
        rows=[r[:3] for r in TABLE5]
        def f(row):
            x=[(v-lo)/(hi-lo) for v,(lo,hi) in zip(row,BOUNDS)]
            return 2-sum((v-t)**2 for v,t in zip(x,(.2,.4,.6)))
        model=fit(rows,[f(row) for row in rows])
        self.assertAlmostEqual(predict(model,[.2,.4,.6]),2)
        ga=optimize_ga(model,seed=7,generations=100)
        self.assertGreater(ga['predicted_performance'],1.9999)
        self.assertEqual(ga,optimize_ga(model,seed=7,generations=100))
        with self.assertRaises(ValueError):predict(model,[0,0,2])
        with self.assertRaises(ValueError):fit([rows[0]]*20,[1]*20)

    def fixture(self):
        case=self.root/'metrics';case.mkdir()
        (case/'case.json').write_text(json.dumps({'geometry':physical_geometry(92.5,.75,1.92,ChenConfig())}))
        (case/'log.checkMesh').write_text('cells: 100\nMesh OK.\n')
        (case/'log.simpleFoam').write_text('Time = 20\nSIMPLE solution converged in 20 iterations\nEnd\n')
        data={'p_inlet':('areaAverage(p)',100/1050),'p_outlet':('areaAverage(p)',0),
              'q_inlet':('sum(phi)',-1.25e-5),'q_outlet':('sum(phi)',1.25e-5),
              'q_walls':('sumMag(phi)',0),'plane_speed':('areaAverage(speed)',.02),
              'plane_speed2':('areaAverage(speed2)',.0005)}
        for name,(column,value) in data.items():
            dest=case/'postProcessing'/name/'0';dest.mkdir(parents=True)
            (dest/'surfaceFieldValue.dat').write_text(f'# Time {column}\n'+''.join(f'{i} {value}\n' for i in range(1,21)))
        return case

    def test_metrics_use_bar_and_reject_truncated_stale_or_leaking_results(self):
        case=self.fixture()
        with patch('flowfield.stability.fields.final_fields',return_value={'status':'passed'}), \
             patch('flowfield.problems.level1_small_continuous.p009.verification.verify_saved',return_value={}):
            m=extract_metrics(case,ChenConfig())
            self.assertAlmostEqual(m['performance_m_s_per_bar'],20)
            self.assertAlmostEqual(m['flow_cv'],.5)
            log=(case/'log.simpleFoam').read_text()
            (case/'log.simpleFoam').write_text('Time = 20\nEnd\n')
            with self.assertRaisesRegex(SimulationError,'convergence'):extract_metrics(case,ChenConfig())
            (case/'log.simpleFoam').write_text(log)
            p=case/'postProcessing/q_walls/0/surfaceFieldValue.dat';original=p.read_text()
            p.write_text(original.replace('20 0','20 0.0001'))
            with self.assertRaisesRegex(SimulationError,'conservation'):extract_metrics(case,ChenConfig())
            p.write_text(original+'21 0\n')
            with self.assertRaisesRegex(SimulationError,'unaligned'):extract_metrics(case,ChenConfig())

    def test_failed_baseline_cannot_be_penalized(self):
        p=P009Problem(failure_penalty=1e6)
        with patch('flowfield.evaluation.CaseEvaluator.evaluate',side_effect=SimulationError('failed')):
            with self.assertRaises(SimulationError):p.evaluate_raw(p.initial_design())
            self.assertEqual(p.evaluate_raw([0,0,0])['status'],'penalized')

    def test_missing_native_runtime_does_not_create_study(self):
        from flowfield.problems.level1_small_continuous.p009.study import paper_study
        with patch.object(P009Problem,'preflight'), patch.object(Runtime,'resolve',side_effect=SimulationError('checkMesh missing')):
            with self.assertRaises(SimulationError):paper_study(self.root/'study',ChenConfig(),Runtime('native'))
        self.assertFalse((self.root/'study').exists())

    def test_study_preserves_repeated_rows_but_never_fits_failures(self):
        from flowfield.problems.level1_small_continuous.p009.study import paper_study
        calls=[]
        def evaluate(problem,a,b,c):
            calls.append((a,b,c))
            return dict(status='ok',case_directory=str(problem.output_root),
                metrics={'performance_m_s_per_bar':2-.001*(a-80)**2-.1*(b-.5)**2-.1*(c-1.9)**2})
        with patch.object(P009Problem,'preflight'),patch.object(Runtime,'resolve',return_value={}),patch.object(P009Problem,'evaluate_physical',evaluate):
            r=paper_study(self.root/'study',ChenConfig(),Runtime('native'))
        self.assertEqual(len(calls),15)
        self.assertEqual(len(r['table5']),20)
        self.assertTrue(r['model']['qualified_cfd_panel'])
        with patch.object(P009Problem,'preflight'),patch.object(Runtime,'resolve',return_value={}),patch.object(P009Problem,'evaluate_physical',side_effect=SimulationError('failed')):
            with self.assertRaises(SimulationError):paper_study(self.root/'failed',ChenConfig(),Runtime('native'))
        self.assertFalse((self.root/'failed/response-surface.json').exists())

    def test_cli_reference_audit_does_not_resolve_cfd(self):
        with patch.object(Runtime,'resolve',side_effect=AssertionError('No CFD')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['reference-audit','--case',str(self.root/'audit.json')]),0)
        self.assertEqual(json.loads((self.root/'audit.json').read_text())['evidence'],
                         'published table/regression audit; zero CFD evaluations')


if __name__=='__main__':unittest.main()
