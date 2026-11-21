"""Paper transcription, finite-volume mesh, immutable inputs and study gates."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.ndimage import label
from flowfield.registry import create_problem
from flowfield.runtime import Runtime,SimulationError
from flowfield.problems.level1_small_continuous.p010 import P010Problem,PEMFCConfig
from flowfield.problems.level1_small_continuous.p010.config import BASELINE,FLOW_FIELDS,normalized
from flowfield.problems.level1_small_continuous.p010.geometry import build_geometry,esc_width
from flowfield.problems.level1_small_continuous.p010.metrics import mesh_arrays,extract_metrics
from flowfield.problems.level1_small_continuous.p010.rsm import reference_audit,maximize,predict,features
from flowfield.problems.level1_small_continuous.p010.study import paper_study
from flowfield.problems.level1_small_continuous.p010.cli import main
from flowfield.stability.profiles import limits,experiments

class P010Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_reference_reproduces_statistics_without_cfd(self):
        audit=reference_audit();json.dumps(audit,allow_nan=False)
        self.assertFalse(audit['cfd_executed']);self.assertFalse(audit['paper_validated'])
        self.assertAlmostEqual(audit['optimum']['predicted_net_power_W'],3.11097,places=5)
        self.assertAlmostEqual(audit['model']['r_squared'],.99424290248,places=9)
        self.assertAlmostEqual(audit['model']['predicted_r_squared'],.9689473633,places=9)
        self.assertEqual(audit['model']['distinct_designs'],25)
        self.assertEqual(audit['model']['observations'],27)
        self.assertEqual(audit['model']['group_validation']['status'],'incomplete')
        interval=audit['optimum_intervals'];self.assertLess(interval['individual_response_interval_W'][0],interval['mean_response_interval_W'][0])

    def test_quadratic_box_global_maximum(self):
        # A known concave four-dimensional quadratic with two boundary optima.
        optimum=np.array([-.3,.4,2,-2]);coef=np.zeros(15);coef[1:5]=2*optimum;coef[5:9]=-1
        model={'coefficients':coef.tolist()};best=maximize(model)
        np.testing.assert_allclose(best['normalized_design'],[.35,.7,1,0],atol=1e-12)
        self.assertAlmostEqual(predict(model,best['normalized_design']),best['predicted_net_power_W'])
        self.assertEqual(features([[0]*4,[1]*4]).shape,(2,15))

    def test_geometry_channels_and_layers(self):
        for kind in FLOW_FIELDS:
            c=PEMFCConfig(flow_field=kind);p=P010Problem(config=c);g=p.decode(p.initial_design());m=build_geometry(g,c)
            self.assertEqual(len(m['ids']),67712);self.assertAlmostEqual(m['z'][-1],3.7338)
            a=np.array(m['ids']).reshape(len(m['z'])-1,len(m['y'])-1,len(m['x'])-1)
            self.assertEqual(label(a[1]==1)[1],2 if kind=='CDLIS' else 1)
            self.assertGreater((a==3).sum(),0);self.assertGreater((a==5).sum(),0)
            if kind.startswith('ESCFF'):
                widths=[esc_width(x,kind) for x in np.linspace(0,23,2001)]
                self.assertGreaterEqual(min(widths),.5);self.assertLessEqual(max(widths),1)

    def test_mesh_orientation_volume_ports_and_immutability(self):
        p=create_problem('P010');case=p.write_case(p.initial_design(),self.root/'case')
        g=p.decode(p.initial_design());a=mesh_arrays(case,g,p.config);ni=a['ni']
        direction=a['centers'][a['n'][:ni]]-a['centers'][a['o'][:ni]]
        self.assertTrue(np.all(np.einsum('ij,ij->i',direction,a['normal'][:ni])>0))
        self.assertAlmostEqual(a['volumes'].sum(),.023**2*.0037338,places=14)
        for port in ('cathodeInlet','anodeInlet'):
            self.assertAlmostEqual(a['area'][a['patches'][port]].sum(),1e-6,places=14)
        from flowfield.evaluation import CaseEvaluator
        CaseEvaluator(p.case_definition)._load_case(case)
        (case/'solver/Gas.H').write_text('// altered')
        with self.assertRaises(ValueError):CaseEvaluator(p.case_definition)._load_case(case)

    def test_no_unqualified_baseline_enters_study(self):
        with patch.object(P010Problem,'preflight'),patch.object(Runtime,'resolve',return_value={}),patch.object(P010Problem,'evaluate_raw',side_effect=SimulationError('not converged')) as run:
            with self.assertRaises(SimulationError):paper_study(self.root/'study',PEMFCConfig(),Runtime())
        self.assertEqual(run.call_count,1)
        self.assertTrue(P010Problem.requires_qualified_baseline)

    def test_study_keeps_duplicate_rows_and_fresh_confirmation(self):
        def evaluate(problem,x):
            # Workflow fixture only: deliberately not represented as real CFD.
            y=3-float(np.sum((np.asarray(x)-.6)**2))
            return dict(status='ok',case_directory=str(problem.output_root),metrics={'net_power_W':y})
        with patch.object(P010Problem,'preflight'),patch.object(Runtime,'resolve',return_value={}),patch.object(P010Problem,'evaluate_raw',evaluate):
            result=paper_study(self.root/'study',PEMFCConfig(),Runtime())
        records=json.loads((self.root/'study/observations.json').read_text())
        self.assertEqual(len(records),27)  # baseline + 25 distinct points + new confirmation
        self.assertEqual(len(result['table6']),27)
        self.assertEqual(sum(r['repeated_design'] for r in result['table6']),2)
        self.assertEqual(records[-1]['label'],'optimum-confirmation')
        self.assertFalse(result['paper_validated'])

    def test_configuration_and_checker(self):
        p=P010Problem();self.assertEqual(p.dimension,4);self.assertEqual(p.initial_design(),normalized(BASELINE))
        for kw in ({'processes':2},{'balance_tolerance':.1},{'field_tolerance':.01},{'cell_voltage':1}):
            with self.assertRaises(ValueError):PEMFCConfig(**kw)
        self.assertIn('membrane_balance',limits('P010',p.config))
        jobs,unsupported=experiments('P010',p.config,['mesh','numerics','parallel'])
        self.assertEqual(len(jobs),4);self.assertEqual(len(unsupported),1)

    def test_cli_audit_and_no_in_place_case_override(self):
        path=self.root/'audit.json'
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['reference-audit','--case',str(path)]),0)
        self.assertFalse(json.loads(path.read_text())['cfd_executed'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['run','--case',str(self.root/'case'),'--mesh','fine'])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['reference-audit','--case',str(path)])

    def test_extraction_rejects_false_completion(self):
        case=self.root/'case';(case/'postProcessing/p010').mkdir(parents=True)
        (case/'case.json').write_text('{"cells":1}')
        (case/'postProcessing/p010/status.json').write_text('{"converged":false}')
        (case/'postProcessing/p010/history.csv').write_text('iteration,field_change\n1,0\n')
        with self.assertRaisesRegex(ValueError,'history'):extract_metrics(case,PEMFCConfig())

if __name__=='__main__':unittest.main()
