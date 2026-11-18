"""Geometry, reported electrochemical inputs, fixed EWM scores and acceptance gates."""
import json
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from flowfield.problems.level1_small_continuous.p008 import P008Problem,AEMConfig
from flowfield.problems.level1_small_continuous.p008.config import MODEL_REVISION
from flowfield.problems.level1_small_continuous.p008.geometry import build_geometry
from flowfield.problems.level1_small_continuous.p008.ewm import fit,score,INDICATORS
from flowfield.problems.level1_small_continuous.p008.metrics import mesh_arrays
from flowfield.evaluation import CaseEvaluator

class P008Tests(unittest.TestCase):
    def test_cli_preserves_existing_or_incomplete_cases(self):
        from unittest.mock import patch
        from flowfield.problems.level1_small_continuous.p008.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp)/'incomplete';case.mkdir()
            marker=case/'keep.txt';marker.write_text('existing data')
            for action,message in (('generate','already exists'),('run','not a complete'),('extract','not a complete')):
                error=io.StringIO()
                with self.subTest(action=action),contextlib.redirect_stderr(error), \
                     patch.object(P008Problem,'write_case') as generate, \
                     patch.object(P008Problem,'run_case') as run, \
                     patch.object(P008Problem,'extract_results') as extract:
                    with self.assertRaises(SystemExit) as stopped:main([action,'--case',str(case)])
                    self.assertEqual(stopped.exception.code,2)
                    generate.assert_not_called();run.assert_not_called();extract.assert_not_called()
                self.assertIn(message,error.getvalue())
                self.assertNotIn('Traceback',error.getvalue())
                self.assertEqual(marker.read_text(),'existing data')
                self.assertEqual(list(case.iterdir()),[marker])

    def test_width_depth_and_fixed_open_area(self):
        p=P008Problem()
        for x,n,w in ((1/6,8,1.5),(.5,10,1.2),(5/6,12,1.)):
            for d in (.4,.6,.8,1.):
                g=p.decode([x,(d-.4)/.6]);self.assertEqual(g['channels'],n)
                self.assertAlmostEqual(g['width_mm'],w);self.assertAlmostEqual(n*w,12)
                self.assertAlmostEqual(n*(w+g['rib_mm']),20)
                m=build_geometry(g,p.config)
                self.assertAlmostEqual(m['z'][-1],4.86)
                self.assertEqual(set(m['ids']),set(range(9)))
        base=p.decode(p.initial_design());self.assertAlmostEqual(base['depth_mm'],.8)
        self.assertEqual(base['width_mm'],1.5)
    def test_reported_kinetics_and_sources(self):
        c=AEMConfig();self.assertAlmostEqual(c.exchange_current_anode,2.19e-3*1e4)
        self.assertAlmostEqual(c.exchange_current_cathode,1.32*1e4)
        # Global electrolysis mass closes after OH- transfer between the sides.
        f=96485.33212;mw=.01801528;mh=.00201588;mo=.0319988
        self.assertAlmostEqual(-mw/(2*f)+mh/(2*f)+mo/(4*f),0,places=12)
    def test_config_guards(self):
        for kw in ({'processes':2},{'iterations':True},{'mesh':'bad'},{'maximum_packing':1},
                   {'field_tolerance':.1},{'balance_tolerance':.1},{'contact_angle_deg':90},
                   {'liquid_density':float('nan')},{'ewm_calibration':'file.json'}):
            with self.subTest(kw=kw),self.assertRaises((ValueError,TypeError)):AEMConfig(**kw)
    def test_mesh_ports_and_pinned_inputs(self):
        import numpy as np
        p=P008Problem()
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case')
            manifest=json.loads((case/'case.json').read_text());m=build_geometry(manifest['design_geometry'],p.config)
            o,nb,xyz,size,v,normal,do,dn,area,patches=mesh_arrays(case,m)
            self.assertEqual(len(m['cells']),9000)
            self.assertAlmostEqual(sum(v),20e-3*20e-3*4.86e-3)
            for name in ('anodeInlet','cathodeInlet'):
                self.assertAlmostEqual(sum(area[patches[name]]),12e-3*.8e-3)
                self.assertTrue(np.all(normal[patches[name],1]==-1))
            for name in ('anodeOutlet','cathodeOutlet'):self.assertTrue(np.all(normal[patches[name],1]==1))
            CaseEvaluator(p.case_definition)._load_case(case)
            with (case/'solver/Transport.H').open('a') as stream:stream.write('\n// mutation\n')
            with self.assertRaises((ValueError,RuntimeError)):CaseEvaluator(p.case_definition)._load_case(case)
    def test_entropy_directions_constants_and_fixed_panel(self):
        rows=[[1.,3.,7.],[2.,2.,7.],[3.,1.,7.]];indicators=(('a','benefit'),('b','cost'),('constant','cost'))
        c=fit(rows,indicators);self.assertEqual(c['weights'][2],0)
        self.assertAlmostEqual(c['weights'][0],.5);self.assertAlmostEqual(c['weights'][1],.5)
        self.assertLess(c['scores'][0],c['scores'][1]);self.assertLess(c['scores'][1],c['scores'][2])
        for row,expected in zip(rows,c['scores']):self.assertAlmostEqual(score(row,c)['score'],expected)
        before=json.dumps(c,sort_keys=True);extrapolated=score([4,0,7],c)
        self.assertGreater(extrapolated['score'],1);self.assertEqual(extrapolated['outside_calibration'],['a','b'])
        self.assertEqual(before,json.dumps(c,sort_keys=True))
        for bad in ([],[[1,1]],[[1,1],[1,1]],[[1,1],[2,float('nan')]]):
            with self.assertRaises(ValueError):fit(bad,(('x','benefit'),('y','cost')))
    def test_no_optimizer_without_qualified_fixed_calibration(self):
        p=P008Problem()
        with self.assertRaises(ValueError):p.validate_optimization()
        c=fit([[1]*9,[2]*9,[3]*9])
        p=P008Problem(config=replace(AEMConfig(),ewm_calibration=c))
        with self.assertRaises(ValueError):p.validate_optimization()
        c.update(qualified_cfd_panel=True,model_revision=MODEL_REVISION,config_sha256=AEMConfig().fingerprint)
        p=P008Problem(config=replace(AEMConfig(),ewm_calibration=c));p.validate_optimization()
        with self.assertRaises(ValueError):P008Problem(config=replace(p.config,mesh='medium')).validate_optimization()
    def test_only_expected_catalyst_aspect_warning_is_accepted(self):
        from flowfield.problems.level1_small_continuous.p008.mesh_checks import assess_mesh
        p=P008Problem()
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case')
            manifest=json.loads((case/'case.json').read_text());m=build_geometry(manifest['design_geometry'],p.config)
            cells=[i for i,v in enumerate(m['ids']) if v in (3,5)]
            folder=case/'constant/polyMesh/sets';folder.mkdir()
            def cell_set(indices):
                (folder/'highAspectRatioCells').write_text('FoamFile {}\n'+str(len(indices))+'\n(\n'+'\n'.join(map(str,indices))+'\n)\n')
            cell_set(cells)
            log='Failed 1 mesh checks.\n ***High aspect ratio cells found, Max aspect ratio: 1333.333333333, number of cells 4800\nMesh non-orthogonality Max: 0 average: 0\nMax skewness = 1e-14 OK.\nMin volume = 1.0416667e-12. Max volume = 1e-9.\n'
            accepted=assess_mesh(case,log,manifest)
            self.assertTrue(accepted['accepted']);self.assertFalse(accepted['openfoam_all_checks_passed'])
            with self.assertRaises(ValueError):assess_mesh(case,log.replace('Failed 1','Failed 2'),manifest)
            with self.assertRaises(ValueError):assess_mesh(case,log.replace('Max: 0 average','Max: 5 average'),manifest)
            cell_set([0])
            with self.assertRaises(ValueError):assess_mesh(case,log,manifest)
    def test_bad_calibration_cannot_create_nan_score(self):
        c=fit([[1,3],[2,2],[3,1]],(('a','benefit'),('b','cost')))
        for key,value in (('epsilon',float('nan')),('epsilon',0),('weights',[True,False]),('weights',[-1,2])):
            bad=dict(c,**{key:value})
            with self.assertRaises(ValueError):score([2,2],bad)
    def test_paper_panels_write_frozen_portable_calibration(self):
        from unittest.mock import patch
        from flowfield.problems.level1_small_continuous.p008.study import paper_study
        def fake(problem,x):
            base=1+x[0]+2*x[1]
            return dict(status='ok',case_directory='qualified-test-case',metrics={k:base*(i+1) for i,(k,_) in enumerate(INDICATORS)})
        with tempfile.TemporaryDirectory() as tmp,patch.object(P008Problem,'evaluate_raw',fake):
            root=Path(tmp)/'panels';result=paper_study(root,AEMConfig(),None)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(len(json.loads((root/'observations.json').read_text())),5)
            config=AEMConfig(**json.loads((root/'optimization-config.json').read_text()))
            P008Problem(config=config).validate_optimization()
            self.assertEqual(config.ewm_calibration['panel_count'],5)
    def test_failed_panel_does_not_publish_calibration(self):
        from unittest.mock import patch
        from flowfield.problems.level1_small_continuous.p008.study import paper_study
        from flowfield.runtime import SimulationError
        with tempfile.TemporaryDirectory() as tmp,patch.object(P008Problem,'evaluate_raw',side_effect=SimulationError('unconverged')):
            root=Path(tmp)/'panels'
            with self.assertRaises(SimulationError):paper_study(root,AEMConfig(),None)
            self.assertFalse((root/'calibration.json').exists())
    def test_stability_profile(self):
        from flowfield.stability.runner import make_plan
        from flowfield.stability.profiles import limits
        plan=make_plan(['P008'],studies=('mesh','numerics','parallel'))
        self.assertEqual(plan['case_count'],4)
        self.assertEqual(plan['problems'][0]['unsupported_studies'][0]['study'],'parallel')
        self.assertIn('energy_balance',limits('P008',AEMConfig()))
    def test_ewm_calibration_portable(self):
        c=fit([[1]*9,[2]*9]);cfg=replace(AEMConfig(),ewm_calibration=c)
        copied=AEMConfig(**json.loads(json.dumps(cfg.to_dict())))
        self.assertEqual(copied.fingerprint,cfg.fingerprint)

if __name__=='__main__':unittest.main()
