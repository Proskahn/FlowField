import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from flowfield import get_problem,list_problems,DualLayerConfig
from flowfield.problems.level1_small_continuous.p005 import P005Problem,BASELINE_X
from flowfield.problems.level1_small_continuous.p005.case import generate_case,INPUTS
from flowfield.problems.level1_small_continuous.p005.geometry import block_mesh
from flowfield.problems.level1_small_continuous.p005.metrics import extract_metrics
from flowfield.problems.level1_small_continuous.p005.evaluation import run_case
from flowfield.optimization import main
from flowfield.runtime import SimulationError


class P005Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_registry_decode_and_operating_condition(self):
        p=get_problem('P005');self.assertIn('P005',list_problems())
        self.assertEqual(p.dimension,2)
        for x,hb,hd in (([0,0],.0005,.0005),([1,1],.002,.004),(BASELINE_X,.001,.003)):
            g=p.decode(x);self.assertAlmostEqual(g['base_height_m'],hb);self.assertAlmostEqual(g['degassing_height_m'],hd)
            self.assertAlmostEqual(g['volume_m3'],.02*(.001*hb+.0016*hd))
        q=P005Problem(config=replace(p.config,oxygen_velocity_m_s=1))
        self.assertNotEqual(p.ioh_name,q.ioh_name)
        self.assertAlmostEqual(q.decode(BASELINE_X)['gas_flow_m3_s'],1e-8)
        for x in ([0],[0,1,2],[0,2],[True,0],[float('nan'),0]):
            with self.assertRaises(ValueError):p.decode(x)

    def test_config_rejects_invalid_physics_and_relaxed_checks(self):
        for kw in ({'mesh':'bad'},{'gas_inlet_side_m':.002},{'gas_inlet_z_m':0},
                   {'gas_threshold':1},{'max_courant':.8},{'temperature_k':353.15},
                   {'end_time_s':True},{'oxygen_velocity_m_s':float('nan')},
                   {'alpha_tolerance':.01},{'water_density_kg_m3':1}):
            with self.assertRaises(ValueError):DualLayerConfig(**kw)
        with self.assertRaises(ValueError):P005Problem(failure_penalty=.1)

    def test_conformal_blocks_refine_gas_inlet_and_preserve_volume(self):
        p=P005Problem();counts=[]
        for level,r in (('coarse',1),('medium',2),('fine',4)):
            c=replace(p.config,mesh=level);text,m=block_mesh(p.decode(BASELINE_X),c)
            self.assertEqual(m['blocks'],24);self.assertEqual(m['gas_inlet_faces'],4*r*r)
            self.assertAlmostEqual(m['expected_volume_m3'],1.16e-7)
            self.assertLessEqual(m['maximum_delta_t_s'],c.max_delta_t_s)
            self.assertIn('gasInlet',text);counts.append(m['cells'])
        self.assertTrue(counts[0]<counts[1]<counts[2])

    def test_no_overwrite_and_changed_input_rejected_before_execution(self):
        p=P005Problem();case=generate_case(self.root/'case',p.decode(BASELINE_X),p.config)
        self.assertEqual(set(json.loads((case/'case.json').read_text())['input_sha256']),INPUTS)
        with self.assertRaises(FileExistsError):generate_case(case,p.decode(BASELINE_X),p.config)
        (case/'0/U').write_text('changed')
        with self.assertRaisesRegex(ValueError,'input changed'):run_case(case)
        self.assertFalse((case/'.execution-started').exists())

    def diagnostics(self,detached=False):
        case=self.root/'metrics';case.mkdir(exist_ok=True)
        c=DualLayerConfig()
        t=.1 if detached else .2
        row=dict(time_s=t,gas_volume_m3=1e-9,attached_volume_m3=0,detached_volume_m3=1e-9,
            injected_volume_m3=5e-8*t,gas_balance_relative=1e-9,continuity_relative=1e-9,
            min_alpha_water=0,max_alpha_water=1,armed=1,confirmed=int(detached),
            event_time_s=t-.0003 if detached else -1,event_bracket_start_s=t-.0004 if detached else -1,
            max_gas_balance_relative=1e-8,max_continuity_relative=1e-8)
        (case/'postProcessing').mkdir(exist_ok=True)
        def save():
            with (case/'postProcessing/detachment.csv').open('w') as stream:
                w=csv.DictWriter(stream,fieldnames=list(row));w.writeheader();w.writerow(row)
        (case/'log.interFoam').write_text('trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\nEnd\n')
        save();return case,c,row,save

    def test_censoring_is_not_detachment_or_solver_failure(self):
        case,c,row,save=self.diagnostics()
        r=extract_metrics(case,c)
        self.assertEqual(r['objective'],.4);self.assertIsNone(r['metrics']['detachment_time_s'])
        self.assertTrue(r['metrics']['right_censored']);self.assertEqual(r['constraints']['detachment_observed'],1)
        row['time_s']=.19;row['injected_volume_m3']=5e-8*.19;save()
        with self.assertRaisesRegex(SimulationError,'horizon'):extract_metrics(case,c)

    def test_detachment_timing_and_conservation_validation(self):
        case,c,row,save=self.diagnostics(True)
        r=extract_metrics(case,c);self.assertAlmostEqual(r['objective'],.0997)
        self.assertFalse(r['metrics']['right_censored'])
        row['max_gas_balance_relative']=.03;save()
        with self.assertRaisesRegex(SimulationError,'conservation'):extract_metrics(case,c)
        row['max_gas_balance_relative']=1e-8;row['event_time_s']=.09999;save()
        with self.assertRaisesRegex(SimulationError,'timing'):extract_metrics(case,c)
        row['min_alpha_water']=float('nan');save()
        with self.assertRaisesRegex(SimulationError,'nonfinite'):extract_metrics(case,c)

    def test_failure_policy_never_fakes_detachment(self):
        p=P005Problem(failure_penalty=100)
        with patch('flowfield.problems.level1_small_continuous.p005.evaluation.evaluate',side_effect=SimulationError('test')):
            result=p.evaluate_raw(BASELINE_X)
        self.assertEqual(result['objective'],100);self.assertFalse(result['metadata']['valid_cfd'])
        self.assertEqual(result['objectives'],{})

    def test_optimization_plan_and_old_all_resume_membership(self):
        out=self.root/'plan';buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(['--problems','P005','--budget','20','--output',str(out),'--dry-run']),0)
        plan=json.loads(buf.getvalue());self.assertEqual(plan['maximum_cfd_attempts'],20)
        self.assertEqual(plan['problems'][0]['algorithm'],'CMA-ES');self.assertFalse(out.exists())
        # A pre-P005 saved 'all' suite must not gain a new problem during resume.
        out.mkdir();old=['P000','P001','P002','P003','P004','P004-hydraulic']
        (out/'plan.json').write_text(json.dumps({'problems':[{'problem':p} for p in old]}))
        (out/'run-options.json').write_text(json.dumps({'problems':['all']}))
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(['--resume','--output',str(out),'--dry-run'])
        plan=json.loads(buf.getvalue())
        self.assertEqual([e['problem'] for e in plan['problems']],old)
        self.assertEqual(plan['maximum_cfd_attempts'],100)

    def test_mesh_diagnostics_reject_wrong_volume_or_disconnected_domain(self):
        from flowfield.problems.level1_small_continuous.p005.metrics import verify_mesh_log
        expected={'expected_volume_m3':1e-7,'cells':100,'gas_inlet_faces':4}
        log='Mesh OK.\nMesh has 3 geometric\nNumber of regions: 1 (OK).\nTotal volume = 1e-7.\ncells: 100\n gasInlet 4 9 ok\n'
        self.assertEqual(verify_mesh_log(log,expected)['cells'],100)
        for changed in (log.replace('1e-7','2e-7'),log.replace('regions: 1','regions: 2'),log.replace('cells: 100','cells: 99')):
            with self.assertRaises(SimulationError):verify_mesh_log(changed,expected)

    def test_general_search_routes_p005_to_its_physical_config(self):
        from flowfield.cli import main as flowfield_main
        with patch('flowfield.optimization.runner._execute_suite',return_value=0) as run, \
             patch('flowfield.runtime.Runtime.resolve',return_value={'backend':'native'}):
            flowfield_main(['search','--problem','P005','--mesh','medium','--optimizer','random',
                            '--budget','1','--output',str(self.root/'search'),'--backend','native'])
        args, problems = run.call_args.args[:2]
        self.assertEqual(args.method, 'random')
        self.assertIsInstance(problems[0],P005Problem)
        self.assertEqual(problems[0].config.mesh,'medium')

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required for actual detector unit tests')
    def test_compiled_detector_connectivity_and_persistence(self):
        from flowfield.problems.level1_small_continuous import p005
        inc=Path(p005.__file__).parent/'monitor'
        source=self.root/'detector.cpp'
        source.write_text(r'''
#include "Detector.H"
#include <cassert>
int main(){
 using namespace flowfieldP005;
 std::vector<std::vector<int>> n{{1},{0,2},{1,3},{2}};
 std::vector<double> v(4,1),a{1,1,1,0};
 std::vector<bool> electrode{true,false,false,false},port{false,false,false,true};
 auto c=components(a,v,n,electrode,port,.5);
 assert(c.attached==3 && c.detached==0);
 Event e;e.update(0,c,.5,.5,.2);assert(e.armed&&!e.confirmed);
 a={1,0,1,0};c=components(a,v,n,electrode,port,.5);
 assert(c.attached==1&&c.detached==1);
 e.update(.1,c,.5,.5,.2);assert(!e.confirmed);
 // Reconnections reset the persistence window.
 auto attached=components({1,1,1,0},v,n,electrode,port,.5);
 e.update(.2,attached,.5,.5,.2);assert(e.candidate<0);
 e.update(.3,c,.5,.5,.2);e.update(.4,c,.5,.5,.2);e.update(.5,c,.5,.5,.2);
 assert(e.confirmed&&e.eventTime==.3&&e.bracketStart==.2);
 Event unarmed;unarmed.update(1,c,2,.5,.2);assert(!unarmed.armed&&!unarmed.confirmed);
 c=components({0,0,1,1},v,n,electrode,port,.5);assert(c.detached==0);
 c=components({1,.6,1,0},v,n,electrode,port,.5);assert(c.detached==0);
 c=components({1,.4,1,0},v,n,electrode,port,.5);assert(c.detached==1);
}
''')
        subprocess.run(['c++','-std=c++14','-I',str(inc),str(source),'-o',str(self.root/'test')],check=True,capture_output=True)
        subprocess.run([str(self.root/'test')],check=True,capture_output=True)
