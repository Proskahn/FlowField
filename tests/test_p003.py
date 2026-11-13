"""P003 mathematical/contract tests. CFD verification is the explicit smoke command."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import get_problem, P003Problem, TopologyConfig
from flowfield.benchmarking.ioh import make_ioh_problem, analyzer_logger
from flowfield.runtime import SimulationError
from flowfield.problems.level4_high_dimensional_topology.p003.config import current_density, CONTINUATION
from flowfield.problems.level4_high_dimensional_topology.p003.design import grid, inverse_project, project, stencil
from flowfield.problems.level4_high_dimensional_topology.p003.case import generate_case, INPUTS
from flowfield.problems.level4_high_dimensional_topology.p003.evaluation import run_case
from flowfield.problems.level4_high_dimensional_topology.p003.metrics import scalarize, values, history, COLUMNS

PREFIX='flowfield.problems.level4_high_dimensional_topology.p003'


class TopologyTests(unittest.TestCase):
    def test_registry_and_uniform_initialization(self):
        p=get_problem('P003');x=p.initial_design();g=p.decode(x)
        self.assertIsInstance(p,P003Problem);self.assertEqual(p.dimension,2145)
        self.assertEqual(g['node_shape'],[65,33]);self.assertAlmostEqual(g['volume_fraction'],.5,14)
        self.assertAlmostEqual(x[0],math.log(2)/50,14)
        self.assertGreater(p.decode([.5]*p.dimension)['volume_fraction'],.999999)

    def test_paper_projection_and_filter_preserve_constants(self):
        for beta in (0,1,50):
            for target in (0,.1,.5,.9,1):
                self.assertAlmostEqual(project(inverse_project(target,beta),beta),target,14)
        p=P003Problem();g=p.decode([.01]*p.dimension)
        self.assertLess(max(abs(v-.01) for v in g['filtered']),1e-15)
        self.assertAlmostEqual(g['volume_fraction'],1-math.exp(-.5)+.01*math.exp(-50),13)
        rows,_=stencil(p.config)
        self.assertTrue(all(abs(sum(w for _,w in row)-1)<1e-14 for row in rows))

    def test_volume_gradient_against_directional_finite_difference(self):
        p=P003Problem();x=[.01+.008*math.sin(i)**2 for i in range(p.dimension)]
        direction=[math.sin(i*.13) for i in range(p.dimension)];h=1e-6
        numerical=(p.decode([a+h*b for a,b in zip(x,direction)])['volume_fraction']-
                   p.decode([a-h*b for a,b in zip(x,direction)])['volume_fraction'])/(2*h)
        exact=sum(a*b for a,b in zip(p.volume_gradient(x),direction))
        self.assertAlmostEqual(numerical,exact,8)

    def test_weighted_volume_on_nonuniform_mesh_and_parallel_seed(self):
        p=P003Problem();g=p.decode(p.preset('parallel'));a=g['cell_areas_m2']
        self.assertNotEqual(min(a),max(a))
        self.assertAlmostEqual(sum(a),.02236**2/2,15)
        self.assertAlmostEqual(g['volume_fraction'],.5,13)
        self.assertLess(min(g['indicator']),1e-12);self.assertGreater(max(g['indicator']),.8)

    def test_depth_average_endpoints_and_continuation_are_explicit(self):
        p=P003Problem();a=p.config.interpolation
        self.assertAlmostEqual(a['eps_rib'],.8*.00038/.00138,14)
        self.assertAlmostEqual(a['diffusion_channel'],(.8**1.5*.00038+.001)/.00138,14)
        for name,s in [('all_rib',0),('all_channel',1)]:
            g=p.decode(p.preset(name))
            self.assertAlmostEqual(g['porosity'][0],a['eps_channel'] if s else a['eps_rib'])
            self.assertAlmostEqual(g['brinkman_pa_s_m2'][0],0 if s else a['alpha_rib'])
        identities=set()
        for stage,(power,q) in enumerate(CONTINUATION):
            conf=replace(p.config,stage=stage);identities.add(P003Problem(config=conf).ioh_name)
            self.assertEqual(conf.interpolation['simp_p'],power);self.assertEqual(conf.interpolation['ramp_q'],q)
        self.assertEqual(len(identities),4)

    def test_invalid_config_and_design_rejected_before_cfd(self):
        for kwargs in ({'design_nx':2},{'stage':True},{'stage':4},{'projection_beta':-1},
                       {'inlet_pressure_pa':float('nan')},{'oxygen_inlet':.99},
                       {'balance_tolerance':.1},{'power_scale_w':0},{'mesh':'fake'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):TopologyConfig(**kwargs)
        p=P003Problem()
        for x in ([],[.1]*(p.dimension-1),[math.nan]*p.dimension,[1.01]*p.dimension,[True]*p.dimension):
            with self.assertRaises(ValueError):p.decode(x)
        for kwargs in ({'mesh':'medium'},{'power_scale_w':.2},{'design_nx':64}):
            self.assertNotEqual(p.ioh_name,P003Problem(config=replace(p.config,**kwargs)).ioh_name)

    def test_electrochemistry_satisfies_implicit_butler_volmer(self):
        c=TopologyConfig();previous=-1
        for oxygen in (0,.001,.05,.228,1):
            j=current_density(oxygen,c);eta=c.open_circuit_voltage_v-c.cell_voltage_v-c.ohmic_resistance_ohm_m2*j
            z=96485.3*eta/(8.3144*c.temperature_k)
            rhs=.17*(oxygen/.032)/30*(math.exp(.5*z)-math.exp(-.5*z))
            self.assertAlmostEqual(j,rhs,9);self.assertGreaterEqual(j,previous);previous=j
            self.assertLessEqual(j,(1.22-.4)/7e-5)
        self.assertLess(current_density(.228,replace(c,cell_voltage_v=.6)),current_density(.228,c))

    def test_scalar_criteria_and_explicit_constraint_penalty(self):
        c=TopologyConfig();r=scalarize(.2,100,.6,c)
        self.assertAlmostEqual(r['unpenalized_objective'],-.2);self.assertAlmostEqual(r['objective'],.8)
        self.assertFalse(r['feasible']);self.assertAlmostEqual(r['constraints']['channel_volume'],.1)
        r=scalarize(.2,100,.5,replace(c,uniformity_weight=.5,current_std_scale_a_m2=200,power_scale_w=.4))
        self.assertAlmostEqual(r['objective'],-.25);self.assertTrue(r['feasible'])

    def test_case_port_geometry_and_fresh_input_guard(self):
        p=P003Problem();x,y=grid(p.config)
        self.assertIn(.0012,x);self.assertIn(p.config.width_m-.0012,x)
        self.assertAlmostEqual(y[-1],.02236)
        with tempfile.TemporaryDirectory() as tmp:
            case=generate_case(Path(tmp)/'case',p.decode(p.initial_design()),p.config)
            manifest=json.loads((case/'case.json').read_text());self.assertEqual(set(manifest['input_sha256']),INPUTS)
            self.assertEqual(manifest['cells'],2176)
            text=(case/'constant/polyMesh/boundary').read_text()
            self.assertIn('type symmetryPlane;',text);self.assertIn('type empty;',text)
            with self.assertRaises(FileExistsError):generate_case(case,p.decode(p.initial_design()),p.config)
            with (case/'0/oxygen').open('a') as f:f.write('// tampered\n')
            with self.assertRaisesRegex(ValueError,'input changed'):run_case(case)
            self.assertFalse((case/'.execution-started').exists())

    def test_raw_contract_and_failure_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=P003Problem(tmp);x=p.initial_design()
            r=dict(status='ok',objective=-.2,objectives={'negative_power_domain_W':-.2},constraints={},
                   case_directory=tmp,runtime={'backend':'test'})
            with patch(PREFIX+'.evaluation.evaluate',return_value=r) as backend:self.assertEqual(p(x),-.2)
            backend.assert_called_once();self.assertFalse(p.last_result['metadata']['paper_validated'])
            self.assertEqual(p.last_result['metadata']['dimension'],2145)
            p=P003Problem(tmp,failure_penalty=100)
            with patch(PREFIX+'.evaluation.evaluate',side_effect=SimulationError('failed')):r=p.evaluate_raw(x)
            self.assertEqual(r['objective'],100);self.assertFalse(r['metadata']['valid_cfd'])

    def test_field_and_history_corruption_detection(self):
        self.assertEqual(values('internalField nonuniform List<scalar> 2 (1 2);','internalField',2),[1,2])
        for text in ('internalField uniform nan;','internalField nonuniform List<scalar> 1 (1);'):
            with self.assertRaises(ValueError):values(text,'internalField',2)
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);folder=case/'postProcessing/p003';folder.mkdir(parents=True)
            path=folder/'history_0.csv'
            rows=[','.join([str(i)]+['0']*(len(COLUMNS)-1)) for i in range(1,6)]
            path.write_text(','.join(COLUMNS)+'\n'+'\n'.join(rows)+'\n');self.assertEqual(len(history(case)),5)
            path.write_text(path.read_text().replace('5,0','7,0'))
            with self.assertRaises(ValueError):history(case)

    @unittest.skipUnless(importlib.util.find_spec('ioh'),'optional IOH unavailable')
    def test_high_dimensional_actual_ioh_logger_calls_backend_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=P003Problem(tmp);objective=make_ioh_problem(p)
            r=dict(status='ok',objective=-.2,objectives={},constraints={},case_directory=tmp,runtime={'backend':'test'})
            with patch(PREFIX+'.evaluation.evaluate',return_value=r) as backend:
                with analyzer_logger(objective,Path(tmp)/'ioh',algorithm_name='unit-test'):
                    self.assertEqual(objective(p.initial_design()),-.2)
            backend.assert_called_once();self.assertEqual(objective.state.evaluations,1)
            self.assertEqual(objective.meta_data.n_variables,2145)
            data=next((Path(tmp)/'ioh').rglob('*.dat')).read_text().splitlines()
            self.assertEqual(len(data[0].split()),2147);self.assertEqual(len(data),2)


if __name__=='__main__':unittest.main()
