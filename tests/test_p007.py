"""Numerical structure and integration tests for the 2D Jin reconstruction."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from flowfield import create_problem, CaseEvaluator
from flowfield.problems.level4_high_dimensional_topology.p007 import TopologyConfig, P007Problem
from flowfield.problems.level4_high_dimensional_topology.p007.design import operators,grid,projection
from flowfield.problems.level4_high_dimensional_topology.p007.metrics import scalarize,values
from flowfield.optimization import optimize
from flowfield.runtime import SimulationError
from flowfield.stability.runner import make_plan
from flowfield.stability.profiles import stricter


class P007Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.p=P007Problem(config=TopologyConfig(design_nx=4,design_ny=4))
    def tearDown(self):self.temp.cleanup()

    def test_published_values_and_explicit_estimates(self):
        c=TopologyConfig()
        self.assertEqual((c.reynolds,c.prandtl,c.volume_limit,c.ramp_q,c.projection_beta),(130,6.78,.635,.01,8))
        self.assertEqual(c.dimension,128);self.assertEqual(c.weights,(.8,.2))
        for key,value in (('mesh','tiny'),('scheme',6),('reynolds',float('nan')),('darcy',0),
                          ('outer_tolerance',1e-3),('mirror_y',1),('design_ny',3),('iterations',True)):
            with self.subTest(key=key), self.assertRaises(ValueError):replace(c,**{key:value})
        for key in ('darcy','heat_coefficient','filter_radius'):
            self.assertNotEqual(c.fingerprint,replace(c,**{key:getattr(c,key)*2}).fingerprint)

    def test_mesh_port_area_and_control_dimension_stay_fixed_under_refinement(self):
        counts=[]
        for mesh in ('coarse','medium','fine'):
            c=replace(self.p.config,mesh=mesh);x,y=map(np.asarray,grid(c))
            dy=np.diff(y);cy=(y[1:]+y[:-1])/2
            self.assertAlmostEqual(float(dy[(cy>2.425)&(cy<2.575)].sum()),.15)
            self.assertEqual(c.dimension,8)
            counts.append((len(x)-1)*(len(y)-1))
        self.assertEqual(counts,[4480,10080,17920])

    def test_helmholtz_preserves_constant_volume_and_maximum_principle(self):
        c=self.p.config;solve,areas,mapping=operators(c)
        raw=np.random.default_rng(12).uniform(0,1,c.dimension)[mapping]
        filtered=solve.solve(areas*raw)
        self.assertAlmostEqual(float(areas@filtered),float(areas@raw),places=10)
        self.assertGreaterEqual(filtered.min(),raw.min()-1e-12)
        self.assertLessEqual(filtered.max(),raw.max()+1e-12)
        self.assertLess(np.var(filtered),np.var(raw))
        g=self.p.decode(self.p.initial_design());self.assertAlmostEqual(g['volume_fraction'],.5)
        np.testing.assert_allclose(g['brinkman'],c.alpha_s*c.ramp_q*.5/(c.ramp_q+.5))
        for raw,target in ((0,0),(1,1)):
            self.assertAlmostEqual(self.p.decode([raw]*c.dimension)['volume_fraction'],target)

    def test_design_mirroring_and_volume_chain_rule(self):
        x=np.random.default_rng(3).uniform(.35,.65,self.p.dimension)
        g=self.p.decode(x);xx,yy=grid(self.p.config)
        gamma=np.array(g['indicator']).reshape(len(yy)-1,len(xx)-1)
        np.testing.assert_allclose(gamma,gamma[::-1],atol=1e-12)
        analytic=self.p.volume_gradient(x);eps=1e-6
        for i in range(len(x)):
            xp=x.copy();xm=x.copy();xp[i]+=eps;xm[i]-=eps
            fd=(self.p.decode(xp)['volume_fraction']-self.p.decode(xm)['volume_fraction'])/(2*eps)
            self.assertAlmostEqual(analytic[i],fd,places=7)

    def test_volume_repair_is_feasible_and_does_not_change_feasible_rows(self):
        rows=np.array([self.p.initial_design(),[1.]*self.p.dimension])
        repaired=self.p.repair_candidates(rows)
        np.testing.assert_array_equal(rows[0],repaired[0])
        self.assertTrue(np.all((repaired>=0)&(repaired<=rows)))
        self.assertLessEqual(self.p.decode(repaired[1])['volume_fraction'],self.p.config.volume_limit+1e-12)
        with self.assertRaises(ValueError):self.p.repair_candidates(rows,anchor=np.ones(self.p.dimension))

    def test_five_objectives_use_raw_paper_weights_and_explicit_volume_constraint(self):
        for i,(w1,w2) in enumerate(((.25,.75),(.4,.6),(.6,.4),(.8,.2),(1,0)),1):
            c=replace(self.p.config,scheme=i);r=scalarize(3,2,.5,c)
            self.assertAlmostEqual(r['objective'],-w1*3+w2*2)
            self.assertTrue(r['feasible'])
            excess=scalarize(3,2,.7,c)
            self.assertFalse(excess['feasible']);self.assertGreater(excess['objective'],r['objective'])

    def test_sources_and_boundaries_are_pinned_and_case_is_readable_without_solver(self):
        case=self.p.write_case(self.p.initial_design(),self.root/'case')
        CaseEvaluator(self.p.case_definition)._load_case(case)
        m=json.loads((case/'case.json').read_text());self.assertFalse(m['paper_validated'])
        self.assertIn('pressureInletVelocity',(case/'0/U').read_text())
        self.assertIn('gamma*fvm::div(phi,T)',(case/'solver/flowfieldP007Foam.C').read_text())
        (case/'0/T').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Case input changed'):CaseEvaluator(self.p.case_definition)._load_case(case)

    def test_stability_plan_and_tighter_numerics_preserve_physics(self):
        plan=make_plan(['P007'],studies=['repeat','mesh','numerics'])
        self.assertEqual(plan['case_count'],5)
        c=self.p.config;t=stricter('P007',c)
        for key in ('reynolds','prandtl','darcy','heat_coefficient','filter_radius','pumping_power'):
            self.assertEqual(getattr(c,key),getattr(t,key))

    def test_failed_baseline_stops_optimizer_even_with_failure_penalty(self):
        p=self.p
        with patch.object(p,'evaluate_raw',side_effect=SimulationError('unqualified')) as evaluate:
            with self.assertRaisesRegex(SimulationError,'baseline did not qualify'):
                optimize(p,optimizer='random',budget=3,output=self.root/'optimization',with_ioh=False,failure_penalty=100)
        self.assertEqual(evaluate.call_count,1)
        summary=json.loads((self.root/'optimization/summary.json').read_text())
        self.assertEqual(summary['attempts'],1)
        self.assertIsNone(summary['best_objective'])

    def test_field_parser_rejects_truncated_and_nonfinite_output(self):
        np.testing.assert_equal(values('value uniform 0;', 'value',3),[0,0,0])
        np.testing.assert_equal(values('value nonuniform List<vector> 2((1 2 3) (4 5 6));','value',2,3),[[1,2,3],[4,5,6]])
        for text in ('value uniform nan;','value nonuniform List<scalar> 3(1 2);'):
            with self.assertRaises(ValueError):values(text,'value',3)


if __name__=='__main__':unittest.main()
