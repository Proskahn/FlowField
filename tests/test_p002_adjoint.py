"""Coupled residual derivatives, discrete-adjoint identity and GCMMA accounting.

These tests do not require Docker. Independent end-to-end CFD directional
checks are available in the CLI and are enabled by default for a new campaign.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(m) for m in ('numpy','scipy','mmapy','pymetis'))


@unittest.skipUnless(AVAILABLE,'optional GCMMA dependencies unavailable')
class AdjointTests(unittest.TestCase):
    def test_adjoint_includes_direct_and_coupled_state_effects(self):
        import numpy as np
        from scipy import sparse
        from scipy.optimize import root
        from flowfield.problems.level2_medium_parametric.p002.adjoint import solve_adjoint
        x = np.array([.2,.3])
        def state(v):
            return root(lambda y:[y[0]**2+y[1]-v[0],y[0]+3*y[1]-v[1]],[.1,.1]).x
        y = state(x)
        ry = np.array([[2*y[0],1.],[1.,3.]])
        rx = -np.eye(2)
        # J1 = y0 + x1*y1; J2 = y1^2 + x0^2.
        jy = np.array([[1.,x[1]],[0.,2*y[1]]])
        jx = np.array([[0.,y[1]],[2*x[0],0.]])
        g,info,_ = solve_adjoint(sparse.csr_matrix(np.hstack([ry,rx])),np.hstack([jy,jx]),np.array([2.,.5]))
        def objective(v):
            z = state(v); return np.array([z[0]+v[1]*z[1],z[1]**2+v[0]**2])
        expected = np.column_stack([(objective(x+1e-5*d)-objective(x-1e-5*d))/2e-5 for d in np.eye(2)])
        np.testing.assert_allclose(g,expected,rtol=1e-6,atol=1e-8)
        self.assertLess(max(info['linear_relative_residual']),1e-8)

    def test_mesh_transition_is_rejected_and_smooth_geometry_matches_volume(self):
        import numpy as np
        from flowfield.problems.level2_medium_parametric.p002 import BASELINE_X,PAPER_OPTIMUM_X,PEMConfig
        from flowfield.problems.level2_medium_parametric.p002.adjoint_residual import mesh_derivatives,NondifferentiableMesh
        with self.assertRaises(NondifferentiableMesh):
            mesh_derivatives(BASELINE_X,PEMConfig())
        mesh,jac = mesh_derivatives(PAPER_OPTIMUM_X,PEMConfig())
        self.assertEqual(jac.shape,(sum(len(mesh[a]) for a in 'xyz'),7))
        self.assertTrue(np.all(np.isfinite(jac)))
        # External dimensions and therefore total cell volume are invariant.
        start = 0
        for a in 'xyz':
            np.testing.assert_allclose(jac[start],0.,atol=1e-9)
            np.testing.assert_allclose(jac[start+len(mesh[a])-1],0.,atol=1e-9)
            start += len(mesh[a])

    def test_full_coupled_residual_state_and_geometry_jacobians(self):
        import numpy as np
        from flowfield.problems.level2_medium_parametric.p002 import PAPER_OPTIMUM_X,PEMConfig
        from flowfield.problems.level2_medium_parametric.p002.adjoint_residual import Residual
        config = PEMConfig()
        model = Residual(PAPER_OPTIMUM_X,config)
        y = np.zeros(model.size)
        mat = model.material
        cells = np.asarray(model.mesh['cells'])
        field_values = dict(Ux=np.full(model.nc,.1),Uy=np.full(model.nc,.01),Uz=np.full(model.nc,.001),
            p=100.-cells[:,0],Cgas=10.+cells[:,1]*.02,T=334.+cells[:,2]*.01,
            phiS=np.where(mat>=5,2.2,.01),phiE=np.full(model.nc,.13),
            iVol=np.where(mat==5,5e8,-5e8),massPhi=np.full(model.nf,1e-6))
        for name,part in model.slices.items():
            y[part] = field_values[name][model.indices[name]]
        residual,obj = model.evaluate(y)
        rng = np.random.default_rng(812)
        v = rng.normal(size=model.size)*model.scales
        h = 1e-6
        plus,jplus = model.evaluate(y+h*v,derivatives=False)
        minus,jminus = model.evaluate(y-h*v,derivatives=False)
        expected = residual.j[:,:model.size]@v
        actual = (plus.v-minus.v)/(2*h)
        scale = np.asarray(abs(residual.j[:,:model.size]).multiply(model.scales).sum(axis=1)).ravel()
        np.testing.assert_allclose(actual/scale,expected/scale,rtol=2e-6,atol=1e-6)
        np.testing.assert_allclose((jplus.v-jminus.v)/(2*h),obj.j[:,:model.size]@v,rtol=1e-7,atol=1e-7)
        direction = np.array([1.,-1.,.4,.3,-.2,.5,-.3])
        x = np.asarray(PAPER_OPTIMUM_X)
        plus,jplus = Residual(x+h*direction,config,differentiate_geometry=False).evaluate(y,derivatives=False)
        minus,jminus = Residual(x-h*direction,config,differentiate_geometry=False).evaluate(y,derivatives=False)
        expected = residual.j[:,model.size:]@direction
        actual = (plus.v-minus.v)/(2*h)
        np.testing.assert_allclose(actual/scale,expected/scale,rtol=5e-5,atol=2e-6)
        np.testing.assert_allclose((jplus.v-jminus.v)/(2*h),obj.j[:,model.size:]@direction,rtol=5e-5,atol=1e-7)

    def test_scalar_weights_match_cmaes(self):
        from flowfield.problems.level2_medium_parametric.p002.adjoint import weights
        from flowfield.problems.level2_medium_parametric.p002.config import PEMConfig,scalar_objective
        for mode in ('total_power_w','voltage_v','pressure_sum_pa'):
            config = PEMConfig(scalarization=mode)
            self.assertAlmostEqual(float(weights(config)@[2.24,140.]),scalar_objective(2.24,140.,config))

    def test_rejected_trial_shrinks_without_fake_objective(self):
        import numpy as np
        from flowfield.optimization.gcmma import GCMMA,GCMMAOptions
        optimizer = GCMMA([.3],options=GCMMAOptions(move=.1,max_inner=2))
        optimizer.propose(1.,[-1.],[-1.],[[0.]])
        before = optimizer.f.copy()
        trial = optimizer.reject_trial()
        self.assertLessEqual(abs(trial[0]-.3),.05)
        np.testing.assert_array_equal(before,optimizer.f)
        with self.assertRaisesRegex(RuntimeError,'exhausted'):
            optimizer.reject_trial()


class Toy:
    def __init__(self):
        self.calls = []

    def evaluate(self,x,config,root):
        import numpy as np
        self.calls.append(x)
        voltage = 2.+.01*float(np.dot(x,x))
        pressure = 100.+float(sum(x))
        from flowfield.problems.level2_medium_parametric.p002.config import scalar_objective
        return dict(status='ok',objective=scalar_objective(voltage,pressure,config),x=x,
                    objectives=dict(voltage_V=voltage,pressure_sum_Pa=pressure),
                    case_directory=str(root/str(len(self.calls))))

    def differentiate(self,result):
        import numpy as np
        x = np.asarray(result['x'])
        return dict(x=x.tolist(),case_directory=result['case_directory'],gradient=(1.21*.02*x+2.5e-7).tolist(),
                    objective_gradients=dict(voltage_V=(.02*x).tolist(),pressure_sum_Pa=[1.]*7))


@unittest.skipUnless(AVAILABLE,'optional GCMMA dependencies unavailable')
class CampaignTests(unittest.TestCase):
    def test_failed_cfd_trial_is_counted_then_rejected_without_a_penalty(self):
        from flowfield.problems.level2_medium_parametric.p002.gcmma import run,Options
        from flowfield.runtime import SimulationError
        with tempfile.TemporaryDirectory() as directory:
            toy = Toy()
            calls = []
            def evaluate(x,config,root):
                calls.append(x)
                if len(calls)==2:
                    raise SimulationError('deliberate invalid CFD trial')
                return toy.evaluate(x,config,root)
            output = Path(directory)/'run'
            summary = run(output,options=Options(iterations=1,move=.001,check_gradients=False),
                          max_evaluations=3,evaluate=evaluate,differentiate=toy.differentiate)
            self.assertEqual(summary['status'],'completed_schedule')
            self.assertEqual(summary['cfd_attempts'],3)
            self.assertEqual(summary['accepted_iterations'],1)
            rows = [json.loads(line) for line in (output/'evaluations.jsonl').read_text().splitlines()]
            self.assertEqual([r['valid'] for r in rows],[True,False,True])
            self.assertIsNone(rows[1]['objective'])

    def test_budget_resume_and_default_checks_use_real_adjoint_gradient(self):
        from flowfield.problems.level2_medium_parametric.p002.gcmma import run,Options
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            toy = Toy()
            options = Options(iterations=2,move=.001)
            first = run(output,options=options,max_evaluations=10,evaluate=toy.evaluate,differentiate=toy.differentiate)
            self.assertEqual(first['status'],'budget_exhausted')
            self.assertEqual(first['accepted_iterations'],1)
            self.assertTrue(first['gradient_checked'])
            self.assertEqual(len(toy.calls),10) # baseline + eight checks + one trial
            last = run(output,options=options,max_evaluations=11,resume=True,
                       evaluate=toy.evaluate,differentiate=toy.differentiate)
            self.assertEqual(last['status'],'completed_schedule')
            self.assertEqual(last['accepted_iterations'],2)
            self.assertEqual(len(toy.calls),11)
            self.assertEqual(last['new_adjoint_evaluations'],0)
            self.assertEqual(len((output/'evaluations.jsonl').read_text().splitlines()),11)
            check = json.loads((output/'gradient-check.json').read_text())
            self.assertTrue(check['passed'])

    def test_failed_gradient_check_blocks_optimization(self):
        from flowfield.problems.level2_medium_parametric.p002.gcmma import run,Options
        with tempfile.TemporaryDirectory() as directory:
            toy = Toy()
            def wrong(result):
                d = toy.differentiate(result); d['objective_gradients']['voltage_V']=[1.]*7
                return d
            output = Path(directory)/'run'
            with self.assertRaisesRegex(RuntimeError,'directional checks'):
                run(output,options=Options(iterations=1),evaluate=toy.evaluate,differentiate=wrong)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['accepted_iterations'],0)
            self.assertEqual(len(toy.calls),9)

    def test_cli_dry_run_leaves_no_output(self):
        from flowfield.problems.level2_medium_parametric.p002.cli import main
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with patch('builtins.print'):
                self.assertEqual(main(['gcmma','--output',str(output),'--dry-run']),0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
