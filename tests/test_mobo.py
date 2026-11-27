"""Actual BoTorch acquisitions, constraint/sign semantics and journal replay."""

import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowfield.problem import FlowFieldProblem
from flowfield.optimization import optimize, main, QLogNEHVISampler, QLogNParEGOSampler
from flowfield.optimization.mobo import MOBO_METHODS, mobo_settings
from flowfield.optimization.registry import resolve_optimizer
from flowfield.optimization.state import ResumeError, validate_experiment
from flowfield.runtime import SimulationError

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch','botorch','gpytorch'))


class Tradeoff(FlowFieldProblem):
    def __init__(self, *, failures=(), scalar_sign=1.):
        super().__init__('MoboTest',2,[0.,0.],[1.,1.])
        self.calls=[]
        self.failures,self.scalar_sign=failures,scalar_sign

    def decode(self,x):
        return self.validate_design(x)

    def evaluate_raw(self,x):
        self.calls.append(list(x))
        if len(self.calls) in self.failures:
            raise SimulationError('Deliberate CFD failure')
        a,b=x
        return dict(status='ok',objective=self.scalar_sign*(a+10*b),
                    objectives={'right':(a-1)**2+b*b,'left':a*a+b*b},
                    constraints={'limit':b-.8},metadata={'valid_cfd':True})


class MoboPlanningTests(unittest.TestCase):
    def test_settings_validation_and_optional_imports(self):
        for change in (dict(initial_points=1),dict(candidates=1),dict(mc_samples=17),
                       dict(mc_samples=True),dict(reference_point=[1,2]),
                       dict(reference_point={'a':1,'b':math.nan})):
            with self.subTest(change=change),self.assertRaises(ValueError):
                mobo_settings('qlognehvi',**change)
        with self.assertRaises(ValueError):
            mobo_settings('qlognparego',reference_point={'a':1,'b':1})
        p=Tradeoff();p.variable_type='integer'
        for method in MOBO_METHODS:
            with self.assertRaises(ValueError):resolve_optimizer(p,method)
        subprocess.run([sys.executable,'-c',
            'import sys; from flowfield.optimization import list_optimizers; '
            'assert "qlognehvi" in list_optimizers(); assert "torch" not in sys.modules'],check=True)

    def test_both_clis_plan_full_dimension_without_running_cfd(self):
        from flowfield.benchmarking.runner import main as benchmark_main
        with tempfile.TemporaryDirectory() as tmp:
            for method in MOBO_METHODS:
                stream=io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(main(['--problem','P003','--dimension','500','--optimizer',method,
                        '--bo-initial-points','3','--bo-candidates','16','--mobo-mc-samples','16',
                        '--output',tmp+'/preview','--dry-run']),0)
                entry=json.loads(stream.getvalue())['problems'][0]
                self.assertEqual(entry['dimension'],500)
                self.assertEqual(entry['optimization_mode'],'multi_objective')
                self.assertEqual(entry['optimizer_settings']['mc_samples'],16)
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(benchmark_main(['--problem','P003','--dimension','6000',
                    '--optimizers',*MOBO_METHODS,'--seeds','42','43',
                    '--mobo-reference-point','{"negative_power_domain_W":0,"current_std_A_m2":1000}',
                    '--output',tmp+'/matrix','--dry-run']),0)
            plan=json.loads(stream.getvalue())
            self.assertEqual(plan['run_count'],4)
            self.assertEqual(plan['instances'][0]['dimension'],6000)
            self.assertIsNone(plan['optimizer_options']['qlognparego']['mobo_reference_point'])
            self.assertEqual(list(Path(tmp).iterdir()),[])


@unittest.skipUnless(AVAILABLE,'optional BoTorch dependencies unavailable')
class MoboTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def run_problem(self,p,folder,method,**options):
        return optimize(p,output=self.root/folder,optimizer=method,
                        **(dict(budget=6,bo_initial_points=3,bo_candidates=16,
                                mobo_mc_samples=16,with_ioh=False)|options))

    def sampler(self,cls,problem=None,**options):
        p=problem or Tradeoff()
        return cls(p,seed=0,initial=p.initial_design(),bo_initial_points=3,
                   bo_candidates=16,mobo_mc_samples=16,**options)

    def test_real_acquisitions_ignore_scalar_value_and_preserve_rng(self):
        import numpy as np
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.acquisition.multi_objective.parego import qLogNParEGO
        for cls,acq_type in ((QLogNEHVISampler,qLogNoisyExpectedHypervolumeImprovement),
                             (QLogNParEGOSampler,qLogNParEGO)):
            first,second=Tradeoff(),Tradeoff(scalar_sign=-1000.)
            a,b=self.sampler(cls,first),self.sampler(cls,second)
            numpy_state=np.random.get_state()
            torch_state=torch.random.get_rng_state().clone()
            threads=torch.get_num_threads()
            for sampler,p in ((a,first),(b,second)):
                sampler.initialize(p.initial_design(),p.evaluate_raw(p.initial_design()))
                for _ in range(4):
                    batch=sampler.ask()
                    with self.assertRaises(RuntimeError):sampler.ask()
                    with self.assertRaises(ValueError):sampler.tell(batch,[])
                    sampler.tell(batch,[p.evaluate_raw(batch[0])])
                self.assertIsInstance(sampler.acquisition,acq_type)
                self.assertTrue(all(0<=v<=1 for row in p.calls for v in row))
                self.assertEqual(len(set(map(tuple,p.calls))),len(p.calls))
            self.assertEqual(first.calls,second.calls)
            np.testing.assert_array_equal(numpy_state[1],np.random.get_state()[1])
            self.assertEqual(numpy_state[2:],np.random.get_state()[2:])
            self.assertTrue(torch.equal(torch_state,torch.random.get_rng_state()))
            self.assertEqual(threads,torch.get_num_threads())

    def test_objective_sign_constraint_direction_reference_and_weights(self):
        import torch
        for cls in (QLogNEHVISampler,QLogNParEGOSampler):
            sampler=self.sampler(cls,**({'mobo_reference_point':{'left':4.,'right':5.}} if cls==QLogNEHVISampler else {}))
            p=sampler.problem
            raw=p.evaluate_raw(p.initial_design())
            sampler.initialize(p.initial_design(),raw)
            self.assertEqual(sampler.y[0],[-raw['objectives']['left'],-raw['objectives']['right'],raw['constraints']['limit']])
            weights=[]
            for _ in range(4):
                batch=sampler.ask()
                if sampler.acquisition is not None:
                    constraints=(sampler.acquisition.constraints if cls==QLogNEHVISampler
                                 else sampler.acquisition._constraints)
                    constraint=constraints[0]
                    values=constraint(torch.tensor([[1.,2.,-.1],[1.,2.,.1]],dtype=torch.double))
                    self.assertLess(values[0],0)
                    self.assertGreater(values[1],0)
                    if cls==QLogNEHVISampler:
                        self.assertEqual(sampler.reference_point.tolist(),[-4.,-5.])
                        self.assertEqual(sampler.diagnostics['mobo_acquisition_reference_point'],{'left':4.,'right':5.})
                    else:
                        weights.append(sampler.weights)
                        self.assertAlmostEqual(sum(sampler.weights),1.)
                        self.assertTrue(all(v>0 for v in sampler.weights))
                sampler.tell(batch,[p.evaluate_raw(batch[0])])
            if weights:self.assertNotEqual(weights[0],weights[1])

    def test_schema_errors_do_not_become_scalar_fallback(self):
        sampler=self.sampler(QLogNEHVISampler)
        initial=sampler.initial
        for result in (dict(objective=0.),dict(objectives={'a':1.}),
                       dict(objectives={'a':True,'b':2}),dict(objectives={'a':math.inf,'b':2})):
            with self.assertRaises(ValueError):sampler.initialize(initial,result)
        self.assertEqual(sampler.seen,[])
        sampler.initialize(initial,sampler.problem.evaluate_raw(initial))
        batch=sampler.ask()
        for result in (dict(objectives={'left':1.,'right':2.},constraints={}),
                       dict(objectives={'left':1.,'new':2.},constraints={'limit':-.1})):
            with self.assertRaises(ValueError):sampler.tell(batch,[result])
        self.assertEqual(len(sampler.seen),1)

    def test_failures_count_but_do_not_enter_gp_or_pareto_set(self):
        for method in MOBO_METHODS:
            problem=Tradeoff(failures=(1,2))
            result=self.run_problem(problem,method,method,failure_penalty=1e8)
            self.assertEqual(result['attempts'],6)
            self.assertEqual(result['failed_evaluations'],2)
            self.assertEqual(result['mobo_excluded_attempts'],2)
            self.assertEqual(result['mobo_valid_observations'],4)
            archive=json.loads((self.root/method/'pareto.json').read_text())
            rows=[json.loads(s) for s in (self.root/method/'evaluations.jsonl').read_text().splitlines()]
            feasible=[r for r in rows if r['feasible']]
            def dominates(a,b):
                x,y=a['result']['objectives'],b['result']['objectives']
                return all(x[k]<=y[k] for k in x) and any(x[k]<y[k] for k in x)
            expected=[r for r in feasible if not any(dominates(q,r) for q in feasible)]
            self.assertEqual({tuple(r['x']) for r in archive['solutions']},{tuple(r['x']) for r in expected})

    def test_initialization_budget_never_fits_an_unused_model(self):
        from flowfield.optimization.mobo import BoTorchMultiObjectiveSampler
        with patch.object(BoTorchMultiObjectiveSampler,'_fit',side_effect=AssertionError('Unused GP fit')):
            for method in MOBO_METHODS:
                for budget in (1,3):
                    result=self.run_problem(Tradeoff(),f'{method}-{budget}',method,budget=budget)
                    self.assertEqual(result['attempts'],budget)

    def test_resume_and_extension_match_uninterrupted_proposals(self):
        for method in MOBO_METHODS:
            reference=Tradeoff()
            self.run_problem(reference,method+'-reference',method)
            before=Tradeoff()
            def stop(name,index,budget,status):
                if index==4 and status=='ok':raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                self.run_problem(before,method,method,progress=stop)
            after=Tradeoff()
            result=self.run_problem(after,method,method,resume=True)
            self.assertEqual(before.calls+after.calls,reference.calls)
            self.assertEqual(result['new_attempts'],2)
            def front(folder):
                saved=json.loads((self.root/folder/'pareto.json').read_text())
                return [(r['x'],r['result']) for r in saved['solutions']]
            self.assertEqual(front(method),front(method+'-reference'))
            shorter=Tradeoff()
            self.run_problem(shorter,method+'-extend',method,budget=3)
            continued=Tradeoff()
            self.run_problem(continued,method+'-extend',method,resume=True,extend_budget=6)
            self.assertEqual(shorter.calls+continued.calls,reference.calls)
            saved=json.loads((self.root/method/'experiment.json').read_text())
            for key in ('torch','botorch','gpytorch','linear-operator'):
                changed=copy.deepcopy(saved);changed['versions'][key]='changed'
                with self.assertRaisesRegex(ResumeError,key):validate_experiment(saved,changed)
            changed=copy.deepcopy(saved);changed['optimizer_settings']['mc_samples']=32
            with self.assertRaisesRegex(ResumeError,'optimizer_settings'):validate_experiment(saved,changed)

    def test_cli_settings_persist_and_resume_without_repeating_options(self):
        for method in MOBO_METHODS:
            problem=Tradeoff();problem.name='P000'
            output=self.root/method
            with patch('flowfield.optimization.cli._make_problem',return_value=problem), \
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--problem','P000','--optimizer',method,'--budget','4',
                    '--bo-initial-points','3','--bo-candidates','16','--mobo-mc-samples','16',
                    '--output',str(output)]),0)
                self.assertEqual(main(['--resume','--extend-budget','5','--output',str(output)]),0)
            self.assertEqual(len(problem.calls),5)
            saved=json.loads((output/'P000/experiment.json').read_text())
            self.assertEqual(saved['optimizer_settings']['mc_samples'],16)
            self.assertEqual(saved['optimization_mode'],'multi_objective')

    def test_benchmark_matrix_runs_both_methods_with_real_p003_mapping(self):
        from flowfield import P003Problem
        from flowfield.benchmarking.runner import main as benchmark_main
        calls=[]
        def evaluate(problem,x):
            calls.append((problem.dimension,list(x)))
            volume=problem.decode(x)['volume_fraction']
            self.assertLessEqual(volume,.5+1e-12)
            mean=sum(x)/len(x)
            return dict(status='ok',objective=mean,
                objectives={'left':mean,'right':sum((v-.02)**2 for v in x)/len(x)},
                constraints={'channel_volume':volume-.5},metadata={'valid_cfd':True})
        stream=io.StringIO()
        with patch.object(P003Problem,'evaluate_raw',autospec=True,side_effect=evaluate), \
             contextlib.redirect_stdout(stream):
            self.assertEqual(benchmark_main(['--problem','P003','--dimension','500',
                '--optimizers',*MOBO_METHODS,'--seeds','42','43','--budget','4',
                '--bo-initial-points','3','--bo-candidates','16','--mobo-mc-samples','16',
                '--no-ioh','--output',str(self.root/'matrix')]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(calls),16)
        self.assertTrue(all(n==len(x)==500 for n,x in calls))
        self.assertEqual(len(result['runs']),4)
        for run in result['runs']:
            self.assertEqual(run['optimization_mode'],'multi_objective')
            self.assertEqual(run['scalar_score_role'],'diagnostic only')
            self.assertTrue((self.root/'matrix'/run['pareto_file']).is_file())

    def test_p003_full_dimension_repair_is_applied_before_model_scoring(self):
        import numpy as np
        from flowfield import P003Problem
        from flowfield.problems.level4_high_dimensional_topology.p003.scalability import scalable_config
        for n in (500,6000):
            p=P003Problem(config=scalable_config(dimension=n))
            for cls in (QLogNEHVISampler,QLogNParEGOSampler):
                sampler=self.sampler(cls,p)
                def result(x):
                    volume=p.decode(x)['volume_fraction']
                    self.assertLessEqual(volume,p.config.volume_limit+1e-12)
                    return dict(objective=100.,objectives={'negative_power_domain_W':-float(np.mean(x)),
                        'current_std_A_m2':float(np.std(x))},constraints={'channel_volume':volume-.5})
                sampler.initialize(p.initial_design(),result(p.initial_design()))
                for _ in range(3):
                    batch=sampler.ask()
                    self.assertEqual(len(batch[0]),n)
                    sampler.tell(batch,[result(batch[0])])
                self.assertIsNotNone(sampler.model)
                self.assertEqual(sampler.model.num_outputs,2)
                self.assertEqual(sampler.modeled_constraint_names,())
                self.assertTrue(sampler.proposal_metadata['applied'])
                # Every acquisition candidate, not just the winner, is feasible.
                original_score=sampler._score
                def score(points):
                    for x in points:result(x.tolist())
                    return original_score(points)
                with patch.object(sampler,'_score',side_effect=score):sampler.ask()


if __name__=='__main__':unittest.main()
