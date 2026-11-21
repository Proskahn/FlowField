"""Boundary tests: standalone cases, shared execution, optimizer plugins and analysis."""

from dataclasses import replace
import io
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import (CaseEvaluator, CFDProblem, FlowFieldProblem, PolyMesh,
                       create_problem, load_config, register_problem)
from flowfield.benchmarking import run_benchmark, compare_runs
from flowfield.optimization import Sampler, optimize, register_optimizer, main
from flowfield.optimization.state import ResumeError
from flowfield.runtime import SimulationError


class Toy(FlowFieldProblem):
    def __init__(self, output_root=None):
        super().__init__('ArchitectureToy', 2, [0, 0], [1, 1])
        self.calls = []

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        x = self.decode(x)
        self.calls.append(x)
        return dict(status='ok', objective=sum(x), objectives={'a': x[0], 'b': x[1]},
                    constraints={'limit': x[0] - .9}, metadata={})


class BatchSampler(Sampler):
    instances = []

    def __init__(self, problem, **options):
        self.generation = 1
        self.feedback = []
        self.baseline = None
        self.instances.append(self)

    def initialize(self, x, result):
        self.baseline = result

    def ask(self):
        if self.generation > 2:
            return []
        return [[.1 * self.generation, .2], [.2, .1 * self.generation]]

    def tell(self, designs, results):
        self.feedback.extend(results)
        self.generation += 1


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_cfd_cases_use_shared_lifecycle_and_preserve_single_execution(self):
        for name in ('P000', 'P001', 'P002', 'P003', 'P004-hydraulic', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010'):
            with self.subTest(problem=name):
                problem = create_problem(name)
                self.assertIsInstance(problem, CFDProblem)
                case = problem.write_case(problem.initial_design(), self.root / name)
                spec = problem.case_definition
                calls = []
                def execute(*args):
                    calls.append('solve')
                def extract(*args):
                    calls.append('extract')
                    return dict(objective=3., objectives={'physical': 3.}, constraints={})
                spec = replace(spec, execute=execute, extract=extract)
                with patch('flowfield.evaluation.Runtime.resolve', return_value={'backend': 'test'}) as resolve:
                    evaluator = CaseEvaluator(spec)
                    result = evaluator.run_case(case)
                    self.assertEqual(result['status'], 'ok')
                    self.assertEqual(calls, ['solve', 'extract'])
                    self.assertEqual(evaluator.extract_results(case)['objective'], 3.)
                    self.assertEqual(resolve.call_count, 1)  # Extraction never resolves a runtime.
                    with self.assertRaises(FileExistsError):
                        evaluator.run_case(case)
                    self.assertEqual(resolve.call_count, 1)
                self.assertEqual(json.loads((case / 'result.json').read_text())['objective'], 3.)

    def test_failure_and_interrupt_leave_evidence_and_cannot_be_retried_in_place(self):
        problem = create_problem('P000')
        for index, error in enumerate((SimulationError('solver stopped'), KeyboardInterrupt())):
            case = problem.write_case(problem.initial_design(), self.root / str(index))
            def fail(*args):
                args[-1]['mesh_validation'] = {'passed': True}
                raise error
            evaluator = CaseEvaluator(replace(problem.case_definition, execute=fail))
            with patch('flowfield.evaluation.Runtime.resolve', return_value={'backend': 'test'}):
                with self.assertRaises(type(error)):
                    evaluator.run_case(case)
            saved = json.loads((case / 'result.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertTrue(saved['mesh_validation']['passed'])
            self.assertIn('wall_time_s', saved)
            with self.assertRaises(FileExistsError):
                evaluator.run_case(case)

    def test_json_is_validated_and_mesh_artifact_checks_required_files(self):
        path = self.root / 'config.json'
        path.write_text('{"mesh":"medium"}')
        problem = create_problem('P005', config=path)
        self.assertEqual(problem.config.mesh, 'medium')
        path.write_text('{"oxygen_velocity_m_s":-1}')
        with self.assertRaises(ValueError):
            create_problem('P005', config=path)
        path.write_text('{"typo":1}')
        with self.assertRaises(TypeError):
            create_problem('P005', config=path)
        path.write_text('[]')
        with self.assertRaises(ValueError):
            load_config(type(problem.config), path)
        with self.assertRaises(ValueError):
            PolyMesh.from_case(self.root)
        p003 = create_problem('P003')
        case = p003.write_case(p003.initial_design(), self.root / 'mesh-case')
        self.assertEqual(set(PolyMesh.required_files) - PolyMesh.from_case(case).checksums().keys(), set())

    def test_plugin_receives_vector_feedback_and_resume_does_not_repeat_cfd(self):
        with patch.dict('flowfield.optimization.registry._OPTIMIZERS'):
            register_optimizer('batch-test', BatchSampler, revision='test-v1')
            first = Toy()
            def interrupt(name, index, budget, state):
                if index == 2 and state == 'ok':
                    raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                optimize(first, optimizer='batch-test', output=self.root / 'run', budget=5,
                         with_ioh=False, progress=interrupt)
            resumed = Toy()
            result = optimize(resumed, optimizer='batch-test', output=self.root / 'run', budget=5,
                              with_ioh=False, resume=True)
            self.assertEqual((len(first.calls), len(resumed.calls), result['attempts']), (2, 3, 5))
            sampler = BatchSampler.instances[-1]
            self.assertEqual(sampler.baseline['objectives'], {'a': .5, 'b': .5})
            self.assertEqual(len(sampler.feedback), 4)
            self.assertTrue(all('limit' in r['constraints'] for r in sampler.feedback))
            # Registered algorithm changes cannot silently resume an old run.
            with patch('flowfield.optimization.registry._OPTIMIZERS', {}):
                register_optimizer('batch-test', BatchSampler, revision='test-v2')
                with self.assertRaisesRegex(ResumeError, 'optimizer_revision'):
                    optimize(Toy(), optimizer='batch-test', output=self.root / 'run', budget=5,
                             with_ioh=False, resume=True)

    def test_selected_cli_default_and_custom_registration(self):
        for args, expected in (([], 'P000'), (['--problem', 'P005', '--optimizer', 'random'], 'P005')):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main([*args, '--budget', '2', '--output', str(self.root / 'dry'), '--dry-run'])
            plan = json.loads(output.getvalue())
            self.assertEqual([entry['problem'] for entry in plan['problems']], [expected])
            self.assertEqual(plan['maximum_attempts'], 2)
            self.assertFalse((self.root / 'dry').exists())

    def test_repeated_runs_and_fixed_target_analysis_use_saved_data(self):
        with patch.dict('flowfield.registry._FACTORIES'):
            register_problem('ArchitectureToy', Toy)
            manifest = run_benchmark(problems=['ArchitectureToy'], optimizers=['random'],
                seeds=[1, 2], budget=3, output=self.root / 'benchmark', with_ioh=False)
        directories = [self.root / 'benchmark' / entry['directory'] for entry in manifest['runs']]
        self.assertEqual(len(directories), 2)
        with patch.object(Toy, 'evaluate_raw', side_effect=AssertionError('Must not reevaluate')):
            comparison = compare_runs(directories, targets=[1., -1.])
        targets = comparison['algorithms']['RandomSearch']['targets']
        self.assertEqual((targets[0]['successes'], targets[0]['ert']), (2, 1.))
        self.assertEqual((targets[1]['successes'], targets[1]['ert']), (0, None))
        with self.assertRaisesRegex(ValueError, 'Duplicate seeds'):
            compare_runs([directories[0], directories[0]])


if __name__ == '__main__':
    unittest.main()
