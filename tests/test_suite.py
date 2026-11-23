"""Suite access and architecture boundaries, without an OpenFOAM installation."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flowfield import (CFDProblem, FlowFieldProblem, ProblemInfo, create_problem,
                       describe_problem, list_problems, problem_info, register_problem)
from flowfield.benchmarking import plan_benchmark, run_benchmark
from flowfield.cli import main


class ExternalProblem(FlowFieldProblem):
    calls = []

    def __init__(self, output_root=None):
        super().__init__('External', 2, [-1, 2], [1, 4])

    def decode(self, x):
        return self.validate_design(x)

    def evaluate_raw(self, x):
        x = self.decode(x)
        self.calls.append(x)
        return dict(status='ok', objective=sum(v*v for v in x),
                    objectives={'energy': sum(v*v for v in x)},
                    constraints={'limit': x[0] - .8}, metadata={'source': 'test'})


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        ExternalProblem.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(args))
        return code, json.loads(stdout.getvalue()) if stdout.getvalue() else None, stderr.getvalue()

    def test_discovery_and_single_selection_work_without_site_packages(self):
        source = str(Path(__file__).resolve().parents[1] / 'src')
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
import flowfield
from flowfield.cli import main
assert flowfield.list_problems(level=2) == ('P001', 'P002')
assert flowfield.problem_info('P008').extra == 'p008'
assert not any(n.startswith(('flowfield.problems', 'flowfield.optimization',
                             'flowfield.runtime', 'numpy', 'scipy', 'ioh')) for n in sys.modules)
flowfield.create_problem('P008')
implementations = [n for n in sys.modules if n.startswith('flowfield.problems.')]
assert all(n == 'flowfield.problems.level1_small_continuous' or
           n.startswith('flowfield.problems.level1_small_continuous.p008') for n in implementations), implementations
'''
        subprocess.run([sys.executable, '-S', '-c', script, source], check=True, capture_output=True)

    def test_optimizer_api_does_not_import_concrete_problems_or_its_cli(self):
        source = str(Path(__file__).resolve().parents[1] / 'src')
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
from flowfield.optimization import optimize
assert 'flowfield.optimization.cli' not in sys.modules
assert not any(n.startswith('flowfield.problems') for n in sys.modules)
'''
        subprocess.run([sys.executable, '-S', '-c', script, source], check=True, capture_output=True)

    def test_catalog_filtering_matches_concrete_variable_types_and_case_support(self):
        self.assertEqual(list_problems(variable_type='integer'), ('P004', 'P004-hydraulic'))
        self.assertEqual(list_problems(evaluation_kind='data'), ('P004',))
        for identifier in list_problems():
            with self.subTest(problem=identifier):
                problem, info = create_problem(identifier), problem_info(identifier)
                self.assertEqual(info.variable_type, getattr(problem, 'variable_type', 'real'))
                self.assertEqual(info.evaluation_kind == 'cfd', isinstance(problem, CFDProblem))
        with self.assertRaises(ValueError):
            list_problems(level=0)
        with self.assertRaises(ValueError):
            list_problems(evaluation_kind='typo')

    def test_inspection_uses_configured_dimensions_and_never_resolves_runtime(self):
        config = self.root / 'config.json'
        config.write_text('{"design_nx":8,"design_ny":8}')
        output = self.root / 'not-created'
        with patch('flowfield.runtime.Runtime.resolve', side_effect=AssertionError('Must not run')):
            info = describe_problem('P007', config=config, output_root=output)
            code, from_cli, _ = self.invoke('describe', 'P007', '--config', str(config))
        self.assertEqual((code, info['dimension'], len(info['initial_design'])), (0, 32, 32))
        self.assertEqual(from_cli['instance'], info['instance'])
        self.assertFalse(output.exists())
        self.assertNotEqual(info['instance'], describe_problem('P007')['instance'])

    def test_custom_problem_is_discoverable_and_cli_evaluates_once(self):
        with patch.dict('flowfield.registry._FACTORIES'):
            register_problem('External', ExternalProblem, info=ProblemInfo('External', 'External energy', level=1))
            self.assertIn('External', list_problems(level=1))
            code, decoded, _ = self.invoke('decode', '--problem', 'External', '--x', '-.5', '3')
            self.assertEqual((code, decoded, ExternalProblem.calls), (0, [-.5, 3.], []))
            design = self.root / 'design.json'
            design.write_text('[-0.5,3]')
            code, result, _ = self.invoke('evaluate', '--problem', 'External', '--design', str(design),
                                          '--output', str(self.root / 'cases'))
            self.assertEqual(code, 0)
            self.assertEqual(result['objective'], 9.25)
            self.assertEqual(result['objectives'], {'energy': 9.25})
            self.assertEqual(ExternalProblem.calls, [[-.5, 3.]])
            code, _, error = self.invoke('evaluate', '--problem', 'External', '--x', '8', '3',
                                         '--output', str(self.root / 'invalid'))
            self.assertEqual(code, 1)
            self.assertIn('within', error)
            self.assertEqual(len(ExternalProblem.calls), 1)

    def test_cli_preserves_the_selected_problems_runtime_defaults(self):
        problem = create_problem('P006')
        result = dict(objective=1., objectives={'temperature_std_K': 1.}, constraints={}, metadata={}, status='ok')
        for flags, timeout in (([], problem.runtime.timeout_s), (['--timeout', '7200'], 7200)):
            with patch('flowfield.cli.create_problem', return_value=problem), \
                 patch.object(problem, 'preflight'), patch.object(problem, 'evaluate_raw', return_value=result) as evaluate:
                code, payload, _ = self.invoke('evaluate', '--problem', 'P006',
                                               '--output', str(self.root / 'cases'), *flags)
            self.assertEqual(code, 0)
            self.assertEqual(problem.runtime.timeout_s, timeout)
            evaluate.assert_called_once()
            self.assertEqual(payload, result)

    def test_generic_case_lifecycle_uses_manifest_and_extraction_never_executes(self):
        from flowfield.problems.p000 import P000Problem
        case = self.root / 'case'
        code, _, _ = self.invoke('generate', '--problem', 'P000', '--case', str(case))
        self.assertEqual(code, 0)
        spec = P000Problem().case_definition
        calls = []
        def execute(*args):
            calls.append('execute')
        def extract(*args):
            calls.append('extract')
            return dict(objective=3., objectives={'pressure_drop_pa': 3.}, constraints={})
        definition = replace(spec, execute=execute, extract=extract)
        with patch.object(P000Problem, 'case_definition', definition), \
             patch('flowfield.runtime.Runtime.resolve', return_value={'backend': 'test'}) as resolve:
            code, result, _ = self.invoke('run', str(case))
            self.assertEqual((code, result['objective']), (0, 3.))
            self.assertEqual(calls, ['execute', 'extract'])
            resolve.assert_called_once()
            resolve.side_effect = AssertionError('Extraction cannot launch a solver')
            code, result, _ = self.invoke('extract', str(case))
            self.assertEqual((code, result['objective']), (0, 3.))
            code, _, error = self.invoke('run', str(case))
            self.assertEqual(code, 1)
            self.assertIn('execution data', error)
            self.assertEqual(calls.count('execute'), 1)

    def test_data_requirements_fail_before_output_and_hydraulics_stays_independent(self):
        for identifier, pattern in (('P004', 'dataset'), ('P008', 'calibration')):
            with self.subTest(problem=identifier), self.assertRaisesRegex((ValueError, RuntimeError), pattern):
                run_benchmark(problems=[identifier], optimizers=['random'], seeds=[1], budget=1,
                              output=self.root / identifier, with_ioh=False)
            self.assertFalse((self.root / identifier).exists())
        plan = plan_benchmark(problems=['P004-hydraulic'], optimizers=['random'], seeds=[1], budget=1)
        self.assertEqual(plan['run_count'], 1)
        with self.assertRaisesRegex(ValueError, 'does not support integer'):
            plan_benchmark(problems=['P004-hydraulic'], optimizers=['cmaes'], seeds=[1], budget=1)

    def test_matrix_planning_never_evaluates_and_execution_uses_fresh_instances(self):
        with patch.dict('flowfield.registry._FACTORIES'):
            register_problem('External', ExternalProblem)
            options = dict(problems=['External'], optimizers=['random'], seeds=[1, 2], budget=3, with_ioh=False)
            with patch.object(ExternalProblem, 'evaluate_raw', side_effect=AssertionError('Plan cannot evaluate')):
                plan = plan_benchmark(**options)
            self.assertEqual((plan['run_count'], plan['maximum_attempts']), (2, 6))
            self.assertEqual(plan['instances'][0]['lower_bounds'], [-1., 2.])
            self.assertEqual(list(self.root.iterdir()), [])
            manifest = run_benchmark(**options, output=self.root / 'runs')
        self.assertEqual(len(ExternalProblem.calls), 6)
        self.assertEqual(len(manifest['runs']), 2)
        self.assertEqual(manifest['instances'], plan['instances'])
        self.assertEqual(manifest['status'], 'completed')
        saved = json.loads((self.root / 'runs/benchmark.json').read_text())
        self.assertEqual(saved['maximum_attempts'], 6)

    def test_dry_run_and_invalid_options_leave_no_output(self):
        output = self.root / 'dry'
        code, plan, _ = self.invoke('benchmark', '--problem', 'P000', '--optimizer', 'random',
                                    '--seeds', '1', '2', '--budget', '2', '--output', str(output), '--dry-run')
        self.assertEqual((code, plan['maximum_attempts']), (0, 4))
        self.assertFalse(output.exists())
        base = dict(problems=['P000'], optimizers=['random'], seeds=[1], budget=2)
        for options in ({'optimizer_options': {'typo': {}}}, {'problem_options': {'P001': {}}},
                        {'optimizer_options': {'random': {'sigma': 9}}}, {'seeds': [1, 1]},
                        {'budget': True}, {'problems': 'P000'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan_benchmark(**(base | options))

    def test_p000_physical_unit_commands_remain_available(self):
        for prefix in ([], ['p000']):
            code, result, _ = self.invoke(*prefix, 'reference', '--width-mm', '1', '--height-mm', '1')
            self.assertEqual(code, 0)
            self.assertEqual(result['backend'], 'analytical_reference_only')


if __name__ == '__main__':
    unittest.main()
