"""Search fidelity must change cost without silently changing acceptance rules."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from flowfield import CaseEvaluator
from flowfield.optimization import main as optimize_main
from flowfield.problems.level1_small_continuous.p006 import ManifoldConfig, P006Problem
from flowfield.problems.level1_small_continuous.p006.case import generate_case
from flowfield.problems.level1_small_continuous.p006.cli import main as case_main
from flowfield.problems.level1_small_continuous.p006.geometry import build_geometry
from flowfield.problems.level1_small_continuous.p006.metrics import extract_metrics, values
from flowfield.problems.level1_small_continuous.p006.warm_start import FIELDS
from flowfield.runtime import Runtime
from flowfield.stability.audit import inspect_case
from flowfield.stability.profiles import refined


class P006SearchTests(unittest.TestCase):
    def test_reference_identity_is_backward_compatible_and_search_is_distinct(self):
        default = ManifoldConfig()
        self.assertEqual(default.field_tolerance, 1e-5)
        # Recorded 3000-iteration cases retain their identity even when the
        # current default iteration budget is raised to 3500.
        self.assertEqual(replace(default, iterations=3000).fingerprint,
                         '86a236583078263b8f3cca62b73ac91761676405e97cb7982fc5a450bd2d464e')
        configs = [default, replace(default, field_tolerance=1e-4),
                   replace(default, mesh='screening'),
                   replace(default, mesh='screening-coarse', field_tolerance=1e-4)]
        self.assertEqual(len({P006Problem(config=c).ioh_name for c in configs}), len(configs))
        self.assertFalse(default.fidelity['reference_recheck_required'])
        for config in configs[1:]:
            self.assertEqual(config.fidelity['purpose'], 'search')
            self.assertTrue(config.fidelity['reference_recheck_required'])
            self.assertNotIn('fidelity', config.to_dict())
        for overrides in ({'field_tolerance': 1.01e-4}, {'field_tolerance': float('nan')},
                          {'field_tolerance': 0}, {'field_tolerance': 1e-4, 'balance_tolerance': .0011}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(default, **overrides)

    def test_search_mesh_preserves_material_volumes_planes_and_design_topology(self):
        def material_volumes(mesh):
            volumes = [0.] * 9
            for (i, j, k), material in zip(mesh['cells'], mesh['ids']):
                volumes[material] += ((mesh['x'][i+1]-mesh['x'][i]) *
                                      (mesh['y'][j+1]-mesh['y'][j]) *
                                      (mesh['z'][k+1]-mesh['z'][k]))
            return volumes

        for manifold in (1., 3.):
            config = ManifoldConfig(manifold_width_mm=manifold)
            problem = P006Problem(config=config)
            reference = build_geometry(problem.decode(problem.initial_design()), config)
            for mesh, baseline_cells in (('screening', 9072), ('screening-coarse', 7056)):
                search = replace(config, mesh=mesh)
                baseline = build_geometry(problem.decode(problem.initial_design()), search)
                if manifold == 1.:
                    self.assertEqual(len(baseline['cells']), baseline_cells)
                self.assertEqual(baseline['x'], reference['x'])
                self.assertEqual(baseline['z'], reference['z'])
                for plane in (0., manifold, 12.5, 25-manifold, 25.):
                    self.assertIn(plane, baseline['y'])
                for x in ([0., 0., 0.], [1., 1., 1.], [.25, .375, .5]):
                    g = problem.decode(x)
                    reduced = build_geometry(g, search)
                    full = build_geometry(g, config)
                    self.assertEqual(reduced['cells'], baseline['cells'])
                    self.assertEqual(reduced['ids'], baseline['ids'])
                    for got, want in zip(material_volumes(reduced), material_volumes(full)):
                        self.assertAlmostEqual(got, want, places=9)

    def test_case_preserves_feed_and_inner_tolerances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = P006Problem().write_case([.375]*3, root/'reference')
            for mesh in ('screening', 'screening-coarse'):
                p = P006Problem(config=ManifoldConfig(mesh=mesh, field_tolerance=1e-4))
                case = p.write_case(p.initial_design(), root/mesh)
                manifest = json.loads((case/'case.json').read_text())
                self.assertEqual(manifest['fidelity'], p.config.fidelity)
                self.assertEqual(manifest['inlet_areas_m2'],
                                 {'cathodeInlet': 1e-6, 'anodeInlet': 1e-6})
                self.assertIn('field_tolerance 0.0001;', (case/'constant/modelProperties').read_text())
                self.assertIn('initial_momentum_fraction 1e-6;', (case/'constant/modelProperties').read_text())
                self.assertNotIn('initial_momentum_fraction', (reference/'constant/modelProperties').read_text())
                self.assertEqual((case/'system/fvSolution').read_bytes(),
                                 (reference/'system/fvSolution').read_bytes())
                for phase, ratio in (('L', 1.), ('G', .1/.9)):
                    text = (case/'0'/('phi'+phase)).read_text()
                    for inlet in ('cathodeInlet', 'anodeInlet'):
                        block = re.search(inlet+r'\s*\{(.*?)\}', text, re.S).group(1)
                        count = int(re.search(r'List<scalar>\s+(\d+)', block).group(1))
                        self.assertAlmostEqual(sum(values(block, 'value', count)),
                                               -15e-6/60*ratio, places=17)

    def test_manifest_metadata_cannot_override_config_and_old_cases_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = P006Problem()
            case = p.write_case(p.initial_design(), Path(tmp)/'case')
            file = case/'case.json'
            manifest = json.loads(file.read_text())
            del manifest['fidelity']  # Existing manifests predate this metadata.
            file.write_text(json.dumps(manifest))
            evaluator = CaseEvaluator(p.case_definition)
            evaluator._load_case(case)
            manifest['fidelity'] = ManifoldConfig(field_tolerance=1e-4).fidelity
            file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Fidelity metadata'):
                evaluator._load_case(case)

    def test_cli_and_audit_record_search_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)/'case'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(case_main(['generate', '--case', str(case), '--mesh', 'screening',
                                            '--field-tolerance', '1e-4']), 0)
            audit = inspect_case(case)
            self.assertEqual(audit['fidelity']['purpose'], 'search')
            self.assertFalse(audit['qualified'])
            self.assertIn('evaluation_fidelity', [check['name'] for check in audit['checks']])
            for action in ('run', 'extract'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    case_main([action, '--case', str(case), '--field-tolerance', '1e-4'])

    def test_relaxed_change_keeps_five_sample_coupling_and_conservation_checks(self):
        config = ManifoldConfig(field_tolerance=1e-4, minimum_iterations=1)
        row = dict(iteration=1, charge_residual=0., field_change=5e-5, bounded_cells=0,
                   liquid_balance=0., gas_balance=0., local_liquid_balance=0., local_gas_balance=0.,
                   energy_balance=0., applied_voltage_V=2.2, liquid_momentum_residual=0.,
                   gas_momentum_residual=0., momentum_fraction=1., thermal_active=1., T_std_K=.8)
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            post = case/'postProcessing/p006'
            post.mkdir(parents=True)
            (case/'case.json').write_text(json.dumps(dict(cells=9072, design_geometry={'constraints': {}})))
            (post/'status.json').write_text(json.dumps(dict(converged=True, iteration=5, cells=9072,
                processes=2, momentum_fraction=1., thermal_active=True)))
            (case/'log.flowfieldP006Foam').write_text(
                'CONVERGED: coupled fields and local/global balances passed.\nEnd\n')

            def history(bad_iteration=0, changes=None):
                with (post/'history.csv').open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    for i in range(1, 6):
                        writer.writerow(dict(row, iteration=i, **(changes if i == bad_iteration else {})))

            history()
            with patch('flowfield.problems.level1_small_continuous.p006.metrics.verify_fields',
                       return_value={'temperature_std_K': .8}) as verify:
                result = extract_metrics(case, config)
                self.assertEqual(result['objective'], .8)
                self.assertTrue(result['fidelity']['reference_recheck_required'])
                verify.assert_called_once()
                with self.assertRaisesRegex(ValueError, 'convergence or conservation'):
                    extract_metrics(case, replace(config, field_tolerance=1e-5))
                verify.reset_mock()
                for key, value in (('charge_residual', 1e-6), ('field_change', 1e-4),
                    ('bounded_cells', 1), ('applied_voltage_V', 1.4), ('momentum_fraction', .5),
                    ('thermal_active', 0), *((key, .001) for key in (
                        'liquid_balance', 'gas_balance', 'local_liquid_balance', 'local_gas_balance',
                        'energy_balance', 'liquid_momentum_residual', 'gas_momentum_residual'))):
                    for i in range(1, 6):
                        with self.subTest(field=key, iteration=i):
                            history(i, {key: value})
                            with self.assertRaisesRegex(ValueError, 'convergence or conservation'):
                                extract_metrics(case, config)
                verify.assert_not_called()
            history()
            with patch('flowfield.problems.level1_small_continuous.p006.metrics.verify_fields',
                       side_effect=ValueError('Independent saved-field conservation failed')):
                with self.assertRaisesRegex(ValueError, 'saved-field conservation'):
                    extract_metrics(case, config)

    def test_reference_recheck_requires_new_case_and_matching_mesh_for_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = P006Problem(config=ManifoldConfig(mesh='screening', field_tolerance=1e-4))
            source = search.write_case(search.initial_design(), root/'search')
            manifest = json.loads((source/'case.json').read_text())
            saved = source/'10'
            saved.mkdir()
            for name in FIELDS:
                shutil.copy2(source/'0'/name, saved/name)
            post = source/'postProcessing/p006'
            post.mkdir(parents=True)
            (post/'status.json').write_text(json.dumps(dict(converged=False, iteration=10,
                cells=manifest['cells'], processes=2, momentum_fraction=1., thermal_active=True)))
            (post/'history.csv').write_text(
                'iteration,momentum_fraction,thermal_active,applied_voltage_V\n10,1,1,2.2\n')
            (source/'log.flowfieldP006Foam').write_text('Iteration = 10\nEnd\n')
            (source/'result.json').write_text(json.dumps(dict(status='failed', input_sha256=manifest['input_sha256'])))
            case = generate_case(root/'reference', search.decode(search.initial_design()),
                                 ManifoldConfig(mesh='screening'), initial_state=source)
            new = json.loads((case/'case.json').read_text())
            self.assertEqual(new['config']['field_tolerance'], 1e-5)
            self.assertFalse(new['initial_state']['qualification_inherited'])
            self.assertFalse((case/'result.json').exists())
            properties = (case/'constant/modelProperties').read_text()
            self.assertEqual(properties.count('initial_momentum_fraction'), 1)
            self.assertIn('initial_momentum_fraction 1;', properties)
            self.assertIn('initial_thermal_active true;', properties)
            status=json.loads((post/'status.json').read_text());status['thermal_active']=False
            (post/'status.json').write_text(json.dumps(status))
            (post/'history.csv').write_text(
                'iteration,momentum_fraction,thermal_active,applied_voltage_V\n10,1,0,2.2\n')
            partial=generate_case(root/'partial',search.decode(search.initial_design()),search.config,
                                  initial_state=source)
            props=(partial/'constant/modelProperties').read_text()
            self.assertEqual(props.count('initial_momentum_fraction'),1)
            self.assertIn('initial_momentum_fraction 1;',props)
            self.assertIn('initial_thermal_active false;',props)
            self.assertNotIn('numerical_initialization',json.loads((partial/'case.json').read_text()))
            with self.assertRaisesRegex(ValueError, 'same physical configuration and mesh'):
                generate_case(root/'different-mesh', search.decode(search.initial_design()),
                              ManifoldConfig(), initial_state=source)
            self.assertFalse((root/'different-mesh').exists())

    def test_optimizer_plans_use_presets_for_all_four_methods(self):
        config = Path(__file__).resolve().parents[1]/'benchmarks/P006/search-config.json'
        with tempfile.TemporaryDirectory() as tmp:
            for method in ('random', 'bo', 'turbo', 'cmaes'):
                output = io.StringIO()
                root = Path(tmp)/method
                with contextlib.redirect_stdout(output):
                    self.assertEqual(optimize_main(['--problem', 'P006', '--optimizer', method,
                        '--budget', '10', '--seed', '42', '--config', str(config),
                        '--output', str(root), '--dry-run']), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan['mesh'], 'screening')
                self.assertEqual(plan['problems'][0]['fidelity']['field_tolerance'], 1e-4)
                self.assertFalse(root.exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                optimize_main(['--problem', 'P000', '--mesh', 'screening', '--output', str(Path(tmp)/'bad'), '--dry-run'])

    def test_resume_preserves_fidelity_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'optimization'
            config_file = Path(tmp)/'config.json'
            config = ManifoldConfig(field_tolerance=1e-4)
            config_file.write_text(json.dumps(config.to_dict()))
            accepted = dict(status='ok', objective=.8, objectives={}, constraints={}, case_directory='/test/qualified')
            with patch.object(P006Problem, 'preflight'), \
                 patch.object(Runtime, 'resolve', return_value={'backend': 'native'}), \
                 patch.object(CaseEvaluator, 'evaluate', return_value=accepted), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(optimize_main(['--problem', 'P006', '--optimizer', 'random', '--budget', '1',
                    '--config', str(config_file), '--output', str(root)]), 0)
            for changes in ({}, {'field_tolerance': 1e-5}, {'mesh': 'screening'}):
                config_file.write_text(json.dumps(replace(config, **changes).to_dict()))
                with patch.object(CaseEvaluator, 'evaluate') as evaluate, \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if changes:
                        with self.assertRaises(SystemExit):
                            optimize_main(['--resume', '--output', str(root), '--dry-run'])
                    else:
                        self.assertEqual(optimize_main(['--resume', '--output', str(root), '--dry-run']), 0)
                    evaluate.assert_not_called()

    def test_mesh_study_includes_reference_levels(self):
        meshes = [c.mesh for _, c in refined(ManifoldConfig(mesh='screening-coarse'))]
        self.assertEqual(meshes, ['screening', 'coarse', 'medium', 'fine'])
        self.assertEqual([c.mesh for _, c in refined(ManifoldConfig())], ['medium', 'fine'])


if __name__ == '__main__':
    unittest.main()
