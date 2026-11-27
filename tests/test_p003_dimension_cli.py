"""Dimension-only P003 commands, including persisted settings and resume."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flowfield import P003Problem, TopologyConfig
from flowfield.benchmarking.runner import main as benchmark_main
from flowfield.optimization.cli import main as optimize_main
from flowfield.problems.level4_high_dimensional_topology.p003.cli import main as p003_main
from flowfield.problems.level4_high_dimensional_topology.p003.scalability import scalable_config


class P003DimensionCliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def call(self, main, args):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(args),0)
        return json.loads(stream.getvalue())

    def test_optimizer_dimension_selects_medium_without_config_or_files(self):
        for n in (500,512,1000,2000,2500,3000,4000,5000,6000):
            for optimizer in ('turbo','cmaes'):
                with self.subTest(n=n,optimizer=optimizer):
                    output=self.root/f'{n}-{optimizer}'
                    plan=self.call(optimize_main,['--problem','P003','--optimizer',optimizer,
                        '--dimension',str(n),'--budget','5','--output',str(output),'--dry-run'])
                    entry=plan['problems'][0]
                    self.assertEqual(entry['dimension'],n)
                    self.assertEqual(entry['initial_design_length'],n)
                    self.assertEqual(entry['cfd_cells'],8320)
                    self.assertEqual(plan['mesh'],'medium')
                    self.assertEqual(entry['mesh'],'medium')
                    self.assertEqual(entry['design_points'][0]*entry['design_points'][1],n)
                    self.assertEqual(entry['config_sha256'],scalable_config(dimension=n).fingerprint)
                    self.assertFalse(output.exists())

    def test_benchmark_dimension_applies_to_every_optimizer_and_seed(self):
        for n in (500,6000):
            plan=self.call(benchmark_main,['--problem','P003','--optimizers','turbo','cmaes',
                '--seeds','42','43','--dimension',str(n),'--budget','5',
                '--output',str(self.root/'benchmark'),'--dry-run'])
            self.assertEqual(plan['run_count'],4)
            self.assertEqual(plan['maximum_attempts'],20)
            instance=plan['instances'][0]
            self.assertEqual(instance['dimension'],n)
            self.assertEqual(instance['cfd_cells'],8320)
            self.assertEqual(instance['config'],scalable_config(dimension=n).to_dict())
        self.assertFalse((self.root/'benchmark').exists())

    def test_invalid_dimension_and_conflicting_selection_fail_before_writes(self):
        for main,prefix in ((optimize_main,['--problem','P003']),
                            (benchmark_main,['--problem','P003','--optimizer','turbo','--seeds','42']),
                            (p003_main,['gcmma'])):
            for extra in (['--dimension','503'],['--dimension','0'],['--dimension','9000'],
                          ['--dimension','500','--config',str(self.root/'missing.json')]):
                with self.subTest(command=main.__module__,extra=extra):
                    stream=io.StringIO()
                    with contextlib.redirect_stderr(stream),self.assertRaises(SystemExit):
                        main([*prefix,*extra,'--output',str(self.root/'bad'),'--dry-run'])
                    self.assertNotIn('No such file',stream.getvalue())
                    self.assertFalse((self.root/'bad').exists())
        for main,prefix in ((optimize_main,[]),
                            (benchmark_main,['--optimizer','turbo','--seeds','42'])):
            stream=io.StringIO()
            with contextlib.redirect_stderr(stream),self.assertRaises(SystemExit):
                main([*prefix,'--problem','P000','--dimension','500',
                      '--output',str(self.root/'bad'),'--dry-run'])
            self.assertIn('supported by P003 only',stream.getvalue())

    def test_omitting_dimension_preserves_legacy_default_and_explicit_mesh_is_honored(self):
        args=['--problem','P003','--output',str(self.root/'preview'),'--dry-run']
        default=self.call(optimize_main,args)
        self.assertEqual(default['mesh'],'coarse')
        self.assertEqual(default['problems'][0]['dimension'],2145)
        self.assertEqual(default['problems'][0]['config_sha256'],TopologyConfig().fingerprint)
        fine=self.call(optimize_main,[*args,'--dimension','6000','--mesh','fine'])
        self.assertEqual(fine['mesh'],'fine')
        self.assertEqual(fine['problems'][0]['cfd_cells'],18816)

    def test_gcmma_dimension_and_resume_use_saved_configuration(self):
        output=self.root/'gcmma'
        args=['gcmma','--dimension','500','--output',str(output),'--dry-run']
        plan=self.call(p003_main,args)
        self.assertEqual(plan['design_variables'],500)
        self.assertEqual(plan['probes_per_gradient'],500)
        self.assertEqual(plan['design_shape'],[25,20])
        self.assertEqual(plan['cfd_cells'],8320)
        self.assertFalse(output.exists())
        output.mkdir()
        (output/'gcmma-run.json').write_text(json.dumps(dict(
            config=scalable_config(dimension=500).to_dict(),options=plan['options'],
            max_evaluations=plan['max_evaluations'],seed=42,discard_probe_cases=False)))
        resumed=self.call(p003_main,['gcmma','--resume','--output',str(output),'--dry-run'])
        self.assertEqual(resumed['config_sha256'],plan['config_sha256'])
        self.assertEqual(resumed['design_variables'],500)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            p003_main(['gcmma','--resume','--dimension','1000','--output',str(output),'--dry-run'])

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'optional sparse repair dependencies unavailable')
    def test_execution_persists_dimension_and_resume_extends_without_config_file(self):
        # Run real CLI/optimizer/repair/journal paths; replace only CFD evaluation.
        output=self.root/'run'
        calls=[]
        def evaluate(problem,x):
            calls.append((problem.config,list(x)))
            return dict(status='ok',objective=sum(x)/len(x),constraints={},
                        metadata={'valid_cfd':True})
        with patch.object(P003Problem,'evaluate_raw',autospec=True,side_effect=evaluate), \
             patch('flowfield.optimization.cli.Runtime.resolve',return_value={'backend':'native'}):
            first=self.call(optimize_main,['--problem','P003','--optimizer','random',
                '--dimension','500','--budget','1','--output',str(output)])
            self.assertEqual(first['total_attempts'],1)
            options=json.loads((output/'run-options.json').read_text())
            self.assertEqual(options['dimension'],500)
            self.assertIsNone(options['config'])
            saved=json.loads((output/'P003/experiment.json').read_text())
            self.assertEqual(saved['config'],scalable_config(dimension=500).to_dict())
            preview=self.call(optimize_main,['--resume','--output',str(output),'--dry-run'])
            self.assertEqual(preview['problems'][0]['dimension'],500)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                optimize_main(['--resume','--dimension','1000','--output',str(output)])
            self.assertEqual(len(calls),1)
            extended=self.call(optimize_main,['--resume','--extend-budget','2','--output',str(output)])
            self.assertEqual(extended['total_attempts'],2)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(len(x)==500 and c==scalable_config(dimension=500) for c,x in calls))

    @unittest.skipUnless(importlib.util.find_spec('scipy'),'optional sparse repair dependencies unavailable')
    def test_benchmark_execution_uses_dimension_without_config_file(self):
        configs=[]
        def evaluate(problem,x):
            configs.append(problem.config)
            return dict(status='ok',objective=sum(x)/len(x),constraints={},metadata={'valid_cfd':True})
        with patch.object(P003Problem,'evaluate_raw',autospec=True,side_effect=evaluate):
            result=self.call(benchmark_main,['--problem','P003','--optimizer','random',
                '--seeds','42','43','--dimension','512','--budget','1','--no-ioh',
                '--output',str(self.root/'matrix')])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(configs),2)
        self.assertTrue(all(c==scalable_config(dimension=512) for c in configs))


if __name__=='__main__':unittest.main()
