import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import shutil
import re
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch
import contextlib
import io
import os
import signal
import subprocess
import sys
import time

from flowfield import create_problem, CaseEvaluator
from flowfield.problems.level1_small_continuous.p006 import P006Problem, ManifoldConfig
from flowfield.problems.level1_small_continuous.p006.config import spline_widths
from flowfield.problems.level1_small_continuous.p006.geometry import build_geometry, PATCHES
from flowfield.problems.level1_small_continuous.p006.metrics import values, vectors, verify_phase_fluxes, extract_metrics
from flowfield.problems.level1_small_continuous.p006.case import generate_case
from flowfield.problems.level1_small_continuous.p006.warm_start import FIELDS
from flowfield.problems.level1_small_continuous.p006.pressure import pressure_forces,verify_pressure_forces
from flowfield.problems.level1_small_continuous.p006.runtime import iteration_limit_reason, _run_command, run_phase
from flowfield.runtime import Runtime
from flowfield.optimization import optimize, main as optimize_main
from flowfield.runtime import SimulationError


class P006Tests(unittest.TestCase):
    def test_default_timeout_reaches_baseline_api_and_cli(self):
        from flowfield.problems.level1_small_continuous.p006.cli import main
        from flowfield.problems.level1_small_continuous.p006.evaluation import CASE
        self.assertEqual(P006Problem().runtime.timeout_s,14400)
        self.assertEqual(CaseEvaluator(CASE).runtime.timeout_s,14400)
        for action in ('evaluate','run'):
            for flags,expected in (([],14400),(['--timeout','600'],600)):
                with self.subTest(action=action,flags=flags),tempfile.TemporaryDirectory() as tmp:
                    output,errors=io.StringIO(),io.StringIO()
                    accepted=dict(status='ok',case_directory=tmp)
                    with patch.object(CaseEvaluator,'run_case',autospec=True,return_value={}) as run, \
                         patch.object(CaseEvaluator,'evaluate',autospec=True,return_value=accepted) as evaluate, \
                         contextlib.redirect_stdout(output),contextlib.redirect_stderr(errors):
                        self.assertEqual(main([action,'--case',tmp,*flags]),0)
                    called=evaluate if action=='evaluate' else run
                    self.assertEqual(called.call_args.args[0].runtime.timeout_s,expected)
                    self.assertIn(f'command timeout = {expected} s',errors.getvalue())

    def test_optimizer_timeout_is_recorded_passed_to_cfd_and_retained_on_resume(self):
        # The per-problem default must reach the actual evaluator, including a
        # mixed suite. Legacy numeric limits and explicit overrides still win.
        scenarios=(([],14400),(['--timeout','1800'],1800))
        for flags,expected in scenarios:
            with self.subTest(flags=flags),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'optimization';errors=io.StringIO()
                accepted=dict(status='ok',objective=.8,objectives={},constraints={},
                              case_directory='/test/p006-qualified')
                with patch.object(P006Problem,'preflight'), \
                     patch.object(Runtime,'resolve',return_value={'backend':'native'}), \
                     patch.object(CaseEvaluator,'evaluate',autospec=True,return_value=accepted) as evaluate, \
                     contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(errors):
                    self.assertEqual(optimize_main(['--problem','P006','--optimizer','random',
                        '--budget','1','--output',str(root),*flags]),0)
                self.assertEqual(evaluate.call_args.args[0].runtime.timeout_s,expected)
                plan=json.loads((root/'plan.json').read_text())
                self.assertEqual(plan['problems'][0]['command_timeout_s'],expected)
                self.assertIn(f'command timeout = {expected} s',errors.getvalue())
                self.assertIn('baseline must qualify',errors.getvalue())
                for override,resumed_limit in (([],expected),(['--timeout','7200'],7200)):
                    output=io.StringIO()
                    # Updating the default must not change a saved experiment's
                    # limit. An explicit resume --timeout can change execution.
                    with patch('flowfield.optimization.cli.P006_TIMEOUT_S',99999), \
                         contextlib.redirect_stdout(output):
                        self.assertEqual(optimize_main(['--resume','--dry-run',
                            '--output',str(root),*override]),0)
                    resumed=json.loads(output.getvalue())
                    self.assertEqual(resumed['problems'][0]['command_timeout_s'],resumed_limit)

    @unittest.skipUnless(hasattr(os,'killpg'),'POSIX process groups required')
    def test_timeout_stops_wrapper_and_its_child(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pidfile=root/'child.pid';lockfile=root/'child.lock'
            child_code=('import fcntl, os, time\nfrom pathlib import Path\n'
                        f'f=open({str(lockfile)!r},"w")\nfcntl.flock(f,fcntl.LOCK_EX)\n'
                        f'Path({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n')
            parent_code=('import subprocess, sys\n'
                         f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}])\n'
                         'p.wait()\n')
            with (root/'log').open('w') as stream:
                with self.assertRaises(subprocess.TimeoutExpired):
                    _run_command([sys.executable,'-c',parent_code],cwd=root,env=os.environ.copy(),
                                 stdout=stream,timeout=1)
            pid=int(pidfile.read_text())
            # The child's OS lock disappears when it exits, even before its
            # adopted zombie is reaped. No global process inspection needed.
            with lockfile.open('r+') as lock:
                deadline=time.monotonic()+2
                while True:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        if time.monotonic()>=deadline:
                            os.kill(pid,signal.SIGKILL)
                            self.fail('Child survived the wrapper timeout and still holds its lock')
                        time.sleep(.01)

    def test_runtime_distinguishes_timeout_from_solver_exit(self):
        scenarios=[(subprocess.TimeoutExpired(['solver'],1800),'timed out after 1800 seconds'),
                   (subprocess.CalledProcessError(1,['solver']),'exited with status 1')]
        for failure,message in scenarios:
            with self.subTest(message=message),tempfile.TemporaryDirectory() as tmp:
                case=Path(tmp)
                with patch('flowfield.problems.level1_small_continuous.p006.runtime._run_command',side_effect=failure):
                    with self.assertRaisesRegex(SimulationError,message) as error:
                        run_phase(case,'solve',Runtime(backend='native',timeout_s=1800),dict(backend='native'))
                self.assertEqual(error.exception.case_directory,str(case.resolve()))

    def test_failed_baseline_stops_before_candidates_with_either_penalty_policy(self):
        methods=['random']+(['cmaes'] if importlib.util.find_spec('cma') else [])
        scenarios=[(m,p,r) for m in methods for p,r in ((None,1e6),(1e6,None),(1e6,1e6))]
        for method, problem_penalty, runner_penalty in scenarios:
            with self.subTest(method=method,problem_penalty=problem_penalty,runner_penalty=runner_penalty), \
                 tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);p=P006Problem(output_root=root/'cases',failure_penalty=problem_penalty)
                with patch.object(p,'preflight'), \
                     patch.object(CaseEvaluator,'evaluate',side_effect=SimulationError(
                         'Pressure/saturation projection did not converge',case_directory='/test/p006-baseline')) as evaluate:
                    with self.assertRaisesRegex(SimulationError,'baseline did not qualify') as error:
                        optimize(p,method=method,population=8,budget=100,output=root/'optimization',
                                 with_ioh=False,failure_penalty=runner_penalty)
                self.assertEqual(evaluate.call_count,1)
                self.assertEqual(error.exception.case_directory,'/test/p006-baseline')
                summary=json.loads((root/'optimization/summary.json').read_text())
                self.assertEqual(summary['attempts'],1)
                self.assertEqual(summary['failed_evaluations'],1)
                self.assertEqual(summary['successful_evaluations'],0)
                self.assertEqual(summary['complete_generations'],0)
                self.assertIsNone(summary['best_objective'])
                self.assertIn('Pressure/saturation projection',summary['error'])
                self.assertFalse((root/'optimization/best.json').exists())

    def test_candidate_penalties_remain_available_after_a_qualified_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=P006Problem(output_root=root/'cases')
            def accepted(value):
                return dict(status='ok',objective=value,objectives={},constraints={},feasible=True,
                            case_directory='/test/p006-qualified')
            with patch.object(p,'preflight'),patch.object(CaseEvaluator,'evaluate',side_effect=[
                accepted(.8),SimulationError('candidate failed'),accepted(.7)]) as evaluate:
                summary=optimize(p,method='random',budget=3,output=root/'optimization',
                                 with_ioh=False,failure_penalty=1e6)
            self.assertEqual(evaluate.call_count,3)
            self.assertEqual(summary['successful_evaluations'],2)
            self.assertEqual(summary['failed_evaluations'],1)
            self.assertEqual(summary['baseline_objective'],.8)
            self.assertEqual(summary['best_objective'],.7)

    def test_missing_eigen_stops_optimization_before_cases_or_penalties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            # The runner's penalty and the problem's own penalty cannot turn a
            # missing build dependency into repeated, invalid observations.
            p=P006Problem(output_root=root/'cases',failure_penalty=1e6)
            with patch('flowfield.problems.level1_small_continuous.p006.runtime.distribution',
                       side_effect=PackageNotFoundError('cmeel-eigen')), \
                 patch.object(p,'evaluate_raw') as evaluate:
                with self.assertRaisesRegex(SimulationError,'requires Eigen headers'):
                    optimize(p,method='random',budget=100,output=root/'optimization',
                             with_ioh=False,failure_penalty=1e6)
                evaluate.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_cli_missing_eigen_reports_install_command_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'optimization'
            errors=io.StringIO()
            with patch('flowfield.problems.level1_small_continuous.p006.runtime.distribution',
                       side_effect=PackageNotFoundError('cmeel-eigen')), \
                 patch('flowfield.optimization.cli.Runtime.resolve') as resolve, \
                 contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit) as exc:
                    optimize_main(['--problem','P006','--optimizer','random','--budget','100',
                                   '--failure-penalty','1000000','--output',str(output)])
                self.assertEqual(exc.exception.code,1)
                resolve.assert_not_called()
            self.assertIn('python -m pip install cmeel-eigen==3.4.1',errors.getvalue())
            self.assertFalse(output.exists())

    def test_dry_run_keeps_p006_available_without_eigen(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan'
            capture=io.StringIO()
            with patch('flowfield.problems.level1_small_continuous.p006.runtime.distribution',
                       side_effect=PackageNotFoundError('cmeel-eigen')) as dependency, \
                 contextlib.redirect_stdout(capture):
                result=optimize_main(['--problem','P006','--optimizer','random',
                                     '--output',str(output),'--dry-run'])
                dependency.assert_not_called()
            self.assertEqual(result,0)
            self.assertEqual(json.loads(capture.getvalue())['problems'][0]['problem'],'P006')
            self.assertFalse(output.exists())

    def test_pressure_trace_matches_layered_darcy_solution(self):
        p=P006Problem();m=build_geometry(p.decode(p.initial_design()),p.config)
        nx,ny=len(m['x'])-1,len(m['y'])-1;stride=nx*ny
        # Analytical constant through-plane Darcy flux: pressure is continuous
        # and its slope changes by the permeability ratio at each interface.
        edges=[0,.5,1.5,1.8,1.85,1.8678,1.9178,2.2178,3.2178,3.7178]
        permeability=[1e-6,1e-6,1e-12,1e-13,1e-6,1e-13,1e-12,1e-6,1e-6]
        flux=1e-6;pressure=[]
        for _,_,k in m['cells']:
            z=(m['z'][k]+m['z'][k+1])/2;drop=0.
            for a,b,perm in zip(edges,edges[1:],permeability):
                drop+=max(0,min(z,b)-a)*.001*flux*p.config.liquid_viscosity/perm
            pressure.append(100-drop)
        actual=pressure_forces(m,p.config,pressure,[0.]*len(pressure))
        checked=0
        for c,mat in enumerate(m['ids']):
            if mat not in (2,6) or any(m['ids'][c+d*stride] not in (1,2,3,5,6,7) for d in (-1,1)):continue
            for phase in ('L','G'):
                self.assertAlmostEqual(actual[phase][c][0],0,places=7)
                self.assertAlmostEqual(actual[phase][c][1],0,places=7)
                self.assertAlmostEqual(actual[phase][c][2],flux*p.config.liquid_viscosity/1e-12,places=7)
            checked+=1
        self.assertGreater(checked,100)

    def test_pressure_force_matches_manufactured_linear_field_and_wall_conditions(self):
        p=P006Problem();m=build_geometry(p.decode(p.initial_design()),p.config)
        nx,ny=len(m['x'])-1,len(m['y'])-1
        gradient=1000.;pressure=[];expected=[]
        for c,((i,j,k),mat) in enumerate(zip(m['cells'],m['ids'])):
            x=(m['x'][i]+m['x'][i+1])*.0005
            pressure.append(gradient*(x-(.018 if mat>=5 else 0)))
            count=0
            if mat in (1,2,3,5,6,7):
                for direction in (-1,1):
                    if 0<=i+direction<nx:
                        count+=m['ids'][c+direction] in (1,2,3,5,6,7)
                    else:
                        y=(m['y'][j]+m['y'][j+1])/2
                        count+=(mat==7 and direction==1 and y>24) or (mat==1 and direction==-1 and y<1)
            expected.append([-gradient*count/2,0.,0.])
        # Zero capillary pressure isolates the pressure-gradient operator.
        saturation=[0.]*len(pressure)
        actual=pressure_forces(m,p.config,pressure,saturation)
        for phase in ('L','G'):
            for got,want in zip(actual[phase],expected):
                for a,b in zip(got,want):self.assertAlmostEqual(a,b,places=7)
        self.assertLess(verify_pressure_forces(m,p.config,pressure,saturation,
            {'L':expected,'G':expected}),1e-10)
        broken=[row[:] for row in expected]
        cell=next(i for i,v in enumerate(m['ids']) if v==7)
        broken[cell][0]+=100
        with self.assertRaisesRegex(ValueError,'physical pressure force'):
            verify_pressure_forces(m,p.config,pressure,saturation,{'L':broken,'G':expected})

    def finished_state(self, root, processes=1):
        p=P006Problem(config=ManifoldConfig(processes=processes));case=p.write_case(p.initial_design(),root/'source')
        manifest=json.loads((case/'case.json').read_text())
        saved=case/'10';saved.mkdir()
        for name in FIELDS:shutil.copy2(case/'0'/name,saved/name)
        text=(saved/'T').read_text()
        text=re.sub(r'\binternalField\b[^;]+;','internalField uniform 354;',text,count=1)
        # The guess must not replace the target's prescribed inlet temperature.
        (saved/'T').write_text(text.replace('value uniform 353.0','value uniform 351.0'))
        post=case/'postProcessing/p006';post.mkdir(parents=True)
        (post/'status.json').write_text(json.dumps(dict(converged=False,iteration=10,
            cells=manifest['cells'],processes=processes)))
        (case/'log.flowfieldP006Foam').write_text('Iteration = 10\nEnd\n')
        (case/'result.json').write_text(json.dumps(dict(status='failed',input_sha256=manifest['input_sha256'])))
        return p,case

    def test_finished_state_is_a_new_guess_with_fresh_boundaries_and_no_inherited_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            original={name:(source/'10'/name).read_bytes() for name in FIELDS}
            # CLI millimetres-to-normalized conversion differs by one ulp.
            cli_design=[(1.-.7)/.8]*3
            case=generate_case(root/'continued',p.decode(cli_design),
                ManifoldConfig(processes=2,iterations=4000),initial_state=source)
            manifest=json.loads((case/'case.json').read_text())
            self.assertEqual(values((case/'0/T').read_text(),'internalField',manifest['cells']),[354]*manifest['cells'])
            self.assertIn('value uniform 353.0;', (case/'0/T').read_text())
            self.assertNotIn('value uniform 351.0;', (case/'0/T').read_text())
            self.assertFalse(manifest['initial_state']['qualification_inherited'])
            self.assertFalse(manifest['initial_state']['source_converged'])
            self.assertFalse((case/'postProcessing').exists())
            self.assertEqual(original,{name:(source/'10'/name).read_bytes() for name in FIELDS})
            CaseEvaluator(p.case_definition)._load_case(case)

    def test_initial_guess_refuses_running_case_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            (source/'result.json').unlink()
            with self.assertRaisesRegex(ValueError,'finished run'):
                generate_case(root/'continued',p.decode(p.initial_design()),p.config,initial_state=source)
            self.assertFalse((root/'continued').exists())

    def coupled_state(self, root, processes=1):
        p,source=self.finished_state(root,processes)
        path=source/'postProcessing/p006/status.json'
        status=json.loads(path.read_text());status.update(momentum_fraction=1.,thermal_active=True)
        path.write_text(json.dumps(status))
        (source/'postProcessing/p006/history.csv').write_text(
            'iteration,momentum_fraction,thermal_active,applied_voltage_V\n10,1,1,2.2\n')
        return p,source

    def test_recovery_reconstructs_a_copy_and_preserves_completed_coupling(self):
        from flowfield.problems.level1_small_continuous.p006.recovery import recover_completed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.coupled_state(root,processes=2)
            (source/'processor0').mkdir();shutil.move(source/'10',source/'processor0/10')
            original={str(x.relative_to(source)):x.read_bytes() for x in source.rglob('*') if x.is_file()}
            def reconstruct(argv,**kw):
                copied=Path(argv[2]);self.assertNotEqual(copied,source)
                self.assertEqual(argv[-2:],['-time','10'])
                shutil.copytree(copied/'processor0/10',copied/'10')
            with patch.dict(os.environ,WM_PROJECT_VERSION='v2312'), \
                 patch('flowfield.problems.level1_small_continuous.p006.recovery.shutil.which',return_value='/test/reconstructPar'), \
                 patch('flowfield.problems.level1_small_continuous.p006.recovery._run_command',side_effect=reconstruct) as command, \
                 contextlib.redirect_stdout(io.StringIO()):
                result=recover_completed(source,root/'recovery')
            self.assertEqual(command.call_count,1);self.assertEqual(result['status'],'prepared')
            self.assertEqual(original,{str(x.relative_to(source)):x.read_bytes() for x in source.rglob('*') if x.is_file()})
            continued=Path(result['continuation_case']);m=json.loads((continued/'case.json').read_text())
            self.assertEqual(m['config'],p.config.to_dict())
            self.assertEqual(m['design_geometry'],p.decode(p.initial_design()))
            self.assertTrue(m['initial_state']['coupling']['restored'])
            self.assertFalse(m['initial_state']['qualification_inherited'])
            self.assertFalse((continued/'result.json').exists())
            self.assertEqual(json.loads((Path(result['recovered_case'])/'result.json').read_text())['status'],'failed')
            CaseEvaluator(p.case_definition)._load_case(continued)

    def test_recovery_rejects_live_or_nested_sources_before_writing(self):
        from flowfield.problems.level1_small_continuous.p006.recovery import recover_completed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,source=self.coupled_state(root)
            with self.assertRaisesRegex(ValueError,'inside the source'):
                recover_completed(source,source/'recovery')
            self.assertFalse((source/'recovery').exists())
            with (source/'log.flowfieldP006Foam').open('a') as stream:stream.write('Iteration = 11\n')
            with self.assertRaisesRegex(ValueError,'must have finished'):
                recover_completed(source,root/'recovery')
            self.assertFalse((root/'recovery').exists())

    def test_failed_reconstruction_never_prepares_a_continuation(self):
        from flowfield.problems.level1_small_continuous.p006.recovery import recover_completed
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,source=self.coupled_state(root,processes=2)
            with patch.dict(os.environ,WM_PROJECT_VERSION='v2312'), \
                 patch('flowfield.problems.level1_small_continuous.p006.recovery.shutil.which',return_value='/test/reconstructPar'), \
                 patch('flowfield.problems.level1_small_continuous.p006.recovery._run_command',side_effect=subprocess.CalledProcessError(1,['reconstructPar'])), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):recover_completed(source,root/'recovery')
            self.assertFalse((root/'recovery/continuation').exists())
            record=json.loads((root/'recovery/recovery.json').read_text())
            self.assertEqual(record['status'],'failed');self.assertIn('CalledProcessError',record['error'])

    def test_unconverged_full_coupled_guess_retains_stage_and_fresh_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.coupled_state(root)
            original={x:x.read_bytes() for x in source.rglob('*') if x.is_file()}
            case=generate_case(root/'continued',p.decode(p.initial_design()),p.config,initial_state=source)
            props=(case/'constant/modelProperties').read_text()
            self.assertIn('initial_momentum_fraction 1;',props)
            self.assertIn('initial_thermal_active true;',props)
            m=json.loads((case/'case.json').read_text())
            self.assertTrue(m['initial_state']['coupling']['restored'])
            self.assertFalse(m['initial_state']['source_converged'])
            self.assertFalse(m['initial_state']['qualification_inherited'])
            self.assertEqual(original,{x:x.read_bytes() for x in source.rglob('*') if x.is_file()})
            self.assertFalse((case/'result.json').exists())
            CaseEvaluator(p.case_definition)._load_case(case)

    def test_initial_guess_cannot_invent_full_coupling_from_inconsistent_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.coupled_state(root)
            history=source/'postProcessing/p006/history.csv'
            for row in ('10,1,0,2.2','9,1,1,2.2','10,.5,1,2.2','10,1,1,1.4','10,1,1,nan'):
                history.write_text('iteration,momentum_fraction,thermal_active,applied_voltage_V\n'+row+'\n')
                with self.assertRaisesRegex(ValueError,'coupling evidence'):
                    generate_case(root/'invalid',p.decode(p.initial_design()),p.config,initial_state=source)
                self.assertFalse((root/'invalid').exists())

    def test_explicit_lower_initial_voltage_keeps_normal_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.coupled_state(root)
            config=ManifoldConfig(processes=1,initial_voltage=1.4)
            case=generate_case(root/'continued',p.decode(p.initial_design()),config,initial_state=source)
            self.assertNotIn('initial_thermal_active true;', (case/'constant/modelProperties').read_text())
            m=json.loads((case/'case.json').read_text())
            self.assertFalse(m['initial_state']['coupling']['restored'])

    def test_iteration_budget_exit_is_distinct_from_fatal_or_missing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            status_path=source/'postProcessing/p006/status.json'
            status=json.loads(status_path.read_text());status['iteration']=p.config.iterations
            status_path.write_text(json.dumps(status))
            self.assertIn('3000-iteration limit',iteration_limit_reason(source))
            (source/'log.flowfieldP006Foam').write_text('FOAM FATAL ERROR\nEnd\n')
            self.assertIsNone(iteration_limit_reason(source))
            (source/'log.flowfieldP006Foam').write_text('End\n')
            status['converged']=True;status_path.write_text(json.dumps(status))
            self.assertIsNone(iteration_limit_reason(source))

    def test_initial_guess_refuses_changed_geometry_or_invalid_saved_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            with self.assertRaisesRegex(ValueError,'same P006 geometry'):
                generate_case(root/'changed',p.decode([.5,.375,.375]),p.config,initial_state=source)
            (source/'10/T').write_text('internalField uniform nan;')
            with self.assertRaises(ValueError):
                generate_case(root/'invalid',p.decode(p.initial_design()),p.config,initial_state=source)
            self.assertFalse((root/'invalid').exists())

    def test_initial_guess_preserves_face_fluxes_and_fresh_inlet_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            saved=source/'10/phiL'
            text=re.sub(r'\binternalField\b[^;]+;','internalField uniform 1e-12;',saved.read_text(),count=1)
            text=re.sub(r'\banodeInlet\s*\{.*?\}',
                'anodeInlet { type calculated; value uniform -1; }',text,flags=re.S)
            text=re.sub(r'\banodeOutlet\s*\{.*?\}',
                'anodeOutlet { type calculated; value uniform 2e-8; }',text,flags=re.S)
            saved.write_text(text)
            case=generate_case(root/'continued',p.decode(p.initial_design()),p.config,initial_state=source)
            copied=(case/'0/phiL').read_text()
            self.assertIn('oriented oriented;',copied)
            self.assertIn('oriented oriented;', (case/'0/phiG').read_text())
            self.assertIn('internalField uniform 1e-12;',copied)
            outlet=re.search(r'\banodeOutlet\s*\{(.*?)\}',copied,re.S).group(1)
            self.assertIn('value uniform 2e-8;',outlet)
            inlet=re.search(r'\banodeInlet\s*\{(.*?)\}',copied,re.S).group(1)
            original=re.search(r'\banodeInlet\s*\{(.*?)\}',(source/'0/phiL').read_text(),re.S).group(1)
            self.assertEqual(inlet,original)
            self.assertTrue(json.loads((case/'case.json').read_text())['initial_state']['preserves_face_fluxes'])

    def test_initial_guess_refuses_incomplete_face_fluxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,source=self.finished_state(root)
            (source/'10/phiG').write_text('internalField nonuniform List<scalar> 2 (0);')
            with self.assertRaises(ValueError):
                generate_case(root/'invalid',p.decode(p.initial_design()),p.config,initial_state=source)
            self.assertFalse((root/'invalid').exists())

    def test_spline_knots_symmetry_and_area(self):
        w=spline_widths([.7,1.5,.7])
        self.assertEqual((w[0],w[4],w[8]),(.7,1.5,.7))
        self.assertEqual(w,list(reversed(w)))
        self.assertEqual(spline_widths([1,1,1]),[1]*9)
        p=create_problem('P006');g=p.decode(p.initial_design())
        self.assertEqual(p.dimension,3)
        self.assertAlmostEqual(g['contact_area_mm2'],207.)
        self.assertAlmostEqual(g['constraints']['contact_area'],-.05)
        self.assertTrue(all(v<=0 for v in g['constraints'].values()))

    def test_natural_spline_matches_independent_scipy(self):
        if not importlib.util.find_spec('scipy'):self.skipTest('optional scipy absent')
        from scipy.interpolate import CubicSpline
        for controls in ([.7,1.5,.7],[1.5,.7,.7],[.7,1.2,1.5]):
            expected=CubicSpline([0,4,8],controls,bc_type='natural')(range(9))
            for a,b in zip(spline_widths(controls),expected):self.assertAlmostEqual(a,b,places=13)

    def test_spline_overshoot_is_a_constraint(self):
        p=P006Problem();g=p.decode([1,0,0])
        self.assertGreater(g['constraints']['minimum_width'],0)
        self.assertGreater(g['constraints']['contact_area'],0)

    def test_layer_volumes_and_counterflow_ports(self):
        p=P006Problem();g=p.decode(p.initial_design());m=build_geometry(g,p.config)
        volumes={i:0. for i in range(9)}
        for (i,j,k),mat in zip(m['cells'],m['ids']):
            volumes[mat]+=(m['x'][i+1]-m['x'][i])*(m['y'][j+1]-m['y'][j])*(m['z'][k+1]-m['z'][k])
        self.assertAlmostEqual(sum(volumes.values()),18*25*3.7178,places=7)
        self.assertAlmostEqual(volumes[4],18*25*.0178,places=8)
        self.assertAlmostEqual(volumes[5],18*25*.05,places=8)
        self.assertAlmostEqual(volumes[7],18*2+9*23,places=8)
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case')
            data=json.loads((case/'case.json').read_text())
            self.assertEqual(data['cells'],len(m['cells']))
            self.assertTrue(all(abs(a-1e-6)<1e-15 for a in data['inlet_areas_m2'].values()))
            velocity=(case/'0/Ul').read_text()
            self.assertIn('uniform (-0.277777777778 0 0)',velocity)
            self.assertIn('uniform (0.277777777778 0 0)',velocity)
            for phase,ratio in [('L',1),('G',1/9)]:
                for patch in ('anodeInlet','cathodeInlet'):
                    block=re.search(r'\b'+patch+r'\s*\{(.*?)\}',(case/'0'/('phi'+phase)).read_text(),re.S).group(1)
                    count=int(re.search(r'List<scalar>\s+(\d+)',block).group(1))
                    self.assertAlmostEqual(sum(values(block,'value',count)),-15e-6/60*ratio,places=16)
            CaseEvaluator(p.case_definition)._load_case(case)
            with self.assertRaises(FileExistsError):p.write_case(p.initial_design(),case)
            (case/'0/T').write_text('altered')
            with self.assertRaises(ValueError):CaseEvaluator(p.case_definition)._load_case(case)

    def test_manifold_sweep_keeps_footprint_and_contact_reference(self):
        for w in (1,1.5,2,2.5,3):
            p=P006Problem(config=ManifoldConfig(manifold_width_mm=w))
            g=p.decode(p.initial_design())
            self.assertEqual(g['footprint_mm'],[18,25])
            self.assertEqual(g['clear_branch_length_mm'],25-2*w)
            self.assertAlmostEqual(g['contact_area_mm2']/g['reference_contact_area_mm2'],1.)

    def test_channel_width_optimization_preserves_mesh_topology(self):
        p=P006Problem()
        meshes=[build_geometry(p.decode(x),p.config) for x in ([.375]*3,[.475,.375,.375],[0,1,0],[1,0,1])]
        self.assertTrue(all(m['cells']==meshes[0]['cells'] for m in meshes))
        self.assertTrue(all(m['ids']==meshes[0]['ids'] for m in meshes))

    def test_config_rejects_invalid_inputs_and_changes_identity(self):
        for kw in (dict(manifold_width_mm=4),dict(processes=0),dict(processes=2.5),
                   dict(contact_tolerance=0),dict(permeability_ptl=-1),dict(cell_voltage=3),
                   dict(field_tolerance=.1),dict(inlet_liquid_saturation=1)):
            with self.subTest(kw=kw),self.assertRaises(ValueError):ManifoldConfig(**kw)
        self.assertNotEqual(ManifoldConfig().fingerprint,ManifoldConfig(manifold_width_mm=2).fingerprint)

    def test_field_parser_refuses_nonfinite_and_truncated_output(self):
        self.assertEqual(values('internalField uniform 353;','internalField',2),[353,353])
        for text in ('internalField uniform nan;','internalField nonuniform List<scalar> 2 (1);'):
            with self.assertRaises(ValueError):values(text,'internalField',2)

    def test_vector_parser_refuses_incomplete_or_nonfinite_velocity(self):
        self.assertEqual(vectors('internalField uniform (1 2 3);','internalField',2),[(1,2,3)]*2)
        self.assertEqual(vectors('internalField nonuniform List<vector> 2 ((1 2 3) (4 5 6));','internalField',2),[(1,2,3),(4,5,6)])
        for text in ('internalField uniform (1 nan 3);',
                     'internalField nonuniform List<vector> 2 ((1 2 3));',
                     'internalField uniform (1 2);'):
            with self.assertRaises(ValueError):vectors(text,'internalField',2)

    def test_saved_flux_verification_detects_local_error_even_when_global_balances(self):
        p=P006Problem();m=build_geometry(p.decode(p.initial_design()),p.config)
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case');out=case/'1';out.mkdir()
            zero='internalField uniform 0;\nboundaryField\n{\n'
            for phase in ('L','G'):
                text=zero
                for patch in PATCHES:
                    phi=-1e-7 if phase=='L' and patch=='anodeInlet' else 1e-7 if phase=='L' and patch=='anodeOutlet' else 0
                    text+=f'{patch} {{ type calculated; value uniform {phi}; }}\n'
                (out/('phi'+phase)).write_text(text+'}\n')
            n=len(m['cells'])
            checks=verify_phase_fluxes(case,out,m['ids'],[1e-12]*n,{'Sl':[0.]*n,'Sg':[0.]*n},p.config)
            self.assertAlmostEqual(checks['saved_global_L_balance'],0.)
            self.assertGreater(checks['saved_local_L_balance'],.1)

    def test_converged_voltage_continuation_stage_is_not_the_requested_case(self):
        p=P006Problem(config=ManifoldConfig(minimum_iterations=1))
        with tempfile.TemporaryDirectory() as tmp:
            case=p.write_case(p.initial_design(),Path(tmp)/'case')
            manifest=json.loads((case/'case.json').read_text())
            post=case/'postProcessing/p006';post.mkdir(parents=True)
            (post/'status.json').write_text(json.dumps(dict(converged=True,iteration=5,
                cells=manifest['cells'],processes=p.config.processes,momentum_fraction=1,thermal_active=True)))
            (case/'log.flowfieldP006Foam').write_text('CONVERGED: coupled fields and local/global balances passed.\nEnd\n')
            row=dict(iteration=1,charge_residual=0,field_change=0,bounded_cells=0,
                liquid_balance=0,gas_balance=0,local_liquid_balance=0,local_gas_balance=0,
                energy_balance=0,applied_voltage_V=1.4,liquid_momentum_residual=0,
                gas_momentum_residual=0,momentum_fraction=1,thermal_active=1)
            # Small field increments do not prove that the steady momentum
            # equations are satisfied; nor does a converged warm-up voltage.
            for changes in ({},dict(applied_voltage_V=2.2,liquid_momentum_residual=.1),
                            dict(applied_voltage_V=2.2,gas_momentum_residual=.1),
                            dict(applied_voltage_V=2.2,momentum_fraction=.5),
                            dict(applied_voltage_V=2.2,thermal_active=0)):
                with self.subTest(changes=changes):
                    with (post/'history.csv').open('w') as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                        for i in range(1,6):writer.writerow(dict(row,**changes,iteration=i))
                    with self.assertRaisesRegex(ValueError,'convergence or conservation'):
                        extract_metrics(case,p.config)

            # The last iteration alone cannot qualify a continuation stage.
            row.update(applied_voltage_V=2.2,T_std_K=1.)
            for bad_iteration in range(1,6):
                with (post/'history.csv').open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                    for i in range(1,6):
                        writer.writerow(dict(row,iteration=i,thermal_active=0 if i==bad_iteration else 1))
                with self.assertRaisesRegex(ValueError,'convergence or conservation'):
                    extract_metrics(case,p.config)
            with (post/'history.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                for i in range(1,6):writer.writerow(dict(row,iteration=i))
            with patch('flowfield.problems.level1_small_continuous.p006.metrics.verify_fields',
                       return_value={'temperature_std_K':1.}) as verify:
                self.assertEqual(extract_metrics(case,p.config)['objective'],1.)
                verify.assert_called_once()


if __name__=='__main__':unittest.main()
