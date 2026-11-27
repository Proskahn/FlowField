"""Durable optimization journals and exclusive ownership of a run directory.

State is reconstructed from JSON observations, never executable pickle files.
The original observation stream remains append-only, including on recovery.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import socket


class ResumeError(RuntimeError):
    """History cannot be resumed without changing the experiment."""


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_json(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


@contextmanager
def run_lock(path):
    """An OS lock releases on process exit; stale files are safe to retain.

Do not unlink the lock file: another process might already hold its inode.
Requires a filesystem supporting POSIX flock (Linux/macOS supported).
"""
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResumeError(f'Run is already active (locked): {path}') from exc
        try:
            stream.seek(0)
            stream.truncate()
            append_json(stream, dict(pid=os.getpid(), host=socket.gethostname()))
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def check_legacy_pid(output):
    """Respect the nohup PID sidecar used before the locking runner existed."""
    # Linux legacy runners have no advisory lock. Also inspect their command
    # lines, because a new nohup launch can overwrite the old .pid sidecar.
    proc = Path('/proc')
    if proc.is_dir():
        for process in proc.iterdir():
            if not process.name.isdigit() or int(process.name)==os.getpid():
                continue
            try:
                if process.stat().st_uid != os.getuid():
                    continue
                argv = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
                if 'flowfield.optimization' not in argv:
                    continue
                targets = [arg.split('=',1)[1] for arg in argv if arg.startswith('--output=')]
                targets += [argv[i+1] for i,arg in enumerate(argv[:-1]) if arg=='--output']
                cwd = (process/'cwd').resolve(strict=True)
                matches = any((cwd/target).resolve()==Path(output).resolve() for target in targets)
            except OSError:
                continue
            if matches:
                raise ResumeError(f'Optimization process {process.name} is still using {output}')
    path = Path(str(output) + '.pid')
    if not path.exists():
        return
    try:
        pid = int(path.read_text().strip())
        if pid <= 0:
            raise ValueError()
    except ValueError as exc:
        raise ResumeError(f'Invalid PID file: {path}') from exc
    # A new nohup launcher writes our own PID before main reaches this check.
    if pid == os.getpid():
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        pass
    raise ResumeError(f'PID {pid} from {path} still exists. Verify the old run has stopped; '
                      'do not resume alongside it. If this PID belongs to another process, '
                      'rename the stale PID file after verifying that fact.')


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_text()
    # Fail closed: an incomplete started record could represent spent budget.
    if data and not data.endswith('\n'):
        raise ResumeError(f'Truncated journal: {path}; preserve it for recovery')
    try:
        rows = [json.loads(line) for line in data.splitlines()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError('Expected objects')
        return rows
    except (ValueError, TypeError) as exc:
        raise ResumeError(f'Invalid journal: {path}') from exc


def pristine_initialization(output):
    """A crash before the atomic manifest write cannot have entered CFD yet."""
    return all(path.name in ('.optimizer.lock','experiment.json.tmp') for path in Path(output).iterdir())


def load_history(output, budget, penalty):
    """Return chronological observations and explicit pending-attempt recoveries."""
    output = Path(output)
    events = read_jsonl(output/'events.jsonl')
    starts = [event for event in events if event.get('event') == 'started']
    if [row.get('evaluation') for row in starts] != list(range(1, len(starts)+1)):
        raise ResumeError('Started evaluations are not a contiguous unique sequence')
    if len(starts) > budget:
        raise ResumeError('Saved attempts exceed the original budget')
    observations = read_jsonl(output/'evaluations.jsonl')
    recoveries = read_jsonl(output/'recoveries.jsonl')
    by_index, recovered = {}, {}
    for rows, target in ((observations, by_index), (recoveries, recovered)):
        for row in rows:
            index = row.get('evaluation')
            if type(index) is not int or not 1 <= index <= len(starts) or index in target:
                raise ResumeError('Duplicate or unstarted evaluation in journal')
            start = starts[index-1]
            if any(row.get(key) != start.get(key) for key in ('x', 'phase', 'generation', 'proposal')):
                raise ResumeError(f'Observation differs from started evaluation {index}')
            target[index] = row
    history, pending = [], []
    for start in starts:
        index = start['evaluation']
        row = by_index.get(index)
        if row is not None and row.get('objective') is not None:
            if index in recovered:
                raise ResumeError(f'Completed evaluation {index} also has a recovery')
        elif index in recovered:
            row = recovered[index]
        else:
            if penalty is None:
                raise ResumeError(f'Evaluation {index} has no scalar result. Resuming it requires '
                                  'an originally configured --failure-penalty; no CFD is retried.')
            raw = dict(status='penalized', objective=penalty, objectives={}, constraints={},
                       error='Attempt interrupted or failed before a scalar result was saved',
                       metadata=dict(valid_cfd=False, recovered_interruption=True,
                                     failure_penalty=penalty,
                                     case_directory=(row or {}).get('case_directory')))
            row = dict(evaluation=index, x=start['x'], phase=start['phase'],
                       generation=start['generation'], objective=penalty, valid=False,
                       feasible=False, result=raw, wall_time_s=0., recovered=True)
            if 'proposal' in start:
                row['proposal'] = start['proposal']
            pending.append(row)
        value = row.get('objective')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ResumeError(f'Invalid scalar at evaluation {index}')
        if not isinstance(row.get('result'), dict) or row['result'].get('objective') != value:
            raise ResumeError(f'Missing or inconsistent raw result at evaluation {index}')
        history.append(row)
    return history, pending


def validate_budget_extension(previous, target):
    if type(target) is not int or target < 1:
        raise ResumeError('Extended budget must be a positive integer total')
    if target < previous:
        raise ResumeError(f'Cannot decrease the saved budget from {previous} to {target}')


def extend_budget_metadata(saved, target):
    """Return an audited increase; an identical target is an idempotent retry."""
    previous = saved['budget']
    validate_budget_extension(previous, target)
    result = dict(saved)
    if target != previous:
        result.update(budget=target, original_budget=saved.get('original_budget', previous),
            budget_extensions=[*saved.get('budget_extensions', []), dict(
                previous_budget=previous, budget=target,
                timestamp_utc=datetime.now(timezone.utc).isoformat())])
    return result


def validate_experiment(saved, current, *, allow_budget_extension=False):
    # Dataclass configurations can contain tuples; JSON stores those as lists.
    current = json.loads(json.dumps(current,allow_nan=False))
    keys = ('algorithm', 'problem', 'problem_name', 'dimension', 'budget', 'seed',
            'initial_design', 'population', 'sigma_normalized', 'diagonal_covariance',
            'lower_bounds', 'upper_bounds', 'failure_penalty', 'ioh_enabled', 'config',
            'max_seconds_soft', 'optimization_mode', 'objective_source')
    for key in keys:
        if key == 'budget' and allow_budget_extension:
            validate_budget_extension(saved[key], current[key])
            continue
        if saved.get(key) != current.get(key):
            raise ResumeError(f'Cannot resume with changed {key}')
    if saved.get('schema_version', 0) >= 3:
        for key in ('optimizer', 'optimizer_revision'):
            if saved.get(key) != current.get(key):
                raise ResumeError(f'Cannot resume with changed {key}')
    repair = current.get('optimizer_settings', {}).get('candidate_repair')
    if current.get('optimizer') in ('random', 'bo', 'turbo', 'cobyla', 'lqcmaes', 'qlognehvi', 'qlognparego') or repair is not None:
        if saved.get('optimizer_settings') != current.get('optimizer_settings'):
            raise ResumeError('Cannot resume with changed optimizer_settings')
        dependencies = ('numpy', 'scipy') if current.get('optimizer') == 'cobyla' else ('numpy', 'scipy', 'scikit-learn')
        if current.get('optimizer') == 'lqcmaes':
            # Pycma uses SciPy's Kendall tau when installed, its own fallback otherwise.
            dependencies = ('cma', 'numpy', 'scipy')
        if current.get('optimizer') == 'random':
            dependencies = ('python', 'numpy', 'scipy') if current['optimizer_settings']['candidate_repair'] is not None else ('python',)
        if current.get('optimizer') == 'cmaes':
            dependencies = ('cma', 'numpy', 'scipy')
        if current.get('optimizer') == 'nsga2':
            dependencies = ('python', 'numpy', 'scipy')
        if current.get('optimizer') in ('qlognehvi', 'qlognparego'):
            dependencies = ('python', 'numpy', 'scipy', 'torch', 'botorch', 'gpytorch', 'linear-operator')
    else:
        dependencies = ('cma', 'numpy') if current['population'] is not None and current.get('optimizer') != 'nsga2' else ()
    for key in dependencies:
        if saved['versions'].get(key) != current['versions'].get(key):
            raise ResumeError(f'Resume requires the original {key} version: {saved["versions"].get(key)}')


def next_session(output):
    parent = Path(output)/'resume-sessions'
    parent.mkdir(exist_ok=True)
    for index in range(1, 1000000):
        path = parent/f'{index:06d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise ResumeError('Too many resume sessions')


def restore_options(args, supplied):
    """Use the recorded CLI settings, or recover the v0.6.0 suite settings."""
    output = args.output.resolve()
    if not (output/'plan.json').is_file():
        raise ResumeError(f'Missing {output}/plan.json; the console log is not a checkpoint')
    if (output/'run-options.json').exists():
        options = json.loads((output/'run-options.json').read_text())
    else:
        plan = json.loads((output/'plan.json').read_text())
        experiments = [json.loads(path.read_text()) for path in sorted(output.glob('*/experiment.json'))]
        if not experiments:
            raise ResumeError('Legacy run has no experiment.json. No recoverable experiment settings exist.')
        options = dict(problems=[p['problem'] for p in plan['problems']],
            budget=plan['budget_per_problem'],seed=plan['seed'],mesh=plan['mesh'],
            population=4,sigma=.1,p003_sigma=.005,method='auto',
            backend=plan.get('runtime_preflight',{}).get('backend','auto'),
            image=plan.get('runtime_preflight',{}).get('image',args.image),timeout=1800.,
            p004_dataset=None,p004_model=None)
        for option, key in (('ioh','ioh_enabled'),('failure_penalty','failure_penalty'),
                            ('max_seconds_per_problem','max_seconds_soft')):
            values = [e.get(key) for e in experiments]
            if any(value != values[0] for value in values):
                raise ResumeError(f'Conflicting legacy settings: {key}')
            options[option] = values[0]
        populations = {e['population'] for e in experiments if e.get('population') is not None}
        if len(populations)>1:
            raise ResumeError('Conflicting legacy population sizes')
        if populations:
            options['population'] = populations.pop()
        for entry in plan['problems']:
            if entry.get('sigma_normalized') is not None:
                options['p003_sigma' if entry['problem']=='P003' else 'sigma'] = entry['sigma_normalized']
        if any(p.get('algorithm')=='RandomSearch' for p in plan['problems']):
            options['method'] = 'random'
        # Hash validation through problem_name checks the supplied data/model.
        if any(p['problem']=='P004' and p.get('status')!='skipped' for p in plan['problems']):
            if args.p004_dataset is None and args.p004_model is None:
                raise ResumeError('Legacy P004 resume requires the original --p004-dataset or --p004-model')
            options.update(p004_dataset=args.p004_dataset,p004_model=args.p004_model)
    # run-options.json is the atomic commit point for a suite budget increase.
    # A crash may leave plan.json or an individual experiment at the old budget;
    # the recorded authorization lets a plain --resume finish the update.
    saved_plan = json.loads((output/'plan.json').read_text())
    # Runs created before --dimension used their original config/defaults.
    # Reject adding a dimension during resume rather than changing the problem.
    options.setdefault('dimension', None)
    recorded_budget = options.get('budget', saved_plan.get('budget_per_problem'))
    if recorded_budget is None and args.extend_budget is not None:
        raise ResumeError('Cannot extend a run with no recorded original budget')
    if recorded_budget is not None:
        options['budget'] = recorded_budget
    args._saved_budget = options.get('budget', args.budget)
    args._budget_metadata = dict(budget=args._saved_budget, **{key: options[key] for key in
                               ('original_budget', 'budget_extensions') if key in options})
    args._allow_budget_extension = bool(options.get('budget_extensions'))
    saved_entries = saved_plan['problems']
    saved_ids = [entry['problem'] for entry in saved_entries]
    if options.get('problems') == ['all']:
        options['problems'] = saved_ids
    for key,value in options.items():
        if key in ('output','resume','dry_run','extend_budget','original_budget','budget_extensions'):
            continue
        flag = '--'+key.replace('_','-')
        old = getattr(args,key)
        # Execution settings may change; objective and optimizer settings may not.
        if flag in supplied and key in ('backend','image','timeout','p004_dataset','p004_model'):
            continue
        comparable = str(old.resolve()) if isinstance(old, Path) else old
        if flag in supplied and comparable != value:
            # --problems all represents the same suite when all IDs were saved.
            if not (key=='problems' and old==['all'] and value==saved_ids):
                raise ResumeError(f'{flag} differs from the saved experiment; use --extend-budget to increase '
                                  'the budget, and keep the other experiment settings unchanged')
        setattr(args,key,Path(value) if key in ('p004_dataset','p004_model','p005_config','config') and value is not None else value)
    if args.extend_budget is not None:
        validate_budget_extension(args._saved_budget, args.extend_budget)
        args.budget = args.extend_budget
        args._allow_budget_extension = True
    args.command_timeouts = {entry['problem']: entry['command_timeout_s'] for entry in saved_entries
                             if 'command_timeout_s' in entry}
    return args
