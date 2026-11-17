"""Isolated StartLux serving controls and bounded, PID-reuse-safe ownership.

Historical environments/checkpoints are read-only. Every token/start/measurement
worker is the sole child of a Linux subreaper; surviving descendants are signalled
through kernel process descriptors (including the tested RHEL8 proc backend).
Old PID/PGID receipts never grant authority to signal a process.
"""
from contextlib import contextmanager
import copy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .. import runtime as frozen
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..direct_matrix.runtime import slurm_job_state
from ..learnability_profile import ownership, pidfd
from ..learnability_profile import study as life
from ..runtime_compat import legacy_environment
from ..service_study.identity import fixed_write
from ..size_control import runtime as size_runtime
from ..upgrade_study import runtime as old
from ..upgrade_study.launcher import engine_specs
from ..v3 import runtime as v3
from ..v3.cli import slurm_hours_left

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = 'anli-budget-serving-runtime-v1'
ROLES = ('startlux4b', 'startlux9b')
REVISIONS = {'startlux4b': '416088f77cfce09633906a3675c7cd2e5dabcf10',
             'startlux9b': '18a17f734966b213b2f00fa9c915e8b9af2a9b20'}
REPOS = {'startlux4b': 'startlux-models/StartLux-Decision-v1.1-4B-Instant',
         'startlux9b': 'startlux-models/StartLux-Decision-v1.1-9B-Instant'}
PORTS = {'startlux4b': (18380, 18381), 'startlux9b': (18390, 18391)}
CONDITIONS = ('direct4b', 'direct9b', 'threshold', 'budget', 'random')
CONTROL_LAYOUT = {'direct4b': ('startlux4b', .9), 'direct9b': ('startlux9b', .9),
                  'pair4b': ('startlux4b', .4), 'pair9b': ('startlux9b', .5)}
CONTROL_NAMES = ('lock.json', 'installations.json', 'selected_profiles.json', 'selected_memory.json')
MAX_NUM_SEQS, MAX_MODEL_LEN, STARTUP_SECONDS, CLEANUP_MARGIN = 64, 8192, 1200, 180
normal, safe = life.normal, life.safe


def condition_roles(condition):
    if condition not in CONDITIONS:
        raise StudyError(f'Unknown budget-serving condition: {condition}')
    return ('startlux4b',) if condition == 'direct4b' else ('startlux9b',) if condition == 'direct9b' else ROLES


def control(output, condition, role):
    if role not in condition_roles(condition):
        raise StudyError('Role does not belong to this condition')
    key = condition if condition in ('direct4b', 'direct9b') else ('pair4b' if role == 'startlux4b' else 'pair9b')
    return safe(output, 'runtime-controls/' + key)


def _plan(runtime_root, output, *, check_snapshots=True):
    root, output = normal(runtime_root), normal(output)
    v1 = root/'results/solo-v1'
    size = root/'results/aux-size-control-v1-r2'
    if output.parent != root/'results' or not output.name.startswith('aux-anli-budget-serving-v1'):
        raise StudyError('Use a new aux-anli-budget-serving-v1 directory immediately under runtime/results')
    for source in (PROJECT, v1, size, root/'results/aux-anli-learnability-v1-r2'):
        if output == source or output in source.parents or source in output.parents:
            raise StudyError('Budget-serving output overlaps a source or historical result root')
    # safe() rejects symlinked control/output roots before any writes.
    safe(output, 'runtime/binding.json')
    paths = {'v1_lock': safe(v1, 'runtime/lock.json'), 'v1_installations': safe(v1, 'runtime/installations.json'),
             'nine_lock': safe(size, 'runtime-controls/startlux9b/runtime/lock.json'),
             'nine_installations': safe(size, 'runtime-controls/startlux9b/runtime/installations.json'),
             'nine_binding': safe(size, 'runtime/binding.json')}
    base, installed = read_json(paths['v1_lock']), read_json(paths['v1_installations'])
    defaults = read_json(PROJECT/'configs/runtime_defaults.json')
    expected_hash = fingerprint(defaults)
    if ({k: base.get(k) for k in defaults} != defaults or base.get('defaults_hash') != expected_hash
            or base.get('lock_hash') != expected_hash or installed.get('lock_hash') != expected_hash):
        raise StudyError('Historical V1 defaults/installation identity changed')
    name = 'vllm-' + base['packages']['vllm_primary']
    env = installed.get('environments', {}).get(name)
    if not env:
        raise StudyError('Pinned V1 vLLM environment is missing')
    python = Path(env.get('python', ''))
    work = v1/'runtime/work'
    if not python.is_absolute() or python != work/'envs'/name/'bin/python':
        raise StudyError('Historical vLLM interpreter path is not the original runtime environment')
    if check_snapshots and not python.is_file():
        raise StudyError('Pinned historical vLLM interpreter is missing; no environment is replaced')
    nine, nine_installed = read_json(paths['nine_lock']), read_json(paths['nine_installations'])
    nine_defaults = copy.deepcopy(defaults)
    nine_defaults['models']['startlux'].update(repo=REPOS['startlux9b'], revision=REVISIONS['startlux9b'], native_temperature=1.1014)
    h = fingerprint(nine_defaults)
    if ({k: nine.get(k) for k in defaults} != nine_defaults or nine.get('lock_hash') != h
            or nine.get('defaults_hash') != h or nine_installed.get('lock_hash') != h
            or nine.get('size-control_source', {}).get('parent_lock_hash') != base['lock_hash']
            or nine_installed.get('environments', {}).get(name) != env):
        raise StudyError('Historical StartLux9B lock/environment is not the expected size-control derivation')
    size_work = Path(read_json(paths['nine_binding']).get('work_dir', ''))
    if size_work != size/'runtime-work':
        raise StudyError('Historical 9B work directory changed')
    models = {'startlux4b': installed.get('models', {}).get('startlux'),
              'startlux9b': nine_installed.get('models', {}).get('startlux')}
    for role, model in models.items():
        path = Path((model or {}).get('model_path', ''))
        allowed = work if role == 'startlux4b' else size_work
        if (not model or model.get('revision') != REVISIONS[role] or not path.is_absolute()
                or path.name != REVISIONS[role] or allowed not in path.parents):
            raise StudyError(f'Historical {role} checkpoint record changed')
        if check_snapshots:
            if not path.is_dir() or not (path/'config.json').is_file() or not list(path.glob('*.safetensors')):
                raise StudyError(f'Pinned {role} snapshot is missing; no download or substitution')
            if role == 'startlux9b':
                size_runtime._verify_model_metadata(path)
    controls = {}
    for key, (role, share) in CONTROL_LAYOUT.items():
        parent = base if role == 'startlux4b' else nine
        derived = copy.deepcopy(parent)
        derived['models']['startlux'].update(port=PORTS[role][0], engine_port=PORTS[role][1])
        derived['startup_timeout_seconds'] = STARTUP_SECONDS
        derived_defaults = {k: derived[k] for k in defaults}
        derived.update(lock_hash=fingerprint(derived_defaults), defaults_hash=fingerprint(derived_defaults))
        derived['budget-serving_source'] = {'role': role, 'parent_lock_hash': parent['lock_hash'],
                'derivation': 'new ports and bounded startup; frozen checkpoint/interface/precision unchanged'}
        controls[key] = {'lock.json': derived,
                        'installations.json': {'lock_hash': derived['lock_hash'], 'environments': {name: copy.deepcopy(env)},
                                               'models': {'startlux': copy.deepcopy(models[role])}},
                        'selected_profiles.json': {'startlux': 'primary'}, 'selected_memory.json': {'startlux': share}}
    binding = {'schema_version': 1, 'protocol': PROTOCOL, 'runtime_root': str(root), 'v1_dir': str(v1),
               'source_work_dir': str(work), 'work_dir': str(output/'runtime-work'),
               'source_files': {str(p): file_hash(p) for p in paths.values()},
               'models': {r: {'revision': REVISIONS[r], 'model_path': models[r]['model_path']} for r in ROLES},
               'ports': {r: list(PORTS[r]) for r in ROLES}, 'max_num_seqs': MAX_NUM_SEQS,
               'max_model_len': MAX_MODEL_LEN, 'startup_timeout_seconds': STARTUP_SECONDS,
               'control_identity': {k: {n: fingerprint(v) for n, v in plan.items()} for k, plan in controls.items()}}
    return binding, controls


def prepare_controls(runtime_root, output):
    binding, controls = _plan(runtime_root, output)
    contents = {safe(output, 'runtime/binding.json'): binding}
    for key, plan in controls.items():
        contents.update({safe(output, f'runtime-controls/{key}/runtime/{n}'): v for n, v in plan.items()})
    for path, value in contents.items():
        if path.exists() and read_json(path) != value:
            raise StudyError(f'Changed budget-serving runtime identity: {path}')
    for path, value in contents.items():
        fixed_write(path, value)
    return binding


def validate_controls(output, *, offline=False):
    binding = read_json(safe(output, 'runtime/binding.json'))
    if (binding.get('protocol') != PROTOCOL or binding.get('max_num_seqs') != MAX_NUM_SEQS
            or binding.get('max_model_len') != MAX_MODEL_LEN or binding.get('ports') != {r: list(PORTS[r]) for r in ROLES}
            or set(binding.get('control_identity', {})) != set(CONTROL_LAYOUT)):
        raise StudyError('Budget-serving runtime binding identity differs')
    for key, identity in binding['control_identity'].items():
        if set(identity) != set(CONTROL_NAMES):
            raise StudyError('Runtime control evidence inventory differs')
        for name, digest in identity.items():
            if fingerprint(read_json(safe(output, f'runtime-controls/{key}/runtime/{name}'))) != digest:
                raise StudyError('Runtime control evidence changed')
    defaults = read_json(PROJECT/'configs/runtime_defaults.json')
    env_name = 'vllm-' + defaults['packages']['vllm_primary']
    environments = []
    for key, (role, share) in CONTROL_LAYOUT.items():
        target = safe(output, f'runtime-controls/{key}/runtime/lock.json').parent
        lock = read_json(target/'lock.json')
        installed = read_json(target/'installations.json')
        expected_defaults = copy.deepcopy(defaults)
        expected_defaults['startup_timeout_seconds'] = STARTUP_SECONDS
        expected_defaults['models']['startlux'].update(repo=REPOS[role], revision=REVISIONS[role],
            native_temperature=1.1103 if role == 'startlux4b' else 1.1014,
            port=PORTS[role][0], engine_port=PORTS[role][1])
        expected_lock_hash = fingerprint(expected_defaults)
        env = installed.get('environments', {}).get(env_name, {})
        model = installed.get('models', {}).get('startlux', {})
        snapshot = Path(model.get('model_path', ''))
        source_work = Path(binding['source_work_dir'])
        allowed = source_work if role == 'startlux4b' else Path(binding['runtime_root'])/'results/aux-size-control-v1-r2/runtime-work'
        if ({k: lock.get(k) for k in defaults} != expected_defaults
                or lock.get('lock_hash') != expected_lock_hash or lock.get('defaults_hash') != expected_lock_hash
                or installed.get('lock_hash') != expected_lock_hash
                or set(installed.get('environments', {})) != {env_name} or set(installed.get('models', {})) != {'startlux'}
                or env.get('python') != str(source_work/'envs'/env_name/'bin/python')
                or model.get('revision') != REVISIONS[role] or snapshot.name != REVISIONS[role]
                or allowed not in snapshot.parents or binding['models'][role].get('model_path') != str(snapshot)
                or binding['models'][role].get('revision') != REVISIONS[role]
                or read_json(target/'selected_memory.json') != {'startlux':share}
                or read_json(target/'selected_profiles.json') != {'startlux':'primary'}):
            raise StudyError('Runtime control semantics differ from fixed protocol')
        environments.append(env)
    if any(env != environments[0] for env in environments[1:]):
        raise StudyError('Runtime roles do not share the same pinned environment')
    if not offline:
        expected, _ = _plan(binding['runtime_root'], output)
        if binding != expected:
            raise StudyError('Historical runtime source changed after initialization')
    return binding


def session(output, *, current=True):
    value = read_json(safe(output, 'runtime/session.json'))
    if (value.get('binding_hash') != fingerprint(validate_controls(output, offline=not current))
            or not isinstance(value.get('deadline_epoch'), (int, float)) or isinstance(value.get('deadline_epoch'), bool)
            or not math.isfinite(value['deadline_epoch'])):
        raise StudyError('Allocation receipt does not match runtime binding')
    if current and (value.get('allocation') != v3._allocation() or value['deadline_epoch'] <= time.time()):
        raise StudyError('Current allocation differs or its recorded allowance expired')
    return value


def initialize_allocation(output):
    binding = validate_controls(output)
    hours, allocation = slurm_hours_left(), v3._allocation()
    if not allocation.get('slurm_job_id') or not math.isfinite(hours) or not 0 < hours <= 24:
        raise StudyError('An actual RUNNING Slurm allocation with at most24 hours is required')
    path = safe(output, 'runtime/session.json')
    previous = read_json(path) if path.exists() else None
    deadline = time.time() + hours * 3600
    if previous:
        if previous.get('binding_hash') != fingerprint(binding):
            raise StudyError('Previous session belongs to another runtime binding')
        if previous.get('allocation') == allocation:
            deadline = min(deadline, previous['deadline_epoch'])
        else:
            old_job = previous.get('allocation', {}).get('slurm_job_id')
            if not old_job or old_job == allocation.get('slurm_job_id'):
                raise StudyError('Allocation changed without an explicitly different Slurm job')
            proof = slurm_job_state(old_job)
            fixed_write(safe(output, f'runtime/allocation-checks/{time.time_ns()}.json'), proof)
            if not proof.get('terminal') or proof.get('job_id') != old_job:
                raise StudyError('Cannot prove that the previous allocation ended')
        fixed_write(safe(output, f'runtime/session-history/{fingerprint(previous)}.json'), previous)
    if deadline <= time.time() + CLEANUP_MARGIN:
        raise StudyError('Insufficient current allocation allowance beyond cleanup margin')
    value = {'allocation': allocation, 'deadline_epoch': deadline, 'binding_hash': fingerprint(binding),
             'updated_at': utc_now(), 'notice': 'Actual squeue allowance, no same-allocation extension'}
    write_json(path, value)
    for key in CONTROL_LAYOUT:
        write_json(safe(output, f'runtime-controls/{key}/runtime/session.json'), value)
    return value


def remaining(output):
    return session(output)['deadline_epoch'] - time.time()


def _recover_descendants(since, *, check_gpu=True, max_seconds=60, term_seconds=10):
    """Only current descendants, with stable descriptor signals; caller owns subreaper."""
    began, seen, signals, empty = time.monotonic(), {}, [], 0
    while time.monotonic() - began < max_seconds:
        owned = life.owned_descendants(since)
        seen.update(owned)
        if not owned:
            empty += 1
            if empty >= 2:
                gpu = old.assert_gpu_owned_by(set()) if check_gpu else None
                return {'observed_owned_processes': list(seen.values()), 'signals': signals,
                        'gpu_checked': check_gpu, 'gpu_compute_processes': gpu,
                        'ownership': 'current descendants of sole-worker Linux subreaper; stable kernel descriptors; no historical PID signals',
                        'seconds': time.monotonic() - began}
        else:
            empty = 0
            sig = signal.SIGTERM if time.monotonic() - began < term_seconds else signal.SIGKILL
            for pid in sorted(owned, reverse=True):
                if time.monotonic() - began >= max_seconds:
                    break
                if life.signal_descendant(pid, since, sig):
                    signals.append({'pid': pid, 'signal': sig.name})
        time.sleep(.1)
    raise StudyError('Owned descendants remain after bounded parent cleanup')


def native_selftest(output):
    """Run the existing double-fork/decoy actor with this study's own receipt binding."""
    binding, sess = validate_controls(output), session(output)
    if sess['deadline_epoch'] - time.time() < CLEANUP_MARGIN + 90:
        raise StudyError('Insufficient allocation allowance for native ownership self-test; resume in next allocation')
    folder = safe(output, f'ownership-selftest/{time.time_ns()}')
    folder.mkdir(parents=True)
    since, actor, decoy = time.time(), None, None
    with life.subreaper():
        try:
            with (folder/'actor.log').open('x') as log:
                actor = subprocess.Popen([sys.executable, '-B', '-m', 'decision_study.learnability_profile.ownership',
                                          '--actor', str(folder)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            ownership.wait_file(folder/'actor-ready.json')
            created = time.time()
            decoy = subprocess.Popen([sys.executable, '-c', ownership.DECOY], start_new_session=True)
            fixed_write(folder/'decoy.json', {'pid': decoy.pid, 'created_epoch': created})
            actor.wait(timeout=25)
            result = read_json(folder/'actor-result.json')
            if (actor.returncode != 0 or decoy.poll() is not None or result.get('status') != 'PASS'
                    or result.get('decoy_pid') != decoy.pid or result.get('decoy_survived') is not True):
                raise StudyError('Native ownership self-test did not preserve unrelated decoy')
        finally:
            if actor is not None:
                life.stop_child(actor, since)
            stop = _recover_descendants(since, check_gpu=False, max_seconds=10, term_seconds=.1)
            fixed_write(folder/'outer-stop.json', stop)
    files = ('actor-ready.json', 'decoy.json', 'actor-result.json', 'actor-cleanup.jsonl', 'outer-stop.json')
    receipt = {'status': 'PASS', 'binding_hash': fingerprint(binding), 'allocation': sess['allocation'],
               'files': {n: file_hash(safe(folder, n)) for n in files}, 'folder': str(folder.relative_to(normal(output)))}
    fixed_write(folder/'receipt.json', receipt)
    pointer = {'file': str((folder/'receipt.json').relative_to(normal(output))), 'sha256': file_hash(folder/'receipt.json')}
    write_json(safe(output, 'runtime/ownership-selftest.json'), pointer)
    return pointer


def verify_native_selftest(output, pointer=None, *, allocation=None, offline=False):
    binding = validate_controls(output, offline=offline)
    pointer = pointer or read_json(safe(output, 'runtime/ownership-selftest.json'))
    path = safe(output, pointer['file'])
    if file_hash(path) != pointer['sha256']:
        raise StudyError('Native ownership receipt changed')
    receipt = read_json(path)
    expected_allocation = allocation or session(output, current=not offline)['allocation']
    if (receipt.get('status') != 'PASS' or receipt.get('binding_hash') != fingerprint(binding)
            or receipt.get('allocation') != expected_allocation or path.parent != safe(output, receipt['folder'])):
        raise StudyError('Native ownership receipt belongs to another binding/allocation')
    files = {'actor-ready.json', 'decoy.json', 'actor-result.json', 'actor-cleanup.jsonl', 'outer-stop.json'}
    if set(receipt.get('files', {})) != files or any(file_hash(safe(path.parent, n)) != h for n, h in receipt['files'].items()):
        raise StudyError('Native ownership self-test evidence changed')
    result = read_json(path.parent/'actor-result.json')
    if result.get('status') != 'PASS' or result.get('decoy_survived') is not True:
        raise StudyError('Native ownership actor did not pass')
    return pointer


def recover_previous(output, attempt_dir):
    """Prove stopped; never kill a historical PID after controller restart."""
    folder = normal(attempt_dir)
    plan_path = safe(folder, 'supervision.json')
    if not plan_path.exists():
        return None
    plan = read_json(plan_path)
    sess = session(output)
    if plan.get('binding_hash') != sess['binding_hash']:
        raise StudyError('Earlier attempt runtime binding differs')
    for name in ('parent-stop.json', 'recovered-stop.json'):
        p = safe(folder, name)
        if p.exists():
            stop = read_json(p)
            if stop.get('plan_hash') != fingerprint(plan) or stop.get('gpu_checked') is not True or stop.get('gpu_compute_processes') != []:
                raise StudyError('Earlier attempt lacks valid cleanup evidence')
            if name == 'recovered-stop.json':
                proof = stop.get('old_allocation_proof', {})
                if (proof.get('job_id') != plan['allocation']['slurm_job_id'] or not proof.get('terminal')
                        or stop.get('new_allocation', {}).get('slurm_job_id') == proof.get('job_id')):
                    raise StudyError('Earlier allocation recovery evidence differs')
            return stop
    old_job = plan.get('allocation', {}).get('slurm_job_id')
    if not old_job or old_job == sess['allocation'].get('slurm_job_id'):
        raise StudyError('Incomplete attempt lacks cleanup proof in this allocation; no historical PID is signalled')
    proof = slurm_job_state(old_job)
    if proof.get('job_id') != old_job or not proof.get('terminal'):
        raise StudyError('Cannot prove interrupted attempt allocation ended')
    gpu = old.assert_gpu_owned_by(set())
    stop = {'plan_hash': fingerprint(plan), 'old_allocation_proof': proof, 'new_allocation': sess['allocation'],
            'gpu_checked': True, 'gpu_compute_processes': gpu, 'no_historical_pid_signalled': True}
    fixed_write(safe(folder, 'recovered-stop.json'), stop)
    return stop


def supervise(output, attempt_dir, command, timeout_seconds, *, minimum_start_seconds=None):
    """Bound the entire worker, including imports/tokenization/startup/draining/cleanup."""
    if (not isinstance(command, (list, tuple)) or not command or not all(isinstance(a, str) and a for a in command)
            or isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise StudyError('Invalid supervised worker command or deadline')
    sess = session(output)
    minimum = timeout_seconds if minimum_start_seconds is None else minimum_start_seconds
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool) or not math.isfinite(minimum) or not 0 < minimum <= timeout_seconds:
        raise StudyError('Invalid minimum allowance for next worker')
    allowance = sess['deadline_epoch'] - time.time() - CLEANUP_MARGIN
    if allowance < minimum:
        return {'status': 'CHECKPOINT', 'remaining_seconds': allowance, 'required_seconds': minimum}
    pointer = verify_native_selftest(output)
    folder = normal(attempt_dir)
    if normal(output) not in folder.parents:
        raise StudyError('Attempt must belong to this output')
    safe(folder, 'supervision.json')
    if (folder/'supervision.json').exists():
        raise StudyError('Supervision attempt already exists; preserve it and use a new attempt')
    since = time.time()
    deadline = min(since + timeout_seconds, sess['deadline_epoch'] - CLEANUP_MARGIN)
    began_monotonic = time.monotonic()
    monotonic_deadline = began_monotonic + deadline - since
    plan = {'command': list(command), 'cwd': str(PROJECT), 'started_epoch': since, 'deadline_epoch': deadline,
            'allocation': sess['allocation'], 'binding_hash': sess['binding_hash'], 'ownership_selftest': pointer,
            'started_monotonic': began_monotonic, 'deadline_monotonic': monotonic_deadline}
    fixed_write(folder/'supervision.json', plan)
    worker = None
    with life.subreaper():
        try:
            with (folder/'worker.log').open('x') as log:
                worker = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                                'DECISION_STUDY_DEADLINE_EPOCH': str(deadline),
                                                'DECISION_BUDGET_WORKER_OUTPUT': str(normal(output)),
                                                'DECISION_BUDGET_WORKER_ATTEMPT': str(folder)})
            fixed_write(folder/'worker-process.json', {'pid': worker.pid, 'started_epoch': since,
                        'ownership': 'direct child of sole-worker subreaper; no historical group signals'})
            while worker.poll() is None:
                budget_left = min(deadline-time.time(), monotonic_deadline-time.monotonic())
                if budget_left <= 0:
                    raise StudyError('Budget-serving worker exceeded absolute deadline')
                try:
                    worker.wait(timeout=min(.25, max(.001, budget_left)))
                except subprocess.TimeoutExpired:
                    pass
            exited, exited_monotonic = time.time(), time.monotonic()
            if worker.returncode != 0 or exited > deadline or exited_monotonic > monotonic_deadline:
                raise StudyError(f'Budget-serving worker did not finish successfully on time (exit {worker.returncode})')
            stop = _recover_descendants(since)
            stop['plan_hash'] = fingerprint(plan)
            fixed_write(folder/'parent-stop.json', stop)
            receipt = {'status': 'COMPLETE', 'returncode': worker.returncode, 'exited_epoch': exited,
                       'deadline_epoch': deadline, 'plan_hash': fingerprint(plan), 'allocation': sess['allocation'],
                       'parent_stop_sha256': file_hash(folder/'parent-stop.json'),
                       'supervision_sha256': file_hash(folder/'supervision.json'),
                       'exited_monotonic': exited_monotonic, 'deadline_monotonic': monotonic_deadline}
            fixed_write(folder/'supervised.json', receipt)
            return receipt
        except BaseException as exc:
            failure = {'error': str(exc), 'type': type(exc).__name__, 'at': utc_now(), 'plan_hash': fingerprint(plan)}
            if worker is not None:
                try:
                    life.stop_child(worker, since)
                except BaseException as error:
                    failure['worker_stop_error'] = str(error)
            try:
                stop = _recover_descendants(since)
                stop['plan_hash'] = fingerprint(plan)
                fixed_write(folder/'parent-stop.json', stop)
                failure['parent_stop_sha256'] = file_hash(folder/'parent-stop.json')
            except BaseException as error:
                failure['cleanup_error'] = str(error)
            fixed_write(folder/'supervision-failure.json', failure)
            raise


def verify_supervised(output, attempt_dir, *, offline=False):
    """Acceptance requires an on-time zero exit AND the bound parent cleanup proof."""
    folder = normal(attempt_dir)
    if normal(output) not in folder.parents:
        raise StudyError('Attempt must belong to this output')
    binding = validate_controls(output, offline=offline)
    plan_path, stop_path = safe(folder,'supervision.json'), safe(folder,'parent-stop.json')
    plan = read_json(plan_path)
    receipt = read_json(safe(folder,'supervised.json'))
    stop = read_json(stop_path)
    numeric = ('started_epoch','deadline_epoch','started_monotonic','deadline_monotonic')
    if (any(type(plan.get(k)) not in (int,float) or not math.isfinite(plan[k]) for k in numeric)
            or any(type(receipt.get(k)) not in (int,float) or not math.isfinite(receipt[k])
                   for k in ('exited_epoch','exited_monotonic'))):
        raise StudyError('Invalid supervised timing receipt')
    if (receipt.get('status') != 'COMPLETE' or receipt.get('returncode') != 0
            or safe(folder,'supervision-failure.json').exists()
            or plan.get('binding_hash') != fingerprint(binding) or receipt.get('plan_hash') != fingerprint(plan)
            or receipt.get('supervision_sha256') != file_hash(plan_path)
            or receipt.get('parent_stop_sha256') != file_hash(stop_path)
            or receipt.get('allocation') != plan.get('allocation')
            or receipt.get('deadline_epoch') != plan['deadline_epoch']
            or receipt.get('deadline_monotonic') != plan['deadline_monotonic']
            or not plan['started_epoch'] <= receipt['exited_epoch'] <= plan['deadline_epoch']
            or not plan['started_monotonic'] <= receipt['exited_monotonic'] <= plan['deadline_monotonic']
            or stop.get('plan_hash') != fingerprint(plan) or stop.get('gpu_checked') is not True
            or stop.get('gpu_compute_processes') != []):
        raise StudyError('Worker acceptance lacks on-time exit and matching parent cleanup')
    verify_native_selftest(output, plan['ownership_selftest'], allocation=plan['allocation'], offline=offline)
    return receipt


def verify_engine_record(output, condition, record, *, offline=False):
    """Recheck actual deployed commands, not only a metadata concurrency field."""
    validate_controls(output, offline=offline)
    roles = condition_roles(condition)
    if record.get('condition') != condition or [v.get('role') for v in record.get('roles',[])] != list(roles):
        raise StudyError('Deployment role inventory differs')
    for value in record['roles']:
        role = value['role']
        target = control(output, condition, role)
        lock = read_json(safe(target,'runtime/lock.json'))
        installed = read_json(safe(target,'runtime/installations.json'))
        expected = engine_specs(PROJECT,target,'startlux','primary',lock,installed,MAX_NUM_SEQS)
        actual = value.get('processes',[])
        if (value.get('startup_failed') is not False
                or [(x.get('name'),x.get('command')) for x in actual] != [(x['name'],x['command']) for x in expected]
                or value.get('endpoint') != f'http://127.0.0.1:{PORTS[role][0]}'
                or value.get('health',{}).get('prompt_attention') != 'bidirectional'
                or value.get('health',{}).get('defaults',{}).get('think_mode') != 'never'
                or type(value.get('startup_seconds')) not in (int,float)
                or not 0 <= value['startup_seconds'] <= STARTUP_SECONDS):
            raise StudyError('Actual engine commands/readiness differ from fixed deployment')
        config = value.get('runtime',{})
        share = .9 if condition in ('direct4b','direct9b') else (.4 if role=='startlux4b' else .5)
        required = {'model':'startlux','model_revision':REVISIONS[role],'model_repo':REPOS[role],
                    'precision':'bfloat16','max_model_len':8192,'prefix_cache':False,'truncate':False,
                    'native_temperature':1.1103 if role=='startlux4b' else 1.1014,'backend':'vllm',
                    'bidirectional':True,'think_mode':'never','none_check':False,'allow_none':False,
                    'gpu_memory_utilization':share,'chunked_prefill':False,'speculative_decoding':False}
        if any(config.get(k)!=v for k,v in required.items()):
            raise StudyError('Observed runtime computation identity differs')
    return record


@contextmanager
def worker_environment(output, condition, *, cuda=False):
    binding = validate_controls(output)
    sess = session(output)
    # Helpers that may spawn/compile must only run below the supervisor watchdog.
    if os.environ.get('DECISION_BUDGET_WORKER_OUTPUT') != str(normal(output)):
        raise StudyError('GPU/token work must run inside the bounded budget-serving worker')
    with legacy_environment(binding, PROJECT), v3._environment(binding, sess):
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        roles = condition_roles(condition)
        for role in roles:
            target = control(output, condition, role)
            report = frozen.doctor(target, require_gpu=True)
            gpus = report.get('gpu', [])
            if len(gpus) != 1 or gpus[0].get('name') != 'NVIDIA L40S' or gpus[0].get('memory_mb', 0) < 44000:
                raise StudyError('Exactly one L40S with at least44,000MiB is required')
            write_json(safe(target, 'runtime/session.json'), sess)
        if cuda:
            target = control(output, condition, roles[0])
            installed = read_json(safe(target, 'runtime/installations.json'))
            os.environ.update(v3._cuda_preflight(target, binding, installed))
        yield binding, sess


def _signal_handle(fd, sig):
    try:
        pidfd.send_signal(fd, sig)
    except ProcessLookupError:
        pass


def _stop_handles(handles):
    errors = []
    for process, fd in reversed(handles):
        try:
            _signal_handle(fd, signal.SIGTERM)
        except OSError as exc:
            errors.append(str(exc))
    for process, fd in reversed(handles):
        try:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_handle(fd, signal.SIGKILL)
                process.wait(timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
        finally:
            os.close(fd)
    if errors:
        raise StudyError('Direct service shutdown failed: ' + '; '.join(errors))


@contextmanager
def deployment(output, condition, *, sink):
    """Launch exact StartLux engines; PID handles are bound before readiness polling."""
    condition_roles(condition)
    handles, owned, begun, started_controls = [], set(), time.time(), set()
    sink.update(condition=condition, started_at=utc_now(), roles=[], processes=[])
    with worker_environment(output, condition, cuda=True) as (binding, sess):
        sink['allocation'] = sess['allocation']
        try:
            for role in condition_roles(condition):
                target = control(output, condition, role)
                lock = read_json(safe(target, 'runtime/lock.json'))
                installed = read_json(safe(target, 'runtime/installations.json'))
                info = next(iter(installed['environments'].values()))
                frozen._verify_environment(target, info)
                before = old.assert_gpu_owned_by(owned)
                record = {'role': role, 'gpu_before_start': before, 'processes': [], 'startup_failed': True}
                sink['roles'].append(record)
                meta = {'started_at': utc_now(), 'models': {}, 'processes': [], 'max_num_seqs': MAX_NUM_SEQS}
                meta_path = safe(target, 'runtime/last_services.json')
                started = time.monotonic()
                for spec in engine_specs(PROJECT, target, 'startlux', 'primary', lock, installed, MAX_NUM_SEQS):
                    frozen._check_port(spec['port'])
                    log_path = safe(target, f'runtime/logs/{spec["name"]}-{time.time_ns()}.log')
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open('x') as log:
                        process = subprocess.Popen(spec['command'], env=spec['env'], stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    # Even descriptor/readiness failure must leave startup evidence.
                    row = {'name': spec['name'], 'pid': process.pid, 'command': spec['command'], 'log': str(log_path)}
                    meta['processes'].append(row); record['processes'].append(row); sink['processes'].append(row)
                    write_json(meta_path, meta)
                    started_controls.add(target)
                    try:
                        fd = pidfd.open_process(process.pid)
                    except BaseException:
                        life.stop_child(process, begun)
                        raise
                    handles.append((process, fd)); owned.add(process.pid)
                    allowance = STARTUP_SECONDS - (time.monotonic()-started)
                    if allowance <= 0:
                        raise StudyError('Role startup exceeded fixed1200s allowance')
                    health = frozen._wait_ready(process, spec, log_path, allowance)
                    if time.monotonic() - started > STARTUP_SECONDS:
                        raise StudyError('Role readiness arrived after fixed startup deadline')
                    if spec['name'] == 'startlux' and (health.get('prompt_attention') != 'bidirectional'
                            or health.get('defaults', {}).get('think_mode') != 'never'):
                        raise StudyError('StartLux frontend attention/thinking differs')
                    meta['models']['startlux'] = {'health': health, 'config': frozen.runtime_config(target, 'startlux')}
                    write_json(meta_path, meta)
                endpoint = f'http://127.0.0.1:{lock["models"]["startlux"]["port"]}'
                record.update(startup_failed=False, endpoint=endpoint, startup_seconds=time.monotonic()-started,
                              health=meta['models']['startlux']['health'], runtime=meta['models']['startlux']['config'],
                              engine_log=old.engine_log_facts(record['processes']))
                meta['ready_at'] = utc_now(); write_json(meta_path, meta)
            sink['gpu_after_start'] = old.assert_gpu_owned_by(owned)
            sink['ready_at'] = utc_now()
            yield {r['role']: r['endpoint'] for r in sink['roles']}
        finally:
            try:
                _stop_handles(handles)
            finally:
                sink['stopped_at'] = utc_now()
                sink['worker_shutdown'] = 'Stable handles for direct children; sole-worker parent must recover all descendants and confirm empty GPU'
                for target in started_controls:
                    path = safe(target, 'runtime/last_services.json')
                    if path.exists():
                        meta = read_json(path); meta['stopped_at'] = utc_now(); write_json(path, meta)


def token_check(output, condition, role, examples):
    with worker_environment(output, condition):
        target = control(output, condition, role)
        frozen.token_preflight_examples(PROJECT, target, 'startlux', examples)
        path = target/'runtime'/f'tokens-interactive-startlux-{frozen._hash(examples)[:16]}.json'
        receipt = {'file': str(path.relative_to(normal(output))), 'sha256': file_hash(path),
                   'runtime_hash': fingerprint(frozen.runtime_config(target, 'startlux'))}
        verify_tokens(output, condition, role, examples, receipt)
        return receipt


def verify_tokens(output, condition, role, examples, receipt, *, offline=False):
    validate_controls(output, offline=offline)
    target = control(output, condition, role)
    expected = target/'runtime'/f'tokens-interactive-startlux-{frozen._hash(examples)[:16]}.json'
    path = safe(output, receipt['file'])
    if path != expected or file_hash(path) != receipt.get('sha256'):
        raise StudyError('Raw tokenizer report changed or points outside expected control')
    value = read_json(path)
    rows = value.get('rows', [])
    if (value.get('status') != 'ok' or value.get('model') != 'startlux'
            or value.get('model_revision') != REVISIONS[role] or value.get('max_model_len') != MAX_MODEL_LEN
            or value.get('n_examples') != len(examples) or value.get('truncated') is not False
            or value.get('overlong') != [] or [r.get('sample_id') for r in rows] != [e['sample_id'] for e in examples]
            or value.get('effective_config_hash') != receipt.get('runtime_hash')
            or receipt.get('runtime_hash') != fingerprint(frozen.runtime_config(target, 'startlux'))
            or any(type(r.get('input_tokens')) is not int or r['input_tokens'] < 1
                   or r.get('reserved_output_tokens') != 1 or r['input_tokens']+1 > MAX_MODEL_LEN for r in rows)):
        raise StudyError('Token report does not cover every exact input without truncation')
