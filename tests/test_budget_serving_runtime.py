"""Runtime isolation, runtime identity and ownership gates, with hardware boundaries faked."""
from contextlib import contextmanager, nullcontext
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from decision_study.budget_serving import runtime as r
from decision_study.common import StudyError, fingerprint, read_json, write_json


@pytest.fixture
def controls(tmp_path, monkeypatch):
    root = tmp_path/'runtime'
    output = root/'results/aux-anli-budget-serving-v1'
    v1 = root/'results/solo-v1'
    size = root/'results/aux-size-control-v1-r2'
    defaults = read_json(r.PROJECT/'configs/runtime_defaults.json')
    base = {**copy.deepcopy(defaults), 'lock_hash': fingerprint(defaults), 'defaults_hash': fingerprint(defaults),
            'created_at': 'historical', 'verified_sources': []}
    name = 'vllm-' + defaults['packages']['vllm_primary']
    python = v1/'runtime/work/envs'/name/'bin/python'
    python.parent.mkdir(parents=True); python.write_text('stub interpreter')
    env = {'python': str(python), 'freeze_sha256': 'a'*64}
    models = {}
    for role, work in [('startlux4b', v1/'runtime/work'), ('startlux9b', size/'runtime-work')]:
        path = work/'cache/huggingface/hub/models--pinned/snapshots'/r.REVISIONS[role]
        path.mkdir(parents=True)
        (path/'config.json').write_text('{}')
        (path/'model.safetensors').write_bytes(b'fixture')
        models[role] = {'model_path': str(path), 'revision': r.REVISIONS[role]}
    # Deliberately no Qwen model entry, directory or environment.
    write_json(v1/'runtime/lock.json', base)
    write_json(v1/'runtime/installations.json', {'lock_hash': base['lock_hash'], 'environments': {name: env},
                                               'models': {'startlux': models['startlux4b']}})
    nine = copy.deepcopy(base)
    nine['models']['startlux'].update(repo=r.REPOS['startlux9b'], revision=r.REVISIONS['startlux9b'], native_temperature=1.1014)
    nine_hash = fingerprint({k: nine[k] for k in defaults})
    nine.update(lock_hash=nine_hash, defaults_hash=nine_hash,
                **{'size-control_source': {'parent_lock_hash': base['lock_hash']}})
    write_json(size/'runtime-controls/startlux9b/runtime/lock.json', nine)
    write_json(size/'runtime-controls/startlux9b/runtime/installations.json',
               {'lock_hash': nine_hash, 'environments': {name: env}, 'models': {'startlux': models['startlux9b']}})
    write_json(size/'runtime/binding.json', {'work_dir': str(size/'runtime-work')})
    monkeypatch.setattr(r.size_runtime, '_verify_model_metadata', lambda path: None)
    binding = r.prepare_controls(root, output)
    return root, output, binding


def test_isolated_derived_controls_have_fixed_ports_shares_and_no_qwen_dependency(controls):
    root, output, binding = controls
    before = {p: Path(p).read_bytes() for p in binding['source_files']}
    assert set(binding['models']) == {'startlux4b', 'startlux9b'}
    assert r.validate_controls(output) == binding
    for condition, role, share, ports in [('direct4b','startlux4b',.9,(18380,18381)),
                                        ('direct9b','startlux9b',.9,(18390,18391)),
                                        ('budget','startlux4b',.4,(18380,18381)),
                                        ('budget','startlux9b',.5,(18390,18391))]:
        target = r.control(output, condition, role)
        lock = read_json(target/'runtime/lock.json')
        assert (lock['models']['startlux']['port'],lock['models']['startlux']['engine_port']) == ports
        assert lock['startup_timeout_seconds'] == 1200
        assert lock['budget-serving_source']['parent_lock_hash'] != lock['lock_hash']
        assert read_json(target/'runtime/installations.json')['lock_hash'] == lock['lock_hash']
        assert read_json(target/'runtime/selected_memory.json') == {'startlux':share}
        specs = r.engine_specs(r.PROJECT,target,'startlux','primary',lock,read_json(target/'runtime/installations.json'),64)
        engine=specs[0]['command']
        assert engine[-2:]==['--max-num-seqs','64']
        assert engine[engine.index('--gpu-memory-utilization')+1]==str(share)
        assert engine[engine.index('--max-model-len')+1]=='8192'
    assert all(Path(p).read_bytes()==raw for p,raw in before.items())
    assert r.prepare_controls(root,output)==binding


def test_changed_control_rejected_online_and_offline(controls):
    _,output,_=controls
    path=r.control(output,'budget','startlux9b')/'runtime/selected_memory.json'
    write_json(path,{'startlux':.6})
    for offline in (False,True):
        with pytest.raises(StudyError,match='control evidence changed'):
            r.validate_controls(output,offline=offline)


@pytest.mark.parametrize("name,change", [
    ('selected_memory.json',lambda v:{'startlux':.6}),
    ('selected_profiles.json',lambda v:{'startlux':'reference'}),
    ('lock.json',lambda v:{**v,'max_model_len':4096}),
])
def test_offline_semantics_refuse_jointly_rehashed_control_and_binding(controls,name,change):
    _,output,binding=controls
    path=r.control(output,'budget','startlux9b')/'runtime'/name
    value=change(read_json(path));write_json(path,value)
    binding['control_identity']['pair9b'][name]=fingerprint(value)
    write_json(output/'runtime/binding.json',binding)
    with pytest.raises(StudyError,match='semantics differ'):
        r.validate_controls(output,offline=True)


def test_missing_4b_snapshot_and_changed_9b_environment_are_not_substituted(controls):
    root,output,binding=controls
    path=Path(binding['models']['startlux4b']['model_path'])/'config.json'
    path.unlink()
    with pytest.raises(StudyError,match='snapshot is missing'):
        r.validate_controls(output)
    path.write_text('{}')
    installed=root/'results/aux-size-control-v1-r2/runtime-controls/startlux9b/runtime/installations.json'
    value=read_json(installed)
    next(iter(value['environments'].values()))['freeze_sha256']='wrong'
    write_json(installed,value)
    with pytest.raises(StudyError,match='lock/environment'):
        r.validate_controls(output)


def test_same_allocation_does_not_extend_and_new_requires_terminal_proof(controls,monkeypatch):
    _,output,_=controls
    allocation={'hostname':'node','slurm_job_id':'123'}
    monkeypatch.setattr(r.v3,'_allocation',lambda:allocation.copy())
    monkeypatch.setattr(r,'slurm_hours_left',lambda:2)
    first=r.initialize_allocation(output)
    monkeypatch.setattr(r,'slurm_hours_left',lambda:3)
    assert r.initialize_allocation(output)['deadline_epoch']==first['deadline_epoch']
    allocation['slurm_job_id']='124'
    monkeypatch.setattr(r,'slurm_job_state',lambda job:{'job_id':job,'terminal':False})
    with pytest.raises(StudyError,match='previous allocation ended'):
        r.initialize_allocation(output)
    assert read_json(output/'runtime/session.json')['allocation']['slurm_job_id']=='123'
    monkeypatch.setattr(r,'slurm_job_state',lambda job:{'job_id':job,'terminal':True})
    assert r.initialize_allocation(output)['allocation']['slurm_job_id']=='124'


def test_same_job_on_new_host_cannot_renew(controls,monkeypatch):
    _,output,_=controls
    allocation={'hostname':'node','slurm_job_id':'123'}
    monkeypatch.setattr(r.v3,'_allocation',lambda:allocation.copy())
    monkeypatch.setattr(r,'slurm_hours_left',lambda:2)
    r.initialize_allocation(output)
    allocation['hostname']='different'
    with pytest.raises(StudyError,match='different Slurm job'):
        r.initialize_allocation(output)


def fake_supervisor(monkeypatch, output, *, deadline=10000):
    sess={'allocation':{'hostname':'node','slurm_job_id':'123'},'deadline_epoch':deadline,'binding_hash':'fixed'}
    monkeypatch.setattr(r,'session',lambda *a,**k:sess)
    monkeypatch.setattr(r,'verify_native_selftest',lambda *a,**k:{'file':'selftest','sha256':'digest'})
    monkeypatch.setattr(r.life,'subreaper',lambda:nullcontext())
    monkeypatch.setattr(r,'_recover_descendants',lambda *a,**k:{'gpu_checked':True,'gpu_compute_processes':[]})
    return sess


def test_supervisor_checkpoints_before_spawning(monkeypatch,tmp_path):
    fake_supervisor(monkeypatch,tmp_path,deadline=time.time()+130)
    monkeypatch.setattr(r.subprocess,'Popen',lambda *a,**k:pytest.fail('checkpoint must not spawn'))
    result=r.supervise(tmp_path,tmp_path/'attempt',['python','worker'],300)
    assert result['status']=='CHECKPOINT'
    assert not (tmp_path/'attempt/supervision.json').exists()


def test_late_zero_exit_never_accepts_completed_worker(monkeypatch,tmp_path):
    fake_supervisor(monkeypatch,tmp_path)
    times=iter([0.,0.,2.])
    monkeypatch.setattr(r.time,'time',lambda:next(times))
    class Process:
        pid=4321;returncode=0
        def __init__(self,*args,**kwargs):
            (tmp_path/'attempt/complete.json').write_text('{"status":"complete"}')
        def poll(self):return 0
    monkeypatch.setattr(r.subprocess,'Popen',Process)
    monkeypatch.setattr(r.life,'stop_child',lambda *a:None)
    with pytest.raises(StudyError,match='successfully on time'):
        r.supervise(tmp_path,tmp_path/'attempt',['python','worker'],1)
    assert (tmp_path/'attempt/complete.json').exists()
    assert (tmp_path/'attempt/parent-stop.json').exists()
    assert not (tmp_path/'attempt/supervised.json').exists()


def test_parent_cleanup_failure_prevents_acceptance(monkeypatch,tmp_path):
    fake_supervisor(monkeypatch,tmp_path,deadline=time.time()+10000)
    class Process:
        pid=4321;returncode=0
        def poll(self):return 0
    monkeypatch.setattr(r.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(r.life,'stop_child',lambda *a:None)
    def failed(*a,**k):raise StudyError('owned descendants remain')
    monkeypatch.setattr(r,'_recover_descendants',failed)
    with pytest.raises(StudyError,match='owned descendants remain'):
        r.supervise(tmp_path,tmp_path/'attempt',['python','worker'],1)
    assert not (tmp_path/'attempt/supervised.json').exists()
    assert 'cleanup_error' in read_json(tmp_path/'attempt/supervision-failure.json')


def test_unproved_interrupted_attempt_never_signals_historical_pid(monkeypatch,tmp_path):
    sess=fake_supervisor(monkeypatch,tmp_path)
    plan={'binding_hash':'fixed','allocation':sess['allocation'],'started_epoch':0}
    write_json(tmp_path/'attempt/supervision.json',plan)
    monkeypatch.setattr(r.pidfd,'send_signal',lambda *a:pytest.fail('must not signal historical PID'))
    with pytest.raises(StudyError,match='this allocation'):
        r.recover_previous(tmp_path,tmp_path/'attempt')
    sess['allocation']={'hostname':'newnode','slurm_job_id':'456'}
    monkeypatch.setattr(r,'slurm_job_state',lambda job:{'job_id':job,'terminal':True})
    monkeypatch.setattr(r.old,'assert_gpu_owned_by',lambda *a,**k:[])
    proof=r.recover_previous(tmp_path,tmp_path/'attempt')
    assert proof['no_historical_pid_signalled'] is True
    assert proof['old_allocation_proof']['job_id']=='123'


def test_completed_cleanup_receipt_is_bound_to_attempt(monkeypatch,tmp_path):
    sess=fake_supervisor(monkeypatch,tmp_path)
    write_json(tmp_path/'attempt/supervision.json',{'binding_hash':'fixed','allocation':sess['allocation']})
    write_json(tmp_path/'attempt/parent-stop.json',{'plan_hash':'wrong','gpu_checked':True,'gpu_compute_processes':[]})
    with pytest.raises(StudyError,match='valid cleanup'):
        r.recover_previous(tmp_path,tmp_path/'attempt')


def test_worker_environment_refuses_unsupervised_gpu_calls(controls,monkeypatch):
    _,output,binding=controls
    monkeypatch.setattr(r,'session',lambda *a,**k:{'deadline_epoch':time.time()+1000})
    monkeypatch.delenv('DECISION_BUDGET_WORKER_OUTPUT',raising=False)
    monkeypatch.setattr(r.frozen,'doctor',lambda *a,**k:pytest.fail('GPU work before worker gate'))
    with pytest.raises(StudyError,match='bounded budget-serving worker'):
        with r.worker_environment(output,'direct4b'):pass


@pytest.mark.parametrize('late',[False,True])
def test_failed_frontend_is_registered_and_stopped_with_stable_handles(controls,monkeypatch,late):
    _,output,binding=controls
    @contextmanager
    def environment(*a,**k):yield binding,{'allocation':{'slurm_job_id':'123'}}
    monkeypatch.setattr(r,'worker_environment',environment)
    monkeypatch.setattr(r.frozen,'_verify_environment',lambda *a:None)
    monkeypatch.setattr(r.frozen,'_check_port',lambda *a:None)
    monkeypatch.setattr(r.old,'assert_gpu_owned_by',lambda *a:[])
    monkeypatch.setattr(r.frozen,'runtime_config',lambda *a:{})
    monkeypatch.setattr(r.old,'engine_log_facts',lambda *a:{})
    processes=[];signals=[];closed=[]
    class Process:
        def __init__(self,*a,**k):self.pid=100+len(processes);self.returncode=None;processes.append(self)
        def wait(self,timeout):self.returncode=-signal.SIGTERM;return self.returncode
    monkeypatch.setattr(r.subprocess,'Popen',Process)
    monkeypatch.setattr(r.pidfd,'open_process',lambda pid:pid+1000)
    monkeypatch.setattr(r.pidfd,'send_signal',lambda fd,sig:signals.append((fd,sig)))
    monkeypatch.setattr(r.os,'close',lambda fd:closed.append(fd))
    clock=[0.]
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    def ready(process,spec,*args):
        if spec['name']=='startlux':
            if late:
                clock[0]=1201.
                return {'prompt_attention':'bidirectional','defaults':{'think_mode':'never'}}
            raise RuntimeError('frontend readiness failed')
        return {'status':'ok'}
    monkeypatch.setattr(r.frozen,'_wait_ready',ready)
    sink={}
    with pytest.raises((RuntimeError,StudyError),match='startup deadline' if late else 'frontend readiness'):
        with r.deployment(output,'direct4b',sink=sink):pytest.fail('failed startup yielded')
    count=2
    assert len(sink['processes'])==count and sink['roles'][0]['startup_failed'] is True
    meta=read_json(r.control(output,'direct4b','startlux4b')/'runtime/last_services.json')
    assert len(meta['processes'])==count and 'stopped_at' in meta
    assert {fd for fd,sig in signals}==set(range(1100,1100+count))
    assert set(closed)==set(range(1100,1100+count))
    assert all(p.returncode==-signal.SIGTERM for p in processes)


@pytest.mark.skipif(sys.platform!='linux',reason='actual Linux subreaper/process descriptors required')
def test_native_ownership_actor_preserves_decoy(controls,monkeypatch):
    _,output,binding=controls
    sess={'allocation':{'hostname':'test','slurm_job_id':'123'},'binding_hash':fingerprint(binding),
          'deadline_epoch':time.time()+300}
    monkeypatch.setattr(r,'session',lambda *a,**k:sess)
    pointer=r.native_selftest(output)
    assert r.verify_native_selftest(output,pointer)==pointer
    receipt=read_json(r.safe(output,pointer['file']))
    actor=read_json(r.safe(output,receipt['folder'])/'actor-result.json')
    assert actor['decoy_survived'] is True
    assert any(s['signal']=='SIGKILL' for s in actor['stop']['signals'])


@pytest.mark.skipif(sys.platform!='linux',reason='actual Linux subreaper/process descriptors required')
def test_native_parent_watchdog_recovers_detached_term_ignoring_descendant(monkeypatch,tmp_path):
    sess={'allocation':{'hostname':'test','slurm_job_id':'123'},'binding_hash':'fixed','deadline_epoch':time.time()+300}
    monkeypatch.setattr(r,'session',lambda *a,**k:sess)
    monkeypatch.setattr(r,'verify_native_selftest',lambda *a,**k:{'file':'selftest','sha256':'digest'})
    monkeypatch.setattr(r.old,'assert_gpu_owned_by',lambda *a,**k:[])
    original=r._recover_descendants
    monkeypatch.setattr(r,'_recover_descendants',lambda since,**kw:original(since,max_seconds=3,term_seconds=.1))
    code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'],start_new_session=True);time.sleep(60)"
    with pytest.raises(StudyError,match='absolute deadline'):
        r.supervise(tmp_path,tmp_path/'attempt',[sys.executable,'-c',code],.5)
    stop=read_json(tmp_path/'attempt/parent-stop.json')
    assert stop['gpu_checked'] is True and stop['gpu_compute_processes']==[]
    assert any(s['signal']=='SIGKILL' for s in stop['signals'])
    assert not (tmp_path/'attempt/supervised.json').exists()


@pytest.mark.parametrize('change',['late_wall','late_monotonic','bad_stop','failure_receipt'])
def test_verify_supervised_rejects_rehashed_invalid_acceptance(controls,monkeypatch,change):
    _,output,binding=controls
    folder=output/'attempt'
    plan={'binding_hash':fingerprint(binding),'allocation':{'slurm_job_id':'123'},'ownership_selftest':{},
          'started_epoch':100.,'deadline_epoch':110.,'started_monotonic':10.,'deadline_monotonic':20.}
    stop={'plan_hash':fingerprint(plan),'gpu_checked':True,'gpu_compute_processes':[]}
    write_json(folder/'supervision.json',plan)
    write_json(folder/'parent-stop.json',stop)
    receipt={'status':'COMPLETE','returncode':0,'plan_hash':fingerprint(plan),'allocation':plan['allocation'],
             'exited_epoch':105.,'deadline_epoch':110.,'exited_monotonic':15.,'deadline_monotonic':20.,
             'supervision_sha256':r.file_hash(folder/'supervision.json'),
             'parent_stop_sha256':r.file_hash(folder/'parent-stop.json')}
    write_json(folder/'supervised.json',receipt)
    monkeypatch.setattr(r,'verify_native_selftest',lambda *a,**k:{})
    assert r.verify_supervised(output,folder,offline=True)==receipt
    if change=='late_wall':receipt['exited_epoch']=111.
    elif change=='late_monotonic':receipt['exited_monotonic']=21.
    elif change=='bad_stop':
        stop['gpu_compute_processes']=[{'pid':123}]
        write_json(folder/'parent-stop.json',stop)
        receipt['parent_stop_sha256']=r.file_hash(folder/'parent-stop.json')
    else:write_json(folder/'supervision-failure.json',{'error':'failure'})
    write_json(folder/'supervised.json',receipt)
    with pytest.raises(StudyError,match='on-time exit'):
        r.verify_supervised(output,folder,offline=True)


def test_verify_engine_record_requires_actual_sequence_flag_on_both_engines(controls):
    _,output,_=controls
    record={'condition':'budget','roles':[]}
    for role in r.ROLES:
        target=r.control(output,'budget',role)
        write_json(target/'runtime/doctor.json',{'gpu':[]})
        lock=read_json(target/'runtime/lock.json'); installed=read_json(target/'runtime/installations.json')
        specs=r.engine_specs(r.PROJECT,target,'startlux','primary',lock,installed,64)
        record['roles'].append({'role':role,'startup_failed':False,'startup_seconds':1.,
            'endpoint':f'http://127.0.0.1:{r.PORTS[role][0]}',
            'health':{'prompt_attention':'bidirectional','defaults':{'think_mode':'never'}},
            'runtime':r.frozen.runtime_config(target,'startlux'),
            'processes':[{'name':s['name'],'command':s['command']} for s in specs]})
    assert r.verify_engine_record(output,'budget',record,offline=True)==record
    record['roles'][1]['processes'][0]['command']=record['roles'][1]['processes'][0]['command'][:-2]
    record['max_num_seqs']=64  # Claimed metadata does not replace actual command evidence.
    with pytest.raises(StudyError,match='Actual engine commands'):
        r.verify_engine_record(output,'budget',record,offline=True)


def test_parent_cleanup_checks_its_deadline_between_pid_signals(monkeypatch):
    clock=[0.]; sent=[]
    monkeypatch.setattr(r.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(r.time,'sleep',lambda *a:None)
    monkeypatch.setattr(r.life,'owned_descendants',lambda since:{1:{},2:{}})
    def send(pid,since,sig):sent.append(pid);clock[0]+=60;return True
    monkeypatch.setattr(r.life,'signal_descendant',send)
    with pytest.raises(StudyError,match='bounded parent cleanup'):
        r._recover_descendants(0,max_seconds=60)
    assert sent==[2]
