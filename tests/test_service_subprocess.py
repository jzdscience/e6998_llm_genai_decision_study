"""Actual cell subprocess + synthetic HTTP fixtures, never task/model inference."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from decision_study.common import StudyError, file_hash, fingerprint, read_json, read_jsonl, write_json
from decision_study.service_study import study
from decision_study.service_study.cpu import ACCEPTED_MODELS, LABELS
from decision_study.service_study.measurement import summarize, verify_cell


class FixtureHTTP(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.calls.append(request)
        text = request['text']
        if text == 'authored-rejected-fixture':
            code, body = 429, {'error': 'software fixture server rejection'}
        else:
            code = 200
            body = {'request_id': request['request_id'], 'recipe': 'minilm400',
                    'classifier_sha256': ACCEPTED_MODELS['minilm400'],
                    'instance_id': 'SOFTWARE_FIXTURE_NOT_MODEL', 'labels': list(LABELS),
                    'label': 'oos', 'predicted_label': 'oos',
                    'probabilities': [0.] * 15 + [1. if text == 'authored-ok-fixture' else 0.],
                    'is_software_fixture': True}
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers(); self.wfile.write(raw)


class CellSubprocessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHTTP)
        self.server.calls = []
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01})
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def plan(self):
        workload = [{'sample_id': f'fixture-{i}', 'state': f'authored-{kind}-fixture'}
                    for i, kind in enumerate(('ok', 'rejected', 'malformed'))]
        schedule = [{'offer_id': i, 'sample_id': row['sample_id'], 'arrival_seconds': .01 + i*.02}
                    for i, row in enumerate(workload)]
        write_json(self.root / 'workload.json', workload)
        write_json(self.root / 'schedule.json', schedule)
        write_json(self.root / 'warmup.json', [{'sample_id': 'authored-warmup'}])
        write_json(self.root / 'service.json', {'is_software_fixture': True, 'pid': os.getpid()})
        write_json(self.root / 'warmup-record.json', {'rows': [{'sample_id': 'authored-warmup', 'status': 'ok'}], 'is_software_fixture': True})
        p = {'duration_seconds': .12, 'workers': 1, 'max_inflight': 4, 'slo_ms': 1000,
             'drain_seconds': 2, 'request_timeout_seconds': 1, 'watchdog_grace_seconds': 2,
             'is_software_fixture': True}
        cell = {'cell_id': 'software-fixture-cell', 'condition': 'minilm400', 'repeat': 0,
                'rate': 'fixture', 'schedule_file': 'schedule.json'}
        config = {'protocol': p, 'schedules': {'schedule.json': file_hash(self.root / 'schedule.json')},
                  'workload_sha256': file_hash(self.root / 'workload.json')}
        folder = self.root / 'cells' / cell['cell_id'] / 'attempt-0001'
        folder.mkdir(parents=True)
        plan = {**cell, 'protocol': p, 'settings_hash': fingerprint(config),
                'schedule_sha256': config['schedules']['schedule.json'],
                'workload_file': 'workload.json', 'workload_sha256': config['workload_sha256'],
                'endpoint': f'http://127.0.0.1:{self.server.server_port}', 'service_pids': [os.getpid()],
                'service_record_file': 'service.json', 'service_record_sha256': file_hash(self.root / 'service.json'),
                'warmup_record_file': 'warmup-record.json', 'warmup_record_sha256': file_hash(self.root / 'warmup-record.json')}
        return plan, folder, config, cell

    def test_actual_cell_subprocess_streams_all_http_outcomes_and_valid_completion(self):
        plan, folder, config, cell = self.plan()
        with patch.object(study.runtime, 'check_time', return_value={}):
            study._run_process(self.root, folder, plan)
        receipt = verify_cell(folder, fingerprint(plan))
        rows = sorted(read_jsonl(folder / 'traces.jsonl'), key=lambda r: r['offer_id'])
        self.assertEqual([r['offer_id'] for r in rows], [0, 1, 2])
        self.assertEqual([r['status'] for r in rows], ['ok', 'http_rejected', 'error'])
        self.assertEqual(rows[0]['result']['prediction'], 'oos')
        self.assertTrue(rows[0]['result']['raw_response']['is_software_fixture'])
        self.assertEqual(rows[1]['result']['error']['http_code'], 429)
        self.assertEqual(rows[2]['result']['error']['type'], 'invalid_response')
        self.assertEqual(len(self.server.calls), 3)
        self.assertEqual({r['request_id'] for r in self.server.calls},
                         {f"{plan['cell_id']}-{i}" for i in range(3)})
        self.assertTrue(all(row['result']['attempts'] == 1 for row in rows))
        self.assertEqual(receipt['report']['offered'], 3)
        self.assertEqual(receipt['report']['successful'], 1)
        self.assertEqual(receipt['report']['failed_admitted'], 2)
        self.assertEqual(receipt['report']['rejected'], 0)
        self.assertTrue(read_jsonl(folder / 'occupancy.jsonl'))
        samples = read_jsonl(folder / 'resources.jsonl')
        self.assertTrue(samples)
        self.assertIn('gpu', samples[0])
        self.assertTrue(all(len(sample['owned_root_pids']) == 2 for sample in samples))
        for name, digest in receipt['files'].items():
            self.assertEqual(file_hash(folder / name), digest)
        self.assertEqual(study.completed(self.root, config, cell), (folder, receipt))
        scored = summarize(rows, .12, 1000, {f'fixture-{i}': 'oos' for i in range(3)})
        self.assertEqual(scored['correct'], 1)
        self.assertEqual(scored['correct_fraction_all_offered'], 1/3)

    def test_real_child_rejects_changed_workload_before_any_http(self):
        plan, folder, _, _ = self.plan()
        write_json(self.root / 'workload.json', [{'sample_id': 'changed'}])
        with patch.object(study.runtime, 'check_time', return_value={}):
            with self.assertRaisesRegex(StudyError, 'Cell process exited'):
                study._run_process(self.root, folder, plan)
        self.assertEqual(self.server.calls, [])
        self.assertFalse((folder / 'complete.json').exists())
        self.assertIn('Changed cell workload', (folder / 'controller.log').read_text())

    def test_watchdog_kills_owned_child_and_quarantines_racing_completion(self):
        plan, folder, _, _ = self.plan()
        plan['protocol'] = {**plan['protocol'], 'duration_seconds': .01,
                            'drain_seconds': .02, 'watchdog_grace_seconds': .03}
        original_popen = subprocess.Popen
        children = []
        def artificial_hang(command, **kwargs):
            self.assertIn('decision_study.service_study.cell', command)
            # This deliberately bogus completion emulates a child racing the
            # watchdog. It is never valid model or measurement evidence.
            script = ('import pathlib,time; '
                      f'pathlib.Path({str(folder / "complete.json")!r}).write_text("{{\\\"racing_software_fixture\\\":true}}"); '
                      'time.sleep(10)')
            process = original_popen([sys.executable, '-c', script], **kwargs)
            children.append(process)
            until = time.monotonic() + 1
            while not (folder / 'complete.json').exists() and process.poll() is None and time.monotonic() < until:
                time.sleep(.005)
            self.assertTrue((folder / 'complete.json').exists())
            return process
        try:
            with patch.object(study.subprocess, 'Popen', side_effect=artificial_hang), \
                 patch.object(study.runtime, 'check_time', return_value={}):
                with self.assertRaisesRegex(StudyError, 'absolute parent watchdog'):
                    study._run_process(self.root, folder, plan)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())
            self.assertFalse((folder / 'complete.json').exists())
            self.assertEqual(read_json(folder / 'unaccepted-complete.json'), {'racing_software_fixture': True})
        finally:
            for process in children:
                if process.poll() is None:
                    study.runtime.stop(process)

    def test_exit_observed_after_watchdog_is_not_accepted_as_success(self):
        plan, folder, _, _ = self.plan()
        plan['protocol'] = {**plan['protocol'], 'duration_seconds': .01,
                            'drain_seconds': .02, 'watchdog_grace_seconds': .03}
        class Finished:
            returncode = 0
            def poll(self):
                return 0
        process = Finished()
        def finished_child(*_args, **_kwargs):
            write_json(folder / 'complete.json', {'late_software_fixture': True})
            return process
        # The first monotonic reading arms the deadline; a zero exit code is
        # subsequently observed after it. This is not a real child/process PID.
        with patch.object(study.subprocess, 'Popen', side_effect=finished_child), \
             patch.object(study.time, 'monotonic', side_effect=[100., 101.]), \
             patch.object(study.runtime, 'stop') as stop:
            with self.assertRaisesRegex(StudyError, 'watchdog'):
                study._run_process(self.root, folder, plan)
            stop.assert_called_once_with(process)
        self.assertFalse((folder / 'complete.json').exists())
        self.assertEqual(read_json(folder / 'unaccepted-complete.json'), {'late_software_fixture': True})


if __name__ == '__main__':
    unittest.main()
