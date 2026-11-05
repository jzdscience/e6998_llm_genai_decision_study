"""Actual loopback responses and direct construction checks; no model execution."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch

from decision_study.common import StudyError
from decision_study.service_study import client
from decision_study.service_study.cpu import ACCEPTED_MODELS, LABELS


EXAMPLE = {'sample_id': 'authored-client-fixture', 'task': 'clinc', 'split': 'development',
           'state': 'This is an authored fixture, not dataset text.',
           'labels': list(LABELS), 'criteria': {label: label for label in LABELS},
           'instructions': 'Select one fixed intent label.'}


def cpu_response():
    return {'request_id': 'r1', 'recipe': 'minilm400',
            'classifier_sha256': ACCEPTED_MODELS['minilm400'], 'instance_id': 'fixture',
            'label': 'oos', 'predicted_label': 'oos', 'labels': list(LABELS),
            'probabilities': [0.] * 15 + [1.], 'encoder_ms': 1., 'service_ms': 1.1,
            'classifier_ms': .1, 'queue_ms': 0.}


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = self.rfile.read(int(self.headers['Content-Length']))
        self.server.calls.append({'path': self.path, 'payload': json.loads(payload)})
        raw = self.server.raw if self.server.raw is not None else json.dumps(self.server.answer).encode()
        self.send_response(self.server.code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ServiceClientTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        self.server.calls, self.server.answer = [], cpu_response()
        self.server.code, self.server.raw = 200, None
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01})
        self.thread.start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def call(self, condition='minilm400'):
        return client.call_http(condition, self.endpoint, EXAMPLE, 'r1', 1.0)

    def test_cpu_success_sends_original_text_and_retains_full_response(self):
        result = self.call()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['prediction'], 'oos')
        self.assertEqual(result['raw_response'], self.server.answer)
        self.assertEqual(result['attempts'], 1)
        self.assertEqual(result['probabilities']['oos'], 1.)
        self.assertEqual(self.server.calls, [{'path': '/predict', 'payload': {'text': EXAMPLE['state'], 'request_id': 'r1'}}])

    def test_cpu_wrong_condition_request_or_model_is_retained_as_error(self):
        for key, value in (('recipe', 'minilm1600'), ('request_id', 'other-request'),
                           ('classifier_sha256', 'other-model'), ('predicted_label', 'balance')):
            with self.subTest(key=key):
                self.server.answer = cpu_response(); self.server.answer[key] = value
                self.server.calls.clear()
                result = self.call()
                self.assertEqual(result['status'], 'error')
                self.assertIsNone(result['prediction'])
                self.assertEqual(result['raw_response'], self.server.answer)
                self.assertEqual(result['error']['type'], 'invalid_response')
                self.assertEqual(len(self.server.calls), 1)

    def test_cpu_invalid_probability_vector_and_axis_are_rejected(self):
        vectors = [[0.] * 16, [0.] * 15 + [True], [0.] * 15 + [float('nan')],
                   [0.] * 15 + [float('inf')], [0.] * 15 + [1.1], [1.] + [0.] * 15,
                   [1.], None]
        for vector in vectors:
            with self.subTest(vector=vector):
                self.server.answer = cpu_response(); self.server.answer['probabilities'] = vector
                result = self.call()
                self.assertEqual(result['status'], 'error')
                self.assertIsNone(result['prediction'])
        self.server.answer = cpu_response(); self.server.answer['labels'] = sorted(LABELS)
        self.assertEqual(self.call()['status'], 'error')

    def test_cpu_http_rejection_and_failure_have_one_attempt_and_observed_body(self):
        for code in (429, 500, 503):
            with self.subTest(code=code):
                self.server.code, self.server.answer = code, {'error': f'fixture-{code}'}
                self.server.calls.clear()
                result = self.call()
                self.assertEqual(result['attempts'], 1)
                self.assertEqual(result['status'], 'http_rejected' if code == 429 else 'error')
                self.assertEqual(result['error']['http_code'], code)
                self.assertIn(f'fixture-{code}', result['error']['detail'])
                self.assertEqual(len(self.server.calls), 1)

    def test_cpu_bad_json_is_error_without_retry(self):
        self.server.raw = b'{not json'
        result = self.call()
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error']['type'], 'invalid_response')
        self.assertEqual(len(self.server.calls), 1)

    def test_gpu_constructor_explicitly_disables_frozen_default_retry(self):
        for condition, model in (('startlux4b', 'startlux'), ('qwen4b', 'qwen')):
            with self.subTest(condition=condition), patch.object(client, 'ModelClient') as model_client:
                model_client.return_value.predict.return_value = {'status': 'fixture'}
                self.assertEqual(self.call(condition), {'status': 'fixture'})
                model_client.assert_called_once_with(model, self.endpoint, timeout=1.0, max_attempts=1)
                model_client.return_value.predict.assert_called_once_with(EXAMPLE)

    def test_real_gpu_adapter_records_retryable_failure_once(self):
        self.server.code, self.server.answer = 503, {'error': 'fixture busy'}
        for condition in ('startlux4b', 'qwen4b'):
            self.server.calls.clear()
            with patch.dict('os.environ', {'DECISION_STUDY_DEADLINE_EPOCH': ''}):
                result = self.call(condition)
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['error']['code'], 503)
            self.assertEqual(len(result['attempts']), 1)
            self.assertEqual(len(self.server.calls), 1)
            self.assertEqual(self.server.calls[0]['path'], '/v1/systemone' if condition == 'startlux4b' else '/v1/chat/completions')

    def test_unknown_condition_and_nonowned_endpoint_make_no_http_request(self):
        with self.assertRaises(StudyError):
            self.call('unknown')
        for endpoint in ('https://127.0.0.1:1', 'http://example.com:1', 'http://127.0.0.1:1/wrong'):
            with self.subTest(endpoint=endpoint), self.assertRaises(StudyError):
                client.call_http('minilm400', endpoint, EXAMPLE, 'r1', 1)
        self.assertEqual(self.server.calls, [])


if __name__ == '__main__':
    unittest.main()
