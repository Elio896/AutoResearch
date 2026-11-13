import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CloudTests(unittest.TestCase):
    def test_protocol_usage_and_no_redirect_or_error_body_leak(self):
        from cloud_provider import DeepSeekClient, CloudError
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                received.append((self.path, self.headers.get('Authorization'),
                                 json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                if len(received) == 2:
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'sensitive-provider-error')
                    return
                if len(received) == 3:
                    self.send_response(302)
                    self.send_header('Location', '/must-not-follow')
                    self.end_headers()
                    return
                body = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
                    'content': '{"model":"random_forest","parameters":{"max_depth":3}}'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 30}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = DeepSeekClient('test-only-not-a-secret', base_url=f'http://127.0.0.1:{server.server_port}')
            content = client([{'role': 'user', 'content': 'Return JSON'}], timeout=2)
            self.assertEqual(json.loads(content)['model'], 'random_forest')
            self.assertEqual(client.last_usage['prompt_tokens'], 100)
            self.assertEqual(received[0][0], '/chat/completions')
            self.assertEqual(received[0][1], 'Bearer test-only-not-a-secret')
            self.assertEqual(received[0][2]['max_tokens'], 256)
            self.assertEqual(received[0][2]['thinking'], {'type': 'disabled'})
            with self.assertRaises(CloudError) as failure:
                client([{'role': 'user', 'content': 'Return JSON'}], timeout=2)
            self.assertNotIn('sensitive-provider-error', str(failure.exception))
            with self.assertRaisesRegex(CloudError, 'HTTP 302'):
                client([{'role': 'user', 'content': 'Return JSON'}], timeout=2)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)

    def test_missing_key_rejected_without_network(self):
        from cloud_provider import DeepSeekClient
        with self.assertRaises(ValueError):
            DeepSeekClient('')

    def test_cloud_rescue_uses_same_validator_and_keeps_holdout_private(self):
        from sklearn.datasets import load_iris
        from executor import ResearchSession
        from orchestrator import ResearchOrchestrator
        calls = []
        def cloud(messages, timeout):
            calls.append(messages)
            if len(calls) == 1:
                return '{"model":"logistic_regression","parameters":{"max_depth":3}}'
            return '{"model":"random_forest","parameters":{"max_depth":3}}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'iris.csv'
            load_iris(as_frame=True).frame.to_csv(path, index=False)
            session = ResearchSession(path, 'target')
            result = ResearchOrchestrator(session, None, max_calls=0, max_experiments=2,
                        cloud_proposer=cloud, max_cloud_calls=2).run()
            self.assertEqual(result['cloud_calls'], 2)
            self.assertEqual(len(result['experiments']), 2)
            self.assertEqual(result['stop_reason'], 'cloud_recovery')
            self.assertFalse(session.closed)
            self.assertNotIn('test_f1', json.dumps(calls))

    def test_cloud_budget_and_experiment_budget_prevent_extra_calls(self):
        from sklearn.datasets import load_iris
        from executor import ResearchSession
        from orchestrator import ResearchOrchestrator
        def forbidden(*args, **kwargs):
            self.fail('Cloud should not have been called')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'iris.csv'
            load_iris(as_frame=True).frame.to_csv(path, index=False)
            for experiment_budget, cloud_budget in [(1, 1), (2, 0)]:
                session = ResearchSession(path, 'target')
                result = ResearchOrchestrator(session, None, max_calls=0,
                    max_experiments=experiment_budget, cloud_proposer=forbidden,
                    max_cloud_calls=cloud_budget).run()
                self.assertEqual(result['cloud_calls'], 0)
