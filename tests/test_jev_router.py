import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class JevRouterTests(unittest.TestCase):
    def test_official_noul_protocol_usage_and_sanitized_failure(self):
        from jev_router import JevClient, JevError
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                received.append((self.path, self.headers.get('Authorization'),
                    json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                if len(received) > 1:
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'test-secret-must-not-appear-in-errors')
                    return
                payload = json.dumps({'model': 'jev-test', 'answers': {
                    'worth_cloud_diagnosis': {'type': 'noul', 'noul': 0.85}},
                    'usage': {'input_tokens': 90, 'output_tokens': 8}}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = JevClient('test-only-not-a-real-key', endpoint=f'http://127.0.0.1:{server.server_port}/v1/systemone')
            result = client({'trigger': 'validation_plateau'}, timeout=2)
            self.assertEqual(result['probability'], 0.85)
            self.assertEqual(result['usage']['input_tokens'], 90)
            self.assertEqual(received[0][0], '/v1/systemone')
            self.assertEqual(received[0][1], 'Bearer test-only-not-a-real-key')
            self.assertEqual(received[0][2]['questions']['worth_cloud_diagnosis']['type'], 'noul')
            with self.assertRaises(JevError) as failure:
                client({}, timeout=2)
            self.assertNotIn('test-secret', str(failure.exception))
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=2)

    def test_missing_key_and_insecure_remote_endpoint_rejected(self):
        from jev_router import JevClient
        with self.assertRaises(ValueError):
            JevClient('')
        with self.assertRaises(ValueError):
            JevClient('test-only-not-a-real-key', endpoint='http://example.com/v1/systemone')

    def run_research(self, probability=None, error=False, **kwargs):
        from sklearn.datasets import load_iris
        from executor import ResearchSession
        from orchestrator import ResearchOrchestrator
        from jev_router import JevError, JevGate
        seen = []
        def client(state, timeout):
            seen.append(state)
            if error:
                raise JevError('safe error code')
            return {'probability': probability, 'model': 'jev-test',
                    'usage': {'input_tokens': 90, 'output_tokens': 8}}
        cloud_requests = []
        def cloud(messages, timeout):
            cloud_requests.append(messages)
            return '{"model":"random_forest","parameters":{"max_depth":3}}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'iris.csv'
            load_iris(as_frame=True).frame.to_csv(path, index=False)
            session = ResearchSession(path, 'target')
            result = ResearchOrchestrator(session, None, max_calls=0,
                cloud_proposer=cloud, jev_gate=JevGate(client),
                **{'max_experiments': 2, 'max_cloud_calls': 1, **kwargs}).run()
            self.assertFalse(session.closed)
            self.assertNotIn('test_f1', json.dumps(seen))
            return result, seen, cloud_requests

    def test_low_probability_stops_before_cloud_and_high_probability_allows(self):
        low, seen, cloud = self.run_research(probability=0.2)
        self.assertEqual(low['stop_reason'], 'jev_declined')
        self.assertEqual(low['jev_calls'], 1)
        self.assertFalse(cloud)
        high, seen, cloud = self.run_research(probability=0.95)
        self.assertEqual(high['cloud_calls'], 1)
        self.assertEqual(high['jev_calls'], 1)
        self.assertEqual(len(cloud), 1)
        self.assertEqual(len(high['experiments']), 2)

    def test_failed_router_falls_back_to_existing_rule_and_records_it(self):
        result, _, cloud = self.run_research(error=True)
        self.assertEqual(len(cloud), 1)
        self.assertEqual(result['jev_calls'], 1)
        self.assertEqual(result['escalation_gate']['source'], 'rule_fallback')

    def test_router_cannot_override_cloud_experiment_or_router_budget(self):
        for settings in [{'max_cloud_calls': 0}, {'max_experiments': 1}, {'max_jev_calls': 0}]:
            with self.subTest(settings=settings):
                result, seen, cloud = self.run_research(probability=1, **settings)
                self.assertFalse(seen)
                self.assertEqual(result['jev_calls'], 0)
                if settings.get('max_jev_calls') != 0:
                    self.assertFalse(cloud)

    def test_bad_probability_is_not_used_as_an_approval(self):
        for probability in [True, float('nan'), 2]:
            with self.subTest(probability=probability):
                result, _, _ = self.run_research(probability=probability)
                self.assertEqual(result['escalation_gate']['source'], 'rule_fallback')
