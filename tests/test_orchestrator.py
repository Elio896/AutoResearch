import json
import tempfile
import unittest
from pathlib import Path


class OrchestratorTests(unittest.TestCase):
    def test_model_parameter_mismatch_is_rejected(self):
        from orchestrator import validate_proposal
        with self.assertRaises(ValueError):
            validate_proposal('{"model":"logistic_regression","parameters":{"max_depth":3}}')

    def test_ranges_types_and_extra_actions_are_rejected(self):
        from orchestrator import validate_proposal
        for proposal in [
            {'model': 'random_forest', 'parameters': {'n_estimators': 999999}},
            {'model': 'random_forest', 'parameters': {'max_depth': True}},
            {'model': 'logistic_regression', 'parameters': {'C': float('nan')}},
            {'model': 'logistic_regression', 'parameters': {}, 'action': 'finalize'},
        ]:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                validate_proposal(json.dumps(proposal))

    def test_complete_json_fence_is_normalized(self):
        from orchestrator import validate_proposal
        result = validate_proposal('```json\n{"model":"random_forest","parameters":{"max_depth":3}}\n```')
        self.assertEqual(result['parameters']['max_depth'], 3)

    def run_research(self, responses, **kwargs):
        from sklearn.datasets import load_iris
        from executor import ResearchSession
        from orchestrator import ResearchOrchestrator
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'iris.csv'
            load_iris(as_frame=True).frame.to_csv(path, index=False)
            session = ResearchSession(path, 'target')
            replies = iter(responses)
            def propose(messages, timeout):
                value = next(replies)
                if isinstance(value, Exception):
                    raise value
                return {'message': {'content': value}, 'eval_count': 20, 'prompt_eval_count': 100}
            result = ResearchOrchestrator(session, propose, **kwargs).run()
            self.assertFalse(session.closed)
            self.assertNotIn('test_f1', json.dumps(result))
            return result

    def test_invalid_then_valid_candidate_does_not_train_invalid_model(self):
        result = self.run_research(['{"model":"logistic_regression","parameters":{"max_depth":3}}',
                                    '{"model":"random_forest","parameters":{"max_depth":3}}'],
                                   max_experiments=2, max_calls=2)
        self.assertEqual(len(result['experiments']), 2)
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['experiments'][1]['model'], 'random_forest')

    def test_busy_and_invalid_responses_cannot_exceed_call_budget(self):
        from orchestrator import LocalModelError
        result = self.run_research([LocalModelError('busy', retryable=True), 'invalid'],
                                   max_experiments=3, max_calls=2, retry_delay=0)
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(len(result['experiments']), 1)
        self.assertEqual(result['status'], 'escalation_required')

    def test_experiment_budget_one_never_calls_model(self):
        result = self.run_research([], max_experiments=1)
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(len(result['experiments']), 1)

    def test_real_http_adapter_handles_busy_then_ollama_envelope(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from orchestrator import LocalModelError, LocalQwenClient
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append((self.path, body))
                if len(received) == 1:
                    self.send_response(429)
                    self.end_headers()
                    return
                payload = json.dumps({'message': {'content': '{"model":"random_forest","parameters":{}}'},
                                      'done_reason': 'stop', 'eval_count': 20}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = LocalQwenClient(f'http://127.0.0.1:{server.server_port}')
            with self.assertRaises(LocalModelError) as failure:
                client([{'role': 'user', 'content': 'test'}], timeout=2)
            self.assertTrue(failure.exception.retryable)
            result = client([{'role': 'user', 'content': 'test'}], timeout=2)
            self.assertEqual(result['eval_count'], 20)
            self.assertEqual(received[1][0], '/api/chat')
            self.assertEqual(received[1][1]['options']['num_predict'], 160)
            self.assertFalse(received[1][1]['stream'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
