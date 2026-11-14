"""轻量模型 API 客户端的验收测试：用一个本地假服务把真实 HTTP 路径跑通。

不花钱、不碰真 key；验证的是协议、记账、上限、以及失败时不留假象。
"""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from light_model import LightModelClient, LightModelError, from_env


class _Handler(BaseHTTPRequestHandler):
    """按类变量里的剧本回话。"""

    script: dict = {}
    received: list = []

    def do_POST(self):                      # noqa: N802 - 标准库命名
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        type(self).received.append({'path': self.path, 'body': body,
                                    'auth': self.headers.get('Authorization')})
        status = self.script.get('status', 200)
        payload = self.script.get('payload', {'choices': [{'message': {'content': 'ok'}}],
                                              'usage': {'prompt_tokens': 11, 'completion_tokens': 5}})
        raw = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):           # 静音
        pass


class ApiClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(('127.0.0.1', 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _Handler.received = []
        _Handler.script = {}
        self.client = LightModelClient('sk-test-not-a-real-key', base_url=self.base,
                                       model='light-test')

    def test_successful_call_returns_content_and_counts_usage(self):
        text = self.client([{'role': 'user', 'content': '你好'}])
        self.assertEqual(text, 'ok')
        self.assertEqual(self.client.usage['calls'], 1)
        self.assertEqual(self.client.usage['prompt_tokens'], 11)
        self.assertEqual(self.client.usage['completion_tokens'], 5)

        sent = _Handler.received[0]
        self.assertEqual(sent['path'], '/v1/chat/completions')
        self.assertEqual(sent['body']['model'], 'light-test')
        self.assertEqual(sent['body']['messages'][0]['content'], '你好')
        self.assertTrue(sent['auth'].startswith('Bearer '))

    def test_cost_is_unknown_not_zero(self):
        self.client([{'role': 'user', 'content': 'x'}])
        summary = self.client.usage_summary()
        self.assertIsNone(summary['cost_cny'])
        self.assertEqual(summary['cost_basis'], 'unknown')

    def test_http_error_raises_without_leaking_key(self):
        _Handler.script = {'status': 401, 'payload': {'error': {'message': 'bad key'}}}
        with self.assertRaises(LightModelError) as caught:
            self.client([{'role': 'user', 'content': 'x'}])
        message = str(caught.exception)
        self.assertIn('401', message)
        self.assertNotIn('sk-test-not-a-real-key', message)

    def test_empty_choices_is_an_error(self):
        _Handler.script = {'status': 200, 'payload': {'choices': []}}
        with self.assertRaises(LightModelError):
            self.client([{'role': 'user', 'content': 'x'}])
        self.assertEqual(len(_Handler.received), 1, 'Must reach the real TCP fixture; transport failure is not parser verification.')

    def test_non_json_body_is_an_error(self):
        _Handler.script = {'status': 200, 'payload': {'choices': [{'message': {'content': 123}}]}}
        with self.assertRaises(LightModelError):
            self.client([{'role': 'user', 'content': 'x'}])
        self.assertEqual(len(_Handler.received), 1, 'Must reach the real TCP fixture; transport failure is not parser verification.')

    def test_call_budget_is_enforced(self):
        client = LightModelClient('k', base_url=self.base, max_calls=2)
        client([{'role': 'user', 'content': 'a'}])
        client([{'role': 'user', 'content': 'b'}])
        with self.assertRaises(LightModelError):
            client([{'role': 'user', 'content': 'c'}])
        self.assertEqual(len(_Handler.received), 2)     # 第三次没发出去

    def test_empty_key_rejected(self):
        with self.assertRaises(LightModelError):
            LightModelClient('   ')

    def test_from_env_returns_none_without_key(self):
        self.assertIsNone(from_env({}))
        self.assertIsNone(from_env({'DEEPSEEK_API_KEY': '  '}))

    def test_from_env_reads_overrides(self):
        client = from_env({'LIGHT_MODEL_API_KEY': 'k1', 'LIGHT_MODEL_BASE_URL': self.base,
                           'LIGHT_MODEL_NAME': 'tiny', 'LIGHT_MODEL_MAX_CALLS': '3'})
        self.assertIsNotNone(client)
        self.assertEqual(client.model, 'tiny')
        self.assertEqual(client.max_calls, 3)
        self.assertEqual(client.base_url, self.base)

    def test_deepseek_key_fallback(self):
        client = from_env({'DEEPSEEK_API_KEY': 'k2', 'LIGHT_MODEL_BASE_URL': self.base})
        self.assertIsNotNone(client)
        self.assertEqual(client.model, 'deepseek-chat')


if __name__ == '__main__':
    unittest.main()
