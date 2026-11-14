import unittest
from unittest.mock import patch
from light_model import LightModelClient, LightModelError, from_env


class ModelConfigurationTests(unittest.TestCase):
    def test_public_api_respects_configured_https_proxy(self):
        with patch('urllib.request.getproxies', return_value={'https': 'http://127.0.0.1:9999'}):
            client = LightModelClient('test-ascii-key')
            self.assertEqual(client._proxy_handler().proxies, {'https': 'http://127.0.0.1:9999'})

    def test_loopback_fixture_remains_direct(self):
        with patch('urllib.request.getproxies', return_value={'https': 'http://127.0.0.1:9999'}):
            client = LightModelClient('test-ascii-key', base_url='http://127.0.0.1:12345/v1')
            self.assertEqual(client._proxy_handler().proxies, {})

    def test_unicode_example_is_rejected_before_transport_without_echoing_key(self):
        with self.assertRaises(LightModelError) as caught:
            LightModelClient('你的API密钥')
        self.assertIn('non-ASCII', str(caught.exception))
        self.assertNotIn('你的API密钥', str(caught.exception))

    def test_environment_configuration_uses_the_same_preflight(self):
        with self.assertRaises(LightModelError):
            from_env({'LIGHT_MODEL_API_KEY': '示例密钥'})
        self.assertIsNotNone(from_env({'LIGHT_MODEL_API_KEY': 'test-ascii-key'}))
