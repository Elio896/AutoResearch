"""模型分层调度的测试。"""
import unittest

from model_router import ModelRouter


class FakeClient:
    def __init__(self, model, prompt=10, completion=5):
        self.model = model
        self.prompt = prompt
        self.completion = completion
        self.calls = 0

    def __call__(self, messages, timeout=None):
        self.calls += 1
        return {'message': {'content': '{}'},
                'prompt_eval_count': self.prompt,
                'eval_count': self.completion}


def factory_for(models):
    created = {}

    def _factory(model):
        created[model] = created.get(model, 0) + 1
        return FakeClient(model)
    _factory.created = created
    return _factory


class ModelRouterTests(unittest.TestCase):
    def test_routes_roles_to_different_models(self):
        f = factory_for({})
        router = ModelRouter(f, {'scientist': 'big', 'redteam': 'small'})
        sci = router.for_role('scientist')
        red = router.for_role('redteam')
        self.assertIsNotNone(sci)
        self.assertIsNotNone(red)
        self.assertEqual(router._clients['big'].model, 'big')
        self.assertEqual(router._clients['small'].model, 'small')

    def test_unconfigured_role_returns_none(self):
        router = ModelRouter(factory_for({}), {'scientist': 'big'})
        self.assertIsNone(router.for_role('analyst'))
        self.assertIsNone(router.for_role('manager'))

    def test_usage_is_attributed_per_role(self):
        router = ModelRouter(factory_for({}), {'scientist': 'big', 'redteam': 'small'})
        router.for_role('scientist')([], timeout=5)
        router.for_role('scientist')([], timeout=5)
        router.for_role('redteam')([], timeout=5)
        summary = router.summary()
        self.assertEqual(summary['by_role']['scientist']['calls'], 2)
        self.assertEqual(summary['by_role']['redteam']['calls'], 1)
        self.assertEqual(summary['by_role']['scientist']['prompt_tokens'], 20)
        self.assertEqual(summary['total']['calls'], 3)
        self.assertEqual(summary['total']['prompt_tokens'], 30)

    def test_same_model_is_created_once(self):
        f = factory_for({})
        router = ModelRouter(f, {'scientist': 'big', 'redteam': 'big'})
        router.for_role('scientist')
        router.for_role('redteam')
        self.assertEqual(f.created['big'], 1)          # 复用同一个客户端

    def test_empty_routes_rejected_gracefully(self):
        router = ModelRouter(factory_for({}), {})
        self.assertIsNone(router.for_role('scientist'))
        self.assertEqual(router.summary()['total']['calls'], 0)

    def test_non_callable_factory_raises(self):
        with self.assertRaises(TypeError):
            ModelRouter('not callable', {'scientist': 'big'})


if __name__ == '__main__':
    unittest.main()
