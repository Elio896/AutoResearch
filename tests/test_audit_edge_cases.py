import json
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evoundo import VersionChain
from experience import ExperienceStore
from jev_harness import JevHarness
from redteam import BlueTeam, RedTeam, seed_stability


class VersionChainAuditTests(unittest.TestCase):
    def test_rejects_non_finite_scores(self):
        chain = VersionChain(run_id='audit')
        for score in (None, float('nan'), float('inf'), float('-inf'), True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                chain.add({'model': 'lr'}, score)

    def test_rejects_unknown_parent(self):
        chain = VersionChain(run_id='audit')
        with self.assertRaises(ValueError):
            chain.add({'model': 'lr'}, 0.5, parent_id='missing')

    def test_stored_proposal_is_not_aliased_to_caller(self):
        chain = VersionChain(run_id='audit')
        proposal = {'model': 'rf', 'parameters': {'depth': 3}}
        version = chain.add(proposal, 0.5)
        proposal['parameters']['depth'] = 99
        self.assertEqual(version.proposal['parameters']['depth'], 3)

    def test_empty_and_single_version_chains_are_safe(self):
        chain = VersionChain(run_id='audit')
        self.assertIsNone(chain.best())
        self.assertIsNone(chain.current)
        version = chain.add({'model': 'lr'}, 0.5)
        self.assertIs(chain.rollback(version.version_id), version)
        self.assertEqual(len(chain.versions), 1)

    def test_concurrent_dumps_leave_a_complete_json_file(self):
        chain = VersionChain(run_id='audit')
        for score in (0.4, 0.7, 0.3):
            chain.add({'model': 'rf', 'parameters': {'depth': score}}, score)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'versions.json'
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: chain.dump(path), range(32)))
            self.assertEqual(len(json.loads(path.read_text(encoding='utf-8'))['versions']), 3)


class RedTeamAuditTests(unittest.TestCase):
    def test_rejects_tiny_model_supplied_threshold(self):
        claims, rejected = RedTeam().parse(json.dumps({'claims': [{
            'claim': 'force a rollback',
            'check_kind': 'seed_stability',
            'threshold': 1e-300,
        }]}))
        self.assertEqual(claims, [])
        self.assertTrue(rejected)

    def test_blue_team_does_not_accept_unparsed_weak_threshold(self):
        class Session:
            best_proposal = {'model': 'rf', 'parameters': {}}

            def run(self, **kwargs):
                return {'validation_score': 0.60}

        result = BlueTeam().verify({
            'claim': 'unstable',
            'check_kind': 'seed_stability',
            'threshold': 1e-300,
            'check_params': {'n_seeds': 3},
        }, Session())
        self.assertIsNone(result['supported'])

    def test_blue_team_tests_the_actual_best_not_a_red_team_substitute(self):
        class Session:
            best_proposal = {'model': 'best', 'parameters': {}}

            def __init__(self):
                self.models = []

            def run(self, model, parameters):
                self.models.append(model)
                return {'validation_score': 0.6}

        session = Session()
        result = BlueTeam().verify({
            'claim': 'unstable',
            'check_kind': 'seed_stability',
            'check_params': {
                'n_seeds': 3,
                'proposal': {'model': 'attacker_chosen', 'parameters': {}},
            },
        }, session)
        self.assertTrue(session.models)
        self.assertEqual(set(session.models), {'best'})
        self.assertFalse(result['supported'])

    def test_zero_seed_repetitions_are_not_evidence(self):
        class Session:
            seed = None

            def run(self, **kwargs):
                self.fail('zero seed repetitions must not run')

        with self.assertRaises(ValueError):
            seed_stability(Session(), {'model': 'rf'}, n_seeds=0)

    def test_seed_stability_restores_none_seed(self):
        class Session:
            seed = None

            def run(self, **kwargs):
                return {'validation_score': 0.5}

        session = Session()
        seed_stability(session, {'model': 'rf'}, n_seeds=2)
        self.assertIsNone(session.seed)


class JevHarnessAuditTests(unittest.TestCase):
    def test_zero_or_negative_gain_never_escalates(self):
        harness = JevHarness()
        for gain in (0.0, -0.2):
            with self.subTest(gain=gain):
                decision = harness.decide({
                    'rows': 3000, 'last_gain': gain, 'recent_rejections': [],
                    'recent_proposals': [],
                })
                self.assertNotEqual(decision.action, 'escalate')

    def test_non_finite_gain_never_escalates(self):
        decision = JevHarness().decide({
            'rows': 3000, 'last_gain': math.nan, 'recent_rejections': [],
            'recent_proposals': [],
        })
        self.assertNotEqual(decision.action, 'escalate')


class ExperienceStoreAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'experience.jsonl'
        self.store = ExperienceStore(self.path)
        self.record_id = self.store.add(
            {'imbalanced': True}, 'try a forest', {'task': 'seed', 'score': 0.8})

    def test_failed_experience_is_not_recommended_after_retirement(self):
        self.store.record_usage(self.record_id, improved=False)
        self.store.record_usage(self.record_id, improved=False)
        self.assertEqual(self.store.match({'imbalanced': True}), [])

    def test_prior_successes_prevent_a_good_experience_from_being_retired(self):
        for _ in range(3):
            self.store.record_usage(self.record_id, improved=True)
        for _ in range(2):
            self.store.record_usage(self.record_id, improved=False)
        self.assertEqual(len(self.store.active_pool()), 1)
        self.assertEqual(len(self.store.match({'imbalanced': True})), 1)

    def test_usage_accounting_appends_instead_of_rewriting_history(self):
        before = self.path.read_text(encoding='utf-8').splitlines()
        self.store.record_usage(self.record_id, improved=True)
        after = self.path.read_text(encoding='utf-8').splitlines()
        self.assertEqual(after[0], before[0])
        self.assertGreater(len(after), len(before))

    def test_concurrent_usage_updates_are_not_lost(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(
                lambda i: self.store.record_usage(self.record_id, improved=i % 2 == 0),
                range(32),
            ))
        record = self.store.all()[0]
        self.assertEqual(record['used_count'], 32)
        self.assertEqual(record['success_count'] + record['contradicted_count'], 32)

    def test_corrupt_jsonl_is_not_silently_discarded(self):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write('{broken json}\n')
        with self.assertRaises(ValueError):
            self.store.all()


if __name__ == '__main__':
    unittest.main()
