import tempfile
import unittest
from pathlib import Path


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = Path(__file__).resolve().parents[1] / 'bench/tasks/t1_missing/data.csv'

    def test_confirmed_benchmark_leak_is_blocked_and_cannot_be_acknowledged_as_safe(self):
        from executor import ResearchSession
        session = ResearchSession(self.fixture, 'target')
        self.assertTrue(session.risk_report['blocked'])
        self.assertIn('record_id', session.risk_report['unresolved_columns'])
        with self.assertRaises(RuntimeError):
            session.run()
        with self.assertRaises(ValueError):
            ResearchSession(self.fixture, 'target', feature_decisions={
                'record_id': {'action': 'keep', 'reason': 'pretend safe'}})

    def test_explicit_exclusion_is_applied_to_every_model(self):
        from executor import ResearchSession
        session = ResearchSession(self.fixture, 'target', feature_decisions={
            'record_id': {'action': 'exclude', 'reason': 'Generator creates this ID after sorting target.'}})
        self.assertFalse(session.risk_report['blocked'])
        self.assertNotIn('record_id', session.x.columns)
        for model in ['logistic_regression', 'random_forest']:
            result = session.run(model)
            self.assertEqual(result['excluded_columns'], ['record_id'])
            self.assertNotIn('test_f1', result)

    def test_ordinary_id_can_be_explicitly_kept_and_legitimate_strong_feature_is_not_banned(self):
        import pandas as pd
        from executor import ResearchSession
        path = self.root / 'ordinary.csv'
        pd.DataFrame({'patient_id': range(100), 'measurement': [0, 1] * 50,
                      'target': [0, 1] * 50}).to_csv(path, index=False)
        session = ResearchSession(path, 'target')
        # candidate 只告警不阻断（confirmed 才阻断），但必须出现在 unresolved 里让人看见
        self.assertTrue(session.risk_report['warning_columns'])
        self.assertFalse(session.risk_report['blocked'])
        decisions = {name: {'action': 'keep', 'reason': 'Measured and available before prediction.'}
                     for name in session.risk_report['unresolved_columns']}
        reviewed = ResearchSession(path, 'target', feature_decisions=decisions)
        self.assertFalse(reviewed.risk_report['blocked'])
        self.assertIn('measurement', reviewed.x.columns)

    def test_store_review_precedes_training_and_cannot_change_after_experiments(self):
        from task_store import TaskStore
        store = TaskStore(self.root / 'tasks.db')
        task = store.create(self.fixture, 'target')
        with self.assertRaises(RuntimeError):
            store.run(task)
        self.assertEqual(len(store.show(task)['experiments']), 0)
        store.review(task, {'record_id': {'action': 'exclude', 'reason': 'Confirmed generator leakage.'}})
        store.run(task)
        with self.assertRaises(RuntimeError):
            store.review(task, {})

    def test_candidate_warns_but_does_not_block(self):
        """candidate 是启发式猜测，只告警；只有 confirmed 才拒绝训练。

        否则阈值放宽后每个新数据集都得先人工写 decisions，和"自主"冲突。
        """
        from fastapi.testclient import TestClient
        from api import create_legacy_app as create_app
        with TestClient(create_app(self.root)) as client:
            response = client.post('/tasks', json={'csv_text': self.fixture.read_text(), 'target': 'target'})
            self.assertEqual(response.status_code, 201)
            task = response.json()['task_id']
            report = response.json()['risk_report']
            # 这份 fixture 的哈希不在 known_leaks 里，所以是 candidate
            self.assertFalse(report['blocked'])
            self.assertTrue(report['warning_columns'])
            # 能直接跑，不需要先写 decisions
            self.assertEqual(client.post(f'/tasks/{task}/experiments', json={}).status_code, 200)

    def test_renamed_scrambled_identifier_is_still_a_candidate(self):
        import pandas as pd
        from executor import ResearchSession
        path = self.root / 'renamed.csv'
        frame = pd.read_csv(self.fixture).rename(columns={'record_id': 'ordinary_feature'})
        frame.to_csv(path, index=False)
        report = ResearchSession(path, 'target').risk_report
        finding = next(item for item in report['findings'] if item['column'] == 'ordinary_feature')
        self.assertEqual(finding['level'], 'candidate')
        self.assertIn('unique_numeric_separates_training_classes', finding['signals'])
        # 候选只告警，不阻断
        self.assertIn('ordinary_feature', report['warning_columns'])
        self.assertFalse(report['blocked'])

    def test_legacy_scores_are_kept_but_marked_at_risk(self):
        import json
        from task_store import TaskStore
        from executor import ResearchSession
        store = TaskStore(self.root / 'legacy.db')
        task = store.create(self.fixture, 'target')
        # Reuse a real fitted result solely to simulate a pre-guard DB row.
        clean = ResearchSession(self.fixture, 'target', feature_decisions={
            'record_id': {'action': 'exclude', 'reason': 'Confirmed fixture evidence.'}})
        result = clean.run()
        with store.connection() as db:
            db.execute('UPDATE tasks SET risk_report=NULL WHERE id=?', (task,))
            db.execute('INSERT INTO experiments (task_id,number,proposal,status,result) VALUES (?,?,?,?,?)',
                       (task, 1, json.dumps({'model': 'logistic_regression', 'parameters': {}}),
                        'success', json.dumps(result)))
        shown = store.show(task)
        self.assertTrue(shown['legacy_results_at_risk'])
        self.assertEqual(shown['experiments'][0]['result'], result)
        with self.assertRaises(RuntimeError):
            store.finalize(task)

    def test_signals_do_not_depend_on_holdout_values(self):
        import pandas as pd
        from executor import ResearchSession
        from integrity import inspect_features
        from sklearn.datasets import load_iris
        path = self.root / 'scope.csv'
        frame = load_iris(as_frame=True).frame
        frame.to_csv(path, index=False)
        session = ResearchSession(path, 'target')
        training = session.train_indices
        before = inspect_features(frame.drop(columns='target').iloc[training],
            frame.target.astype(str).iloc[training], 'not-a-known-hash', 'target')
        frame.loc[session.test_indices, frame.columns[0]] = 999999
        after = inspect_features(frame.drop(columns='target').iloc[training],
            frame.target.astype(str).iloc[training], 'not-a-known-hash', 'target')
        self.assertEqual(before, after)

    def test_flags_post_hoc_column_like_chargeback_filed(self):
        """二德子报的缺陷：事后字段名字不含 id、相关不到 0.98，老规则全漏。

        真实后果：fraud 数据保留它 F1=0.955，剔除后 0.635，差 0.32。
        修法：加事后性列名规则 + 把相关阈值放到 0.85（仍标 candidate，要人审）。
        """
        import numpy as np
        import pandas as pd
        from integrity import inspect_features

        rng = np.random.default_rng(7)
        n = 600
        labels = pd.Series(rng.choice(['0', '1'], size=n, p=[0.95, 0.05]))
        fraud = (labels == '1').to_numpy()
        frame = pd.DataFrame({
            'amount': rng.normal(100, 40, n).round(2),
            'hour': rng.integers(0, 24, n),
            # 事后列：只有立案后才是 1，带 10% 噪声
            'chargeback_filed': (fraud.astype(int) ^ (rng.random(n) < 0.10).astype(int)),
        })
        findings = inspect_features(frame, labels, 'synthetic', 'target')
        hit = [f for f in findings if f['column'] == 'chargeback_filed']
        self.assertTrue(hit, 'chargeback_filed 应该被标出来')
        self.assertIn('post_hoc_name', hit[0]['signals'])
        self.assertEqual(hit[0]['level'], 'candidate')   # 候选，仍需人审

    def test_normal_features_are_not_flagged(self):
        """别修过头：普通特征不该被误标。"""
        import numpy as np
        import pandas as pd
        from integrity import inspect_features

        rng = np.random.default_rng(11)
        n = 600
        labels = pd.Series(rng.choice(['0', '1'], size=n))
        frame = pd.DataFrame({
            'amount': rng.normal(100, 40, n).round(2),
            'hour': rng.integers(0, 24, n),
            'merchant': rng.integers(0, 5, n),
        })
        findings = inspect_features(frame, labels, 'synthetic2', 'target')
        self.assertEqual(findings, [])
    def test_successful_run_still_surfaces_risk_warnings(self):
        """二德子指出的告警不出口问题。

        场景：candidate 泄漏列不阻断训练，于是 status=success、F1 很高，
        但如果 result 里没有 findings，用户会以为这份数据是干净的。
        """
        import numpy as np
        import pandas as pd
        from executor import ResearchSession

        rng = np.random.default_rng(7)
        n = 600
        labels = rng.choice(['0', '1'], n, p=[0.9, 0.1])
        fraud = (labels == '1').astype(int)
        frame = pd.DataFrame({
            'amount': rng.normal(100, 40, n).round(2),
            'hour': rng.integers(0, 24, n),
            'chargeback_filed': fraud ^ (rng.random(n) < 0.1).astype(int),
            'target': labels,
        })
        path = self.root / 'warn.csv'
        frame.to_csv(path, index=False)

        session = ResearchSession(path, 'target')
        result = session.run('logistic_regression', {})

        # 训练照常成功（candidate 不阻断）
        self.assertEqual(result['status'], 'success')
        self.assertFalse(result['risk_blocked'])
        # 但风险必须出现在 result 里
        columns = [w['column'] for w in result['risk_warnings']]
        self.assertIn('chargeback_filed', columns)
        self.assertIn('post_hoc_name',
                      next(w['signals'] for w in result['risk_warnings']
                           if w['column'] == 'chargeback_filed'))
        # 人的决策为空也不能把机器的发现一起吞掉
        self.assertEqual(result['feature_review'], {})