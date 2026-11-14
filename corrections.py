"""User feedback compiled into scoped, durable, executable constraints.

Natural language is never executable code or a privileged model instruction.
Recognized constraints are enforced before training; ambiguous feedback remains
pending. Guarantees cover these checked invariants, not arbitrary future errors.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

MODELS = ('logistic_regression', 'random_forest', 'hist_gradient_boosting')
KINDS = ('reject_duplicates', 'forbid_model', 'require_missing_indicator', 'no_paid_calls', 'protect_holdout')


def fingerprint(proposal):
    """Compare effective defaults too: {} and C=1 are the same LR experiment."""
    defaults = {
        'logistic_regression': {'C': 1.0, 'max_iter': 1000, 'class_weight': None},
        'random_forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None},
        'hist_gradient_boosting': {'max_iter': 100, 'max_depth': None, 'min_samples_leaf': 20,
                                   'learning_rate': 0.1, 'class_weight': None},
    }
    model = proposal.get('model')
    params = {**defaults.get(model, {}), **(proposal.get('parameters') or {})}
    prep = {'impute': 'median', 'encode': 'onehot', 'imbalance_strategy': 'none',
            'add_missing_indicator': False, **(proposal.get('preprocessing') or {})}
    if prep['imbalance_strategy'] == 'class_weight' and params.get('class_weight') is None:
        params['class_weight'] = 'balanced'
    # class_weight is the effective operation; normalize its equivalent prep spelling.
    prep['imbalance_strategy'] = 'none'
    params = {k: float(v) if type(v) in (int, float) else v for k, v in params.items()}
    body = json.dumps({'model': model, 'parameters': params, 'preprocessing': prep}, sort_keys=True)
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def interpret(text):
    if re.search(r'不要.{0,8}测试集.{0,8}(调参|训练)|never.{0,15}test.{0,15}tun', text, re.I):
        return 'protect_holdout', None
    if re.search(r'忽略预算|绕过预算|改.{0,4}测试.{0,4}(标签|分数)|删除失败记录|ignore.{0,5}budget', text, re.I):
        return 'unsafe', None
    if re.search(r'(不要|别|禁止|不再|避免).{0,10}(重复|相同|同样).{0,10}(实验|训练|运行|跑)|never repeat|no duplicate', text, re.I):
        return 'reject_duplicates', None
    if re.search(r'(不要|禁止|不许|别|never|no).{0,12}(收费|付费|paid|云端)', text, re.I):
        return 'no_paid_calls', None
    if re.search(r'(加上|需要|必须|请加|require).{0,12}(缺失指示|缺失标记|missing indicator)', text, re.I):
        return 'require_missing_indicator', None
    if re.search(r'不要|禁止|不许|别|never|avoid', text, re.I):
        for name, aliases in [('random_forest', r'随机森林|random.?forest'),
                              ('logistic_regression', r'逻辑回归|logistic'),
                              ('hist_gradient_boosting', r'梯度提升|hist_gradient_boosting')]:
            if re.search(aliases, text, re.I):
                return 'forbid_model', name
    return None, None


class CorrectionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS corrections (
                  id TEXT PRIMARY KEY, text TEXT NOT NULL, kind TEXT, value TEXT,
                  scope TEXT NOT NULL, dataset_sha256 TEXT, status TEXT NOT NULL,
                  created REAL NOT NULL, source_run_id TEXT);
                CREATE TABLE IF NOT EXISTS correction_events (
                  seq INTEGER PRIMARY KEY AUTOINCREMENT, correction_id TEXT,
                  kind TEXT NOT NULL, run_id TEXT, details TEXT NOT NULL, created REAL NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def audit(db, rule_id, kind, run_id='', **details):
        db.execute('INSERT INTO correction_events(correction_id,kind,run_id,details,created) VALUES(?,?,?,?,?)',
                   (rule_id, kind, run_id, json.dumps(details, ensure_ascii=False), time.time()))

    def add(self, text, *, kind=None, value=None, scope='global', dataset_sha256=None, source_run_id=None):
        if not isinstance(text, str) or not 3 <= len(text.strip()) <= 2000:
            raise ValueError('Feedback must contain 3..2000 characters.')
        if re.search(r'\b(?:sk-|apikey_)[A-Za-z0-9]{24,}\b', text):
            raise ValueError('Do not include API credentials in feedback.')
        if scope not in ('global', 'dataset'):
            raise ValueError('Unknown feedback scope.')
        if scope == 'dataset' and not re.fullmatch('[0-9a-f]{64}', dataset_sha256 or ''):
            raise ValueError('Dataset feedback requires its exact SHA256.')
        if scope == 'global':
            dataset_sha256 = None
        inferred, inferred_value = interpret(text)
        if kind is not None and kind not in KINDS:
            raise ValueError('Feedback cannot modify code, metrics, budgets or test labels.')
        kind, value = (kind, value) if kind else (inferred, inferred_value)
        if kind == 'forbid_model' and value not in MODELS:
            raise ValueError('Unknown forbidden model.')
        if kind != 'forbid_model':
            value = None
        status = 'rejected' if inferred == 'unsafe' else ('active' if kind else 'pending')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = [dict(r) for r in db.execute("SELECT * FROM corrections WHERE status='active'")]
            same = next((r for r in existing if r['kind'] == kind and r['value'] == value and
                         r['scope'] == scope and r['dataset_sha256'] == dataset_sha256), None)
            if same and status == 'active':
                self.audit(db, same['id'], 'duplicate_feedback', source_run_id or '')
                return {**same, 'duplicate': True}
            if kind == 'forbid_model' and status == 'active':
                contexts = {dataset_sha256} if scope == 'dataset' else {None} | {
                    r['dataset_sha256'] for r in existing if r['scope'] == 'dataset'}
                for context in contexts:
                    relevant = [r for r in existing if r['scope'] == 'global' or
                                (context is not None and r['dataset_sha256'] == context)]
                    banned = {r['value'] for r in relevant if r['kind'] == 'forbid_model'} | {value}
                    if banned == set(MODELS):
                        status = 'rejected_conflict'
                        break
            rule_id = uuid.uuid4().hex
            db.execute('INSERT INTO corrections VALUES(?,?,?,?,?,?,?,?,?)',
                       (rule_id, text.strip(), kind, value, scope, dataset_sha256, status, time.time(), source_run_id))
            self.audit(db, rule_id, 'feedback_' + status, source_run_id or '', constraint=kind)
            return dict(db.execute('SELECT * FROM corrections WHERE id=?', (rule_id,)).fetchone())

    def all(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT * FROM corrections ORDER BY created,id')]

    def snapshot(self, dataset_sha256=None):
        rows = [r for r in self.all() if r['status'] == 'active' and
                (r['scope'] == 'global' or r['dataset_sha256'] == dataset_sha256)]
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode('utf-8')).hexdigest()
        return {'rules': rows, 'digest': digest, 'dataset_sha256': dataset_sha256, 'pinned_at': time.time()}

    def revoke(self, rule_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM corrections WHERE id=?', (rule_id,)).fetchone()
            if row is None:
                raise KeyError(rule_id)
            db.execute("UPDATE corrections SET status='revoked' WHERE id=?", (rule_id,))
            self.audit(db, rule_id, 'revoked')
            return {'id': rule_id, 'status': 'revoked'}

    def record(self, rule_id, event, run_id, **details):
        with self.connection() as db:
            self.audit(db, rule_id, event, run_id, **details)

    def events(self):
        with self.connection() as db:
            return [{**dict(r), 'details': json.loads(r['details'])}
                    for r in db.execute('SELECT * FROM correction_events ORDER BY seq')]


class CorrectionViolation(ValueError):
    def __init__(self, rule, reason):
        super().__init__(reason)
        self.rule_id = rule['id']
        self.kind = rule['kind']


class CorrectionGuard:
    def __init__(self, snapshot):
        self.snapshot = copy.deepcopy(snapshot)
        self.rules = self.snapshot['rules']

    @property
    def allowed_models(self):
        banned = {r['value'] for r in self.rules if r['kind'] == 'forbid_model'}
        return [model for model in MODELS if model not in banned]

    @property
    def no_paid_calls(self):
        return any(r['kind'] == 'no_paid_calls' for r in self.rules)

    def instructions(self):
        # Controlled vocabulary only; raw user text never becomes system instructions.
        return [{'constraint': r['kind'], 'value': r['value'], 'id': r['id']} for r in self.rules]

    def prepare(self, proposal, previous):
        result = copy.deepcopy(proposal)
        applied = []
        for rule in self.rules:
            if rule['kind'] == 'forbid_model' and result.get('model') == rule['value']:
                raise CorrectionViolation(rule, 'User correction forbids model ' + rule['value'])
            if rule['kind'] == 'require_missing_indicator':
                prep = result.setdefault('preprocessing', {})
                if prep.get('add_missing_indicator') is not True:
                    prep['add_missing_indicator'] = True
                    applied.append(rule['id'])
        for rule in self.rules:
            if rule['kind'] == 'reject_duplicates' and any(fingerprint(result) == fingerprint(p) for p in previous):
                raise CorrectionViolation(rule, 'User correction rejects a duplicate effective experiment before training.')
        return result, applied
