"""One shared model, multiple roles, one auditable budget. No automatic retries."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime,timezone
from model_costs import cost_breakdown,reference_prices


class ModelBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelLimits:
    max_calls: int = 24
    max_tokens: int = 100000
    max_seconds: float = 180

    def __post_init__(self):
        if (type(self.max_calls) is not int or self.max_calls < 0 or
                type(self.max_tokens) is not int or self.max_tokens < 0 or
                self.max_seconds <= 0):
            raise ValueError('Invalid shared model budget.')


class SharedModel:
    """Reserve before transport; reconcile known usage and retain unknown reserves.

    UTF-8 byte length plus message overhead is a conservative admission estimate,
    not provider token usage. No tokenizer can be assumed for arbitrary endpoints.
    A provider exceeding that reservation closes the gate for subsequent calls.
    """
    source = 'single_model'

    def __init__(self, client, limits=None, clock=time.monotonic, price_feed=None):
        self.client = client
        self.price_feed=price_feed
        self.limits = limits or ModelLimits()
        self.clock = clock
        self.started = clock()
        self.records = []
        self.last_usage = {}
        self._lock = threading.RLock()

    def for_role(self, role):
        owner = self

        class RoleClient:
            source = 'single_model'
            last_usage = {}

            def __call__(self, messages, timeout=30):
                self.last_usage = {}
                try:
                    return owner.call(role, messages, timeout)
                finally:
                    self.last_usage = dict(owner.last_usage)

        return RoleClient()

    def call(self, role, messages, timeout=30):
        with self._lock:
            self.last_usage = {}
            state = self.snapshot()
            encoded = json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            reserve = len(encoded) + 64 + int(getattr(self.client, 'max_output_tokens', 512))
            left = self.limits.max_seconds - (self.clock() - self.started)
            if (state['calls'] >= self.limits.max_calls or left <= 0 or
                    state['budget_tokens'] + reserve > self.limits.max_tokens):
                raise ModelBudgetExceeded('Shared model call/token/time budget exhausted.')
            if self.price_feed and getattr(self.price_feed,'auto_refresh',False):self.price_feed.refresh_async()
            record = {'price_snapshot':self.price_feed.snapshot_for_request() if self.price_feed else reference_prices(),**getattr(self.client,'billing_identity',{}),'request_utc':datetime.now(timezone.utc).isoformat(),'role': role, 'status': 'attempted', 'input_bytes': len(encoded),
                      'request_sha256': hashlib.sha256(encoded).hexdigest(),
                      'reserved_tokens': reserve, 'usage': None}
            self.records.append(record)
            began = self.clock()
            try:
                raw = self.client(messages, timeout=min(float(timeout), left))
                usage = dict(getattr(self.client, 'last_usage', {}) or {})
                if isinstance(raw, dict):
                    for dest, src in (('prompt_tokens', 'prompt_eval_count'),
                                      ('completion_tokens', 'eval_count')):
                        if type(raw.get(src)) is int:
                            usage[dest] = raw[src]
                if all(type(usage.get(k)) is int and usage[k] >= 0
                       for k in ('prompt_tokens', 'completion_tokens')):
                    self.last_usage = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens')}
                    record['usage'] = dict(self.last_usage)
                record['status'] = 'returned'
                return raw
            except Exception:
                # No provider body, key, raw prompt, or arbitrary exception is persisted.
                usage=dict(getattr(self.client,'last_usage',{}) or {})
                if all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                    self.last_usage={k:usage[k] for k in ('prompt_tokens','completion_tokens')}
                    record['usage']=dict(self.last_usage)
                    record['status']='failed_with_usage'
                else: record['status'] = 'failed_usage_unknown'
                raise
            finally:
                record['billing_usage']=dict(getattr(self.client,'last_billing_usage',{}) or {})
                record['elapsed_seconds'] = round(self.clock() - began, 6)

    def snapshot(self):
        known = sum(sum(r['usage'].values()) for r in self.records if r['usage'] is not None)
        reserved = sum(r['reserved_tokens'] for r in self.records if r['usage'] is None)
        return {'costs':cost_breakdown(self.records),'calls': len(self.records), 'known_tokens': known,
                'unreconciled_calls': sum(r['usage'] is None for r in self.records),
                'budget_tokens': known + reserved, 'reserved_unknown_tokens': reserved,
                'cost_cny': None if self.records else 0,
                'cost_basis': 'unknown' if self.records else 'no_model_calls',
                'elapsed_seconds': round(self.clock() - self.started, 6),
                'limits': vars(self.limits), 'records': list(self.records)}


def compact_profile(profile):
    """Training aggregates only; never send raw rows or column names to the model."""
    return {k: profile[k] for k in ('rows', 'features', 'class_count', 'minority_fraction',
                                   'risks', 'selection_metric', 'missing_fraction') if k in profile}
