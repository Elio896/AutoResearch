"""Validated local-model proposals around the LLM-free experiment executor."""
import argparse
import json
import math
import os
import time
from pathlib import Path
from urllib import error, request

from executor import ResearchSession
from cloud_provider import CloudError, DeepSeekClient


class LocalModelError(Exception):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def validate_proposal(text):
    """Accept JSON or one complete JSON fence, never code or extra actions."""
    if not isinstance(text, str) or len(text) > 12000:
        raise ValueError('Expected a short JSON object.')
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4].strip()
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field.')
            result[key] = value
        return result
    proposal = json.loads(text, object_pairs_hook=unique_object)
    if type(proposal) is not dict or set(proposal) != {'model', 'parameters'}:
        raise ValueError('Only model and parameters are allowed.')
    model, parameters = proposal['model'], proposal['parameters']
    fields = {
        'logistic_regression': {'C', 'max_iter', 'class_weight'},
        'random_forest': {'n_estimators', 'max_depth', 'min_samples_leaf', 'class_weight'},
    }
    if not isinstance(model, str) or model not in fields or type(parameters) is not dict:
        raise ValueError('Unsupported model or invalid parameters object.')
    if set(parameters) - fields[model]:
        raise ValueError('Parameter does not belong to this model.')
    ranges = {'C': (0.001, 100, False), 'max_iter': (100, 2000, True),
              'n_estimators': (10, 200, True), 'max_depth': (1, 20, True),
              'min_samples_leaf': (1, 20, True)}
    for key, value in parameters.items():
        if key == 'class_weight':
            if value is not None and value != 'balanced':
                raise ValueError('class_weight must be balanced or null.')
            continue
        if key == 'max_depth' and value is None:
            continue
        low, high, integer = ranges[key]
        if type(value) not in (int, float) or (integer and type(value) is not int):
            raise ValueError(f'{key} has an invalid numeric type.')
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} is outside the allowed range.')
    return proposal


class LocalQwenClient:
    def __init__(self, base_url='http://127.0.0.1:11435',
                 model='Qwen2.5-0.5B-Instruct:latest', max_output_tokens=160):
        self.url = base_url.rstrip('/') + '/api/chat'
        self.model = model
        self.max_output_tokens = max_output_tokens

    def __call__(self, messages, timeout):
        body = json.dumps({'model': self.model, 'messages': messages, 'stream': False,
                           'options': {'num_predict': self.max_output_tokens}}).encode('utf-8')
        req = request.Request(self.url, data=body, headers={'Content-Type': 'application/json'})
        # Do not send localhost traffic through system HTTP proxies.
        opener = request.build_opener(request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise LocalModelError('Response too large.')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise LocalModelError('Invalid response envelope.')
            if result.get('done_reason') == 'length':
                raise LocalModelError('Local output truncated.', retryable=True)
            return result
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise LocalModelError(f'HTTP {status}', retryable=status in (429, 503)) from None
        except (error.URLError, TimeoutError, OSError):
            raise LocalModelError('Local endpoint unavailable or timed out.') from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise LocalModelError('Malformed local response.') from None


class ResearchOrchestrator:
    def __init__(self, session, proposer, max_experiments=3, max_calls=4,
                 max_seconds=120, patience=2, retry_delay=1, emit=None,
                 cloud_proposer=None, max_cloud_calls=0, jev_gate=None, max_jev_calls=1):
        if type(max_experiments) is not int or not 1 <= max_experiments <= 20:
            raise ValueError('max_experiments must be an integer between 1 and 20.')
        if type(max_calls) is not int or not 0 <= max_calls <= 30:
            raise ValueError('max_calls must be an integer between 0 and 30.')
        if type(max_cloud_calls) is not int or not 0 <= max_cloud_calls <= 3:
            raise ValueError('max_cloud_calls must be an integer between 0 and 3.')
        if type(max_jev_calls) is not int or max_jev_calls not in (0, 1):
            raise ValueError('max_jev_calls must be 0 or 1.')
        if not math.isfinite(max_seconds) or max_seconds <= 0 or patience < 1 or retry_delay < 0:
            raise ValueError('Invalid time, patience or retry budget.')
        self.session, self.proposer = session, proposer
        self.max_experiments, self.max_calls = max_experiments, max_calls
        self.max_seconds, self.patience = max_seconds, patience
        self.retry_delay, self.emit = retry_delay, emit
        self.cloud_proposer, self.max_cloud_calls = cloud_proposer, max_cloud_calls
        self.jev_gate, self.max_jev_calls = jev_gate, max_jev_calls
        self.used = False

    def run(self):
        if self.used or self.session.closed or self.session.experiments:
            raise RuntimeError('Start one orchestration per fresh session.')
        self.used = True
        started = time.monotonic()
        experiments, events, seen = [], [], set()
        calls = no_improvement = 0
        tokens = {'prompt': 0, 'completion': 0}
        feedback = ''
        context = self.session.planning_context() if hasattr(self.session, 'planning_context') else {}
        def record(kind, **values):
            event = {'event': kind, **values}
            events.append(event)
            if self.emit:
                self.emit(event)
        def train(proposal):
            result = self.session.run(**proposal)
            experiments.append(result)
            defaults = {'logistic_regression': {'max_iter': 1000},
                        'random_forest': {'n_estimators': 100}}
            canonical = {'model': proposal['model'],
                         'parameters': {**defaults[proposal['model']], **proposal['parameters']}}
            seen.add(json.dumps(canonical, sort_keys=True))
            record('experiment', result=result)
            return result
        best = train({'model': 'logistic_regression', 'parameters': {}})
        status, reason = 'completed', 'experiment_budget'
        while len(experiments) < self.max_experiments:
            remaining = self.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                status, reason = 'stopped', 'time_budget'
                break
            if calls >= self.max_calls:
                status, reason = 'escalation_required', 'local_call_budget'
                break
            if no_improvement >= self.patience:
                status, reason = 'escalation_required', 'validation_plateau'
                break
            history = [{'model': item['model'], 'parameters': item['parameters'],
                        'selection_metric': item.get('selection_metric', 'f1_macro'), 'validation_score': item.get('validation_score', item['validation_f1']), 'validation_class_metrics': item.get('validation_class_metrics', [])} for item in experiments[-5:]]
            messages = [
                {'role': 'system', 'content': 'Propose ONE classifier experiment. Return only a JSON object with model and parameters. No explanation. No test evaluation. Never repeat an experiment.'},
                {'role': 'user', 'content': 'Allowed: logistic_regression parameters C (0.001..100), max_iter (100..2000), class_weight (null or balanced); random_forest parameters n_estimators (10..200), max_depth (1..20), min_samples_leaf (1..20), class_weight (null or balanced). Example: {"model":"random_forest","parameters":{"max_depth":3,"n_estimators":50}}\nValidation history: '
                 + json.dumps(history) + '\nTraining profile: ' + json.dumps(context) + '\nResearch guidance: If only one model family has been tried, prioritize an untried model family before more tuning. Use class recall feedback for imbalance. This is guidance, not evidence that a different family will win.\nLast rejection: ' + feedback},
            ]
            calls += 1  # Includes failed, busy, malformed, and duplicate attempts.
            try:
                response = self.proposer(messages, timeout=min(30, remaining))
                for key, field in [('prompt', 'prompt_eval_count'), ('completion', 'eval_count')]:
                    value = response.get(field, 0)
                    if type(value) is int and value >= 0:
                        tokens[key] += value
                content = response.get('message', {}).get('content')
                proposal = validate_proposal(content)
                defaults = {'logistic_regression': {'max_iter': 1000}, 'random_forest': {'n_estimators': 100}}
                canonical = {'model': proposal['model'], 'parameters': {**defaults[proposal['model']], **proposal['parameters']}}
                if json.dumps(canonical, sort_keys=True) in seen:
                    raise ValueError('Duplicate experiment; choose different parameters.')
            except LocalModelError as exc:
                feedback = str(exc)
                record('local_error', attempt=calls, reason=feedback)
                if not exc.retryable:
                    status, reason = 'escalation_required', 'local_unavailable'
                    break
                time.sleep(min(self.retry_delay, max(0, self.max_seconds - (time.monotonic() - started))))
                continue
            except (ValueError, TypeError, AttributeError):
                feedback = 'Invalid or duplicate JSON. Use the exact allowed schema and parameter ranges.'
                record('proposal_rejected', attempt=calls, reason=feedback)
                continue
            if time.monotonic() - started >= self.max_seconds:
                status, reason = 'stopped', 'time_budget'
                break
            record('proposal_accepted', attempt=calls, proposal=proposal)
            try:
                result = train(proposal)
            except ValueError:
                # A valid but data-incompatible experiment consumes an experiment slot.
                status, reason = 'escalation_required', 'experiment_failed'
                record('experiment_failed', reason='Valid candidate could not fit this dataset.')
                break
            if result.get('validation_score', result['validation_f1']) > best.get('validation_score', best['validation_f1']) + 0.0001:
                best, no_improvement = result, 0
            else:
                no_improvement += 1
            feedback = ''
        cloud_calls = 0
        cloud_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        cloud_usage_complete = True
        jev_calls = 0
        gate_decision = {'source': 'rules', 'allow_cloud': None, 'reason': 'not_evaluated'}
        cloud_eligible = (status == 'escalation_required' and self.cloud_proposer is not None
                          and self.max_cloud_calls > 0 and len(experiments) < self.max_experiments
                          and time.monotonic() - started < self.max_seconds)
        if cloud_eligible:
            gate_decision = {'source': 'rules', 'allow_cloud': True, 'reason': 'rule_escalation'}
            if self.jev_gate is not None and self.max_jev_calls > 0:
                jev_calls += 1  # Failed attempts still consume the explicit Jev call budget.
                gate_state = {'trigger': reason, 'local_calls': calls,
                    'remaining_experiments': self.max_experiments - len(experiments),
                    'remaining_cloud_calls': self.max_cloud_calls,
                    'remaining_seconds': round(max(0, self.max_seconds - (time.monotonic() - started)), 2),
                    'validation_history': [{'model': item['model'], 'parameters': item['parameters'],
                        'selection_metric': item.get('selection_metric', 'f1_macro'), 'validation_score': item.get('validation_score', item['validation_f1']), 'validation_class_metrics': item.get('validation_class_metrics', [])} for item in experiments[-5:]]}
                try:
                    gate_decision = self.jev_gate(gate_state, timeout=min(10, max(0.001,
                        self.max_seconds - (time.monotonic() - started))))
                    if type(gate_decision.get('allow_cloud')) is not bool:
                        raise ValueError('Invalid escalation decision.')
                except Exception:
                    gate_decision = {'source': 'rule_fallback', 'allow_cloud': True,
                                     'reason': 'jev_gate_failed', 'usage': {}}
            elif self.jev_gate is not None:
                gate_decision = {'source': 'rule_fallback', 'allow_cloud': True,
                                 'reason': 'jev_call_budget', 'usage': {}}
            record('escalation_gate', decision=gate_decision)
            if not gate_decision['allow_cloud']:
                status, reason = 'stopped', 'jev_declined'
        if cloud_eligible and gate_decision['allow_cloud']:
            record('cloud_escalation_requested', reason=reason)
            while cloud_calls < self.max_cloud_calls and len(experiments) < self.max_experiments:
                remaining = self.max_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    status, reason = 'stopped', 'time_budget'
                    break
                # This is a bounded expert recovery proposal, not arbitrary tools or code.
                history = [{'model': item['model'], 'parameters': item['parameters'],
                            'selection_metric': item.get('selection_metric', 'f1_macro'), 'validation_score': item.get('validation_score', item['validation_f1']), 'validation_class_metrics': item.get('validation_class_metrics', [])} for item in experiments[-5:]]
                messages = [
                    {'role': 'system', 'content': 'Diagnose validation stagnation or invalid local proposals. Return JSON only: model and parameters. Select a different bounded experiment, never evaluate test data. Allowed models: logistic_regression (C 0.001..100, max_iter 100..2000, class_weight null or balanced), random_forest (n_estimators 10..200, max_depth 1..20, min_samples_leaf 1..20, class_weight null or balanced).'},
                    {'role': 'user', 'content': json.dumps({'escalation_reason': reason,
                        'validation_history': history, 'last_rejection': feedback})},
                ]
                cloud_calls += 1
                try:
                    content = self.cloud_proposer(messages, timeout=min(30, remaining))
                    proposal = validate_proposal(content)
                    defaults = {'logistic_regression': {'max_iter': 1000}, 'random_forest': {'n_estimators': 100}}
                    canonical = {'model': proposal['model'], 'parameters': {**defaults[proposal['model']], **proposal['parameters']}}
                    if json.dumps(canonical, sort_keys=True) in seen:
                        raise ValueError('Duplicate cloud proposal.')
                except CloudError:
                    # Do not log arbitrary exception messages or provider response bodies.
                    record('cloud_error', attempt=cloud_calls, reason='Cloud request failed; no automatic retry.')
                    status, reason = 'escalation_required', 'cloud_unavailable'
                    break
                except (ValueError, TypeError, AttributeError):
                    feedback = 'Invalid or duplicate cloud JSON; follow the allowed schema.'
                    record('cloud_proposal_rejected', attempt=cloud_calls)
                    continue
                finally:
                    usage = getattr(self.cloud_proposer, 'last_usage', {})
                    if not isinstance(usage, dict):
                        usage = {}
                    for field in cloud_usage:
                        value = usage.get(field)
                        if type(value) is int and value >= 0:
                            cloud_usage[field] += value
                        else:
                            cloud_usage_complete = False
                if time.monotonic() - started >= self.max_seconds:
                    status, reason = 'stopped', 'time_budget'
                    break
                record('cloud_proposal_accepted', attempt=cloud_calls, proposal=proposal)
                try:
                    candidate = train(proposal)
                except ValueError:
                    status, reason = 'escalation_required', 'cloud_experiment_failed'
                    record('experiment_failed', reason='Cloud candidate could not fit dataset.')
                    break
                if candidate.get('validation_score', candidate['validation_f1']) > best.get('validation_score', best['validation_f1']) + 0.0001:
                    best = candidate
                status, reason = 'completed', 'cloud_recovery'
                break
        result = {'status': status, 'stop_reason': reason, 'experiments': experiments,
                  'best_experiment_id': best['experiment_id'], 'model_calls': calls,
                  'reported_local_tokens': tokens, 'cloud_calls': cloud_calls,
                  'reported_cloud_tokens': cloud_usage,
                  'cloud_usage_complete': cloud_usage_complete,
                  'cloud_cost': None,
                  'jev_calls': jev_calls, 'escalation_gate': gate_decision,
                  'total_remote_calls': cloud_calls + jev_calls, 'jev_cost': None,
                  'planning_policy': 'context_v2', 'training_profile': context,
                  'elapsed_seconds': time.monotonic() - started, 'events': events,
                  'test_evaluated': False}
        record('research_stopped', status=status, reason=reason)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--target', required=True)
    parser.add_argument('--metric', choices=['f1_macro', 'f1_minority'], default='f1_macro')
    parser.add_argument('--pos-label', default='1')
    parser.add_argument('--base-url', default='http://127.0.0.1:11435')
    parser.add_argument('--model', default='Qwen2.5-0.5B-Instruct:latest')
    parser.add_argument('--max-experiments', type=int, default=3)
    parser.add_argument('--max-calls', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=120)
    parser.add_argument('--enable-cloud', action='store_true', help='Opt in to paid DeepSeek recovery')
    parser.add_argument('--max-cloud-calls', type=int, default=1)
    parser.add_argument('--cloud-model', default=os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'))
    parser.add_argument('--enable-jev', action='store_true', help='Opt in to a paid Jev cloud-escalation gate')
    parser.add_argument('--max-jev-calls', type=int, choices=[0, 1], default=1)
    parser.add_argument('--jev-threshold', type=float, default=0.75)
    parser.add_argument('--jev-model', default='jev-latest')
    parser.add_argument('--database', default='artifacts/research.sqlite3', help='Persistent task database')
    parser.add_argument('--feature-decisions', default='{}', help='Explicit feature review JSON')
    parser.add_argument('--output', required=True, help='New JSON result file; existing files are never overwritten')
    args = parser.parse_args()
    if args.enable_jev and (not args.enable_cloud or args.max_cloud_calls < 1):
        parser.error('--enable-jev requires cloud recovery enabled with a positive call budget.')
    jev_gate = None
    if args.enable_jev:
        from jev_router import JevClient, JevGate
        jev_gate = JevGate(JevClient(os.environ.get('TYPESAFE_API_KEY') or os.environ.get('JEV_API_KEY', ''),
                                    model=args.jev_model), threshold=args.jev_threshold)
    cloud = DeepSeekClient(os.environ.get('DEEPSEEK_API_KEY', ''), model=args.cloud_model) if args.enable_cloud else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive journal creation prevents accidentally reusing an existing task path.
    with output.with_suffix(output.suffix + '.events.jsonl').open('x', encoding='utf-8') as journal:
        if output.exists():
            raise FileExistsError('Result path already exists; use a new task path.')
        from task_store import StoredResearchSession, TaskStore
        store = TaskStore(args.database)
        task_id = store.create(args.csv, args.target, max_experiments=args.max_experiments,
                               feature_decisions=json.loads(args.feature_decisions), metric=args.metric, pos_label=args.pos_label)
        def emit(event):
            store.add_event(task_id, event)
            journal.write(json.dumps(event, ensure_ascii=False) + '\n')
            journal.flush()
        session = StoredResearchSession(store, task_id)
        result = ResearchOrchestrator(session, LocalQwenClient(args.base_url, args.model),
                    max_experiments=args.max_experiments, max_calls=args.max_calls,
                    max_seconds=args.max_seconds, emit=emit,
                    cloud_proposer=cloud, max_cloud_calls=args.max_cloud_calls if args.enable_cloud else 0,
                    jev_gate=jev_gate, max_jev_calls=args.max_jev_calls).run()
        result['task_id'] = task_id
        store.save_report(task_id, result)
        with output.open('x', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
