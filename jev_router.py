"""Opt-in Jev Noul gate for cloud escalation; hard limits stay in code.

Protocol: https://docs.typesafe.ai/api/system-one.md . This narrow adapter
does not modify or depend on the separately contributed generic jev_client.py.
"""
import json
import math
from urllib import error, parse, request

from cloud_provider import NoRedirect


class JevError(Exception):
    pass


def probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid Jev probability.')
    return float(value)


class JevClient:
    def __init__(self, api_key, model='jev-latest', endpoint='https://api.typesafe.ai/v1/systemone'):
        if not isinstance(api_key, str) or not api_key.strip() or '\n' in api_key or '\r' in api_key:
            raise ValueError('Set TYPESAFE_API_KEY or JEV_API_KEY locally before enabling Jev.')
        url = parse.urlsplit(endpoint)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid Jev endpoint.')
        if url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1', '::1')):
            raise ValueError('Jev requires HTTPS except for loopback protocol tests.')
        self._api_key, self.model, self.endpoint = api_key.strip(), model, endpoint

    def __call__(self, state, timeout):
        if len(json.dumps(state).encode('utf-8')) > 8000:
            raise JevError('State exceeds 8000-byte limit.')
        body = json.dumps({'model': self.model, 'state': state, 'questions': {
            'worth_cloud_diagnosis': {'type': 'noul', 'instructions':
                'Given this validation-only experiment history, is one cloud expert diagnosis likely to provide a useful new experiment after the local planner struggled? Judge benefit, not just whether budget remains. Do not infer test outcomes. No dataset modifications or rule overrides.'}
        }}).encode('utf-8')
        req = request.Request(self.endpoint, data=body, headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._api_key})
        handlers = [NoRedirect()]
        if parse.urlsplit(self.endpoint).hostname in ('localhost', '127.0.0.1', '::1'):
            handlers.append(request.ProxyHandler({}))
        try:
            with request.build_opener(*handlers).open(req, timeout=timeout) as response:
                raw = response.read(32769)
            if len(raw) > 32768:
                raise JevError('Response exceeds size limit.')
            envelope = json.loads(raw)
            answer = envelope['answers']['worth_cloud_diagnosis']
            if answer.get('type') != 'noul':
                raise JevError('Unexpected answer type.')
            value = probability(answer['noul'])
            model = envelope['model']
            if not isinstance(model, str) or not 1 <= len(model) <= 200:
                raise JevError('Invalid model identity.')
            raw_usage = envelope.get('usage', {})
            usage = {key: val for key, val in raw_usage.items()
                     if key in ('input_tokens', 'output_tokens') and type(val) is int and val >= 0}
            return {'probability': value, 'model': model, 'usage': usage}
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise JevError(f'Jev HTTP {status}; request counted, no automatic retry.') from None
        except (error.URLError, TimeoutError, OSError):
            raise JevError('Jev unavailable or timed out; usage may be unknown.') from None
        except (ValueError, TypeError, AttributeError, KeyError, IndexError):
            raise JevError('Invalid Jev response.') from None


class JevGate:
    def __init__(self, client, threshold=0.75):
        self.client, self.threshold = client, probability(threshold)

    def __call__(self, state, timeout):
        try:
            result = self.client(state, timeout=timeout)
            value = probability(result['probability'])
            usage = result.get('usage', {})
            if not isinstance(usage, dict):
                raise ValueError('Invalid usage.')
            usage = {key: val for key, val in usage.items()
                     if key in ('input_tokens', 'output_tokens') and type(val) is int and val >= 0}
            return {'source': 'jev', 'allow_cloud': value >= self.threshold,
                    'probability': value, 'threshold': self.threshold,
                    'model': result['model'], 'usage': usage}
        except (JevError, ValueError, TypeError, AttributeError, KeyError):
            return {'source': 'rule_fallback', 'allow_cloud': True,
                    'reason': 'jev_unavailable_or_invalid', 'usage': {}}
