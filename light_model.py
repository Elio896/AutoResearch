"""轻量模型 API 客户端（OpenAI 兼容 /chat/completions）。

为什么换：本地 7B 在"按格式说话"上不可靠——同一个提示，它会回 JSON 壳、把句子写进 key、
或者干脆回 `{}`。基础操作与分类这层要的是稳定输出，所以走 API。

代价必须先说清：
- **API 是花钱的**。所以默认关闭，没有 key 就直接不可用，绝不偷偷回退成"已经调用过"；
- 调用次数有硬上限（`max_calls`），超了抛错，不会悄悄多花；
- 记账只记真实 usage；**不知道账单就标 unknown，不记 0**；
- key 只存在进程内存里，不进日志、不进异常消息。

环境变量（按顺序取第一个非空）：
    LIGHT_MODEL_API_KEY / DEEPSEEK_API_KEY
    LIGHT_MODEL_BASE_URL  默认 https://api.deepseek.com/v1
    LIGHT_MODEL_NAME      默认 deepseek-chat
    LIGHT_MODEL_MAX_CALLS 默认 6
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from cloud_provider import NoRedirect

DEFAULT_BASE_URL = 'https://api.deepseek.com/v1'
DEFAULT_MODEL = 'deepseek-chat'
DEFAULT_MAX_CALLS = 6
KEY_ENV_NAMES = ('LIGHT_MODEL_API_KEY', 'DEEPSEEK_API_KEY')


class LightModelError(RuntimeError):
    """API 层失败。消息里不会出现 key。"""


class LightModelTransportError(LightModelError):
    """Transient transport error; only shared budget may retry."""


def _mask(value: str | None) -> str:
    if not value:
        return '<empty>'
    return f'{value[:3]}…{value[-2:]}（len={len(value)}）'


class LightModelClient:
    """一个只会做一件事的客户端：发 messages，拿回字符串。"""

    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, max_output_tokens: int = 512,
                 max_calls: int = DEFAULT_MAX_CALLS, timeout: float = 60.0,
                 temperature: float = 0.0, structured_output: bool = False):
        if not isinstance(api_key, str) or not api_key.strip():
            raise LightModelError('缺少 API key：先设置 LIGHT_MODEL_API_KEY 或 DEEPSEEK_API_KEY')
        if '\n' in api_key or '\r' in api_key:
            raise LightModelError('API key 里含换行，拒绝使用')
        if not api_key.strip().isascii():
            raise LightModelError('Model API key contains non-ASCII characters. Replace the example text with the actual provider API key in the server environment.')
        self._api_key = api_key.strip()
        url = urlsplit(base_url)
        if (not url.hostname or url.username or url.password or url.query or url.fragment or
                (url.scheme != 'https' and not (url.scheme == 'http' and
                  url.hostname in ('localhost', '127.0.0.1', '::1')))):
            raise LightModelError('Model endpoint must use HTTPS (HTTP only for loopback).')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.max_calls = int(max_calls)
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.structured_output = structured_output
        self.usage = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        self.last_usage = {}
        self.last_billing_usage = {}
        self.usage_complete = True

    # ---------- 对外 ----------

    @property
    def descriptor(self) -> dict:
        return {'provider': 'api', 'model': self.model, 'base_url': self.base_url}

    @property
    def billing_account_identity(self):
        import hashlib
        return hashlib.sha256(((urlsplit(self.base_url).hostname or '')+'\0'+self._api_key).encode()).hexdigest()

    def balance_snapshot(self,timeout=3):
        # Only the documented official endpoint. Never send an API key to a
        # guessed billing service, reseller endpoint or redirect target.
        if urlsplit(self.base_url).hostname!='api.deepseek.com':return []
        req=urllib.request.Request('https://api.deepseek.com/user/balance',headers={'Authorization':'Bearer '+self._api_key,'Accept':'application/json'})
        try:
            with urllib.request.build_opener(NoRedirect(),self._proxy_handler()).open(req,timeout=timeout) as response:
                raw=response.read(16385)
            if len(raw)>16384:raise ValueError('Balance response too large')
            data=json.loads(raw)
            return [{'currency':r['currency'],'balance':r['total_balance']} for r in data.get('balance_infos',[])]
        except urllib.error.HTTPError as exc:
            exc.close();raise LightModelError('Account balance unavailable.') from None

    def __call__(self, messages, timeout: float | None = None) -> str:
        if self.usage['calls'] >= self.max_calls:
            raise LightModelError(f'已经调用 {self.usage["calls"]} 次，达到上限 {self.max_calls}')
        payload = {
            'model': self.model,
            'messages': messages,
            'max_tokens': self.max_output_tokens,
            'temperature': self.temperature,
            'stream': False,
        }
        if self.structured_output:
            payload['response_format'] = {'type':'json_object'}
            if urlsplit(self.base_url).hostname == 'api.deepseek.com':
                payload['thinking'] = {'type':'disabled'}
                # DeepSeek 的 json_object 模式要求提示词里出现 "json"，否则直接 400。
                # 这里统一补一句约束，避免每个调用点各自记得写。
                joined = ' '.join(str(m.get('content','')) for m in messages
                                  if isinstance(m,dict))
                if 'json' not in joined.lower():
                    payload['messages'] = list(messages) + [
                        {'role':'user','content':'只输出一个 JSON 对象，不要解释。'}]
        self.last_usage = {}
        self.last_billing_usage = {}
        # Reserve attempts before transport: a timeout can still be billed.
        self.usage['calls'] += 1
        try:
            data = self._post(payload, self.timeout if timeout is None else float(timeout))
        except Exception:
            self.usage_complete = False
            raise
        usage = data.get('usage') or {}
        self.last_usage = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens')
                           if type(usage.get(k)) is int and usage[k] >= 0}
        self.last_billing_usage={k:usage[k] for k in ('prompt_cache_hit_tokens','prompt_cache_miss_tokens') if type(usage.get(k)) is int and usage[k]>=0}
        if 'prompt_cache_hit_tokens' not in self.last_billing_usage:
            cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            if type(cached) is int and cached>=0:self.last_billing_usage['prompt_cache_hit_tokens']=cached
        self.usage_complete = self.usage_complete and len(self.last_usage) == 2
        for key, value in self.last_usage.items():
            self.usage[key] += value
        return self._content(data)

    def usage_summary(self) -> dict:
        return {
            **self.usage,
            'model': self.model,
            'cost_cny': None,          # 没有对账单之前不写 0
            'cost_basis': 'unknown',
            'usage_complete': self.usage_complete,
            'note': '按 provider 账单结算；未取到账单前不记 0 成本',
        }

    # ---------- 内部 ----------

    def _proxy_handler(self):
        # Respect the user's normal HTTPS route. Only loopback fixtures must
        # bypass proxies; forcing public APIs direct breaks proxy-based setups.
        if urlsplit(self.base_url).hostname in ('localhost', '127.0.0.1', '::1'):
            return urllib.request.ProxyHandler({})
        return urllib.request.ProxyHandler()

    def _post(self, payload: dict, timeout: float) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + self._api_key,
                'Accept': 'application/json',
            },
            method='POST',
        )
        opener = urllib.request.build_opener(NoRedirect(), self._proxy_handler())
        try:
            with opener.open(request, timeout=timeout) as response:
                raw_bytes = response.read(65537)
                if len(raw_bytes) > 65536:
                    raise LightModelError('Model response exceeds size limit.')
                raw = raw_bytes.decode('utf-8', 'replace')
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise (LightModelTransportError if code in (408,429,500,502,503,504) else LightModelError)(f'HTTP {code}; provider response withheld.') from None
        except urllib.error.URLError as exc:
            import ssl
            if isinstance(exc.reason,ssl.SSLCertVerificationError):
                raise LightModelError('TLS certificate verification failed.') from None
            raise LightModelTransportError('Model network transport failed; usage unknown.') from None
        except TimeoutError:
            raise LightModelTransportError('Model request timed out; usage unknown.') from None

        if not raw.strip():
            raise LightModelError('返回空响应体')
        try:
            return json.loads(raw)
        except Exception:                   # noqa: BLE001
            raise LightModelError('Model returned malformed JSON.') from None

    @staticmethod
    def _content(data: dict) -> str:
        choices = data.get('choices')
        if not isinstance(choices, list) or not choices:
            raise LightModelError('Model response has no choices.')
        if choices[0].get('finish_reason') == 'length':
            raise LightModelError('模型输出达到长度上限，JSON 可能被截断。')
        message = choices[0].get('message') or {}
        content = message.get('content')
        if not isinstance(content, str):
            raise LightModelError('choices[0].message.content 不是字符串')
        return content


def from_env(environ: dict | None = None) -> LightModelClient | None:
    """没有 key 就返回 None——调用方必须自己决定怎么退，不允许假装调过。"""
    environ = os.environ if environ is None else environ
    key = ''
    for name in KEY_ENV_NAMES:
        value = (environ.get(name) or '').strip()
        if value:
            key = value
            break
    if not key:
        return None
    base_url = (environ.get('LIGHT_MODEL_BASE_URL') or DEFAULT_BASE_URL).strip()
    model = (environ.get('LIGHT_MODEL_NAME') or DEFAULT_MODEL).strip()
    try:
        max_calls = int((environ.get('LIGHT_MODEL_MAX_CALLS') or DEFAULT_MAX_CALLS))
    except ValueError:
        max_calls = DEFAULT_MAX_CALLS
    return LightModelClient(key, base_url=base_url, model=model, max_calls=max_calls)
