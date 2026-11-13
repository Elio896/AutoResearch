"""Small DeepSeek chat adapter; credentials are never persisted or logged."""
import json
from urllib import error, parse, request


class CloudError(Exception):
    pass


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A credential-bearing request must not follow a provider redirect.
        return None


class DeepSeekClient:
    def __init__(self, api_key, model='deepseek-flash',
                 base_url='https://api.deepseek.com', max_output_tokens=256):
        if not isinstance(api_key, str) or not api_key.strip() or '\n' in api_key or '\r' in api_key:
            raise ValueError('Set DEEPSEEK_API_KEY locally before enabling cloud calls.')
        url = parse.urlsplit(base_url)
        if url.username or url.password or url.query or url.fragment or not url.hostname:
            raise ValueError('Invalid cloud base URL.')
        if url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost', '::1')):
            raise ValueError('Cloud credentials require HTTPS; HTTP is allowed only for loopback tests.')
        if type(max_output_tokens) is not int or not 32 <= max_output_tokens <= 1024:
            raise ValueError('Cloud output limit must be between 32 and 1024 tokens.')
        self._api_key = api_key.strip()
        self.model = model
        self.url = base_url.rstrip('/') + '/chat/completions'
        self.max_output_tokens = max_output_tokens
        self.last_usage = {}

    def __call__(self, messages, timeout):
        self.last_usage = {}
        encoded_messages = json.dumps(messages, ensure_ascii=False).encode('utf-8')
        if len(encoded_messages) > 12000:
            raise CloudError('Cloud prompt exceeds 12000-byte limit.')
        body = json.dumps({'model': self.model, 'messages': messages, 'stream': False,
                           'max_tokens': self.max_output_tokens, 'temperature': 0,
                           'thinking': {'type': 'disabled'},
                           'response_format': {'type': 'json_object'}}).encode('utf-8')
        req = request.Request(self.url, data=body, headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._api_key})
        handlers = [NoRedirect()]
        if parse.urlsplit(self.url).hostname in ('127.0.0.1', 'localhost', '::1'):
            handlers.append(request.ProxyHandler({}))
        try:
            with request.build_opener(*handlers).open(req, timeout=timeout) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise CloudError('Cloud response exceeds size limit.')
            envelope = json.loads(raw)
            usage = envelope.get('usage', {})
            if isinstance(usage, dict):
                self.last_usage = {key: value for key, value in usage.items()
                                   if key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                   and type(value) is int and value >= 0}
            choice = envelope['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise CloudError('Cloud response incomplete or not a final text answer.')
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise CloudError('Cloud response is empty.')
            return content
        except error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise CloudError(f'Cloud HTTP {status}; no automatic transport retry.') from None
        except (error.URLError, TimeoutError, OSError):
            raise CloudError('Cloud endpoint unavailable or timed out; billing may be unknown.') from None
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise CloudError('Malformed cloud response.') from None
