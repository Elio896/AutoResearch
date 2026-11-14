"""Credential-free HTTPS diagnostics. Never disables TLS verification."""
import json
import os
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

endpoint = 'https://api.deepseek.com/v1/models'
report = {'endpoint': endpoint, 'python': sys.version.split()[0], 'openssl': ssl.OPENSSL_VERSION,
          'proxy_configuration_detected': sorted(k for k in urllib.request.getproxies() if k in ('http','https','all','no')),
          'credentials_used': False}
try:
    req = urllib.request.Request(endpoint, method='GET')
    with urllib.request.build_opener().open(req, timeout=12) as response:
        report['python_https'] = {'tls_connected': True, 'http_status': response.status}
except urllib.error.HTTPError as exc:
    report['python_https'] = {'tls_connected': True, 'http_status': exc.code}
    exc.close()
except Exception as exc:
    reason = getattr(exc, 'reason', exc)
    report['python_https'] = {'tls_connected': False, 'error_type': type(reason).__name__,
                              'ssl_reason': getattr(reason, 'reason', None),
                              'errno': getattr(reason, 'errno', None), 'winerror': getattr(reason, 'winerror', None)}

curl = shutil.which('curl.exe')
if curl:
    try:
        result = subprocess.run([curl, '--silent', '--output', os.devnull, '--write-out', '%{http_code}',
                                 '--connect-timeout', '5', '--max-time', '12', endpoint],
                                capture_output=True, text=True, timeout=15)
        status = result.stdout.strip()
        report['windows_curl'] = {'exit_code': result.returncode,
                                   'http_status': status if status.isdigit() else None,
                                   'tls_connected': result.returncode == 0 and status != '000'}
    except Exception as exc:
        report['windows_curl'] = {'error_type': type(exc).__name__}
else:
    report['windows_curl'] = {'available': False}

path = Path(__file__).resolve().parents[1] / 'artifacts/model-connection-diagnostic.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
print('Saved:', path)
