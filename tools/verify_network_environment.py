"""Actual loopback TCP probe, without mocks, skipped cases or system changes."""
import json
import socket
from pathlib import Path

listener = socket.socket()
client = socket.socket()
listener.settimeout(2)
client.settimeout(2)
report = {'probe': 'real_loopback_tcp', 'bound': False, 'connected': False, 'accepted': False}
try:
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    report['bound'] = True
    client.connect(listener.getsockname())
    report['connected'] = True
    accepted, _ = listener.accept()
    accepted.close()
    report['accepted'] = True
    report['status'] = 'passed'
except OSError as exc:
    report.update(status='blocked_or_failed', error_type=type(exc).__name__,
                  errno=exc.errno, winerror=getattr(exc, 'winerror', None))
finally:
    client.close()
    listener.close()
path = Path(__file__).resolve().parents[1] / 'artifacts/network-environment.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
raise SystemExit(0 if report['accepted'] else 1)
