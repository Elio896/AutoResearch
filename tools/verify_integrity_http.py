"""Real HTTP verification of review gates; never evaluates the test set."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    base = 'http://127.0.0.1:8002'
    opener = build_opener(ProxyHandler({}))
    def call(method, path, payload=None):
        request = Request(base + path, method=method,
            data=json.dumps(payload).encode('utf-8') if payload is not None else None,
            headers={'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=30) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)
    csv = Path('bench/tasks/t1_missing/data.csv').read_text(encoding='utf-8')
    status, created = call('POST', '/tasks', {'csv_text': csv, 'target': 'target'})
    assert status == 201 and created['risk_report']['blocked']
    task_id = created['task_id']
    manual_status, _ = call('POST', f'/tasks/{task_id}/experiments', {})
    agent_status, _ = call('POST', f'/tasks/{task_id}/research', {})
    assert manual_status == agent_status == 409
    _, before = call('GET', f'/tasks/{task_id}')
    assert not before['experiments']
    status, reviewed = call('POST', f'/tasks/{task_id}/review', {'decisions': {
        'record_id': {'action': 'exclude', 'reason': 'Confirmed generator sorts target before assigning this ID.'}}})
    assert status == 200 and not reviewed['risk_report']['blocked']
    status, experiment = call('POST', f'/tasks/{task_id}/experiments', {})
    assert status == 200 and experiment['excluded_columns'] == ['record_id']
    locked_status, _ = call('POST', f'/tasks/{task_id}/review', {'decisions': {}})
    assert locked_status == 409
    _, task = call('GET', f'/tasks/{task_id}')
    assert task['final_result'] is None
    result = {'task_id': task_id, 'blocked_manual_http': manual_status, 'blocked_agent_http': agent_status,
        'blocked_attempts_consumed': len(before['experiments']), 'review_configuration_locked_http': locked_status,
        'excluded_columns': experiment['excluded_columns'], 'validation_f1': experiment['validation_f1'],
        'test_evaluated': False, 'model_api_calls': 0}
    Path('artifacts/integrity_http_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
