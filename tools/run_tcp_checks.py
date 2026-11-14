"""Run the formerly skipped TCP fixture group without skip decorators or mocks."""
import inspect
import json
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / 'tests')]


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cases = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.cases.append({'test': test.id(), 'status': 'passed'})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.cases.append({'test': test.id(), 'status': 'failed', 'error_type': err[0].__name__})

    def addError(self, test, err):
        super().addError(test, err)
        self.cases.append({'test': test.id(), 'status': 'error', 'error_type': err[0].__name__})


selected = []
for test in flatten(unittest.defaultTestLoader.discover(str(root / 'tests'))):
    klass = type(test)
    method = getattr(klass, test._testMethodName)
    setup = getattr(klass, 'setUpClass', None)
    try:
        code = inspect.getsource(method) + (inspect.getsource(setup) if setup else '')
    except (TypeError, OSError):
        code = ''
    if 'HTTPServer(' in code or 'ThreadingHTTPServer(' in code:
        selected.append(test)
result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(unittest.TestSuite(selected))
report = {'discovered': len(selected), 'run': result.testsRun, 'skipped': len(result.skipped),
          'passed': sum(c['status'] == 'passed' for c in result.cases),
          'failures': len(result.failures), 'errors': len(result.errors),
          'cases': result.cases, 'all_passed': result.wasSuccessful(),
          'note': 'This formerly skipped group includes configuration-only unit cases; their passing does not prove TCP reachability.'}
path = root / 'artifacts/tcp-strict-report.json'
path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)
sys.exit(not result.wasSuccessful())
