"""Run all unit/ASGI/TCP tests by default. --offline is an explicit opt-out."""
import argparse
import inspect
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / 'tests'), str(root / 'tools')]
from asgi_test_runtime import asgi_test_runtime


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


parser = argparse.ArgumentParser()
parser.add_argument('--offline', action='store_true', help='Explicitly skip real TCP fixtures; never a complete verification.')
args = parser.parse_args()
suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
tests = list(flatten(suite))
for test in tests:
    klass = type(test)
    method = getattr(klass, test._testMethodName)
    setup = getattr(klass, 'setUpClass', None)
    try:
        code = inspect.getsource(method) + (inspect.getsource(setup) if setup else '')
    except (TypeError, OSError):
        code = ''
    if args.offline and ('HTTPServer(' in code or 'ThreadingHTTPServer(' in code):
        setattr(klass, test._testMethodName, unittest.skip('Requires real TCP provider fixture, blocked by execution environment.')(method))

with asgi_test_runtime():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
sys.exit(not result.wasSuccessful())
