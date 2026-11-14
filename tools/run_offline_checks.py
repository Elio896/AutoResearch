"""Run existing unit tests that do not need blocked loopback sockets in Work Mode.

This is a selected regression suite, NOT a replacement for full HTTP/browser E2E.
"""
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / 'tests')]
EXCLUDED = {'test_api', 'test_cloud', 'test_demo_flow', 'test_exports', 'test_integrity',
            'test_jev_router', 'test_light_model', 'test_live_demo', 'test_orchestrator',
            'test_research_jobs', 'test_task_store'}
suite = unittest.TestSuite()
for file in sorted((root / 'tests').glob('test_*.py')):
    if file.stem not in EXCLUDED:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(file.stem))
print('Selected offline suite; HTTP/socket-dependent modules excluded:', ', '.join(sorted(EXCLUDED)), flush=True)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not result.wasSuccessful())
