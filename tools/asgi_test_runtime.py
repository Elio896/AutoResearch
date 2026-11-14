"""Socket-free event-loop harness for in-process ASGI tests only.

Uses a thread event for cross-thread callbacks. It cannot register sockets and
must never be used as a production server loop. No network or system settings
are changed. Real TCP/browser and provider tests remain separate.
"""
import asyncio
import selectors
import threading
from contextlib import contextmanager
from unittest.mock import patch


class InProcessSelector(selectors.BaseSelector):
    def __init__(self):
        self.wake = threading.Event()

    def register(self, fileobj, events, data=None):
        raise RuntimeError('In-process ASGI test loop cannot register network descriptors.')

    def unregister(self, fileobj):
        raise KeyError(fileobj)

    def get_map(self):
        return {}

    def select(self, timeout=None):
        self.wake.wait(timeout)
        self.wake.clear()
        return []

    def close(self):
        self.wake.set()


class InProcessLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selector=InProcessSelector())

    def _make_self_pipe(self):
        pass

    def _close_self_pipe(self):
        pass

    def _write_to_self(self):
        self._selector.wake.set()


@contextmanager
def asgi_test_runtime():
    with patch('asyncio.events.new_event_loop', side_effect=InProcessLoop):
        yield
