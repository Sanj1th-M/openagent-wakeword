import uuid
import threading
from types import SimpleNamespace

import pytest

from openagent_wakeword.instance_guard import InstanceAlreadyRunning, WindowsInstanceGuard


class _FakeWinApiFunction:
    def __init__(self, result=None, *, action=None):
        self.result = result
        self.action = action
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        if self.action is not None:
            return self.action(*args)
        return self.result


def _fake_kernel32(wait_result: int):
    closed = []
    released = []
    kernel32 = SimpleNamespace(
        CreateMutexW=_FakeWinApiFunction(123),
        WaitForSingleObject=_FakeWinApiFunction(wait_result),
        CloseHandle=_FakeWinApiFunction(action=lambda handle: closed.append(handle) or 1),
        ReleaseMutex=_FakeWinApiFunction(action=lambda handle: released.append(handle) or 1),
    )
    return kernel32, closed, released


@pytest.mark.parametrize("wait_result", [0, 0x80], ids=["unowned", "abandoned"])
def test_guard_acquires_unowned_or_abandoned_existing_mutex(monkeypatch, wait_result) -> None:
    import openagent_wakeword.instance_guard as module

    kernel32, closed, released = _fake_kernel32(wait_result)
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(WindowsInstanceGuard, "_user_sid", staticmethod(lambda: "S-1-5-21-test"))
    monkeypatch.setattr(module.ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32)
    guard = WindowsInstanceGuard("test")

    guard.acquire()
    assert guard._handle == 123
    assert not closed
    guard.release()
    assert released == [123]
    assert closed == [123]


def test_guard_rejects_currently_owned_mutex(monkeypatch) -> None:
    import openagent_wakeword.instance_guard as module

    kernel32, closed, _released = _fake_kernel32(0x102)
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(WindowsInstanceGuard, "_user_sid", staticmethod(lambda: "S-1-5-21-test"))
    monkeypatch.setattr(module.ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32)

    with pytest.raises(InstanceAlreadyRunning):
        WindowsInstanceGuard("test").acquire()
    assert closed == [123]


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows mutex integration")
def test_named_mutex_rejects_duplicate_and_allows_relaunch_after_release() -> None:
    name = "OpenAgentWakeWordTest-" + uuid.uuid4().hex
    first = WindowsInstanceGuard(name)
    second = WindowsInstanceGuard(name)
    first.acquire()
    try:
        outcomes = []

        def acquire_from_another_thread():
            try:
                second.acquire()
            except Exception as exc:
                outcomes.append(exc)
            else:
                outcomes.append(None)
                second.release()

        thread = threading.Thread(target=acquire_from_another_thread)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(outcomes) == 1 and isinstance(outcomes[0], InstanceAlreadyRunning)
    finally:
        first.release()
    second.acquire()
    second.release()
