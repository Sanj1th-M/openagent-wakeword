from __future__ import annotations

import ctypes
import hashlib
import subprocess
import sys
from ctypes import wintypes


class InstanceAlreadyRunning(RuntimeError):
    """Another listener process already owns the per-user mutex."""


class WindowsInstanceGuard:
    """A global-namespace mutex keyed by the current Windows user SID."""

    def __init__(self, name: str = "OpenAgentWakeWord") -> None:
        self.name = name
        self._handle = None

    @staticmethod
    def _user_sid() -> str:
        result = subprocess.run(
            ["whoami.exe", "/user", "/fo", "csv", "/nh"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        # The final CSV column is the SID; avoid depending on localized headers.
        import csv
        import io
        row = next(csv.reader(io.StringIO(result.stdout)))
        sid = row[-1].strip()
        if not sid.startswith("S-"):
            raise RuntimeError("could not determine current Windows user SID")
        return sid

    def acquire(self) -> None:
        if self._handle is not None:
            return
        if sys.platform != "win32":
            raise OSError("Windows single-instance guard is available only on Windows")
        sid_hash = hashlib.sha256(self._user_sid().encode("ascii")).hexdigest()[:24]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        handle = kernel32.CreateMutexW(None, False, f"Global\\{self.name}-{sid_hash}")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result == 0x102:  # WAIT_TIMEOUT: another process currently owns it.
            kernel32.CloseHandle(handle)
            raise InstanceAlreadyRunning("wake-word listener is already running for this user")
        if wait_result not in (0, 0x80):  # WAIT_OBJECT_0 or WAIT_ABANDONED.
            kernel32.CloseHandle(handle)
            if wait_result == 0xFFFFFFFF:  # WAIT_FAILED
                raise ctypes.WinError(ctypes.get_last_error())
            raise OSError(f"unexpected Windows mutex wait result: {wait_result}")
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.ReleaseMutex(self._handle)
        kernel32.CloseHandle(self._handle)
        self._handle = None

    def __enter__(self) -> "WindowsInstanceGuard":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
