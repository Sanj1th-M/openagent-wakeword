import threading
import time
import types

import numpy as np

from openagent_wakeword.audio import SoundDeviceAudioInput


def test_stream_status_is_not_raised_in_portaudio_callback_and_is_forwarded(monkeypatch) -> None:
    callback_box = {}
    closed = []
    notification = threading.Event()
    callback_threads = []

    class Stream:
        def __init__(self, **kwargs):
            callback_box["callback"] = kwargs["callback"]
            callback_box["finished"] = kwargs["finished_callback"]

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", types.SimpleNamespace(InputStream=Stream))
    audio = SoundDeviceAudioInput()

    def on_error(error):
        callback_threads.append(threading.current_thread())
        notification.set()

    audio.set_error_callback(on_error)
    audio.start(lambda _frame: None)
    callback_box["callback"](np.zeros((1280, 1), dtype=np.int16), 1280, None, "input overflow")
    assert notification.wait(1)
    assert callback_threads[0] is not threading.current_thread()
    audio.stop()
    assert closed == [True]


def test_unexpected_finished_callback_is_reported(monkeypatch) -> None:
    callback_box = {}
    notification = threading.Event()

    class Stream:
        def __init__(self, **kwargs):
            callback_box["finished"] = kwargs["finished_callback"]

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", types.SimpleNamespace(InputStream=Stream))
    audio = SoundDeviceAudioInput()
    audio.set_error_callback(lambda _error: notification.set())
    audio.start(lambda _frame: None)
    callback_box["finished"]()
    assert notification.wait(1)
    audio.stop()


def test_frame_callback_exception_does_not_escape_audio_callback(monkeypatch) -> None:
    callback_box = {}

    class Stream:
        def __init__(self, **kwargs):
            callback_box["callback"] = kwargs["callback"]

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", types.SimpleNamespace(InputStream=Stream))
    audio = SoundDeviceAudioInput()
    audio.start(lambda _frame: (_ for _ in ()).throw(RuntimeError("inference failure")))
    callback_box["callback"](np.zeros((1280, 1), dtype=np.int16), 1280, None, None)
    audio.stop()


def test_failed_stream_start_closes_partially_created_stream(monkeypatch) -> None:
    state = []

    class Stream:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("start failed")

        def stop(self):
            state.append("stop")

        def close(self):
            state.append("close")

    monkeypatch.setitem(__import__("sys").modules, "sounddevice", types.SimpleNamespace(InputStream=Stream))
    from openagent_wakeword.errors import AudioInputError
    import pytest

    with pytest.raises(AudioInputError):
        SoundDeviceAudioInput().start(lambda _frame: None)
    assert state == ["stop", "close"]
