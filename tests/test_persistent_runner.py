import logging
import threading

import pytest

from openagent_wakeword.config import WakeWordConfig
from openagent_wakeword.detector import WakeWordDetector
from openagent_wakeword.errors import AudioInputError, InferenceError, ModelLoadError
from openagent_wakeword.events import WakeWordEventType
from openagent_wakeword.runner import PersistentListener, configure_logging, default_log_dir
from tests.fakes import FakeAudioInput, FakeEngine


def test_logging_uses_bounded_local_file_and_does_not_depend_on_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "app data"))
    assert default_log_dir() == tmp_path / "app data" / "OpenAgent" / "WakeWord" / "logs"

    log_path = configure_logging(tmp_path / "logs")
    logging.getLogger("openagent_wakeword.test").info("startup diagnostic")

    assert log_path.name == "listener.log"
    assert log_path.read_text(encoding="utf-8").endswith("startup diagnostic\n")


def test_runtime_audio_error_is_delivered_and_listener_releases_audio() -> None:
    audio = FakeAudioInput()
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine())
    errors = []
    detector.add_audio_error_callback(errors.append)
    detector.start()
    audio.fail_runtime()
    assert len(errors) == 1
    assert isinstance(errors[0], AudioInputError)
    detector.stop()
    assert not audio.started


def test_listener_retries_runtime_failure_with_backoff_and_stops_cleanly() -> None:
    audio = FakeAudioInput()
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine())
    delays = []

    def wait(delay: float) -> bool:
        delays.append(delay)
        return False

    listener = PersistentListener(detector, initial_backoff=0.1, wait=wait)
    thread = threading.Thread(target=listener.run)
    thread.start()
    deadline = threading.Event()
    for _ in range(100):
        if audio.started:
            break
        deadline.wait(0.01)
    audio.fail_runtime()
    for _ in range(100):
        if audio.started_count >= 2:
            break
        deadline.wait(0.01)
    listener.request_stop()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert delays == [0.1]
    assert audio.stopped_count >= 1
    assert not audio.started


def test_model_failure_is_terminal_and_not_retried() -> None:
    detector = WakeWordDetector(WakeWordConfig(), audio_input=FakeAudioInput(), engine=FakeEngine(fail_load=True))
    listener = PersistentListener(detector, wait=lambda _delay: pytest.fail("must not retry"))
    with pytest.raises(ModelLoadError):
        listener.run()


def test_audio_retries_are_bounded_and_backoff_is_capped() -> None:
    audio = FakeAudioInput(fail_start=True)
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine())
    delays = []
    listener = PersistentListener(
        detector, max_retries=3, initial_backoff=1, max_backoff=2,
        wait=lambda delay: (delays.append(delay) or False),
    )
    with pytest.raises(AudioInputError, match="retries exhausted"):
        listener.run()
    assert delays == [1, 2, 2]
    assert audio.stopped_count >= 4


def test_default_listener_keeps_recovering_after_more_than_five_temporary_errors() -> None:
    class TemporarilyUnavailableAudio(FakeAudioInput):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def start(self, on_frame) -> None:
            self.attempts += 1
            if self.attempts <= 7:
                raise RuntimeError("temporary device startup failure")
            super().start(on_frame)

    audio = TemporarilyUnavailableAudio()
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine())
    delays = []
    listener = PersistentListener(detector, wait=lambda delay: delays.append(delay) or False)
    thread = threading.Thread(target=listener.run)
    thread.start()
    for _ in range(200):
        if audio.started:
            break
        threading.Event().wait(0.01)
    assert audio.started
    assert audio.attempts == 8
    assert delays == [1, 2, 4, 8, 16, 30, 30]
    listener.request_stop()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not audio.started


def test_wake_events_do_not_end_persistent_listening() -> None:
    audio = FakeAudioInput()
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.5, cooldown_seconds=0),
        audio_input=audio,
        engine=FakeEngine([{"hey_jarvis": 0.9}]),
    )
    listener = PersistentListener(detector)
    events = []
    detector.add_callback(lambda event: events.append(event))
    thread = threading.Thread(target=listener.run)
    thread.start()
    for _ in range(100):
        if audio.started:
            break
        threading.Event().wait(0.01)
    audio.push()
    audio.push()
    assert len(events) == 2
    assert all(event.event_type is WakeWordEventType.WAKE_DETECTED for event in events)
    assert thread.is_alive()  # No diagnostic timeout ends the listener.
    listener.request_stop()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not audio.started


def test_inference_failure_stays_fail_closed_and_is_not_retried() -> None:
    audio = FakeAudioInput()
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine(fail_predict=True))
    delays = []
    listener = PersistentListener(detector, wait=lambda delay: delays.append(delay) or False)
    failures = []

    def target():
        try:
            listener.run()
        except Exception as exc:
            failures.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    for _ in range(100):
        if audio.started:
            break
        threading.Event().wait(0.01)
    with pytest.raises(InferenceError):
        audio.push()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], InferenceError)
    assert not delays
    assert not audio.started


def test_cleanup_failure_prevents_retry() -> None:
    audio = FakeAudioInput(fail_stop=True)
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine())
    delays = []
    listener = PersistentListener(detector, wait=lambda delay: delays.append(delay) or False)
    detector.start()
    audio.fail_runtime()
    with pytest.raises(AudioInputError, match="refusing to retry"):
        listener._recover(0)
    assert not delays
