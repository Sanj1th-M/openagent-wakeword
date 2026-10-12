import logging

import pytest

from openagent_wakeword import WakeWordConfig
from openagent_wakeword.detector import WakeWordDetector
from openagent_wakeword.errors import InferenceError

from tests.fakes import FakeAudioInput, FakeEngine


def test_callback_exception_is_logged_and_detector_continues(caplog: pytest.LogCaptureFixture) -> None:
    audio = FakeAudioInput()
    good_events = []
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.5, cooldown_seconds=0),
        audio_input=audio,
        engine=FakeEngine([{"hey_jarvis": 0.9}, {"hey_jarvis": 0.91}]),
    )

    def bad_callback(_event) -> None:
        raise RuntimeError("callback failed")

    detector.add_callback(bad_callback)
    detector.add_callback(good_events.append)

    with caplog.at_level(logging.ERROR):
        detector.start()
        audio.push()
        audio.push()

    assert len(good_events) == 2
    assert "wake-word callback failed" in caplog.text


def test_inference_failure_fails_closed_and_releases_audio() -> None:
    audio = FakeAudioInput()
    engine = FakeEngine(fail_predict=True)
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=engine)

    detector.start()
    with pytest.raises(InferenceError):
        audio.push()

    assert not detector.is_running
    assert not audio.started
    assert engine.closed
