import numpy as np

from openagent_wakeword import WakeWordConfig, WakeWordEventType
from openagent_wakeword.detector import WakeWordDetector

from tests.fakes import FakeAudioInput, FakeEngine


class ManualClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_wake_detected_event_has_no_audio_payload() -> None:
    audio = FakeAudioInput()
    events = []
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.5),
        audio_input=audio,
        engine=FakeEngine([{"hey_jarvis": 0.9}]),
        wall_clock=lambda: 1234.5,
    )
    detector.add_callback(events.append)

    detector.start()
    audio.push()

    assert len(events) == 1
    event = events[0]
    assert event.event_type == WakeWordEventType.WAKE_DETECTED
    assert event.phrase == "Hey Jarvis"
    assert event.model_id == "hey_jarvis"
    assert event.score == 0.9
    assert event.timestamp == 1234.5
    assert not hasattr(event, "audio")


def test_below_threshold_does_not_emit() -> None:
    audio = FakeAudioInput()
    events = []
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.5),
        audio_input=audio,
        engine=FakeEngine([{"hey_jarvis": 0.49}]),
    )
    detector.add_callback(events.append)

    detector.start()
    audio.push()

    assert events == []


def test_cooldown_suppresses_repeated_detection() -> None:
    clock = ManualClock()
    audio = FakeAudioInput()
    events = []
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.5, cooldown_seconds=2.0),
        audio_input=audio,
        engine=FakeEngine([
            {"hey_jarvis": 0.9},
            {"hey_jarvis": 0.95},
            {"hey_jarvis": 0.97},
        ]),
        clock=clock,
    )
    detector.add_callback(events.append)

    detector.start()
    audio.push(np.zeros(1280, dtype=np.int16))
    clock.now += 1.0
    audio.push(np.zeros(1280, dtype=np.int16))
    clock.now += 1.1
    audio.push(np.zeros(1280, dtype=np.int16))

    assert len(events) == 2


def test_versioned_prediction_key_is_accepted() -> None:
    audio = FakeAudioInput()
    events = []
    detector = WakeWordDetector(
        WakeWordConfig(threshold=0.5, prediction_key="hey_jarvis"),
        audio_input=audio,
        engine=FakeEngine([{"hey_jarvis_v0.1": 0.9}]),
    )
    detector.add_callback(events.append)

    detector.start()
    audio.push()

    assert len(events) == 1
