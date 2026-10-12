import pytest

from openagent_wakeword import WakeWordConfig
from openagent_wakeword.detector import WakeWordDetector
from openagent_wakeword.errors import AudioInputError, ModelLoadError

from tests.fakes import FakeAudioInput, FakeEngine


def test_start_stop_are_idempotent_and_cleanup_resources() -> None:
    audio = FakeAudioInput()
    engine = FakeEngine()
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=engine)

    detector.start()
    detector.start()
    assert detector.is_running
    assert not detector.is_paused
    assert engine.loaded
    assert audio.started

    detector.stop()
    detector.stop()
    assert not detector.is_running
    assert not detector.is_paused
    assert not audio.started
    assert engine.closed
    assert audio.stopped_count == 1


def test_pause_releases_audio_and_resume_restarts_it() -> None:
    audio = FakeAudioInput()
    detector = WakeWordDetector(WakeWordConfig(), audio_input=audio, engine=FakeEngine())

    detector.start()
    detector.pause()
    detector.pause()
    assert detector.is_running
    assert detector.is_paused
    assert not audio.started
    assert audio.stopped_count == 1

    detector.resume()
    detector.resume()
    assert detector.is_running
    assert not detector.is_paused
    assert audio.started


def test_model_load_failure_cleaves_detector_stopped() -> None:
    audio = FakeAudioInput()
    detector = WakeWordDetector(
        WakeWordConfig(),
        audio_input=audio,
        engine=FakeEngine(fail_load=True),
    )

    with pytest.raises(ModelLoadError):
        detector.start()

    assert not detector.is_running
    assert not audio.started


def test_microphone_start_failure_cleans_up_model() -> None:
    engine = FakeEngine()
    detector = WakeWordDetector(
        WakeWordConfig(),
        audio_input=FakeAudioInput(fail_start=True),
        engine=engine,
    )

    with pytest.raises(AudioInputError):
        detector.start()

    assert not detector.is_running
    assert engine.closed
