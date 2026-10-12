from pathlib import Path

import pytest

from openagent_wakeword import WakeWordConfig


def test_default_config_targets_hey_jarvis() -> None:
    config = WakeWordConfig()

    assert config.phrase == "Hey Jarvis"
    assert config.model_id == "hey_jarvis"
    assert config.prediction_key == "hey_jarvis"
    assert config.sample_rate == 16000
    assert config.frame_samples == 1280


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phrase", ""),
        ("model_id", ""),
        ("prediction_key", ""),
        ("threshold", 0),
        ("threshold", 1.1),
        ("cooldown_seconds", -1),
        ("sample_rate", 48000),
        ("frame_samples", 1000),
    ],
)
def test_invalid_config_rejected(field: str, value: object) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError):
        WakeWordConfig(**kwargs)


def test_model_dir_string_is_normalized_to_path() -> None:
    config = WakeWordConfig(model_dir=".models")

    assert config.model_dir == Path(".models")
