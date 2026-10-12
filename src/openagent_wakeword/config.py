from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WakeWordConfig:
    """Validated runtime configuration for the wake-word detector."""

    phrase: str = "Hey Jarvis"
    model_id: str = "hey_jarvis"
    prediction_key: str = "hey_jarvis"
    threshold: float = 0.5
    cooldown_seconds: float = 2.0
    sample_rate: int = 16000
    frame_samples: int = 1280
    model_dir: Path | None = None
    device: int | str | None = None

    def __post_init__(self) -> None:
        phrase = self.phrase.strip()
        model_id = self.model_id.strip()
        prediction_key = self.prediction_key.strip()

        if not phrase:
            raise ValueError("phrase must not be empty")
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not prediction_key:
            raise ValueError("prediction_key must not be empty")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be > 0 and <= 1")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if self.sample_rate != 16000:
            raise ValueError("sample_rate must be 16000 for openWakeWord")
        if self.frame_samples <= 0:
            raise ValueError("frame_samples must be > 0")
        if self.frame_samples % 1280 != 0:
            raise ValueError("frame_samples must be a multiple of 1280 samples")
        if self.model_dir is not None and not isinstance(self.model_dir, Path):
            object.__setattr__(self, "model_dir", Path(self.model_dir))

        object.__setattr__(self, "phrase", phrase)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "prediction_key", prediction_key)
