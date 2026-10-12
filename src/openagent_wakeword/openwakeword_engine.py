from __future__ import annotations

from pathlib import Path

import numpy as np

from openagent_wakeword.config import WakeWordConfig
from openagent_wakeword.errors import ModelLoadError
from openagent_wakeword.model_assets import required_model_paths, validate_model_assets


class OpenWakeWordEngine:
    """ONNX openWakeWord adapter for the supported `hey_jarvis` model."""

    def __init__(self, config: WakeWordConfig) -> None:
        if config.model_dir is None:
            raise ModelLoadError("model_dir is required for OpenWakeWordEngine")
        self.config = config
        self.model_dir = Path(config.model_dir)
        self._model = None

    def load(self) -> None:
        try:
            validate_model_assets(self.model_dir)
            from openwakeword.model import Model

            paths = required_model_paths(self.model_dir)
            self._model = Model(
                wakeword_models=[str(paths["hey_jarvis_v0.1.onnx"])],
                inference_framework="onnx",
                melspec_model_path=str(paths["melspectrogram.onnx"]),
                embedding_model_path=str(paths["embedding_model.onnx"]),
            )
        except Exception as exc:
            raise ModelLoadError("failed to load openWakeWord ONNX model assets") from exc

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        if self._model is None:
            raise ModelLoadError("openWakeWord model is not loaded")
        predictions = self._model.predict(frame)
        return {str(key): float(value) for key, value in predictions.items()}

    def close(self) -> None:
        self._model = None
