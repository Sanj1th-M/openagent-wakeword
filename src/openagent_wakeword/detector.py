from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from openagent_wakeword.audio import AudioInput, SoundDeviceAudioInput
from openagent_wakeword.config import WakeWordConfig
from openagent_wakeword.diagnostics import WakeWordDiagnostics
from openagent_wakeword.engines import WakeWordEngine
from openagent_wakeword.errors import AudioInputError, InferenceError, ModelLoadError
from openagent_wakeword.events import WakeWordEvent, WakeWordEventType
from openagent_wakeword.openwakeword_engine import OpenWakeWordEngine

WakeWordCallback = Callable[[WakeWordEvent], None]

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Coordinates audio capture, wake-word inference, and event callbacks."""

    def __init__(
        self,
        config: WakeWordConfig | None = None,
        *,
        audio_input: AudioInput | None = None,
        engine: WakeWordEngine | None = None,
        diagnostics: WakeWordDiagnostics | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or WakeWordConfig()
        self.diagnostics = diagnostics
        self._audio = audio_input or SoundDeviceAudioInput(
            sample_rate=self.config.sample_rate,
            frame_samples=self.config.frame_samples,
            device=self.config.device,
            diagnostics=diagnostics,
        )
        self._engine = engine or OpenWakeWordEngine(self.config)
        self._clock = clock
        self._wall_clock = wall_clock
        self._callbacks: list[WakeWordCallback] = []
        self._lock = threading.RLock()
        self._running = False
        self._paused = False
        self._last_detection_at: float | None = None
        self.last_error: Exception | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def add_callback(self, callback: WakeWordCallback) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: WakeWordCallback) -> None:
        with self._lock:
            self._callbacks.remove(callback)

    def start(self) -> None:
        with self._lock:
            if self._running and not self._paused:
                return
            if not self._running:
                try:
                    self._engine.load()
                except Exception as exc:
                    self.last_error = exc
                    raise ModelLoadError("failed to start wake-word detector: model load failed") from exc
                self._running = True
            self._paused = False
            try:
                self._audio.start(self._handle_audio_frame)
            except Exception as exc:
                self.last_error = exc
                self._cleanup_after_start_failure()
                raise AudioInputError("failed to start wake-word detector: microphone unavailable") from exc

    def stop(self) -> None:
        with self._lock:
            if not self._running and not self._paused:
                return
            self._stop_audio()
            self._engine.close()
            self._running = False
            self._paused = False

    def pause(self) -> None:
        with self._lock:
            if not self._running or self._paused:
                return
            self._stop_audio()
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            if not self._running:
                self.start()
                return
            if not self._paused:
                return
            self._paused = False
            try:
                self._audio.start(self._handle_audio_frame)
            except Exception as exc:
                self.last_error = exc
                self._paused = True
                raise AudioInputError("failed to resume wake-word detector") from exc

    def _handle_audio_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            if not self._running or self._paused:
                return
        if self.diagnostics is not None:
            self.diagnostics.record_frame(frame)
        try:
            prediction_started = self._clock()
            predictions = self._engine.predict(frame)
        except Exception as exc:
            wrapped = InferenceError("wake-word inference failed")
            self.last_error = wrapped
            logger.exception("wake-word inference failed")
            self._fail_closed()
            raise wrapped from exc

        score = self._select_score(predictions)
        if self.diagnostics is not None:
            self.diagnostics.record_prediction(score, self._clock() - prediction_started)
        if score < self.config.threshold:
            return
        now = self._clock()
        with self._lock:
            if self._last_detection_at is not None:
                elapsed = now - self._last_detection_at
                if elapsed < self.config.cooldown_seconds:
                    return
            self._last_detection_at = now

        self._emit(
            WakeWordEvent(
                event_type=WakeWordEventType.WAKE_DETECTED,
                phrase=self.config.phrase,
                model_id=self.config.model_id,
                score=score,
                timestamp=self._wall_clock(),
            )
        )

    def _select_score(self, predictions: dict[str, float]) -> float:
        if self.config.prediction_key in predictions:
            return float(predictions[self.config.prediction_key])
        versioned_key = f"{self.config.prediction_key}_v"
        for key, value in predictions.items():
            if key.startswith(versioned_key):
                return float(value)
        if len(predictions) == 1:
            return float(next(iter(predictions.values())))
        return 0.0

    def _emit(self, event: WakeWordEvent) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:
                self.last_error = exc
                logger.exception("wake-word callback failed")

    def _stop_audio(self) -> None:
        try:
            self._audio.stop()
        except Exception as exc:
            self.last_error = exc
            logger.exception("failed to stop microphone input cleanly")

    def _cleanup_after_start_failure(self) -> None:
        self._stop_audio()
        self._engine.close()
        self._running = False
        self._paused = False

    def _fail_closed(self) -> None:
        with self._lock:
            self._stop_audio()
            self._engine.close()
            self._running = False
            self._paused = False
