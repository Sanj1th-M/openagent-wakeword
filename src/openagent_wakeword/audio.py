from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from openagent_wakeword.diagnostics import WakeWordDiagnostics
from openagent_wakeword.errors import AudioInputError

AudioFrameCallback = Callable[[np.ndarray], None]


class AudioInput(Protocol):
    """Audio capture port used by WakeWordDetector."""

    def start(self, on_frame: AudioFrameCallback) -> None:
        """Start capture and call on_frame with int16 mono PCM frames."""

    def stop(self) -> None:
        """Stop capture and release audio resources."""


class SoundDeviceAudioInput:
    """Microphone adapter based on sounddevice.

    Imports sounddevice lazily so offline unit tests can run without audio
    hardware or the package imported at module import time.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        device: int | str | None = None,
        diagnostics: WakeWordDiagnostics | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.diagnostics = diagnostics
        self._stream = None
        self._stopped = True

    def start(self, on_frame: AudioFrameCallback) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - depends on environment
            raise AudioInputError("sounddevice is not available") from exc

        def callback(indata, frames, _time, status) -> None:
            if status:
                if self.diagnostics is not None:
                    self.diagnostics.record_callback_status(status)
                raise AudioInputError(f"microphone stream status: {status}")
            mono = indata[:, 0] if getattr(indata, "ndim", 1) > 1 else indata
            on_frame(np.asarray(mono, dtype=np.int16).copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                channels=1,
                dtype="int16",
                device=self.device,
                callback=callback,
            )
            self._stream.start()
            self._stopped = False
        except Exception as exc:
            self._stream = None
            self._stopped = True
            raise AudioInputError("failed to start microphone input") from exc

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._stopped = True
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()
