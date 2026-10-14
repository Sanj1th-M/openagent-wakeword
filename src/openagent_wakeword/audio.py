from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import queue
import threading

from openagent_wakeword.diagnostics import WakeWordDiagnostics
from openagent_wakeword.errors import AudioInputError

AudioFrameCallback = Callable[[np.ndarray], None]
AudioErrorCallback = Callable[[AudioInputError], None]


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
        self._error_callback: AudioErrorCallback | None = None
        self._error_queue: queue.Queue[AudioInputError | None] | None = None
        self._error_thread: threading.Thread | None = None
        self._callback_thread_id: int | None = None

    def set_error_callback(self, callback: AudioErrorCallback | None) -> None:
        """Set an optional callback invoked from a worker, never PortAudio's thread."""
        self._error_callback = callback

    def _notify_error(self, error: AudioInputError) -> None:
        work_queue = self._error_queue
        if work_queue is not None:
            try:
                work_queue.put_nowait(error)
            except queue.Full:  # One notification per stream; duplicate failures are coalesced.
                pass

    def start(self, on_frame: AudioFrameCallback) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - depends on environment
            raise AudioInputError("sounddevice is not available") from exc

        self._error_queue = queue.Queue(maxsize=1)
        if self._error_callback is not None:
            self._error_thread = threading.Thread(target=self._dispatch_errors, daemon=True)
            self._error_thread.start()

        def callback(indata, frames, _time, status) -> None:
            self._callback_thread_id = threading.get_ident()
            if status:
                if self.diagnostics is not None:
                    self.diagnostics.record_callback_status(status)
                self._notify_error(AudioInputError(f"microphone stream reported an audio error: {status}"))
                return
            try:
                mono = indata[:, 0] if getattr(indata, "ndim", 1) > 1 else indata
                on_frame(np.asarray(mono, dtype=np.int16).copy())
            except Exception:
                # Detector failures are fail-closed internally. Never propagate into PortAudio.
                return

        def finished_callback() -> None:
            if not self._stopped:
                self._notify_error(AudioInputError("microphone stream stopped unexpectedly"))

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                channels=1,
                dtype="int16",
                device=self.device,
                callback=callback,
                finished_callback=finished_callback,
            )
            self._stopped = False
            self._stream.start()
        except Exception as exc:
            stream = self._stream
            self._stream = None
            self._stopped = True
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            self._finish_error_dispatch()
            raise AudioInputError("failed to start microphone input") from exc

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._stopped = True
        if stream is None:
            self._finish_error_dispatch()
            return
        if threading.get_ident() == self._callback_thread_id:
            # PortAudio APIs must not be called by its stream callback thread.
            threading.Thread(target=self._close_stream, args=(stream,), daemon=True).start()
            return
        self._close_stream(stream)

    def _close_stream(self, stream) -> None:
        try:
            stream.stop()
        finally:
            try:
                stream.close()
            finally:
                self._finish_error_dispatch()

    def _dispatch_errors(self) -> None:
        work_queue = self._error_queue
        if work_queue is None:
            return
        while True:
            error = work_queue.get()
            if error is None:
                return
            callback = self._error_callback
            if callback is not None:
                try:
                    callback(error)
                except Exception:
                    # A user supplied error observer must not kill the audio worker.
                    continue

    def _finish_error_dispatch(self) -> None:
        work_queue, thread = self._error_queue, self._error_thread
        self._error_queue = None
        self._error_thread = None
        if work_queue is not None and thread is not None:
            try:
                work_queue.put_nowait(None)
            except queue.Full:
                # Drain one queued notification before asking the worker to exit.
                try:
                    work_queue.get_nowait()
                except queue.Empty:
                    pass
                work_queue.put_nowait(None)
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
