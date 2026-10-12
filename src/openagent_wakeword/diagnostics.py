from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DiagnosticSummary:
    frame_count: int
    prediction_count: int
    max_score: float | None
    frame_lengths: dict[int, int]
    callback_status_count: int
    callback_status_messages: tuple[str, ...]
    prediction_latency_ms_min: float | None
    prediction_latency_ms_avg: float | None
    prediction_latency_ms_max: float | None
    rms_min: float | None
    rms_avg: float | None
    rms_max: float | None
    peak_max: int
    clipping_frames: int
    near_silence_frames: int


@dataclass
class WakeWordDiagnostics:
    """Bounded local diagnostics that never retain raw audio samples."""

    near_silence_rms: float = 200.0
    clipping_threshold: int = 32760
    max_status_messages: int = 10
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _frame_count: int = field(default=0, init=False)
    _prediction_count: int = field(default=0, init=False)
    _max_score: float | None = field(default=None, init=False)
    _frame_lengths: dict[int, int] = field(default_factory=dict, init=False)
    _callback_status_count: int = field(default=0, init=False)
    _callback_status_messages: list[str] = field(default_factory=list, init=False)
    _latency_count: int = field(default=0, init=False)
    _latency_sum_ms: float = field(default=0.0, init=False)
    _latency_min_ms: float | None = field(default=None, init=False)
    _latency_max_ms: float | None = field(default=None, init=False)
    _rms_count: int = field(default=0, init=False)
    _rms_sum: float = field(default=0.0, init=False)
    _rms_min: float | None = field(default=None, init=False)
    _rms_max: float | None = field(default=None, init=False)
    _peak_max: int = field(default=0, init=False)
    _clipping_frames: int = field(default=0, init=False)
    _near_silence_frames: int = field(default=0, init=False)

    def record_frame(self, frame: np.ndarray) -> None:
        samples = np.asarray(frame, dtype=np.int16)
        length = int(samples.size)
        if length == 0:
            rms = 0.0
            peak = 0
        else:
            as_float = samples.astype(np.float64)
            rms = float(math.sqrt(float(np.mean(as_float * as_float))))
            peak = int(np.max(np.abs(as_float)))

        with self._lock:
            self._frame_count += 1
            self._frame_lengths[length] = self._frame_lengths.get(length, 0) + 1
            self._rms_count += 1
            self._rms_sum += rms
            self._rms_min = rms if self._rms_min is None else min(self._rms_min, rms)
            self._rms_max = rms if self._rms_max is None else max(self._rms_max, rms)
            self._peak_max = max(self._peak_max, peak)
            if peak >= self.clipping_threshold:
                self._clipping_frames += 1
            if rms <= self.near_silence_rms:
                self._near_silence_frames += 1

    def record_prediction(self, score: float, latency_seconds: float) -> None:
        latency_ms = latency_seconds * 1000.0
        with self._lock:
            self._prediction_count += 1
            self._max_score = score if self._max_score is None else max(self._max_score, score)
            self._latency_count += 1
            self._latency_sum_ms += latency_ms
            self._latency_min_ms = (
                latency_ms if self._latency_min_ms is None else min(self._latency_min_ms, latency_ms)
            )
            self._latency_max_ms = (
                latency_ms if self._latency_max_ms is None else max(self._latency_max_ms, latency_ms)
            )

    def record_callback_status(self, status: object) -> None:
        message = str(status)
        with self._lock:
            self._callback_status_count += 1
            if len(self._callback_status_messages) < self.max_status_messages:
                self._callback_status_messages.append(message)

    def summary(self) -> DiagnosticSummary:
        with self._lock:
            latency_avg = (
                self._latency_sum_ms / self._latency_count if self._latency_count else None
            )
            rms_avg = self._rms_sum / self._rms_count if self._rms_count else None
            return DiagnosticSummary(
                frame_count=self._frame_count,
                prediction_count=self._prediction_count,
                max_score=self._max_score,
                frame_lengths=dict(sorted(self._frame_lengths.items())),
                callback_status_count=self._callback_status_count,
                callback_status_messages=tuple(self._callback_status_messages),
                prediction_latency_ms_min=self._latency_min_ms,
                prediction_latency_ms_avg=latency_avg,
                prediction_latency_ms_max=self._latency_max_ms,
                rms_min=self._rms_min,
                rms_avg=rms_avg,
                rms_max=self._rms_max,
                peak_max=self._peak_max,
                clipping_frames=self._clipping_frames,
                near_silence_frames=self._near_silence_frames,
            )
