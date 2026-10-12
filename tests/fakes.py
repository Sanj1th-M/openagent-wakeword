from __future__ import annotations

from collections.abc import Callable

import numpy as np


class FakeAudioInput:
    def __init__(self, *, fail_start: bool = False, fail_stop: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.started = False
        self.stopped_count = 0
        self.callback: Callable[[np.ndarray], None] | None = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        if self.fail_start:
            raise RuntimeError("fake microphone unavailable")
        self.started = True
        self.callback = on_frame

    def stop(self) -> None:
        self.stopped_count += 1
        self.started = False
        self.callback = None
        if self.fail_stop:
            raise RuntimeError("fake stop failed")

    def push(self, frame: np.ndarray | None = None) -> None:
        if self.callback is None:
            raise RuntimeError("fake audio not started")
        self.callback(frame if frame is not None else np.zeros(1280, dtype=np.int16))


class FakeEngine:
    def __init__(
        self,
        predictions: list[dict[str, float]] | None = None,
        *,
        fail_load: bool = False,
        fail_predict: bool = False,
    ) -> None:
        self.predictions = predictions or [{"hey_jarvis": 0.0}]
        self.fail_load = fail_load
        self.fail_predict = fail_predict
        self.loaded = False
        self.closed = False
        self.predict_count = 0

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("fake model load failed")
        self.loaded = True
        self.closed = False

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        if self.fail_predict:
            raise RuntimeError("fake inference failed")
        prediction = self.predictions[min(self.predict_count, len(self.predictions) - 1)]
        self.predict_count += 1
        return prediction

    def close(self) -> None:
        self.closed = True
