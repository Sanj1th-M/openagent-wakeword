from __future__ import annotations

from typing import Protocol

import numpy as np


class WakeWordEngine(Protocol):
    """Inference port used by WakeWordDetector."""

    def load(self) -> None:
        """Load any required model resources."""

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        """Return prediction scores by model/prediction key."""

    def close(self) -> None:
        """Release engine resources."""
