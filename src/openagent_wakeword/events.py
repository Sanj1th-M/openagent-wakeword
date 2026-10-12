from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WakeWordEventType(str, Enum):
    WAKE_DETECTED = "WAKE_DETECTED"


@dataclass(frozen=True)
class WakeWordEvent:
    event_type: WakeWordEventType
    phrase: str
    model_id: str
    score: float
    timestamp: float
