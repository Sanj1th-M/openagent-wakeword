"""Local wake-word detector prototype for openagent."""

from openagent_wakeword.config import WakeWordConfig
from openagent_wakeword.detector import WakeWordDetector
from openagent_wakeword.events import WakeWordEvent, WakeWordEventType

__all__ = [
    "WakeWordConfig",
    "WakeWordDetector",
    "WakeWordEvent",
    "WakeWordEventType",
]
