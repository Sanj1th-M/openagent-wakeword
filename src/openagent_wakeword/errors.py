class WakeWordError(Exception):
    """Base exception for wake-word component failures."""


class ConfigurationError(WakeWordError):
    """Raised when configuration is invalid or incomplete."""


class ModelAssetError(WakeWordError):
    """Raised when model assets are missing or fail validation."""


class ModelLoadError(WakeWordError):
    """Raised when the wake-word engine cannot load."""


class AudioInputError(WakeWordError):
    """Raised when microphone capture cannot start or continue."""


class InferenceError(WakeWordError):
    """Raised when model inference fails."""


class DetectorStateError(WakeWordError):
    """Raised when detector state becomes inconsistent."""
