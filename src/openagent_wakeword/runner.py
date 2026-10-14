from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from openagent_wakeword import WakeWordConfig
from openagent_wakeword.detector import WakeWordDetector
from openagent_wakeword.errors import AudioInputError, InferenceError, ModelLoadError, WakeWordError
from openagent_wakeword.instance_guard import InstanceAlreadyRunning, WindowsInstanceGuard

logger = logging.getLogger(__name__)
LOG_FILENAME = "listener.log"
LOG_MAX_BYTES = 1_048_576
LOG_BACKUP_COUNT = 3


def default_log_dir() -> Path:
    """Return the per-user local log directory without relying on the CWD."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / ".openagent"
    return base / "OpenAgent" / "WakeWord" / "logs"


def configure_logging(log_dir: Path) -> Path:
    """Configure bounded local operational logs for console and task launches."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        handler._openagent_wakeword_handler = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        if getattr(handler, "_openagent_wakeword_handler", False):
            root.removeHandler(handler)
            handler.close()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return log_path


class PersistentListener:
    """Keeps the existing detector alive and serializes bounded audio recovery."""

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        max_retries: int | None = None,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        if (max_retries is not None and max_retries < 0) or initial_backoff <= 0 or max_backoff < initial_backoff:
            raise ValueError("invalid audio recovery policy")
        self.detector = detector
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self._stop = threading.Event()
        self._audio_error = threading.Event()
        self._wait = wait or self._stop.wait
        self._recovery_lock = threading.Lock()
        self.detector.add_audio_error_callback(self._on_audio_error)

    def _on_audio_error(self, error: AudioInputError) -> None:
        # This observer runs on the adapter's worker, and only signals the runner.
        logger.warning("microphone stream reported an error: %s", error)
        self._audio_error.set()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        retries = 0
        while not self._stop.is_set():
            try:
                self._audio_error.clear()
                logger.info("starting wake-word detector")
                self.detector.start()
                logger.info("wake-word detector is listening")
            except AudioInputError as exc:
                logger.warning("microphone failed to start: %s", exc)
                if not self._recover(retries):
                    raise AudioInputError("microphone recovery retries exhausted; check device and permissions")
                retries += 1
                continue
            except ModelLoadError:
                raise

            while not self._stop.is_set():
                if self._audio_error.wait(0.1):
                    break
                if not self.detector.is_running:
                    error = self.detector.last_error
                    if isinstance(error, InferenceError):
                        raise error
                    raise WakeWordError("detector stopped unexpectedly")
            if self._stop.is_set():
                break
            self.detector.stop()
            if not self._recover(retries):
                raise AudioInputError("microphone recovery retries exhausted; check device and permissions")
            retries += 1
        self.detector.stop()
        logger.info("wake-word detector stopped")

    def _recover(self, retries: int) -> bool:
        with self._recovery_lock:
            self.detector.stop()
            if self.detector.audio_cleanup_failed:
                raise AudioInputError("microphone resources could not be released; refusing to retry")
            if (self.max_retries is not None and retries >= self.max_retries) or self._stop.is_set():
                return False
            delay = min(self.initial_backoff * (2 ** min(retries, 30)), self.max_backoff)
            retry_limit = "without limit" if self.max_retries is None else str(self.max_retries)
            logger.warning("microphone unavailable; retry %d/%s in %.1f seconds", retries + 1, retry_limit, delay)
            return not self._wait(delay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local persistent wake-word listener.")
    parser.add_argument("--models", type=Path, required=True, help="Directory containing local ONNX model assets.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=default_log_dir(),
        help="Directory for bounded local operational logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        log_path = configure_logging(args.log_dir)
    except OSError as exc:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")
        logger.error("listener could not create its operational log: %s", exc)
        return 1
    logger.info(
        "listener startup requested: interpreter=%s module=%s models=%s log=%s",
        sys.executable,
        __name__,
        args.models.resolve(),
        log_path,
    )
    try:
        with WindowsInstanceGuard():
            detector = WakeWordDetector(WakeWordConfig(model_dir=args.models))
            listener = PersistentListener(detector)
            detector.add_callback(lambda event: logger.info("%s phrase=%s score=%.3f", event.event_type.value, event.phrase, event.score))
            previous_handlers = {}
            for sig in (signal.SIGINT, getattr(signal, "SIGTERM", signal.SIGINT)):
                try:
                    previous_handlers[sig] = signal.signal(sig, lambda *_: listener.request_stop())
                except (ValueError, OSError):
                    pass
            try:
                listener.run()
                return 0
            except (WakeWordError, OSError) as exc:
                logger.error("listener stopped: %s", exc)
                return 1
            finally:
                listener.request_stop()
                detector.stop()
                logger.info("listener shutdown complete")
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
    except InstanceAlreadyRunning as exc:
        logger.warning("%s", exc)
        return 0
    except OSError as exc:
        logger.error("listener could not acquire its Windows instance guard: %s", exc)
        return 1
    except Exception:
        logger.exception("listener failed before startup completed")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
