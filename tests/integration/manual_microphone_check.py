from __future__ import annotations

import argparse
import threading
from pathlib import Path

from openagent_wakeword import WakeWordConfig, WakeWordDetector
from openagent_wakeword.diagnostics import DiagnosticSummary, WakeWordDiagnostics


def threshold_arg(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number") from exc
    if not 0.0 < threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be > 0 and <= 1")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual microphone wake-word smoke test.")
    parser.add_argument("--models", type=Path, required=True, help="Directory containing ONNX model assets.")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--threshold", type=threshold_arg, default=0.5)
    return parser


def _format_optional(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_diagnostics(summary: DiagnosticSummary, threshold: float) -> None:
    print("\nDiagnostic summary:")
    print(f"  threshold: {threshold:.3f}")
    print(f"  frames: {summary.frame_count}")
    print(f"  predictions: {summary.prediction_count}")
    print(f"  max_score: {_format_optional(summary.max_score)}")
    print(f"  frame_lengths: {summary.frame_lengths}")
    print(f"  callback_status_events: {summary.callback_status_count}")
    if summary.callback_status_messages:
        print(f"  callback_status_samples: {list(summary.callback_status_messages)}")
    print(
        "  prediction_latency_ms: "
        f"min={_format_optional(summary.prediction_latency_ms_min)} "
        f"avg={_format_optional(summary.prediction_latency_ms_avg)} "
        f"max={_format_optional(summary.prediction_latency_ms_max)}"
    )
    print(
        "  rms: "
        f"min={_format_optional(summary.rms_min, digits=1)} "
        f"avg={_format_optional(summary.rms_avg, digits=1)} "
        f"max={_format_optional(summary.rms_max, digits=1)}"
    )
    print(f"  peak_amplitude: {summary.peak_max}")
    print(f"  clipping_frames: {summary.clipping_frames}")
    print(f"  near_silence_frames: {summary.near_silence_frames}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    detected = threading.Event()
    diagnostics = WakeWordDiagnostics()

    config = WakeWordConfig(model_dir=args.models, threshold=args.threshold)
    detector = WakeWordDetector(config, diagnostics=diagnostics)

    def on_event(event) -> None:
        print(
            f"{event.event_type.value} phrase={event.phrase!r} "
            f"model={event.model_id} score={event.score:.3f}"
        )
        detected.set()

    detector.add_callback(on_event)
    print('Listening locally. Say "Hey Jarvis"...')
    try:
        detector.start()
        if not detected.wait(args.timeout):
            print("No wake event detected before timeout.")
            return 2
        return 0
    finally:
        detector.stop()
        print_diagnostics(diagnostics.summary(), args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
