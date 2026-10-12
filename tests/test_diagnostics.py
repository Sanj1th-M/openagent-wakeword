import numpy as np

from openagent_wakeword.diagnostics import WakeWordDiagnostics


def test_diagnostics_record_bounded_audio_statistics_without_raw_audio() -> None:
    diagnostics = WakeWordDiagnostics(near_silence_rms=10.0, clipping_threshold=32000)

    diagnostics.record_frame(np.array([0, 0, 0, 0], dtype=np.int16))
    diagnostics.record_frame(np.array([32767, -32768, 100, -100], dtype=np.int16))
    diagnostics.record_prediction(0.42, 0.012)
    diagnostics.record_prediction(0.8, 0.020)
    diagnostics.record_callback_status("input overflow")

    summary = diagnostics.summary()
    assert summary.frame_count == 2
    assert summary.prediction_count == 2
    assert summary.max_score == 0.8
    assert summary.frame_lengths == {4: 2}
    assert summary.callback_status_count == 1
    assert summary.callback_status_messages == ("input overflow",)
    assert summary.prediction_latency_ms_min == 12.0
    assert summary.prediction_latency_ms_max == 20.0
    assert summary.prediction_latency_ms_avg == 16.0
    assert summary.peak_max == 32768
    assert summary.clipping_frames == 1
    assert summary.near_silence_frames == 1
    assert not hasattr(summary, "samples")
    assert not hasattr(summary, "audio")


def test_callback_status_messages_are_bounded() -> None:
    diagnostics = WakeWordDiagnostics(max_status_messages=2)

    diagnostics.record_callback_status("one")
    diagnostics.record_callback_status("two")
    diagnostics.record_callback_status("three")

    summary = diagnostics.summary()
    assert summary.callback_status_count == 3
    assert summary.callback_status_messages == ("one", "two")
