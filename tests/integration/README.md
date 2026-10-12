# Manual Integration Tests

These checks require a real microphone, Windows microphone permission, installed
runtime dependencies, and downloaded model assets. They are intentionally not
part of the offline unit test suite.

Run from the repository root after setup:

```powershell
python tests\integration\manual_microphone_check.py --models .models\openwakeword\v0.5.1
```

Say "Hey Jarvis" when prompted. The test prints `WAKE_DETECTED` when the detector
emits an event. Do not use this as a benchmark; it is only a smoke test for
hardware, model loading, and event wiring.
