# openagent-wakeword

`openagent-wakeword` is the first isolated component for the future `openagent`
Windows desktop voice assistant. It listens locally for the configured wake
phrase while the assistant is idle and emits a `WAKE_DETECTED` event.

This prototype does not implement Pipecat, Deepgram, TTS, UFO² integration, a
GUI, automatic startup, or a Windows service.

## Supported Phrase

The initial candidate model is openWakeWord's official `hey_jarvis` model. The
supported target phrase for this prototype is **"Hey Jarvis"**.

Do not treat "Jarvis" alone as a reliable supported phrase. The upstream model
documentation says similar phrases such as "jarvis" may work, but likely with
higher false-reject rates.

## Licensing

The `openwakeword` Python code is documented by upstream as Apache 2.0. Upstream
documents the included pretrained model assets as Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International because of training data
licensing uncertainty.

Practical consequence: the pretrained `hey_jarvis` model is development and
testing only for this project. Commercial release is blocked until the model
licensing and intended use are resolved or a suitable replacement model is used.

Model binaries are not committed to this repository.

## Privacy

The detector processes microphone frames locally. It does not save, upload, or
log raw microphone audio by default. Events contain only minimal metadata:
event type, phrase, model id, score, and timestamp.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

Fetch the development model assets explicitly:

```powershell
openagent-wakeword-models download --target .models\openwakeword\v0.5.1
```

This downloads only the ONNX assets needed for the first prototype from the
official openWakeWord GitHub release `v0.5.1`:

- `melspectrogram.onnx`
- `embedding_model.onnx`
- `hey_jarvis_v0.1.onnx`

The GitHub release metadata checked during design exposed asset names, sizes,
and official URLs, but no upstream SHA-256 digests. The setup command therefore
verifies the source host, release tag, file name, and byte size, and writes a
local SHA-256 manifest after download. That local manifest can detect later
local changes, but it is not a substitute for upstream-published checksums.

## Tests

Offline unit tests do not require a microphone, network access, or model files:

```powershell
python -m pytest
```

Manual microphone testing is separate because it needs hardware, permissions,
and downloaded model assets:

```powershell
python tests\integration\manual_microphone_check.py --models .models\openwakeword\v0.5.1
```

Say "Hey Jarvis" after the listener starts. Do not claim real wake-phrase
performance until this manual test succeeds in the actual target environment.

To compare detection thresholds without changing the default configuration, pass
`--threshold`:

```powershell
python tests\integration\manual_microphone_check.py --models .models\openwakeword\v0.5.1 --threshold 0.4
```

The manual check prints a bounded local diagnostic summary after each run. It
includes frame counts, prediction counts, maximum score, frame length counts,
callback status events, prediction latency, RMS/peak level summaries, clipping
frames, and near-silence frames. It does not save or print raw audio samples.

## Troubleshooting

- Missing model files: run the explicit model setup command above.
- Microphone busy or permission denied: close other apps using the microphone
  and check Windows microphone privacy settings.
- Repeated activations: increase `cooldown_seconds` or threshold after manual
  testing.
- Missed activations: confirm that the phrase is "Hey Jarvis", check input
  device selection, and tune threshold only with real environment testing.
