# openagent-wakeword

`openagent-wakeword` is the first isolated component for the future `openagent`
Windows desktop voice assistant. It listens locally for the configured wake
phrase while the assistant is idle and emits a `WAKE_DETECTED` event.

This component does not implement Pipecat, Deepgram, TTS, UFO² integration, a
GUI, or a Windows service. Optional current-user sign-in startup is available
through an explicit Task Scheduler management command.

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

After updating an existing checkout, refresh its virtual environment so newly
declared commands are installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
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

## Persistent Windows listener

Run the listener in the foreground with the configured local model directory:

```powershell
openagent-wakeword-listener --models .models\openwakeword\v0.5.1
```

The listener stays active until Ctrl+C or process termination. It loads only
local model assets and does not save, upload, or log microphone audio. Audio
startup or stream-status failures are retried with delays of 1, 2, 4, 8, 16,
then 30 seconds, capped at 30 seconds. Retries continue until audio recovers or
the process is stopped. Model, inference, and microphone cleanup failures are
terminal. Bounded operational logs are written under
`%LOCALAPPDATA%\OpenAgent\WakeWord\logs\listener.log`.

Task Scheduler startup is opt-in and scoped to the current user's sign-in. Run
these commands from the installed project environment:

```powershell
openagent-wakeword-startup install --python .venv\Scripts\python.exe --models .models\openwakeword\v0.5.1
openagent-wakeword-startup status
openagent-wakeword-startup disable
openagent-wakeword-startup uninstall
```

`install` validates the interpreter and model assets, then creates or updates
the application-owned task `OpenAgentWakeWordListener` only at
`\OpenAgent\WakeWord\`. Ownership requires that dedicated task location, the
exact application description marker, and a single action invoking
`openagent_wakeword.runner`. An existing task that does not meet all three
checks is left unchanged and the command fails clearly. `status` shows whether
the owned task is enabled and its executable, arguments, working directory,
and last result. It also reports battery restrictions; for battery operation,
both should be false. Re-running `install` updates the owned task to allow
starts on battery and to keep running when AC power is disconnected. `disable`
preserves the owned task; `uninstall` removes only the owned task and leaves its task
folder in place. Disabling or uninstalling a missing task reports a clear
failure. These operations do not download models, store credentials, or
require administrator privileges by design. To verify sign-in behavior, inspect
`status` and sign out/in while the configured environment and microphone are
available. Do not install the task from elevated PowerShell if you intend it to
run as your regular account.

The listener uses a named Windows global mutex keyed by the current user's SID.
It prevents duplicate listener instances for that account and Windows releases
the mutex when the owning process exits, including after abnormal termination.
Users who cannot run `whoami.exe` or create named mutexes cannot use the runner.
The guard does not coordinate copies running under different Windows accounts.
The Task Scheduler integration and real microphone recovery should be checked
on the target Windows installation; unit tests do not establish hardware
reliability.

## Troubleshooting

- Missing model files: run the explicit model setup command above.
- Microphone busy or permission denied: close other apps using the microphone
  and check Windows microphone privacy settings.
- Repeated activations: increase `cooldown_seconds` or threshold after manual
  testing.
- Missed activations: confirm that the phrase is "Hey Jarvis", check input
  device selection, and tune threshold only with real environment testing.
