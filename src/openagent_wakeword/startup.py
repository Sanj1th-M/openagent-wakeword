from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

from openagent_wakeword.errors import WakeWordError
from openagent_wakeword.model_assets import validate_model_assets

TASK_NAME = "OpenAgentWakeWordListener"
TASK_PATH = "\\OpenAgent\\WakeWord\\"
TASK_DESCRIPTION = "OpenAgent wake-word listener; managed by openagent-wakeword-startup."
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ps_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_powershell(script: str) -> subprocess.CompletedProcess[str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        check=False, capture_output=True, text=True,
    )


def _ownership_check(
    operation: str,
    *,
    task_name: str = TASK_NAME,
    allow_missing: bool = False,
) -> str:
    missing_message = "" if allow_missing else (
        "Write-Output 'Task not installed.'; exit 0"
        if operation == "report status"
        else "Write-Error 'Task not installed; ownership cannot be verified.'; exit 1"
    )
    missing_check = "" if allow_missing else f"""
if ($null -eq $task) {{
    {missing_message}
}}
"""
    return f"""
$task = Get-ScheduledTask -TaskName {_ps_literal(task_name)} -TaskPath {_ps_literal(TASK_PATH)} -ErrorAction SilentlyContinue
{missing_check}
if ($null -ne $task) {{
    $ownedAction = ($task.Actions.Count -eq 1) -and ($task.Actions[0].Arguments -match '(^|\\s)-m\\s+openagent_wakeword\\.runner(\\s|$)')
    if (($task.Description -ne {_ps_literal(TASK_DESCRIPTION)}) -or (-not $ownedAction)) {{
        Write-Error 'Refusing to {operation}: task ownership could not be verified.'
        exit 1
    }}
}}
"""


def _ensure_task_folder() -> str:
    return """
$service = New-Object -ComObject 'Schedule.Service'
$service.Connect()
$root = $service.GetFolder('\\')
try { $openAgent = $service.GetFolder('\\OpenAgent') } catch { $openAgent = $root.CreateFolder('OpenAgent', $null) }
try { $null = $service.GetFolder('\\OpenAgent\\WakeWord') } catch { $null = $openAgent.CreateFolder('WakeWord', $null) }
"""


def install_task(interpreter: Path, model_dir: Path, *, task_name: str = TASK_NAME) -> None:
    interpreter = interpreter.resolve()
    model_dir = model_dir.resolve()
    if not interpreter.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {interpreter}")
    if not (interpreter.parent / ("python.exe" if os.name == "nt" else "python")).exists():
        raise FileNotFoundError(f"configured Python environment is incomplete: {interpreter.parent}")
    validate_model_assets(model_dir)
    arguments = subprocess.list2cmdline(["-m", "openagent_wakeword.runner", "--models", str(model_dir)])
    script = f"""
$ErrorActionPreference = 'Stop'
{_ownership_check('install or update', task_name=task_name, allow_missing=True)}
{_ensure_task_folder()}
$action = New-ScheduledTaskAction -Execute {_ps_literal(str(interpreter))} -Argument {_ps_literal(arguments)} -WorkingDirectory {_ps_literal(str(PROJECT_ROOT))}
$trigger = New-ScheduledTaskTrigger -AtLogOn -User ($env:USERDOMAIN + '\\' + $env:USERNAME)
$principal = New-ScheduledTaskPrincipal -UserId ($env:USERDOMAIN + '\\' + $env:USERNAME) -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName {_ps_literal(task_name)} -TaskPath {_ps_literal(TASK_PATH)} -Description {_ps_literal(TASK_DESCRIPTION)} -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Write-Output 'Task installed or updated.'
"""
    _execute(script)


def task_operation(operation: str, *, task_name: str = TASK_NAME) -> str:
    if operation == "status":
        script = f"""
$ErrorActionPreference = 'Stop'
{_ownership_check('report status', task_name=task_name)}
$info = Get-ScheduledTaskInfo -TaskName {_ps_literal(task_name)} -TaskPath {_ps_literal(TASK_PATH)}
[pscustomobject]@{{Name=$task.TaskName; TaskPath=$task.TaskPath; State=[string]$task.State; Enabled=$task.Settings.Enabled; DisallowStartIfOnBatteries=$task.Settings.DisallowStartIfOnBatteries; StopIfGoingOnBatteries=$task.Settings.StopIfGoingOnBatteries; MultipleInstances=[string]$task.Settings.MultipleInstances; ExecutionTimeLimit=[string]$task.Settings.ExecutionTimeLimit; StartWhenAvailable=$task.Settings.StartWhenAvailable; Execute=$task.Actions[0].Execute; Arguments=$task.Actions[0].Arguments; WorkingDirectory=$task.Actions[0].WorkingDirectory; LastRunTime=$info.LastRunTime; LastTaskResult=$info.LastTaskResult}} | ConvertTo-Json -Compress
"""
    elif operation == "disable":
        script = f"""
$ErrorActionPreference = 'Stop'
{_ownership_check('disable', task_name=task_name)}
Disable-ScheduledTask -TaskName {_ps_literal(task_name)} -TaskPath {_ps_literal(TASK_PATH)} -ErrorAction Stop | Out-Null
Write-Output 'Task disabled.'
"""
    elif operation == "uninstall":
        script = f"""
$ErrorActionPreference = 'Stop'
{_ownership_check('uninstall', task_name=task_name)}
Unregister-ScheduledTask -TaskName {_ps_literal(task_name)} -TaskPath {_ps_literal(TASK_PATH)} -Confirm:$false -ErrorAction Stop
Write-Output 'Task removed.'
"""
    else:
        raise ValueError(f"unsupported task operation: {operation}")
    return _execute(script)


def _execute(script: str) -> str:
    if os.name != "nt":
        raise OSError("Task Scheduler management is available only on Windows")
    result = _run_powershell(script)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Task Scheduler operation failed: {detail}")
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the current-user wake-word startup task.")
    sub = parser.add_subparsers(dest="operation", required=True)
    install = sub.add_parser("install", help="Install or update the logon task.")
    install.add_argument("--python", type=Path, default=Path(sys.executable))
    install.add_argument("--models", type=Path, default=PROJECT_ROOT / ".models" / "openwakeword" / "v0.5.1")
    sub.add_parser("status")
    sub.add_parser("disable")
    sub.add_parser("uninstall")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "install":
            install_task(args.python, args.models)
            print("Task installed or updated for the current user's sign-in.")
        else:
            print(task_operation(args.operation))
        return 0
    except (OSError, RuntimeError, ValueError, WakeWordError) as exc:
        print(f"Startup task operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
