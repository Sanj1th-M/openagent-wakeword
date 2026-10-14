from pathlib import Path
import tomllib

import pytest

from openagent_wakeword import startup


def test_project_declares_all_console_entry_points() -> None:
    project_root = Path(__file__).parents[1]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["scripts"] == {
        "openagent-wakeword-models": "openagent_wakeword.model_assets:main",
        "openagent-wakeword-listener": "openagent_wakeword.runner:main",
        "openagent-wakeword-startup": "openagent_wakeword.startup:main",
    }


def test_powershell_literal_escapes_quotes() -> None:
    assert startup._ps_literal("C:\\Users\\O'Neil\\models") == "'C:\\Users\\O''Neil\\models'"


def test_install_validates_interpreter_and_model_assets_before_registration(tmp_path, monkeypatch) -> None:
    spaced_path = tmp_path / "folder with spaces"
    spaced_path.mkdir()
    interpreter = spaced_path / "python.exe"
    interpreter.touch()
    models = tmp_path / "model assets"
    monkeypatch.setattr(startup, "validate_model_assets", lambda _path: None)
    scripts = []
    monkeypatch.setattr(startup, "_execute", lambda script: scripts.append(script) or "ok")
    startup.install_task(interpreter, models)
    script = scripts[0]
    assert "New-ScheduledTaskTrigger -AtLogOn" in script
    assert "-RunLevel Limited" in script
    assert "-WorkingDirectory" in script
    assert "Register-ScheduledTask" in script
    assert "-Force" in script
    assert '"' in script
    assert "--models" in script
    assert f"-TaskPath '{startup.TASK_PATH}'" in script
    assert f"-Description '{startup.TASK_DESCRIPTION}'" in script
    assert "Schedule.Service" in script
    assert "Refusing to install or update" in script
    assert "-AllowStartIfOnBatteries" in script
    assert "-DontStopIfGoingOnBatteries" in script
    assert "-MultipleInstances IgnoreNew" in script


def test_install_rejects_missing_interpreter(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="interpreter not found"):
        startup.install_task(tmp_path / "missing.exe", tmp_path / "models")


def test_missing_task_operations_are_safe_and_clear(monkeypatch) -> None:
    scripts = []
    monkeypatch.setattr(startup, "_execute", lambda script: scripts.append(script) or "ok")
    assert startup.task_operation("status") == "ok"
    assert "Task not installed." in scripts[-1]
    assert "exit 0" in scripts[-1]
    startup.task_operation("disable")
    assert "Task not installed;" in scripts[-1]
    assert "ownership cannot be verified" in scripts[-1]
    assert "exit 1" in scripts[-1]
    assert "Disable-ScheduledTask" in scripts[-1]
    startup.task_operation("uninstall")
    assert "Task not installed;" in scripts[-1]
    assert "ownership cannot be verified" in scripts[-1]
    assert "exit 1" in scripts[-1]
    assert "-Confirm:$false" in scripts[-1]


def test_task_operations_refuse_unrelated_tasks_and_target_only_owned_path(monkeypatch) -> None:
    scripts = []
    monkeypatch.setattr(startup, "_execute", lambda script: scripts.append(script) or "ok")
    for operation in ("status", "disable", "uninstall"):
        startup.task_operation(operation)
        script = scripts[-1]
        assert f"-TaskPath '{startup.TASK_PATH}'" in script
        assert f"$task.Description -ne '{startup.TASK_DESCRIPTION}'" in script
        assert f"Refusing to {operation if operation != 'status' else 'report status'}: task ownership could not be verified." in script
        assert "$task.Actions.Count -eq 1" in script
        assert "openagent_wakeword\\.runner" in script
        assert "exit 1" in script
        if operation == "status":
            assert "DisallowStartIfOnBatteries" in script
            assert "StopIfGoingOnBatteries" in script
            assert "MultipleInstances" in script
        else:
            refusal = script.index("task ownership could not be verified")
            mutation = script.index("Disable-ScheduledTask" if operation == "disable" else "Unregister-ScheduledTask")
            assert refusal < mutation


def test_reinstall_updates_only_an_owned_task(monkeypatch, tmp_path) -> None:
    interpreter = tmp_path / "python.exe"
    interpreter.touch()
    monkeypatch.setattr(startup, "validate_model_assets", lambda _path: None)
    scripts = []
    monkeypatch.setattr(startup, "_execute", lambda script: scripts.append(script) or "ok")
    startup.install_task(interpreter, tmp_path / "models")
    script = scripts[-1]
    assert "$null -ne $task" in script
    assert "Refusing to install or update" in script
    assert "$task.Actions.Count -eq 1" in script
    assert script.index("task ownership could not be verified") < script.index("Register-ScheduledTask")
    assert "Register-ScheduledTask" in script
