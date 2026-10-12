from pathlib import Path

import pytest

from openagent_wakeword.errors import ModelAssetError
from openagent_wakeword.model_assets import REQUIRED_ONNX_ASSETS, required_model_paths, validate_model_assets


def test_required_model_paths_are_outside_source_by_caller_choice(tmp_path: Path) -> None:
    paths = required_model_paths(tmp_path)

    assert set(paths) == {asset.filename for asset in REQUIRED_ONNX_ASSETS}
    assert all(path.parent == tmp_path for path in paths.values())


def test_missing_model_assets_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelAssetError, match="missing model assets"):
        validate_model_assets(tmp_path)


def test_wrong_model_asset_size_is_rejected(tmp_path: Path) -> None:
    for asset in REQUIRED_ONNX_ASSETS:
        (tmp_path / asset.filename).write_bytes(b"x")

    with pytest.raises(ModelAssetError, match="has size"):
        validate_model_assets(tmp_path)
