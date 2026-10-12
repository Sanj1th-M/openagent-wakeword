from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openagent_wakeword.errors import ModelAssetError

OFFICIAL_RELEASE_TAG = "v0.5.1"
OFFICIAL_HOST = "github.com"
OFFICIAL_BASE_URL = (
    f"https://github.com/dscripka/openWakeWord/releases/download/{OFFICIAL_RELEASE_TAG}"
)


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    size: int

    @property
    def url(self) -> str:
        return f"{OFFICIAL_BASE_URL}/{self.filename}"


REQUIRED_ONNX_ASSETS = (
    ModelAsset("melspectrogram.onnx", 1087958),
    ModelAsset("embedding_model.onnx", 1326578),
    ModelAsset("hey_jarvis_v0.1.onnx", 1271370),
)


def required_model_paths(model_dir: Path) -> dict[str, Path]:
    return {asset.filename: model_dir / asset.filename for asset in REQUIRED_ONNX_ASSETS}


def validate_model_assets(model_dir: Path) -> None:
    missing = [name for name, path in required_model_paths(model_dir).items() if not path.exists()]
    if missing:
        raise ModelAssetError(f"missing model assets: {', '.join(missing)}")
    for asset in REQUIRED_ONNX_ASSETS:
        path = model_dir / asset.filename
        actual_size = path.stat().st_size
        if actual_size != asset.size:
            raise ModelAssetError(
                f"model asset {asset.filename} has size {actual_size}, expected {asset.size}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_official_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != OFFICIAL_HOST:
        raise ModelAssetError(f"refusing non-official model URL: {url}")
    expected_prefix = f"/dscripka/openWakeWord/releases/download/{OFFICIAL_RELEASE_TAG}/"
    if not parsed.path.startswith(expected_prefix):
        raise ModelAssetError(f"refusing unexpected model URL path: {url}")


def download_required_assets(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str | int]] = {}
    for asset in REQUIRED_ONNX_ASSETS:
        _validate_official_url(asset.url)
        destination = target / asset.filename
        if not destination.exists():
            request = Request(asset.url, headers={"User-Agent": "openagent-wakeword"})
            with urlopen(request, timeout=60) as response:
                data = response.read()
            if len(data) != asset.size:
                raise ModelAssetError(
                    f"downloaded {asset.filename} has size {len(data)}, expected {asset.size}"
                )
            destination.write_bytes(data)
        elif destination.stat().st_size != asset.size:
            raise ModelAssetError(
                f"existing {asset.filename} has size {destination.stat().st_size}, expected {asset.size}"
            )
        manifest[asset.filename] = {
            "url": asset.url,
            "size": asset.size,
            "sha256": sha256_file(destination),
        }

    manifest_path = target / "manifest.sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage openagent-wakeword model assets.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Download required development model assets.")
    download.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "download":
        manifest = download_required_assets(args.target)
        print(f"Downloaded/verified model assets. Manifest: {manifest}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
