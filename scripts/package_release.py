from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path


PROJECT_NAME = "openai-to-claude-proxy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package release artifacts for openai-to-claude-proxy",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True, choices=("windows", "linux"))
    parser.add_argument("--binary-path", required=True, type=Path)
    return parser.parse_args()


def _artifact_name(*, version: str, platform: str) -> str:
    if platform == "windows":
        return f"{PROJECT_NAME}-{version}-windows-x86_64.zip"
    return f"{PROJECT_NAME}-{version}-linux-x86_64.tar.gz"


def _binary_name(platform: str) -> str:
    return f"{PROJECT_NAME}.exe" if platform == "windows" else PROJECT_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_readme_txt(repo_root: Path) -> bytes:
    readme_md = (repo_root / "README.md").read_text(encoding="utf-8")
    return readme_md.encode("utf-8")


def _build_zip(
    artifact_path: Path,
    *,
    binary_source: Path,
    binary_target_name: str,
    config_source: Path,
    readme_bytes: bytes,
) -> None:
    with zipfile.ZipFile(
        artifact_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.write(binary_source, arcname=binary_target_name)
        archive.write(config_source, arcname="config/config.example.json")
        archive.writestr("README.txt", readme_bytes)


def _build_tar(
    artifact_path: Path,
    *,
    binary_source: Path,
    binary_target_name: str,
    config_source: Path,
    readme_bytes: bytes,
) -> None:
    with tarfile.open(artifact_path, "w:gz") as archive:
        archive.add(binary_source, arcname=binary_target_name)
        archive.add(config_source, arcname="config/config.example.json")

        readme_info = tarfile.TarInfo(name="README.txt")
        readme_info.size = len(readme_bytes)
        archive.addfile(readme_info, fileobj=_BytesReader(readme_bytes))


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = len(self._data) - self._offset
        start = self._offset
        end = min(start + size, len(self._data))
        self._offset = end
        return self._data[start:end]


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    binary_path = args.binary_path.resolve()
    if not binary_path.exists():
        raise FileNotFoundError(f"binary path does not exist: {binary_path}")

    config_source = repo_root / "config" / "config.example.json"
    if not config_source.exists():
        raise FileNotFoundError(f"missing config source: {config_source}")

    artifact_path = binary_path.parent / _artifact_name(
        version=args.version,
        platform=args.platform,
    )
    binary_target_name = _binary_name(args.platform)
    readme_bytes = _write_readme_txt(repo_root)

    if args.platform == "windows":
        _build_zip(
            artifact_path,
            binary_source=binary_path,
            binary_target_name=binary_target_name,
            config_source=config_source,
            readme_bytes=readme_bytes,
        )
    else:
        _build_tar(
            artifact_path,
            binary_source=binary_path,
            binary_target_name=binary_target_name,
            config_source=config_source,
            readme_bytes=readme_bytes,
        )

    digest = _sha256(artifact_path)
    sidecar_path = Path(str(artifact_path) + ".sha256")
    sidecar_path.write_text(f"{digest}  {artifact_path.name}\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
