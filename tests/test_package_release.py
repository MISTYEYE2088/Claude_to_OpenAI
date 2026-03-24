from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


def _run_packager(tmp_path: Path, *, version: str, platform: str) -> Path:
    binary_name = "proxy.exe" if platform == "windows" else "proxy"
    binary_path = tmp_path / binary_name
    binary_path.write_bytes(b"binary-bytes")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "package_release.py"

    command = [
        sys.executable,
        str(script_path),
        "--version",
        version,
        "--platform",
        platform,
        "--binary-path",
        str(binary_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    if platform == "windows":
        artifact_name = f"openai-to-claude-proxy-{version}-windows-x86_64.zip"
    else:
        artifact_name = f"openai-to-claude-proxy-{version}-linux-x86_64.tar.gz"
    return tmp_path / artifact_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("platform", "expected_binary"),
    [
        ("windows", "openai-to-claude-proxy.exe"),
        ("linux", "openai-to-claude-proxy"),
    ],
)
def test_packager_creates_expected_archive_and_sha256(
    tmp_path: Path,
    platform: str,
    expected_binary: str,
):
    version = "1.2.3"
    artifact_path = _run_packager(tmp_path, version=version, platform=platform)

    assert artifact_path.exists()
    assert artifact_path.name == (
        f"openai-to-claude-proxy-{version}-windows-x86_64.zip"
        if platform == "windows"
        else f"openai-to-claude-proxy-{version}-linux-x86_64.tar.gz"
    )

    if platform == "windows":
        with zipfile.ZipFile(artifact_path) as archive:
            names = set(archive.namelist())
    else:
        with tarfile.open(artifact_path, "r:gz") as archive:
            names = set(archive.getnames())

    assert expected_binary in names
    assert "config/config.example.json" in names
    assert "README.txt" in names

    sha256_path = Path(str(artifact_path) + ".sha256")
    assert sha256_path.exists()

    sidecar_line = sha256_path.read_text(encoding="utf-8").strip()
    sidecar_hash = sidecar_line.split()[0]
    assert sidecar_hash == _sha256(artifact_path)
