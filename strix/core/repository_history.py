"""Best-effort history lookups against an existing local Git checkout."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class BlameInfo:
    author_name: str
    author_email: str
    commit_sha: str
    commit_timestamp: str
    commit_summary: str


def _parse_blame(output: str) -> BlameInfo | None:
    lines = output.splitlines()
    if not lines or "\x00" in output:
        return None
    header = lines[0].split()
    if not header or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", header[0]):
        return None
    commit_sha = header[0]
    if not commit_sha.strip("0"):
        # Git uses an all-zero object ID for a line with uncommitted changes.
        return None
    metadata: dict[str, str] = {}
    for raw_line in lines[1:]:
        if raw_line.startswith("\t"):
            break
        key, _, value = raw_line.partition(" ")
        metadata[key] = value
    try:
        timestamp = datetime.fromtimestamp(int(metadata["committer-time"]), tz=UTC).isoformat()
        return BlameInfo(
            author_name=metadata["author"],
            author_email=metadata["author-mail"].removeprefix("<").removesuffix(">"),
            commit_sha=commit_sha,
            commit_timestamp=timestamp,
            commit_summary=metadata["summary"],
        )
    except (KeyError, ValueError, OverflowError, OSError):
        return None


def blame_line(
    repository: Path, file_path: object, line: int, *, timeout: float = 3.0
) -> BlameInfo | None:
    """Attribute one working-tree line, or return None when history is unavailable.

    Paths are relative to ``repository``. Git follows committed renames itself;
    missing paths, binary files and uncommitted lines are intentionally omitted.
    Lookups never fetch history and are bounded so enrichment stays optional.
    Callers may supply a shorter timeout to share a budget across lookups.
    """
    if (
        type(line) is not int
        or line < 1
        or not isinstance(file_path, str)
        or not file_path
        or timeout <= 0
    ):
        return None
    try:
        relative = Path(file_path)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        root = repository.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            return None
        with path.open("rb") as source:
            if b"\x00" in source.read(8192):
                return None
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "-C",
                str(root),
                "blame",
                "--line-porcelain",
                "--no-textconv",
                "-L",
                f"{line},{line}",
                "--",
                file_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=min(timeout, 3.0),
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return None
    return _parse_blame(result.stdout) if result.returncode == 0 else None
