"""Local history lookup coverage using real Git history and failure injection."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from strix.core.repository_history import blame_line


if TYPE_CHECKING:
    from pathlib import Path


def _git(repository: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repository), *args],  # noqa: S607
        env={**os.environ, **(env or {})},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "First Author")
    _git(root, "config", "user.email", "first@example.test")
    (root / "vulnerable.py").write_text("first line\nvulnerable line\nlast line\n")
    _git(root, "add", "vulnerable.py")
    _git(
        root,
        "commit",
        "-m",
        "Original source",
        env={
            "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2020-01-02T00:00:00Z",
        },
    )
    (root / "vulnerable.py").write_text("first line\nchanged vulnerable line\nlast line\n")
    _git(root, "add", "vulnerable.py")
    _git(
        root,
        "commit",
        "-m",
        "Change vulnerable line\n\nCommit body.",
        env={
            "GIT_AUTHOR_NAME": "Second Author",
            "GIT_AUTHOR_EMAIL": "second@example.test",
            "GIT_AUTHOR_DATE": "2021-02-03T04:05:06Z",
            "GIT_COMMITTER_DATE": "2021-02-04T05:06:07Z",
        },
    )
    return root


def test_blame_attributes_requested_line_and_commit_timestamp(repository: Path) -> None:
    result = blame_line(repository, "vulnerable.py", 2)

    assert result is not None
    assert result.author_name == "Second Author"
    assert result.author_email == "second@example.test"
    assert result.commit_sha == _git(repository, "rev-parse", "HEAD")
    assert result.commit_timestamp == "2021-02-04T05:06:07+00:00"
    assert result.commit_summary == "Change vulnerable line"
    unchanged = blame_line(repository, "vulnerable.py", 1)
    assert unchanged is not None
    assert unchanged.author_name == "First Author"
    assert unchanged.commit_sha == _git(repository, "rev-parse", "HEAD~1")


def test_committed_rename_follows_history_and_missing_old_path_is_skipped(repository: Path) -> None:
    original_sha = _git(repository, "rev-parse", "HEAD")
    _git(repository, "mv", "vulnerable.py", "renamed.py")
    _git(repository, "commit", "-m", "Rename source")

    result = blame_line(repository, "renamed.py", 2)
    assert result is not None
    assert result.commit_sha == original_sha
    assert blame_line(repository, "vulnerable.py", 2) is None


def test_uncommitted_line_is_skipped_but_unchanged_line_retains_history(repository: Path) -> None:
    (repository / "vulnerable.py").write_text("first line\nuncommitted change\nlast line\n")

    assert blame_line(repository, "vulnerable.py", 2) is None
    assert blame_line(repository, "vulnerable.py", 1) is not None
    _git(repository, "add", "vulnerable.py")
    assert blame_line(repository, "vulnerable.py", 2) is None


@pytest.mark.parametrize("line", [None, 0, -1, True, "2", 1.5, 500])
def test_invalid_missing_or_out_of_bounds_lines_are_skipped(repository: Path, line: Any) -> None:
    assert blame_line(repository, "vulnerable.py", line) is None


@pytest.mark.parametrize(
    "file_path", ["", "missing.py", "../vulnerable.py", "/etc/passwd", "bad\x00"]
)
def test_invalid_or_missing_paths_are_skipped(repository: Path, file_path: str) -> None:
    assert blame_line(repository, file_path, 1) is None


def test_binary_untracked_and_non_repository_files_are_skipped(
    repository: Path, tmp_path: Path
) -> None:
    (repository / "binary.dat").write_bytes(b"binary\x00data\n")
    _git(repository, "add", "binary.dat")
    _git(repository, "commit", "-m", "Add binary")
    (repository / "generated.py").write_text("generated source\n")
    (tmp_path / "plain.py").write_text("not a repository\n")

    assert blame_line(repository, "binary.dat", 1) is None
    assert blame_line(repository, "generated.py", 1) is None
    assert blame_line(tmp_path, "plain.py", 1) is None
    assert blame_line(tmp_path / "absent", "plain.py", 1) is None


def test_tracked_generated_text_can_be_attributed(repository: Path) -> None:
    (repository / "generated.py").write_text("# Generated file\nvalue = 1\n")
    _git(repository, "add", "generated.py")
    _git(repository, "commit", "-m", "Generate source")

    result = blame_line(repository, "generated.py", 2)
    assert result is not None
    assert result.commit_summary == "Generate source"


def test_existing_linked_worktree_uses_its_local_history(repository: Path, tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    _git(repository, "worktree", "add", "--detach", str(worktree), "HEAD")

    result = blame_line(worktree, "vulnerable.py", 2)
    assert result is not None
    assert result.commit_sha == _git(repository, "rev-parse", "HEAD")


def test_symlink_cannot_escape_repository(repository: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("outside source\n")
    (repository / "escape.py").symlink_to(outside)

    with patch("strix.core.repository_history.subprocess.run") as run:
        assert blame_line(repository, "escape.py", 1) is None
    run.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("git missing"),
        PermissionError("denied"),
        subprocess.TimeoutExpired("git", 3),
    ],
)
def test_git_failures_do_not_escape(repository: Path, failure: Exception) -> None:
    with patch("strix.core.repository_history.subprocess.run", side_effect=failure):
        assert blame_line(repository, "vulnerable.py", 2) is None


@pytest.mark.parametrize("output", ["", "malformed", "a" * 40 + " 1 1 1\nauthor Someone\n"])
def test_malformed_output_is_skipped(repository: Path, output: str) -> None:
    result = subprocess.CompletedProcess(["git"], 0, stdout=output)
    with patch("strix.core.repository_history.subprocess.run", return_value=result):
        assert blame_line(repository, "vulnerable.py", 2) is None


@pytest.mark.parametrize("timestamp", ["not-a-time", "9999999999999999999999999999999"])
def test_invalid_commit_timestamp_is_skipped(repository: Path, timestamp: str) -> None:
    output = (
        f"{'a' * 40} 2 2 1\nauthor Name\nauthor-mail <name@example.test>\n"
        f"committer-time {timestamp}\nsummary Message\n\tcode\n"
    )
    result = subprocess.CompletedProcess(["git"], 0, stdout=output)
    with patch("strix.core.repository_history.subprocess.run", return_value=result):
        assert blame_line(repository, "vulnerable.py", 2) is None


def test_nonzero_exit_is_skipped_and_git_is_local_bounded(repository: Path) -> None:
    result = subprocess.CompletedProcess(["git"], 128, stdout="", stderr="fatal")
    with patch("strix.core.repository_history.subprocess.run", return_value=result) as run:
        assert blame_line(repository, "vulnerable.py", 2) is None

    args, kwargs = run.call_args
    assert args[0][-4:] == ["-L", "2,2", "--", "vulnerable.py"]
    assert kwargs["timeout"] <= 3
    assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
    assert "--no-textconv" in args[0]
