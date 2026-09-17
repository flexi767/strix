"""Repository history through the reporting tool, callbacks, and persisted artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING, Any

import pytest
from agents.tool_context import ToolContext

from strix.core.repository_history import BlameInfo
from strix.report import history
from strix.report.history import enrich_report
from strix.report.state import ReportState, set_global_report_state
from strix.tools.reporting.tool import create_vulnerability_report, update_vulnerability_report


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


_ANALYSIS = "The query interpolates attacker-controlled input."
_TARGET = "https://example.test/team/application.git"
_FIRST_AUTHOR = "Alice Original"
_LATEST_AUTHOR = "Bea Reviewer"
_CVSS = {
    "attack_vector": "N",
    "attack_complexity": "L",
    "privileges_required": "N",
    "user_interaction": "N",
    "scope": "U",
    "confidentiality": "H",
    "integrity": "H",
    "availability": "H",
}


def _git(path: Path, *args: str, author: str = _FIRST_AUTHOR) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "-c", "commit.gpgsign=false", "-C", str(path), *args],  # noqa: S607
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": author,
            "GIT_AUTHOR_EMAIL": "author@example.test",
            "GIT_COMMITTER_NAME": author,
            "GIT_COMMITTER_EMAIL": "committer@example.test",
            "GIT_AUTHOR_DATE": "2024-01-02T03:04:05+00:00",
            "GIT_COMMITTER_DATE": "2024-01-02T03:04:05+00:00",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def history_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ReportState, Path, dict[str, str]]]:
    """Use an existing full clone, including an older author for the range start."""
    monkeypatch.chdir(tmp_path)
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--quiet")
    (origin / "app.py").write_text("query = 'initial'\nresult = query\n", encoding="utf-8")
    _git(origin, "add", "app.py")
    _git(origin, "commit", "--quiet", "-m", "Add query handler")
    first_sha = _git(origin, "rev-parse", "HEAD")
    (origin / "app.py").write_text("query = 'initial'\nresult = execute(query)\n", encoding="utf-8")
    _git(origin, "commit", "--quiet", "-am", "Execute the query", author=_LATEST_AUTHOR)
    latest_sha = _git(origin, "rev-parse", "HEAD")
    clone = tmp_path / "application"
    _git(tmp_path, "clone", "--quiet", "--no-hardlinks", str(origin), str(clone))
    assert _git(clone, "rev-parse", "--is-shallow-repository") == "false"

    state = ReportState(run_name="history-run")
    state.set_scan_config(
        {
            "targets": [
                {
                    "type": "repository",
                    "original": _TARGET,
                    "details": {
                        "target_repo": _TARGET,
                        "cloned_repo_path": str(clone),
                        "workspace_subdir": "application",
                    },
                }
            ],
            "local_sources": [{"source_path": str(clone), "workspace_subdir": "application"}],
        }
    )
    set_global_report_state(state)
    try:
        yield state, clone, {"first": first_sha, "latest": latest_sha}
    finally:
        set_global_report_state(None)


async def _create(
    location: dict[str, Any], *additional_locations: dict[str, Any]
) -> dict[str, Any]:
    arguments = {
        "title": "SQL injection in the query handler",
        "description": "The query handler executes unsanitized input.",
        "impact": "An anonymous caller can access other users' records.",
        "target": _TARGET,
        "technical_analysis": _ANALYSIS,
        "poc_description": "Submit a quote in the query parameter.",
        "poc_script_code": "GET /query?q='",
        "remediation_steps": "Use parameterized queries.",
        "evidence": "The response includes another user's record.",
        "assumptions": "The observed database role is available in production.",
        "counterevidence": "No input validation runs before the query.",
        "confidence": "high",
        "severity_change_conditions": "A read-only database role would limit the impact.",
        "fix_effort": "low",
        "cvss_breakdown": _CVSS,
        "code_locations": [location, *additional_locations],
    }
    context = ToolContext(
        context={"agent_id": "root"},
        tool_name="create_vulnerability_report",
        tool_call_id="create-1",
        tool_arguments=json.dumps(arguments),
    )
    result: dict[str, Any] = json.loads(
        await create_vulnerability_report.on_invoke_tool(context, json.dumps(arguments))
    )
    return result


async def _update(report_id: str, **fields: Any) -> dict[str, Any]:
    arguments = {
        "report_id": report_id,
        "update_reason": "Refined the vulnerable location.",
        **fields,
    }
    context = ToolContext(
        context={"agent_id": "root"},
        tool_name="update_vulnerability_report",
        tool_call_id="update-1",
        tool_arguments=json.dumps(arguments),
    )
    result: dict[str, Any] = json.loads(
        await update_vulnerability_report.on_invoke_tool(context, json.dumps(arguments))
    )
    return result


@pytest.mark.parametrize("primary_line", [2, None, -1, 3, "invalid", True])
async def test_tool_enriches_callbacks_and_artifacts_from_full_clone(
    history_run: tuple[ReportState, Path, dict[str, str]], primary_line: Any
) -> None:
    state, _clone, commits = history_run
    callbacks: list[dict[str, Any]] = []
    state.vulnerability_found_callback = lambda report: callbacks.append(dict(report))
    location = {"file": "app.py", "start_line": 1, "end_line": 2, "primary_line": primary_line}

    result = await _create(location)

    assert result["success"] is True
    report = state.vulnerability_reports[0]
    analysis = report["technical_analysis"]
    expected_author = _LATEST_AUTHOR if primary_line == 2 else _FIRST_AUTHOR
    expected_sha = commits["latest"] if primary_line == 2 else commits["first"]
    expected_summary = "Execute the query" if primary_line == 2 else "Add query handler"
    assert analysis.startswith(_ANALYSIS)
    assert analysis.count("Last modified by") == 1
    for value in (expected_author, "author@example.test", expected_sha, expected_summary):
        assert value in analysis
    assert "2024-01-02T03:04:05+00:00" in analysis
    assert callbacks == [report]
    assert "blame" not in report
    assert "author_name" not in report
    saved = json.loads((state.get_run_dir() / "vulnerabilities.json").read_text())
    assert saved == [report]
    markdown = (state.get_run_dir() / "vulnerabilities" / f"{report['id']}.md").read_text()
    technical_section = markdown.split("## Technical Analysis", 1)[1].split(
        "## Proof of Concept", 1
    )[0]
    assert analysis in technical_section


async def test_resume_and_updates_refresh_history_without_stale_or_duplicate_details(
    history_run: tuple[ReportState, Path, dict[str, str]],
) -> None:
    state, _clone, commits = history_run
    assert (await _create({"file": "app.py", "start_line": 1, "end_line": 2}))["success"]
    resumed = ReportState(run_name=state.run_name)
    resumed.hydrate_from_run_dir()
    set_global_report_state(resumed)
    callbacks: list[dict[str, Any]] = []
    resumed.vulnerability_updated_callback = lambda report: callbacks.append(dict(report))
    report_id = resumed.vulnerability_reports[0]["id"]
    assert commits["first"] in resumed.vulnerability_reports[0]["technical_analysis"]

    result = await _update(
        report_id,
        code_locations=[{"file": "app.py", "start_line": 1, "end_line": 2, "primary_line": 2}],
    )
    assert result["success"] is True
    analysis = resumed.vulnerability_reports[0]["technical_analysis"]
    assert commits["latest"] in analysis
    assert commits["first"] not in analysis
    result = await _update(report_id, technical_analysis="Revised root cause.\n\n" + analysis)
    assert result["success"] is True
    analysis = resumed.vulnerability_reports[0]["technical_analysis"]
    assert analysis.count("Last modified by") == 1
    assert analysis.startswith("Revised root cause.")

    result = await _update(
        report_id, code_locations=[{"file": "deleted.py", "start_line": 1, "end_line": 2}]
    )
    assert result["success"] is True
    final_report = resumed.vulnerability_reports[0]
    assert "Last modified by" not in final_report["technical_analysis"]
    assert callbacks[-1] == final_report
    saved = json.loads((resumed.get_run_dir() / "vulnerabilities.json").read_text())
    assert saved == [final_report]
    markdown = (resumed.get_run_dir() / "vulnerabilities" / f"{report_id}.md").read_text()
    assert "Last modified by" not in markdown


@pytest.mark.parametrize(
    "location",
    [
        {"file": "deleted.py", "start_line": 1, "end_line": 1},
        {"file": "app.py", "start_line": 999, "end_line": 999},
        {"file": "app.py", "start_line": True, "end_line": 1},
        {"file": "app.py", "start_line": 1.5, "end_line": 2},
        {"file": "app.py", "end_line": 1},
    ],
)
async def test_unavailable_history_does_not_block_reporting(
    history_run: tuple[ReportState, Path, dict[str, str]], location: dict[str, Any]
) -> None:
    state, _clone, _commits = history_run
    assert (await _create(location))["success"] is True
    assert state.vulnerability_reports[0]["technical_analysis"] == _ANALYSIS
    assert (state.get_run_dir() / "vulnerabilities.json").exists()


async def test_unexpected_history_failure_does_not_block_reporting(
    history_run: tuple[ReportState, Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _clone, _commits = history_run

    def broken_blame(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("history lookup failed")

    monkeypatch.setattr("strix.report.history.blame_line", broken_blame)
    assert (await _create({"file": "app.py", "start_line": 1, "end_line": 1}))["success"] is True
    assert state.vulnerability_reports[0]["technical_analysis"] == _ANALYSIS
    assert (state.get_run_dir() / "vulnerabilities" / "vuln-0001.md").exists()


@pytest.mark.parametrize("target_alias", [_TARGET, "application", "/workspace/application"])
def test_ambiguous_repository_requires_matching_target_and_target_updates_refresh_history(
    history_run: tuple[ReportState, Path, dict[str, str]], tmp_path: Path, target_alias: str
) -> None:
    state, clone, commits = history_run
    other = tmp_path / "other"
    _git(tmp_path, "clone", "--quiet", str(clone), str(other))
    (other / "app.py").write_text("query = 'other'\nresult = execute(query)\n", encoding="utf-8")
    _git(other, "commit", "--quiet", "-am", "Other repository change", author="Other Author")
    state.run_record["local_sources"].append(
        {"source_path": str(other), "workspace_subdir": "other"}
    )
    report = {
        "technical_analysis": _ANALYSIS,
        "code_locations": [{"file": "app.py", "start_line": 1, "end_line": 1}],
    }
    enrich_report(report, state.run_record)
    assert report["technical_analysis"] == _ANALYSIS

    report["target"] = target_alias
    enrich_report(report, state.run_record)
    assert commits["first"] in report["technical_analysis"]
    report["target"] = "other"
    enrich_report(report, state.run_record)
    assert "Other Author" in report["technical_analysis"]
    assert commits["first"] not in report["technical_analysis"]
    assert report["technical_analysis"].count("Last modified by") == 1


@pytest.mark.parametrize("target", [None, "https://example.test/query", "query handler"])
def test_unmatched_target_cannot_select_repository_by_unique_file(
    history_run: tuple[ReportState, Path, dict[str, str]], tmp_path: Path, target: str | None
) -> None:
    state, _clone, _commits = history_run
    other = tmp_path / "other"
    other.mkdir()
    state.run_record["local_sources"].append(
        {"source_path": str(other), "workspace_subdir": "other"}
    )
    report = {
        "target": target,
        "technical_analysis": _ANALYSIS,
        "code_locations": [{"file": "app.py", "start_line": 1, "end_line": 1}],
    }

    enrich_report(report, state.run_record)

    assert report["technical_analysis"] == _ANALYSIS


@pytest.mark.parametrize("target", ["application", "/workspace/application"])
def test_colliding_workspace_alias_cannot_select_repository_by_unique_file(
    history_run: tuple[ReportState, Path, dict[str, str]], tmp_path: Path, target: str
) -> None:
    state, _clone, _commits = history_run
    other = tmp_path / "other"
    other.mkdir()
    state.run_record["local_sources"].append(
        {"source_path": str(other), "workspace_subdir": "application"}
    )
    report = {
        "target": target,
        "technical_analysis": _ANALYSIS,
        "code_locations": [{"file": "app.py", "start_line": 1, "end_line": 1}],
    }

    enrich_report(report, state.run_record)

    assert report["technical_analysis"] == _ANALYSIS


def test_unmatched_target_can_use_only_configured_repository(
    history_run: tuple[ReportState, Path, dict[str, str]],
) -> None:
    state, _clone, commits = history_run
    report = {
        "target": "https://example.test/query",
        "technical_analysis": _ANALYSIS,
        "code_locations": [{"file": "app.py", "start_line": 1, "end_line": 1}],
    }

    enrich_report(report, state.run_record)

    assert commits["first"] in report["technical_analysis"]


@pytest.mark.parametrize("first_lookup_succeeds", [True, False])
async def test_history_budget_is_shared_and_partial_results_are_persisted(
    history_run: tuple[ReportState, Path, dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
    first_lookup_succeeds: bool,
) -> None:
    state, _clone, commits = history_run
    clock = [0.0]
    timeouts: list[float] = []
    info = BlameInfo(
        author_name=_FIRST_AUTHOR,
        author_email="author@example.test",
        commit_sha=commits["first"],
        commit_timestamp="2024-01-02T03:04:05+00:00",
        commit_summary="Add query handler",
    )

    def slow_blame(*_args: Any, timeout: float) -> BlameInfo | None:
        timeouts.append(timeout)
        clock[0] += min(1.0 if len(timeouts) == 1 else 3.0, timeout)
        return info if first_lookup_succeeds and len(timeouts) == 1 else None

    monkeypatch.setattr(history, "monotonic", lambda: clock[0])
    monkeypatch.setattr(history, "blame_line", slow_blame)

    result = await _create(
        {"file": "app.py", "start_line": 1, "end_line": 1},
        {"file": "app.py", "start_line": 2, "end_line": 2},
        {"file": "app.py", "start_line": 999, "end_line": 999},
    )

    assert result["success"] is True
    assert timeouts == pytest.approx([3.0, 2.0])
    assert clock[0] == pytest.approx(3.0)
    report = state.vulnerability_reports[0]
    analysis = report["technical_analysis"]
    if first_lookup_succeeds:
        assert commits["first"] in analysis
        assert "**app.py:1**" in analysis
        assert "**app.py:2**" not in analysis
    else:
        assert analysis == _ANALYSIS
    saved = json.loads((state.get_run_dir() / "vulnerabilities.json").read_text())
    assert saved == [report]
    markdown = (state.get_run_dir() / "vulnerabilities" / f"{report['id']}.md").read_text()
    assert analysis in markdown


def test_invalid_locations_still_consume_shared_history_budget(
    history_run: tuple[ReportState, Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    state, _clone, _commits = history_run
    clock = [0.0]
    location_line = history._location_line

    def slow_location_line(location: dict[str, Any]) -> int | None:
        clock[0] += 0.5
        return location_line(location)

    monkeypatch.setattr(history, "monotonic", lambda: clock[0])
    monkeypatch.setattr(history, "_location_line", slow_location_line)
    report = {
        "target": _TARGET,
        "technical_analysis": _ANALYSIS,
        "code_locations": [{"file": "app.py"} for _ in range(20)],
    }

    enrich_report(report, state.run_record)

    assert clock[0] <= 3.0
    assert report["technical_analysis"] == _ANALYSIS
