"""Optional repository history inside the existing technical analysis text."""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from time import monotonic
from typing import Any

from strix.core.repository_history import blame_line


logger = logging.getLogger(__name__)
_START = "<!-- strix:repository-history -->"
_END = "<!-- /strix:repository-history -->"
_ENRICHMENT_TIMEOUT = 3.0


def _text(value: str) -> str:
    """Keep repository-authored metadata inert in Markdown and HTML renderers."""
    return re.sub(r"([\\`*_\[\]])", r"\\\1", html.escape(" ".join(value.split())))


def _repositories(run_record: dict[str, Any], target: str | None) -> list[Path]:
    sources: list[dict[str, Any]] = run_record.get("local_sources") or []
    roots = {Path(s["source_path"]).resolve() for s in sources if s.get("source_path")}
    matched: set[Path] = set()
    for source in sources:
        if target and target in (
            source.get("source_path"),
            source.get("workspace_subdir"),
            f"/workspace/{source.get('workspace_subdir')}",
        ):
            matched.add(Path(source["source_path"]).resolve())
    targets: list[dict[str, Any]] = run_record.get("targets_info") or []
    for entry in targets:
        details: dict[str, Any] = entry.get("details") or {}
        path = details.get("cloned_repo_path") or details.get("target_path")
        if entry.get("type") not in {"repository", "local_code"} or not path:
            continue
        root = Path(path).resolve()
        roots.add(root)
        if target and target in (
            entry.get("original"),
            details.get("target_repo"),
            details.get("target_path"),
            details.get("workspace_subdir"),
            f"/workspace/{details.get('workspace_subdir')}",
        ):
            matched.add(root)
    # File existence cannot identify the repository for an unmatched target.
    # Fall back only when the scan itself has exactly one possible checkout.
    selected = matched or roots
    return sorted(selected) if len(selected) == 1 else []


def _location_line(location: dict[str, Any]) -> int | None:
    start = location.get("start_line")
    if type(start) is not int or start < 1:
        return None
    primary = location.get("primary_line")
    end = location.get("end_line", start)
    if type(primary) is int and type(end) is int and start <= primary <= end:
        return primary
    return start


def enrich_report(report: dict[str, Any], run_record: dict[str, Any]) -> None:
    """Refresh blame for the current locations, never failing a report operation.

    No new checkout is made. Multiple repositories require a uniquely matching
    target. A shared time budget bounds all lookups in a report. Markers let
    revisions and resumed scans replace only the generated part of the analysis.
    """
    try:
        deadline = monotonic() + _ENRICHMENT_TIMEOUT
        analysis = re.sub(
            re.escape(_START) + r".*?" + re.escape(_END),
            "",
            report.get("technical_analysis") or "",
            flags=re.DOTALL,
        ).rstrip()
        if "technical_analysis" in report:
            report["technical_analysis"] = analysis
        locations: list[dict[str, Any]] = report.get("code_locations") or []
        if not locations:
            return
        roots = _repositories(run_record, report.get("target"))
        if not roots:
            return
        entries: list[str] = []
        seen: set[tuple[str, int]] = set()
        for location in locations:
            if monotonic() >= deadline:
                break
            file_path = location.get("file")
            line = _location_line(location)
            if not isinstance(file_path, str) or line is None or (file_path, line) in seen:
                continue
            seen.add((file_path, line))
            candidates = [root for root in roots if (root / file_path).is_file()]
            if len(candidates) != 1:
                continue
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            info = blame_line(candidates[0], file_path, line, timeout=remaining)
            if info is None:
                continue
            entries.append(
                f"**{_text(file_path)}:{line}**\n\n"
                f"- Author: {_text(info.author_name)} ({_text(info.author_email)})\n"
                f"- Commit: {_text(info.commit_sha)}\n"
                f"- Commit timestamp (UTC): {_text(info.commit_timestamp)}\n"
                f"- Commit summary: {_text(info.commit_summary)}"
            )
        if entries:
            block = "\n\n".join(entries)
            report["technical_analysis"] = (
                f"{analysis}\n\n{_START}\n### Last modified by\n\n{block}\n{_END}"
            ).lstrip()
    except Exception:  # noqa: BLE001 - history must never prevent issue creation or reporting.
        logger.debug("Repository history enrichment failed (non-fatal)", exc_info=True)
