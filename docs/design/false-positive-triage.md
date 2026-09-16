# Local false-positive triage

Design and implementation record, 2026-09-16. Implemented for the local terminal UI and web viewer. This change adds no standalone triage CLI command or Cloud behavior.

**User experience**

The finding detail offers **False positive (f)** in the terminal UI and **Mark as false positive** in the web viewer. Users can close a finding with an optional reason and local note, see **Closed · False positive**, and use **Undo** or **Reopen**. Closing works offline and with telemetry disabled. The original finding is preserved, with the human decision stored separately.

When telemetry is enabled, a small classification event follows a successfully saved decision. Written explanations and finding content stay local. The telemetry policy and feature docs describe this collection; the terminal UI and web viewer contain no telemetry copy, indicators, prompts or toggles. Detailed examples for improving detection belong in a separate, explicit sharing flow.

**Findings from the initial investigation**

| Area | Observed behavior | Implication |
| --- | --- | --- |
| Local viewer | The finding detail renders a status badge and banners, but offers no mutation. `parseOneVulnerability` hardcodes `open` and discards status metadata. | Adding a button alone will not persist a decision. Both the data path and UI need changes. |
| Finding storage | `ReportState` owns the findings list, assigns run-local sequential IDs, and rewrites scan artifacts from memory. | Editing `vulnerabilities.json` from the viewer can be overwritten by a live scan or resume. Deleting findings can also undermine ID allocation and deduplication. |
| Terminal UI | Its finding collection comes directly from `ReportState`, not the viewer's disk reads. | Native close/reopen commands and the collection projection must use the shared triage service. |
| Telemetry | Optional, enabled by default, configured through environment/saved settings. Identity is a random UUID per process. Existing finding events contain severity, CWE and CVE presence. | Respect the effective setting in the backend. There is no existing persistent user/finding identity for longitudinal feedback joins. |
| General feedback | The viewer's support form sends a message and email to a remote relay. | It is unsuitable for closing a local issue: closure should require neither email nor a network call. |

Code anchors: [viewer parser](../../strix/interface/viewer/frontend/src/lib/local-run-parser.ts), [finding detail](../../strix/interface/viewer/frontend/src/components/vulnerability/VulnerabilityDetail.tsx), [report state](../../strix/report/state.py), [terminal collection](../../strix/interface/tui/backend/controller.py), [telemetry policy](../../strix/telemetry/README.md), [feedback form](../../strix/interface/viewer/frontend/src/components/FeedbackView.tsx).

**The user flow**

1. Open an issue in the terminal UI or web viewer and choose **Mark as false positive**. The terminal detail has a visible action and keyboard shortcut; the viewer places the action beside the status.
2. Show a compact form with a reason selector and a note of up to 2,000 characters, rendered natively in the active interface. Primary action: **Close issue**. No required essay, email, account signup, additional confirmation, browser launch, or scan rerun.
3. On successful save, keep the detail open, show **Closed · False positive**, the date and local explanation, and offer **Reopen**. A brief **Undo** action restores the prior state.
4. The issue moves out of the active list. Offer **Open / Closed / All**, retaining the original severity and all evidence in the closed view.

Viewer form copy:

```text
Mark as false positive

This finding is incorrect or does not apply to your target.
Applies to this finding in this run.

Reason (optional)       [ Select a reason                     ]
Note (optional)         [ Add context for your future review  ]

[Cancel]                                      [Close issue]
```

Reason choices: **Incorrect assumption**, **Existing protection prevents the reported exploit**, **Code or dependency is not affected**, **Expected behavior, not a vulnerability**, and **Other**. Omission becomes `unspecified`. These are user assessments. Inconclusive reproduction alone should not be offered as a false-positive reason; provide “Not sure? Keep open for review” as the form's cancel path. Accepted residual risk belongs under accepted risk, even when a mitigation exists.

The form, status banner, terminal footer and success messages contain no telemetry disclosure or controls. Keep that information in documentation. The backend still honors the effective telemetry setting and never enables it as a side effect of closing.

Display counts such as **8 open · 2 false positives · 10 found**. Never count false-positive closures as fixes or show “no vulnerabilities found” when findings were dismissed. If every issue is closed, use “No open findings. 2 marked as false positives.”

A save failure keeps the original state and preserves entered text for retry. An ambiguous response triggers a read of the saved state before another mutation. Keep focus and selection on the reviewed finding instead of making the page jump when it leaves the active list.

**Local status semantics**

| User meaning | Status | Structured resolution |
| --- | --- | --- |
| Awaiting action | `open` | none |
| Incorrect finding | `closed` | `false_positive` |

Use `resolution_reason: false_positive` and a separate optional `reason_code` for the form's category. Keep lifecycle status separate from the reason for closing. The first release offers this false-positive resolution and reopening; accepted-risk and fixed workflows are not required. Never infer false positives from another status or prose notes. Reopening clears the current resolution but preserves history. A user label is not an independently verified scanner error.

**Persistence and integration**

A shared local triage service handles viewer mutations, native terminal commands and the terminal projection. It stores a versioned `triage.json` alongside the run artifacts, keyed by the real finding ID within that run and including the run identity in the document. Legacy runs without this file remain open. Mutations of missing, duplicate or synthetic finding IDs are rejected.

Each decision records current status, resolution reason, reason code, optional note, timestamp, revision, and an audit history of transitions. The actor label is `local_operator`; do not collect an OS username or imply a verified human identity. Record the reviewed finding revision/digest locally so material evidence changes can be detected. Local digests must never enter telemetry.

Protect read-modify-write with a stable per-run cross-process lock and a thread lock, write a temporary file, then atomically replace it. Lock a separate lockfile: replacing the data file must not replace the inode being locked. Validate an expected revision to reject stale edits. If a platform cannot provide the required locking, fail the mutation clearly rather than silently proceeding without it.

The scanner continues to own its original reports, IDs and dedupe inputs. Human triage is not an agent-editable reporting field. Compose the latest triage into reads instead of copying it into the agent's mutable finding dictionary. Closing cannot cause resume or dedupe to re-report the same finding as a new issue.

The viewer uses `POST /api/vulnerabilities/{id}/triage?run=<run>` with both the expected triage revision and the digest of the finding the user reviewed, returning the saved record. A changed finding while the form was open must require fresh review. Reuse the process session capability and run authorization, enforce same-origin browser writes, bound request/note sizes, and reject traversal or symlink escape for writable paths. A note is untrusted text in HTML, terminal and exported reports. Local closure of the launched run needs no email verification; preserve the existing access rule when selecting historical runs through the viewer.

Update the parser, detail controls, active/closed filtering and every displayed count. The terminal backend must read the same overlay and expose native close/reopen commands through its existing command protocol. The terminal finding detail gets a keyboard-accessible action, an optional reason/note form, save errors, Undo and Reopen. Terminal badges, counts, filters and copied reports reflect the same saved decisions. This flow works entirely inside the terminal; **Ctrl+O** remains an optional way to view the run in a browser.

Terminal controls: **F2** opens findings, including on narrow terminals where the sidebar is hidden. **f** opens the false-positive form, **r** reopens a closed finding, and **u** undoes a recent change, scoped to the finding-detail modal. In the detail or focused findings panel, **v** cycles **Open / Closed / All** so closed findings remain reachable. If the current filter is empty, **F2** opens All. Show focusable buttons as well as these shortcuts; preserve **c** to copy, arrow navigation, **Tab / Shift+Tab** to move button focus, **Enter** to activate and **Esc** to go back. Use a separate form state for reason/note input so typing `f`, `r` or `v` cannot trigger a mutation, and preserve the scan composer's draft. See [native triage controls](../../strix/interface/tui/internal/app/triage.go).

The `vulnerability.triage` command in the [terminal backend dispatcher](../../strix/interface/tui/backend/controller.py) handles the native controls. Its request carries the real finding ID, desired status, resolution reason, optional reason/note, expected triage revision and reviewed finding digest. The run is resolved from the terminal session. Saved state returns through the existing request/result protocol, and the collection refreshes only after success. Pending/error state is bound to the finding ID, not its list index. Closing or reopening never sends a chat message to the pentesting agent.

The web viewer refreshes on mutation, window focus and run switch, and checks a small file revision token to pick up terminal or other-tab changes, including for finished runs. The terminal's sidecar revision watcher notifies it of viewer edits even when the scanner is idle. Older in-flight updates cannot overwrite newer successful mutations. UI updates remain scoped to run ID and finding ID when users switch runs mid-request. Read-only run directories expose an unavailable write capability.

**Scope, later evidence and reports**

V1 dismissal applies to one finding in one run and survives restarts/resume. A new scan may rediscover it. Do not silently suppress another finding by title, CWE or file alone. The existing SARIF fingerprints are useful matching inputs, but they do not establish safe cross-scan identity for changing AI-generated findings.

If an agent materially changes the evidence or affected location under an already-reviewed ID, preserve the saved human decision in `triage_status` but project `review_stale: true` and `status: open`. Show **Needs review** in the Open filter and active counts, excluding it from closed counts. Reclosing requires a fresh reviewed digest and creates a new review even if its status/reason are unchanged. This derived staleness is not a human reopen event. Presentation-only edits should not invalidate review. Compare structured evidence/claim inputs, not just the title.

The original scan report and artifacts remain available with an explicit “Original scan report” label. Current triage counts appear beside that report so its original narrative is not mistaken for today's open issue count. The emailed PDF's send/download action, filename and document cover identify it as the original scan report, say subsequent triage is excluded, and label counts as detected findings. The current PDF reads the original findings directly ([PDF generation](../../strix/interface/viewer/report_pdf.py)).

A later curated export should include open findings and an appendix of closed findings/reasons with a triage-as-of timestamp. Freeform notes should be included only when the user selects that export option. Generate exports from one consistent triage revision.

Preserve the original `findings.sarif`. A separately named triaged SARIF export can represent external suppressions where supported, or offer an explicitly selected active-only export. Do not claim that SARIF suppression metadata closes existing GitHub alerts: consumer support and native alert updates need separate verification. Do not retroactively alter a scan's exit code, completion status or coverage assessment after human triage.

**Telemetry contract**

Use PostHog for the richer event through the existing gate. There is no need to duplicate a new classification event across both analytics vendors in v1. Emit from the triage service after a successful, real transition, not from a click handler or the generic `/api/event` endpoint. The latter is currently unauthenticated and cannot prove that a decision was saved; the authenticated successful-feedback handler is a better precedent. See [server events](../../strix/interface/viewer/server.py) and [feedback completion](../../strix/interface/viewer/server.py).

Classification event:

```json
{
  "event": "finding_triage_changed",
  "properties": {
    "schema_version": 1,
    "surface": "viewer",
    "previous_status": "open",
    "new_status": "closed",
    "previous_resolution_reason": null,
    "resolution_reason": "false_positive",
    "reason_code": "incorrect_assumption",
    "severity": "high",
    "cwe": "cwe-79",
    "is_cve": false,
    "scan_mode": "standard"
  }
}
```

Reuse current common app-version/OS properties and the process-only anonymous session identifier. `surface` is `viewer` or `tui`. Validate every category against a closed vocabulary, and validate CWE syntax. Derive classification attributes from persisted data on the backend. Do not forward arbitrary interface fields or arbitrary strings from findings. Missing legacy metadata is `unknown` or omitted.

Exclude notes, titles, descriptions, PoCs, evidence, source code, email, targets, paths, URLs, run names, finding IDs and target-derived hashes. Do not attach the viewer's current model as if it produced an old scan. Model/version comparisons require reliable original-scan provenance and an approved bounded model identifier; leave that dimension out until it exists.

Check the effective telemetry setting before enqueueing and before sending. The sender uses a bounded in-memory background queue; closure should complete without waiting for network timeouts. Delivery may be dropped when the process exits or the queue is full. Failed delivery never reverses the local decision. Do not persist an analytics backlog, replay old triage when telemetry is enabled later, or emit events for read-only loads, failed writes, conflicts and repeated no-op submissions. Notes-only edits need no analytics event. Human reopen, reason-category changes and fresh review of changed evidence are real classification transitions.

Settings are memoized and resolve environment → saved configuration → defaults. Keep the existing controls; add no UI toggle, setting indicator or telemetry copy. The [telemetry policy](../../strix/telemetry/README.md) and local feature documentation describe the event fields, exclusions, best-effort delivery and `STRIX_TELEMETRY=0` opt-out. See [settings loader](../../strix/config/loader.py), [PostHog gate](../../strix/telemetry/posthog.py), and [identity](../../strix/telemetry/_common.py).

This yields useful **user-reported false-positive trends** by reason, severity, CWE and Strix version. It does not yield a defensible scanner false-positive rate: later viewer sessions have different identities, no durable finding correlation exists, users self-select what to review, and undo/reopen actions can repeat. Start with transition counts and reason distributions, labeled honestly. A measured precision/FP rate needs a reviewed sample with a denominator and independent validation.

If detailed examples become necessary, add **Share this finding with Strix** as a separate optional action with a concrete payload preview and deliberate submission. Do not reinterpret enabled basic telemetry as permission to upload content. No backend for this detailed triage submission was verified in this research.

**Why this interaction**

The recommendation follows established review workflows while keeping Strix's run scope explicit. GitHub offers dismissal reasons, optional comments, a closed list and reopening; its API separates state from reason. [GitHub alert management](https://docs.github.com/en/code-security/how-tos/manage-security-alerts/manage-code-scanning-alerts/resolve-alerts), [GitHub alert update API](https://docs.github.com/en/rest/code-scanning/code-scanning#update-a-code-scanning-alert).

Semgrep distinguishes false positives, accepted risk and deferral, and keeps fixed separate. Snyk explicitly names repository-wide ignore scope and supports undoing it. Those are good precedents for precise labels and visible scope, but cross-scan persistence requires identity support Strix has not yet established. [Semgrep triage](https://semgrep.dev/docs/for-developers/resolve-findings-through-app), [Snyk consistent ignores](https://docs.snyk.io/manage-risk/prioritize-issues-for-fixing/ignore-issues/consistent-ignores-for-snyk-code).

Semgrep's metrics documentation also separates counts from source content and warns that deterministic hashes can be pseudonymous. That supports a minimal metadata event rather than uploading the dismissed report. [Semgrep metrics](https://docs.semgrep.dev/metrics).

**Delivery and acceptance**

The implementation includes the local triage service, native terminal close/reopen flow, viewer flow, terminal status projection, original-report labeling (including the emailed PDF), consent-gated metadata event and documentation. Decisions are durable and reversible across both local interfaces. Curated exports, bulk triage, detailed sharing and cross-scan carryover can follow as separate changes.

Acceptance checks:

- Close, reload, restart viewer, resume scan and reopen all preserve evidence, IDs, decisions and audit history.
- A live scan artifact rewrite cannot erase triage; simultaneous terminal and viewer edits either serialize or return a conflict without losing decisions.
- Close and reopen directly inside the terminal, with optional reason/note input, without starting a web viewer. The web viewer reads those same decisions, and viewer changes appear in the terminal.
- Active/closed counts, the selected detail, terminal output and report labels agree. A materially changed finding becomes active for review.
- Unknown findings, duplicate IDs, stale revisions, malformed sidecars, unauthorized/cross-origin requests, path escapes, read-only directories and interrupted writes cannot produce a successful closure.
- Telemetry disabled through either environment or saved settings causes zero analytics HTTP calls while closure and reopening still succeed.
- Sentinel secrets in every freeform field never appear in event payloads; malformed categories cannot bypass allowlisting.
- Telemetry failure cannot delay/fail closure; failed/no-op mutations emit nothing; re-enabling telemetry never replays prior disabled actions.
- Both interfaces omit telemetry copy, prompts and indicators. Documentation describes the collection and opt-out accurately.
- Keyboard/focus behavior, small screens, errors and live-poll/run-switch races work. Build/type-check the viewer and run the relevant Python/Go tests plus repository checks.

Implementation lives in the [shared triage service](../../strix/report/triage.py), [atomic store](../../strix/report/triage_store.py), [native terminal controls](../../strix/interface/tui/internal/app/triage.go), [viewer controls](../../strix/interface/viewer/frontend/src/components/vulnerability/TriageControls.tsx), and [classification telemetry](../../strix/telemetry/triage.py). Tests cover local persistence and real scanner resume, simultaneous processes, invalid storage and failed writes, authenticated viewer and native terminal mutations, and the telemetry network boundary. Synthetic browser and real 80×24 PTY workflows were exercised; original PDF pages were rendered and visually checked. No pentest or cloud account mutation was required.
