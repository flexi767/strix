### Overview

To help make Strix better for everyone, we collect anonymized data that helps us understand how to better improve our AI security agent for our users, guide the addition of new features, and fix common errors and bugs. This feedback loop is crucial for improving Strix's capabilities and user experience.

We use [PostHog](https://posthog.com), an open-source analytics platform, for data collection and analysis, along with [Scarf](https://scarf.sh). Our telemetry implementation is fully transparent - you can review the source code ([posthog.py](https://github.com/usestrix/strix/blob/main/strix/telemetry/posthog.py), [scarf.py](https://github.com/usestrix/strix/blob/main/strix/telemetry/scarf.py)) to see exactly what we track.

### Telemetry Policy

Privacy is our priority. All collected data is anonymized by default. Each session gets a random UUID that is not persisted or tied to you. Your code, scan targets, vulnerability details, and findings always remain private and are never collected.

### What We Track

We collect only very **basic** usage data including:

**Session Errors:** Duration, the failure category, the scan phase, and the exception class name (not messages or stack traces)\
**System Context:** OS type, architecture, Strix version\
**Scan Context:** Scan mode (quick/standard/deep), scan type (whitebox/blackbox)\
**Model Usage:** Which LLM model is being used and whether it runs via an API key or a model subscription (not prompts or responses)\
**Feature Usage:** Which built-in skills were used during a scan (reported once, at scan end)\
**Aggregate Metrics:** Vulnerability counts by severity and weakness category (CWE)\
**Finding Triage:** Saved classification changes from the local web viewer and terminal UI, as described below

### False-Positive Triage

Closing a finding as a false positive and reopening it work locally whether telemetry is enabled or disabled. Classification changes include closing, reopening, a changed reason category, or a fresh review after evidence changes. When enabled, a successfully saved classification change can emit `finding_triage_changed` with:

- Source interface: `viewer` or `tui`
- Previous and new status and resolution, plus an optional predefined reason category (`reason_code`)
- Finding severity, weakness category (CWE), and whether a CVE is present
- Scan mode and the common Strix/Python version, OS, and architecture properties

The event uses the existing random process-only session identifier. It excludes notes, titles, descriptions, evidence, proof-of-concept scripts, code, targets, URLs, paths, email addresses, run names, finding IDs, and local finding digests. Freeform text is never included; categories are validated against a fixed vocabulary.

Events are emitted after the local decision is saved and sent asynchronously on a best-effort basis. Delivery may be dropped when the process exits or the in-memory queue is full. Delivery failure does not affect the saved decision. Failed changes, repeated no-op submissions, reading a run, and notes-only edits do not emit classification events. There is no persisted analytics backlog, and enabling telemetry later does not replay actions taken while it was disabled.

These events describe user-reported classification trends, not an independently verified scanner false-positive rate. They do not provide persistent user or finding tracking across processes.

### What We **Never** Collect

- Usernames, or any identifying information
- Scan targets, file paths, target URLs, or domains
- Vulnerability details, descriptions, code, or written triage notes
- Finding IDs, run names, or local finding digests
- LLM requests and responses

### How to Opt Out

Telemetry in Strix is entirely **optional**:

```bash
export STRIX_TELEMETRY=0
```

You can set this environment variable before running Strix to disable **all** telemetry.
