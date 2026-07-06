---
name: molab-launch-session
description: >-
  Launch a new molab-hosted marimo notebook session (or resume an existing
  one), attach a GPU, and extract the session URL + auth token needed for
  marimo-pair/sync-molab-notebook. Use when asked to start/spin up/create a
  molab session, get GPU compute for the notebook, or obtain a session
  URL/token to pair with.
allowed-tools: Bash(uv run --group molab python **/scripts/launch-session.py *), Read
---

molab (https://molab.marimo.io) has no REST API for creating a session or
attaching a GPU to it -- both are UI-only actions in the molab web app. This
skill drives that UI with Playwright to do what a human would do by hand:
create or open a notebook, attach a GPU, open "Pair with an agent", and
scrape the session URL + auth token out of that panel. It then does a
best-effort connectivity check and writes the result to a JSON file.

**This skill's job ends once connectivity is confirmed.** It does not sync
notebook cells or execute code -- that's `sync-molab-notebook` (pull/push)
and the `marimo-pair` plugin (execute code in a live kernel), both of which
consume the URL/token this skill produces.

## Prerequisites

- `uv run --group molab ...` (Playwright is in the `molab` dependency
  group, kept out of the main GPU-training dependencies).
- Playwright needs a Chromium binary. If one is pre-installed (e.g. this
  environment's `/opt/pw-browsers/chromium`), the script finds it
  automatically via `--chromium-path`'s default. Otherwise run once:
  `uv run --group molab playwright install chromium`.
- Whether molab requires a logged-in account to create/edit notebooks or
  attach a GPU is undocumented. If the script reports a login wall, rerun
  once with `--headed --user-data-dir <path>` to log in interactively by
  hand, then reuse that same `--user-data-dir` headlessly afterward.

## Usage

Create a new GPU-attached notebook:

```bash
uv run --group molab python .claude/skills/molab-launch-session/scripts/launch-session.py \
  --source new --gpu --output .molab/session.json
```

Resume/retry against an already-open notebook (skips creation entirely --
useful if GPU-attach succeeded but a later step failed):

```bash
uv run --group molab python .claude/skills/molab-launch-session/scripts/launch-session.py \
  --resume-url 'https://molab.marimo.io/notebooks/XXXX' --output .molab/session.json
```

Sample output:

```
STEP: launch browser
STEP: open notebook (source=new)
STEP: attach GPU
STEP: open pairing info
STEP: extract url/token
STEP: check connectivity

url:          https://sb-XXXX.sb.molab.run/
token:        ******** (written to .molab/session.json)
gpu_attached: true
connectivity: ok
Wrote /home/user/3DGS-Anchor/.molab/session.json
```

Next step -- hand off to the existing sync skill:

```bash
uv run python .claude/skills/sync-molab-notebook/scripts/sync-notebook.py \
  --url "$(python3 -c 'import json;print(json.load(open(".molab/session.json"))["url"])')" \
  --token "$(python3 -c 'import json;print(json.load(open(".molab/session.json"))["token"])')"
```

## Output format

`--output` (default `.molab/session.json`, written atomically with `0600`
permissions since it holds a live bearer credential):

```json
{
  "url": "https://sb-XXXX.sb.molab.run/",
  "token": "...",
  "created_at": "2026-07-06T12:00:00+00:00",
  "gpu_attached": true,
  "connectivity": "ok",
  "source": "new"
}
```

`connectivity` is one of:
- `"ok"` -- code actually executed in the live kernel via `execute-code.sh`.
- `"partial_http_only"` -- `execute-code.sh` wasn't found, so only a plain
  HTTP reachability check against the URL was done. This does NOT confirm
  the kernel is actually responsive, only that the host answers.
- `"failed"` -- neither check succeeded (the script exits non-zero in this case).
- `"skipped"` -- `--skip-connectivity-check` was passed; nothing was verified.

## Caveats / Troubleshooting

- **Selectors are best-effort, uncalibrated against the live DOM.** No
  ground-truth HTML of molab's actual notebook page was available while
  writing this script -- button locations (the notebook-specs button in
  the header, the actions panel in the upper-right corner, "Pair with an
  agent") are inferred from marimo's own documentation/marketing copy, not
  from inspecting the real page. **Treat the first real run as a
  calibration pass**: if a step fails, it saves a screenshot + HTML dump to
  `--debug-dir` (default `.molab/debug`) and fails loudly (never a silent
  no-op) -- inspect those artifacts and adjust the selector constants near
  the top of `scripts/launch-session.py`.
- Login-wall detection and GPU-attach-state detection are both best-effort
  text/role heuristics, not confirmed against the real UI either.
- If `execute-code.sh` (from the `marimo-pair` plugin) isn't installed or
  discoverable, connectivity checks degrade to `partial_http_only` --
  install/enable that plugin for a real kernel-level check.
- A much older `--output` file is ignored automatically (see
  `--max-age-minutes`); pass `--resume-url` explicitly if you want to
  reconnect to a specific known session regardless of file age.
