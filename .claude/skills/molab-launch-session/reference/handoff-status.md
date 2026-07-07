# Handoff status

`molab-launch-session` was written in a cloud sandbox whose outbound
network **blocks `molab.marimo.io` directly** (`curl` to it returns a
403/tunnel failure). No live HTML of molab's actual notebook UI was ever
obtained during development -- every selector in `scripts/launch-session.py`
is inferred only from marimo's own docs/marketing copy ("notebook specs
button in the app header", "actions panel in the upper right corner ->
Pair with an agent"). The script has **never been run against the real
molab UI**. What has been verified, in that sandbox:

- The script imports cleanly and its CLI surface works
  (`uv run --group molab python scripts/launch-session.py --help`).
- `extract_url_and_token()` (pure regex, no Playwright) correctly parses
  hand-written fixture strings shaped like a plausible "Pair with an
  agent" panel.
- `uv sync --group molab` resolves and installs `playwright` cleanly.

What has **not** been verified: whether any of the browser-driven steps
actually work against the live molab app. That requires an environment
with real network access to `molab.marimo.io` -- this is the point of
this handoff.

## Steps to calibrate and finish verification

1. `uv sync --group molab`
2. `uv run --group molab playwright install chromium` -- this machine
   won't have the cloud sandbox's pre-installed `/opt/pw-browsers/chromium`;
   `resolve_chromium_executable()` in `scripts/launch-session.py` already
   falls back to Playwright's own managed browser when that path doesn't
   exist, so no code change is needed for this, just the one-time install.
3. Run once headed, so you can watch it and intervene:
   ```bash
   uv run --group molab python .claude/skills/molab-launch-session/scripts/launch-session.py \
     --source new --headed --user-data-dir ~/.cache/molab-launch-session/profile
   ```
   If molab shows a login wall (undocumented whether notebook
   creation/GPU attach requires an account), log in by hand in the opened
   browser window; the same `--user-data-dir` reused on later (headless)
   runs keeps that session.
4. When a step fails, it's expected calibration work, not a regression --
   the script saves a screenshot + HTML dump to `--debug-dir` (default
   `.molab/debug`) on every failure. Open those, find the real element,
   and adjust the matching constant near the top of
   `scripts/launch-session.py`:
   - `NEW_NOTEBOOK_RE` / `IMPORT_GITHUB_RE` / `IMPORT_SUBMIT_RE` -- notebook creation/import
   - `SPECS_BUTTON_RE` / `GPU_OPTION_RE` / `GPU_ATTACHED_RE` -- GPU attach
   - `ACTIONS_BUTTON_RE` / `PAIR_MENU_ITEM_RE` -- pairing panel
   - `LOGIN_WALL_RE` -- login-wall detection
   - `SESSION_URL_RE` / `SESSION_URL_FALLBACK_RE` / `TOKEN_PATTERNS` -- extracting the URL/token from the pairing panel's text
5. Success criteria:
   - A `--headless` run produces `.molab/session.json` with
     `"connectivity": "ok"` (meaning `execute-code.sh` was found and a
     sanity snippet actually executed in the live kernel -- not just
     `partial_http_only`, which only means the host answered a plain
     HTTP request).
   - Running the existing sync skill against that URL/token reports a
     real diff, not a connection error:
     ```bash
     uv run python .claude/skills/sync-molab-notebook/scripts/sync-notebook.py \
       --dry-run --url "$(python3 -c 'import json;print(json.load(open(".molab/session.json"))["url"])')" \
       --token "$(python3 -c 'import json;print(json.load(open(".molab/session.json"))["token"])')"
     ```
6. Commit the selector fixes to this same branch
   (`claude/molab-notebook-gpu-script-iadl6t`) and push.

## One thing to know about this branch's origin

This branch was developed in a cloud sandbox where `git remote -v` shows
`origin` as a local proxy URL
(`http://local_proxy@127.0.0.1:<port>/git/diegodox/3DGS-Anchor`) -- that
URL only resolves inside that sandbox. Locally, the real remote is
`https://github.com/diegodox/3DGS-Anchor.git`.
