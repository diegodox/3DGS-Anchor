---
name: sync-molab-notebook
description: >-
  Bidirectionally sync a live molab-hosted marimo session with the git repo:
  pull cell changes from molab into the git-tracked notebook.py, and push a
  local src-layout package's source files into the live sandbox so its cells
  can import it. Use when asked to sync, pull, or push notebook/package
  changes to or from molab, or to reconcile a live marimo pairing session
  with git before committing.
allowed-tools: Bash(bash **/scripts/execute-code.sh *), Bash(python3 **/scripts/sync-notebook.py *), Bash(uv run python **/scripts/sync-notebook.py *), Bash(python3 **/scripts/push-package.py *), Bash(uv run python **/scripts/push-package.py *), Read
---

A live molab-hosted marimo session runs in its own sandbox with its own
filesystem, separate from the local git checkout — editing cells live
(directly or via `marimo-pair`'s `cm` API) does not touch the repo, and a
local package (e.g. one extracted under `src/`) is invisible to the sandbox
unless explicitly pushed there. This skill has two drivers, one per
direction:

- `scripts/sync-notebook.py` (**pull**): fetches the live session's backing
  file, diffs it cell-by-cell against the local notebook, and applies
  new/changed `@app.cell` bodies into the local file automatically. A
  `with app.setup:` block (marimo's shared-imports construct, if present)
  is diffed and applied the same way, as its own pseudo-cell — it is not
  swallowed into the untouched header just because it sits above the first
  `@app.cell`. The driver deliberately leaves the file's actual header
  (`import marimo`, `__generated_with`, `app = marimo.App(...)`) and footer
  (`if __name__ == "__main__": app.run()`) untouched, since the live
  session's copy of those often carries sandbox-only config (e.g. a
  `css_file` path or `__generated_with` version that only makes sense
  inside that container) — any difference there is printed for manual
  review, never auto-applied.
- `scripts/push-package.py` (**push**): writes a local package's `.py` files
  into the live sandbox as a sibling directory next to its notebook file
  (e.g. `gs_anchor/` next to `/marimo/notebook.py`), so `import gs_anchor`
  resolves there via Python's default `sys.path[0]` — no packaging or
  install step needed inside the sandbox.

## Prerequisites

- Python 3 (project uses `uv`; `uv run python ...` works from repo root).
- The sibling `marimo-pair` skill must be present at
  `.claude/skills/marimo-pair/scripts/execute-code.sh` (this driver shells
  out to it for all live-session communication — see that skill for
  starting/discovering a session).
- The molab session URL, and its auth token if the session requires one
  (same as `marimo-pair`: prefer `MARIMO_TOKEN` env, or pass `--token`).

## Pull (agent path)

```bash
uv run python .claude/skills/sync-molab-notebook/scripts/sync-notebook.py \
  --url 'https://sb-XXXX.sb.molab.run/' \
  --token "$(cat /path/to/token.txt)"
```

Add `--dry-run` to preview without writing. `--notebook` overrides the
local file (defaults to `<repo root>/notebook.py`).

Sample output:

```
+ NEW cell applied: claude_status (claude_status)

Wrote /home/diego/src/github.com/diegodox/3DGS-Anchor/notebook.py

NOT applied -- review this manually (sandbox-only config, e.g. App(...) kwargs, __generated_with):
--- local header+footer
+++ remote header+footer
@@ -1,7 +1,11 @@
 import marimo

-__generated_with = "0.23.13"
-app = marimo.App(width="medium")
+__generated_with = "0.23.9"
+app = marimo.App(
+    width="medium",
+    css_file="/usr/local/_marimo/custom.css",
+    auto_download=["html"],
+)
```

`~ CHANGED cell applied: ...` lines mean an existing cell's body differed
and was replaced wholesale. `! REMOVED_UPSTREAM: ...` means a cell exists
locally but not in the live session — it is flagged, never deleted
automatically. A `+ NEW setup block applied: app.setup` or
`~ CHANGED setup block applied: app.setup` line means the live session's
`with app.setup:` block was new or different and got applied the same way.

## Push (agent path)

```bash
uv run python .claude/skills/sync-molab-notebook/scripts/push-package.py \
  --url 'https://sb-XXXX.sb.molab.run/' \
  --token "$(cat /path/to/token.txt)"
```

Add `--dry-run` to list what would be pushed without touching the network.
`--package-dir` overrides the local package (defaults to
`<repo root>/src/gs_anchor`). Every file is `ast.parse`'d locally first —
the script aborts before pushing anything if one has a syntax error.

Sample output:

```
PUSHED:/marimo/gs_anchor/__init__.py
PUSHED:/marimo/gs_anchor/gaussians.py
...
PUSH_BASE:/marimo
GS_ANCHOR_CACHED:False
```

If `GS_ANCHOR_CACHED:True` appears, the package was already imported in the
live kernel before this push — the script prints a warning that the kernel
must be restarted (or the imports cell deleted and re-created) before
dependent cells will see the new code. Do not assume a re-run alone picks
it up; see the cached-module gotcha below.

## After syncing (manual, confirmation-gated)

The driver only touches the notebook file — it never runs git commands.
Once it reports applied changes:

1. `git fetch origin && git status` — confirm local isn't behind/diverged
   from `origin/main` before committing.
2. `git diff notebook.py` — sanity-check the change is the expected cell
   diff, nothing unexpected.
3. Stage **only** the notebook file (`git add notebook.py`), not unrelated
   untracked files.
4. Commit.
5. **Ask the user for explicit confirmation before `git push`** — pushing
   is a shared action and should never be automated by this skill.

## Gotchas

- The live session's file lives on a filesystem separate from the local
  checkout (e.g. `/marimo/notebook.py` inside the sandbox) — you cannot
  just read/write the local file to affect the live kernel, or vice versa.
- `app = marimo.App(...)` kwargs and `__generated_with` seen in the live
  session are often sandbox-specific — this is why the driver never
  auto-applies header/footer differences.
- Cells named `_` (marimo's convention for outputs nobody references) are
  matched by their ordinal position among other `_`-named cells, not by
  name — reordering anonymous cells between local and remote can cause a
  false CHANGED/NEW match. Rename a cell if you need to track it reliably.
- A `with app.setup:` block is matched as a single whole-block pseudo-cell
  (key `__setup__`) — there's no per-statement diffing inside it. If only
  one import in a large setup block changed, the whole block still gets
  replaced wholesale, same as any other changed cell.
- A push does not refresh an already-imported package in the live kernel —
  Python caches modules on first import, so a kernel that already ran
  `import gs_anchor` keeps using the old code until restarted, even though
  the sandbox's files on disk are now updated. See the "Cached module
  availability" gotcha in `.claude/skills/marimo-pair/reference/gotchas.md`.

## Troubleshooting

- Connection errors (no server found, multiple sessions, auth failures):
  see `.claude/skills/marimo-pair/reference/execution-context.md` — this
  driver shells out to the exact same `execute-code.sh` and hits the same
  failure modes.
- `No file named 'notebook.py' (or any .py file) found on the live
  session's sandbox filesystem`: the session's cwd doesn't contain the
  expected file; pass `--notebook` to match the actual basename, or check
  the session is the right one via `--session`/`--port` on
  `execute-code.sh` conventions.
- A much larger diff than expected (many CHANGED cells at once): sanity
  check with `--dry-run` first — a stale or wrong session URL can produce
  a diff against an unrelated notebook.
