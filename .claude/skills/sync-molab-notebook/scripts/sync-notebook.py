#!/usr/bin/env python3
"""Pull @app.cell changes from a live molab-hosted marimo session into a
git-tracked notebook file.

Fetches the live session's backing file from its own sandbox filesystem
(via marimo-pair's execute-code.sh), diffs it cell-by-cell against the
local file, and applies new/changed cell bodies -- never touching the
local file's header (imports, __generated_with, app = marimo.App(...))
or footer (if __name__ ... block), since those can carry sandbox-only
config that shouldn't leak into git.

Usage:
  sync-notebook.py --url URL [--token TOKEN] [--notebook path] [--dry-run]
"""
import argparse
import base64
import difflib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CELL_BOUNDARY_RE = re.compile(r"^@app\.cell")
FOOTER_START_RE = re.compile(r"^if __name__")
DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*\(")


@dataclass
class Cell:
    key: str
    name: str
    text: str


def parse_notebook(src: str) -> tuple[str, list[Cell], str]:
    lines = src.splitlines(keepends=True)
    n = len(lines)
    i = 0

    header_end = 0
    while header_end < n and not CELL_BOUNDARY_RE.match(lines[header_end]):
        header_end += 1
    header = "".join(lines[:header_end])
    i = header_end

    cells: list[Cell] = []
    anon_counter = 0
    while i < n and CELL_BOUNDARY_RE.match(lines[i]):
        start = i
        i += 1
        def_line = lines[i] if i < n else ""
        m = DEF_RE.match(def_line)
        name = m.group(1) if m else "_"
        i += 1
        while i < n and not CELL_BOUNDARY_RE.match(lines[i]) and not FOOTER_START_RE.match(lines[i]):
            i += 1
        text = "".join(lines[start:i])
        if name == "_":
            key = f"_pos{anon_counter}"
            anon_counter += 1
        else:
            key = name
        cells.append(Cell(key=key, name=name, text=text))

    footer = "".join(lines[i:])
    return header, cells, footer


def fetch_remote_source(execute_code_sh: Path, url: str, token: str | None, basename: str) -> str:
    snippet = f"""
import os, base64
target = {basename!r}
def find(match_fn):
    for root, dirs, files in os.walk(os.getcwd()):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__')]
        for f in files:
            if match_fn(f):
                return os.path.join(root, f)
    return None
path = find(lambda f: f == target) or find(lambda f: f.endswith('.py'))
if not path:
    print("SYNC_NOT_FOUND")
else:
    print("SYNC_PATH:" + path)
    print("SYNC_B64_BEGIN")
    print(base64.b64encode(open(path, 'rb').read()).decode())
    print("SYNC_B64_END")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(snippet)
        snippet_path = f.name

    cmd = ["bash", str(execute_code_sh), "--url", url]
    if token:
        cmd += ["--token", token]
    cmd += [snippet_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(
            "Failed to reach the live molab session. See "
            "marimo-pair/reference/execution-context.md for connection "
            "troubleshooting.",
            file=sys.stderr,
        )
        sys.exit(1)

    stdout = result.stdout
    if "SYNC_NOT_FOUND" in stdout:
        print(
            f"No file named {basename!r} (or any .py file) found on the "
            "live session's sandbox filesystem.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        begin = stdout.index("SYNC_B64_BEGIN") + len("SYNC_B64_BEGIN")
        end = stdout.index("SYNC_B64_END")
    except ValueError:
        print("Unexpected output from remote fetch:\n" + stdout, file=sys.stderr)
        sys.exit(1)

    b64_payload = stdout[begin:end].strip()
    return base64.b64decode(b64_payload).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="URL of the live molab/marimo session")
    parser.add_argument("--token", default=None, help="Auth token (falls back to MARIMO_TOKEN env)")
    parser.add_argument("--notebook", default=None, help="Local notebook path (default: <repo root>/notebook.py)")
    parser.add_argument(
        "--execute-code-sh",
        default=None,
        help="Path to marimo-pair's execute-code.sh (default: sibling skill's copy)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing the local file")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    execute_code_sh = (
        Path(args.execute_code_sh)
        if args.execute_code_sh
        else script_dir.parent.parent / "marimo-pair" / "scripts" / "execute-code.sh"
    )
    if not execute_code_sh.exists():
        print(f"execute-code.sh not found at {execute_code_sh}", file=sys.stderr)
        sys.exit(1)

    if args.notebook:
        notebook_path = Path(args.notebook).resolve()
    else:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
        notebook_path = Path(repo_root) / "notebook.py"

    if not notebook_path.exists():
        print(f"Local notebook not found at {notebook_path}", file=sys.stderr)
        sys.exit(1)

    local_src = notebook_path.read_text()
    remote_src = fetch_remote_source(execute_code_sh, args.url, args.token, notebook_path.name)

    local_header, local_cells, local_footer = parse_notebook(local_src)
    remote_header, remote_cells, remote_footer = parse_notebook(remote_src)

    local_by_key = {c.key: c for c in local_cells}
    remote_by_key = {c.key: c for c in remote_cells}

    new_keys = [c.key for c in remote_cells if c.key not in local_by_key]
    changed_keys = [
        c.key for c in remote_cells if c.key in local_by_key and local_by_key[c.key].text != c.text
    ]
    removed_keys = [c.key for c in local_cells if c.key not in remote_by_key]

    if not new_keys and not changed_keys:
        print("Nothing to apply -- local notebook already matches the live session's cells.")
    else:
        out_parts = []
        for c in local_cells:
            text = remote_by_key[c.key].text if c.key in changed_keys else c.text
            out_parts.append(text)
        for c in remote_cells:
            if c.key in new_keys:
                out_parts.append(c.text)

        new_src = local_header + "".join(out_parts) + local_footer

        for key in new_keys:
            print(f"+ NEW cell applied: {remote_by_key[key].name} ({key})")
        for key in changed_keys:
            print(f"~ CHANGED cell applied: {remote_by_key[key].name} ({key})")

        if args.dry_run:
            print("\n--dry-run: not writing local file.")
        else:
            notebook_path.write_text(new_src)
            print(f"\nWrote {notebook_path}")

    for key in removed_keys:
        print(
            f"! REMOVED_UPSTREAM: {local_by_key[key].name} ({key}) exists locally but not in the "
            "live session -- left in place, not deleted."
        )

    if local_header != remote_header or local_footer != remote_footer:
        print("\nNOT applied -- review this manually (sandbox-only config, e.g. App(...) kwargs, "
              "__generated_with):")
        diff = difflib.unified_diff(
            (local_header + local_footer).splitlines(),
            (remote_header + remote_footer).splitlines(),
            fromfile="local header+footer",
            tofile="remote header+footer",
            lineterm="",
        )
        print("\n".join(diff))


if __name__ == "__main__":
    main()
