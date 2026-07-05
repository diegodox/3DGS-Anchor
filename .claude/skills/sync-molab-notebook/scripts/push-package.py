#!/usr/bin/env python3
"""Push a local Python package's source files into a live molab-hosted
marimo session's sandbox filesystem, as a sibling directory next to its
notebook file, so notebook cells can `import` it with zero extra install
step (relying on Python's default sys.path[0] = script's own directory).

Use this after extracting notebook logic into a local src/-layout package:
the live molab sandbox is a fully isolated single-file environment (it does
not have the rest of the git repo), so a package living only in the local
checkout is invisible to the live kernel until pushed here.

Usage:
  push-package.py --url URL [--token TOKEN] [--package-dir path] [--dry-run]
"""
import argparse
import ast
import base64
import subprocess
import sys
import tempfile
from pathlib import Path


def discover_package_files(package_dir: Path) -> list[Path]:
    files = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    for p in files:
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            print(f"SyntaxError in {p}: {e}", file=sys.stderr)
            sys.exit(1)
    return files


def push_files(
    execute_code_sh: Path, url: str, token: str | None, package_dir: Path, notebook_basename: str
) -> str:
    files = discover_package_files(package_dir)
    payload = {
        f"{package_dir.name}/{p.relative_to(package_dir)}": base64.b64encode(p.read_bytes()).decode()
        for p in files
    }

    snippet = f"""
import os, base64, sys

def find_dir(match_fn):
    for root, dirs, fs in os.walk(os.getcwd()):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__')]
        for f in fs:
            if match_fn(f):
                return root
    return None

base = find_dir(lambda f: f == {notebook_basename!r}) or os.getcwd()
files = {payload!r}
for relpath, b64 in files.items():
    full = os.path.join(base, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "wb").write(base64.b64decode(b64))
    print("PUSHED:" + full)
print("PUSH_BASE:" + base)
print("GS_ANCHOR_CACHED:" + str({package_dir.name!r} in sys.modules))
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

    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="URL of the live molab/marimo session")
    parser.add_argument("--token", default=None, help="Auth token (falls back to MARIMO_TOKEN env)")
    parser.add_argument(
        "--package-dir", default=None, help="Local package directory (default: <repo root>/src/gs_anchor)"
    )
    parser.add_argument(
        "--notebook-basename",
        default="notebook.py",
        help="Basename used to locate the sandbox directory to push alongside (default: notebook.py)",
    )
    parser.add_argument(
        "--execute-code-sh",
        default=None,
        help="Path to marimo-pair's execute-code.sh (default: sibling skill's copy)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be pushed, without pushing")
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

    if args.package_dir:
        package_dir = Path(args.package_dir).resolve()
    else:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
        package_dir = Path(repo_root) / "src" / "gs_anchor"

    if not package_dir.is_dir():
        print(f"Package directory not found at {package_dir}", file=sys.stderr)
        sys.exit(1)

    files = discover_package_files(package_dir)
    if args.dry_run:
        print(f"Would push {len(files)} files from {package_dir}:")
        for p in files:
            print(f"  {package_dir.name}/{p.relative_to(package_dir)}")
        return

    stdout = push_files(execute_code_sh, args.url, args.token, package_dir, args.notebook_basename)
    print(stdout, end="")

    if "GS_ANCHOR_CACHED:True" in stdout:
        print(
            "\nWARNING: this package was already imported in the live kernel before this push. "
            "Python caches modules, so the running kernel is still using the OLD code. "
            "Restart the kernel (or delete and re-create the imports cell) before re-running "
            "dependent cells -- do not rely on it picking up the new files automatically.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
