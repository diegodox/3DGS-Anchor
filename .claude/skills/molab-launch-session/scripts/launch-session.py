#!/usr/bin/env python3
"""Launch a molab-hosted marimo notebook session, attach a GPU, and extract
the session URL + auth token needed for marimo-pair / sync-molab-notebook.

molab (https://molab.marimo.io) has no API for session creation or GPU
attachment -- both are UI-only actions. This script drives the web UI with
Playwright to perform them, then scrapes the "Pair with an agent" panel for
the session URL/token and does a best-effort connectivity check.

CAVEAT: the selectors below are inferred from marimo's own documentation,
not from inspecting the live DOM (see SKILL.md's Caveats section). Treat
the first real run as a calibration pass.

Usage:
  launch-session.py --source new --gpu --output .molab/session.json
  launch-session.py --resume-url URL --output .molab/session.json
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeoutError
from playwright.sync_api import sync_playwright

MOLAB_BASE_URL = "https://molab.marimo.io/"
DEFAULT_CHROMIUM_PATH = "/opt/pw-browsers/chromium"

# --- Selector constants (best-effort; see SKILL.md Caveats) ---
NEW_NOTEBOOK_RE = re.compile(r"new notebook", re.I)
IMPORT_GITHUB_RE = re.compile(r"import|open from github", re.I)
IMPORT_SUBMIT_RE = re.compile(r"^(import|open)$", re.I)
SPECS_BUTTON_RE = re.compile(r"specs|resources", re.I)
GPU_OPTION_RE = re.compile(r"gpu|rtx", re.I)
GPU_ATTACHED_RE = re.compile(r"gpu attached|rtx pro 6000|96\s*gb", re.I)
ACTIONS_BUTTON_RE = re.compile(r"actions|more", re.I)
PAIR_MENU_ITEM_RE = re.compile(r"pair with an agent", re.I)
LOGIN_WALL_RE = re.compile(r"sign in|log in|continue with", re.I)

SESSION_URL_RE = re.compile(r"https://sb-[\w.-]+\.sb\.molab\.run/?")
SESSION_URL_FALLBACK_RE = re.compile(r"https://\S+molab\.run\S*")
TOKEN_PATTERNS = [
    re.compile(r"--token\s+(\S+)"),
    re.compile(r"MARIMO_TOKEN[=\s]+(\S+)"),
    re.compile(r"(?:auth )?token[:\s]+([A-Za-z0-9._-]{8,})", re.I),
]

SANITY_SNIPPET = 'print("MOLAB_LAUNCH_SESSION_OK")\n'
SANITY_MARKER = "MOLAB_LAUNCH_SESSION_OK"


class LaunchError(Exception):
    """Raised for any step failure; message names the step that failed."""


class LoginWallError(LaunchError):
    pass


class PairingInfoNotFound(LaunchError):
    def __init__(self, message, scraped_text):
        super().__init__(f"{message}\n\n--- scraped panel text ---\n{scraped_text}")
        self.scraped_text = scraped_text


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["new", "github"], default="new")
    parser.add_argument("--repo-url", help="Required when --source github")
    parser.add_argument("--notebook-path", default="notebook.py")
    parser.add_argument("--start-url", help="Skip create/import, go straight to this URL")
    parser.add_argument("--resume-url", help="Reopen an already-existing notebook instead of creating one")
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", dest="gpu", action="store_true", default=True)
    gpu.add_argument("--no-gpu", dest="gpu", action="store_false")
    parser.add_argument(
        "--user-data-dir",
        default=str(Path.home() / ".cache" / "molab-launch-session" / "profile"),
        help="Persistent Chromium profile dir (cookies/login), kept outside the repo by default",
    )
    parser.add_argument("--chromium-path", default=None)
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--headless", dest="headless", action="store_true", default=True)
    headless.add_argument("--headed", dest="headless", action="store_false")
    parser.add_argument("--output", default=None, help="Default: <repo root>/.molab/session.json")
    parser.add_argument("--timeout", type=int, default=30000, help="Per-step timeout in ms")
    parser.add_argument("--max-age-minutes", type=int, default=60)
    parser.add_argument("--execute-code-sh", default=None)
    parser.add_argument("--skip-connectivity-check", action="store_true")
    parser.add_argument("--debug-dir", default=None, help="Default: <repo root>/.molab/debug")
    args = parser.parse_args()
    if args.source == "github" and not args.repo_url and not (args.resume_url or args.start_url):
        parser.error("--source github requires --repo-url")
    return args


def resolve_repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def finalize_paths(args, repo_root: Path):
    if args.output is None:
        args.output = str(repo_root / ".molab" / "session.json")
    if args.debug_dir is None:
        args.debug_dir = str(repo_root / ".molab" / "debug")


def resolve_chromium_executable(args):
    if args.chromium_path:
        return args.chromium_path
    if os.path.exists(DEFAULT_CHROMIUM_PATH):
        return DEFAULT_CHROMIUM_PATH
    return None


def discover_execute_code_sh(args, repo_root: Path):
    if args.execute_code_sh:
        return Path(args.execute_code_sh)
    env_path = os.environ.get("MARIMO_PAIR_EXECUTE_CODE_SH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    which = shutil.which("execute-code.sh")
    if which:
        return Path(which)
    sibling = repo_root / ".claude" / "skills" / "marimo-pair" / "scripts" / "execute-code.sh"
    if sibling.exists():
        return sibling
    for candidate in glob.glob(
        str(Path.home() / ".claude" / "plugins" / "**" / "marimo-pair" / "scripts" / "execute-code.sh"),
        recursive=True,
    ):
        return Path(candidate)
    return None


def check_connectivity(url: str, token: str | None, args, repo_root: Path) -> str:
    execute_code_sh = discover_execute_code_sh(args, repo_root)
    if execute_code_sh is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(SANITY_SNIPPET)
            snippet_path = f.name
        cmd = ["bash", str(execute_code_sh), "--url", url]
        if token:
            cmd += ["--token", token]
        cmd += [snippet_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(args.timeout / 1000, 30))
            if result.returncode == 0 and SANITY_MARKER in result.stdout:
                return "ok"
        except (subprocess.SubprocessError, OSError):
            pass
        finally:
            os.unlink(snippet_path)

    # Fall back: plain HTTP reachability only -- does not confirm the kernel
    # actually executes code, just that the host answers.
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=min(args.timeout / 1000, 15)) as resp:
            if 200 <= resp.status < 400:
                return "partial_http_only"
    except (urllib.error.URLError, ValueError, OSError):
        pass
    return "failed"


def maybe_reuse_existing_output(args, repo_root: Path):
    if args.resume_url or args.start_url:
        return None
    output_path = Path(args.output)
    if not output_path.exists():
        return None
    try:
        payload = json.loads(output_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    created_at = payload.get("created_at")
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    age_minutes = (datetime.now(timezone.utc) - created).total_seconds() / 60
    if age_minutes > args.max_age_minutes or payload.get("connectivity") != "ok":
        return None
    connectivity = check_connectivity(payload["url"], payload.get("token"), args, repo_root)
    if connectivity != "ok":
        return None
    payload["connectivity"] = connectivity
    return payload


def save_debug_artifacts(page, label: str, args) -> Path:
    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^\w.-]+", "-", label)
    try:
        page.screenshot(path=str(debug_dir / f"{ts}-{safe_label}.png"), full_page=True)
    except Exception:
        pass
    try:
        (debug_dir / f"{ts}-{safe_label}.html").write_text(page.content())
    except Exception:
        pass
    return debug_dir


def click_or_fail(page, locator, step_label: str, args):
    try:
        locator.click(timeout=args.timeout)
    except PWTimeoutError as exc:
        save_debug_artifacts(page, step_label, args)
        raise LaunchError(
            f"step '{step_label}' failed: expected element not found/clickable. "
            f"See --debug-dir ({args.debug_dir}) for a screenshot/HTML dump."
        ) from exc


def detect_login_wall(page) -> bool:
    try:
        content = page.content()
    except Exception:
        return False
    return bool(LOGIN_WALL_RE.search(content))


def open_notebook(page, args):
    if args.resume_url:
        page.goto(args.resume_url, timeout=args.timeout)
    elif args.start_url:
        page.goto(args.start_url, timeout=args.timeout)
    else:
        page.goto(MOLAB_BASE_URL, timeout=args.timeout)
        if detect_login_wall(page):
            raise LoginWallError(
                "molab appears to require login before creating/opening a notebook; "
                "rerun with --headed --user-data-dir <path> to log in interactively once, "
                "then reuse that same --user-data-dir headlessly."
            )
        if args.source == "new":
            click_or_fail(page, page.get_by_role("button", name=NEW_NOTEBOOK_RE), "open notebook (new)", args)
        else:
            click_or_fail(
                page,
                page.get_by_role("button", name=IMPORT_GITHUB_RE),
                "open notebook (github: open import dialog)",
                args,
            )
            try:
                page.get_by_role("textbox").first.fill(args.repo_url, timeout=args.timeout)
            except PWTimeoutError as exc:
                save_debug_artifacts(page, "open notebook (github: fill repo url)", args)
                raise LaunchError("could not find a repo URL input for GitHub import") from exc
            click_or_fail(
                page,
                page.get_by_role("button", name=IMPORT_SUBMIT_RE),
                "open notebook (github: submit import)",
                args,
            )
    try:
        page.wait_for_load_state("networkidle", timeout=args.timeout)
    except PWTimeoutError:
        pass  # best-effort; some notebook UIs keep long-lived connections open
    if detect_login_wall(page):
        raise LoginWallError(
            "molab appears to require login; rerun with --headed --user-data-dir <path> "
            "to log in interactively once, then reuse that same --user-data-dir headlessly."
        )


def attach_gpu(page, args) -> bool:
    if not args.gpu:
        return False
    if GPU_ATTACHED_RE.search(page.content()):
        return False  # already attached -- idempotent, don't re-toggle
    click_or_fail(page, page.get_by_role("button", name=SPECS_BUTTON_RE), "attach GPU (open specs panel)", args)
    click_or_fail(page, page.get_by_text(GPU_OPTION_RE).first, "attach GPU (select GPU option)", args)
    try:
        page.wait_for_selector(f"text=/{GPU_ATTACHED_RE.pattern}/i", timeout=args.timeout)
    except PWTimeoutError as exc:
        save_debug_artifacts(page, "attach GPU (confirm attached)", args)
        raise LaunchError(
            "clicked the GPU option but no attachment confirmation appeared -- "
            f"see --debug-dir ({args.debug_dir})"
        ) from exc
    return True


def open_pairing_info(page, args) -> str:
    click_or_fail(page, page.get_by_role("button", name=ACTIONS_BUTTON_RE), "open pairing info (actions panel)", args)
    click_or_fail(
        page, page.get_by_text(PAIR_MENU_ITEM_RE), "open pairing info (click 'Pair with an agent')", args
    )
    try:
        page.wait_for_timeout(500)
        return page.inner_text("body")
    except Exception as exc:
        save_debug_artifacts(page, "open pairing info (read panel)", args)
        raise LaunchError("could not read the 'Pair with an agent' panel content") from exc


def extract_url_and_token(text: str) -> tuple[str, str | None]:
    url_match = SESSION_URL_RE.search(text) or SESSION_URL_FALLBACK_RE.search(text)
    if not url_match:
        raise PairingInfoNotFound("no session URL found in the pairing panel text", text)
    token = None
    for pattern in TOKEN_PATTERNS:
        m = pattern.search(text)
        if m:
            token = m.group(1)
            break
    return url_match.group(0), token


def write_output(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise
    os.chmod(path, 0o600)


def main():
    args = parse_args()
    repo_root = resolve_repo_root()
    finalize_paths(args, repo_root)

    reused = maybe_reuse_existing_output(args, repo_root)
    if reused is not None:
        print(f"Reusing still-valid session from {args.output} (age < {args.max_age_minutes}m, connectivity ok)")
        print(json.dumps(reused, indent=2))
        return

    print("STEP: launch browser")
    with sync_playwright() as playwright:
        context = launch_context(playwright, args)
        try:
            page = context.pages[0] if context.pages else context.new_page()

            print(f"STEP: open notebook (source={args.source})")
            open_notebook(page, args)

            print("STEP: attach GPU" if args.gpu else "STEP: skip GPU attach (--no-gpu)")
            gpu_attached = attach_gpu(page, args)

            print("STEP: open pairing info")
            panel_text = open_pairing_info(page, args)

            print("STEP: extract url/token")
            url, token = extract_url_and_token(panel_text)

            if args.skip_connectivity_check:
                connectivity = "skipped"
            else:
                print("STEP: check connectivity")
                connectivity = check_connectivity(url, token, args, repo_root)
        finally:
            context.close()

    payload = {
        "url": url,
        "token": token,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_attached": gpu_attached,
        "connectivity": connectivity,
        "source": args.resume_url and "resume" or args.start_url and "start_url" or args.source,
    }
    write_output(Path(args.output), payload)

    print()
    print(f"url:          {url}")
    print(f"token:        {'(none)' if not token else '******** (written to ' + args.output + ')'}")
    print(f"gpu_attached: {gpu_attached}")
    print(f"connectivity: {connectivity}")
    print(f"Wrote {args.output}")

    if connectivity == "failed":
        sys.exit(1)


def launch_context(playwright, args):
    user_data_dir = Path(args.user_data_dir)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(user_data_dir=str(user_data_dir), headless=args.headless, timeout=args.timeout)
    executable_path = resolve_chromium_executable(args)
    if executable_path:
        kwargs["executable_path"] = executable_path
    return playwright.chromium.launch_persistent_context(**kwargs)


if __name__ == "__main__":
    try:
        main()
    except LaunchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
