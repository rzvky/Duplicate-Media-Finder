"""Command line and guided (double-click) entry point."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import APP_NAME, HOMEPAGE, __version__
from .formats import OUTPUT_DIR_NAME, REVIEW_BIN_NAME

# ------------------------------------------------------------------ console helpers


def _console_setup():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


class Progress:
    LABELS = {
        "discover": "Looking for photos and videos",
        "read": "Reading photos",
        "verify": "Double-checking copies",
        "thumbs": "Making previews",
        "videos": "Reading videos",
        "vhash": "Comparing identical videos",
        "vsig": "Comparing videos",
        "vdense": "Double-checking videos",
    }

    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.t0 = time.time()
        self.phase = None
        self.last_pct = -1
        self.last_print = 0.0

    def __call__(self, phase: str, done: int, total: int):
        now = time.time()
        if phase != self.phase:
            if self.phase is not None and self.tty:
                print()
            self.phase, self.t0, self.last_pct = phase, now, -1
            if phase == "discover":
                print(f"{self.LABELS[phase]}...", flush=True)
                return
        if not total:
            return
        pct = int(done * 100 / total)
        if not self.tty:
            if pct >= self.last_pct + 10 or (done == total and self.last_pct != 100):
                self.last_pct = pct
                print(f"  {self.LABELS.get(phase, phase)}: {done:,}/{total:,} ({pct}%)", flush=True)
            return
        if now - self.last_print < 0.15 and done != total:
            return
        self.last_print = now
        el = now - self.t0
        rate = done / el if el > 0.5 else 0
        eta = ""
        if rate and done < total:
            left = (total - done) / rate
            eta = f", about {int(left)} s left" if left < 90 else f", about {int(left / 60) + 1} min left"
        bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
        print(f"\r  {self.LABELS.get(phase, phase)} [{bar}] {done:,}/{total:,}{eta}      ", end="", flush=True)
        if done == total:
            print()
            self.phase = None


def _fmt_size(n: float) -> str:
    from .grouping import fmt_size

    return fmt_size(n)


# ------------------------------------------------------------------ folders


def _clean_path(text: str) -> str:
    p = text.strip().strip('"').strip("'").strip()
    if p.startswith("file://"):
        from urllib.parse import unquote, urlparse

        p = unquote(urlparse(p).path)
    if not os.path.exists(p) and "\\ " in p:  # macOS Terminal drag & drop escapes spaces
        p = p.replace("\\ ", " ")
    return os.path.expanduser(p)


def fallback_out_dir(root: Path) -> Path:
    tag = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:8]
    return Path.home() / "DuplicateMediaFinder_results" / f"{root.name or 'drive'}-{tag}"


def resolve_out_dir(root: Path, out: Optional[str], create: bool) -> Path:
    if out:
        p = Path(_clean_path(out)).resolve()
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p
    primary = root / OUTPUT_DIR_NAME
    fb = fallback_out_dir(root)
    if not create:
        return primary if primary.is_dir() or not fb.is_dir() else fb
    try:
        primary.mkdir(exist_ok=True)
        probe = primary / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return primary
    except OSError:
        fb.mkdir(parents=True, exist_ok=True)
        print(f"  Note: cannot write inside the photo folder, saving results to {fb}")
        return fb


def pick_folder() -> Optional[str]:
    """Open the system's folder picker; fall back to typing/dragging a path."""
    path = ""
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'POSIX path of (choose folder with prompt "Choose the folder with your photos and videos")',
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            path = r.stdout.strip()
        except Exception:
            path = ""
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog

            tk_root = tk.Tk()
            tk_root.withdraw()
            try:
                tk_root.attributes("-topmost", True)
            except Exception:
                pass
            path = filedialog.askdirectory(title="Choose the folder with your photos and videos", mustexist=True) or ""
            tk_root.destroy()
        except Exception:
            path = ""
    if not path:
        try:
            typed = input("Drag your photo or video folder into this window (or type its path) and press Enter:\n> ")
        except EOFError:
            return None
        path = _clean_path(typed)
    return path or None


# ------------------------------------------------------------------ commands


def run_scan(
    folder: str,
    out: Optional[str] = None,
    sensitivity: str = "normal",
    use_ai: bool = True,
    workers: Optional[int] = None,
    include_hidden: bool = False,
    quiet: bool = False,
    media: str = "all",
):
    from . import report
    from .grouping import PRESETS, build_groups
    from .scanner import default_workers, scan
    from .verify import make_verifier
    from .video import build_video_groups

    root = Path(_clean_path(folder)).resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")
    out_dir = resolve_out_dir(root, out, create=True)
    settings = PRESETS[sensitivity]
    workers = workers or default_workers()
    prog = (lambda *a: None) if quiet else Progress()
    if not quiet:
        print(f"\nScanning: {root}")
        print(f"  Sensitivity: {sensitivity}. Using {workers} worker{'s' if workers != 1 else ''}.")
    res = scan(root, out_dir, use_ai=use_ai, workers=workers, include_hidden=include_hidden, progress=prog, media=media)
    if not quiet:
        if res.records or media != "videos":
            print(f"  {res.ai_message}.")
        print("Grouping duplicates...", flush=True)
    verifier = make_verifier(workers, out_dir / "cache.sqlite", prog)
    groups = build_groups(res.records, settings, res.ai_used, verifier=verifier)
    groups += build_video_groups(res.videos, sensitivity)
    groups.sort(key=lambda g: (-g["savings"], -len(g["files"]), g["files"][0]["rel"].lower()))
    for gi, g in enumerate(groups, 1):
        g["id"] = gi
    results = report.build_results(root, res, groups, sensitivity)
    results["out_dir"] = str(out_dir)
    results["bin_dir"] = str(out_dir / REVIEW_BIN_NAME)
    report.make_thumbnails(root, out_dir, groups, workers, prog)
    report.save_all(out_dir, results)
    if not quiet:
        st = results["stats"]
        what = f"{st['photos_read']:,} photos"
        if st.get("videos_found"):
            what += f" and {st['videos_read']:,} videos"
        cached = st["from_cache"] + st.get("videos_from_cache", 0)
        print(
            f"\nDone in {st['seconds']:.0f} s. Checked {what}"
            + (f" ({cached:,} unchanged since last scan)" if cached else "")
            + "."
        )
        if st["unreadable"]:
            print(f"  {st['unreadable']:,} files could not be opened (listed in the review screen).")
        for n in results["notices"]:
            print("  " + n)
        k = st["groups_by_kind"]
        what_copies = (
            "same photo or video in different quality" if st.get("videos_found") else "same photo in different quality"
        )
        print(
            f"  Found {st['groups']:,} groups: {k.get('exact', 0):,} exact copies, "
            f"{k.get('copies', 0):,} {what_copies}, {k.get('similar', 0):,} similar shots."
        )
        if st.get("video_groups"):
            print(f"  Of these, {st['video_groups']:,} groups are videos.")
        print(f"  Suggested to move: {st['suggested_files']:,} files ({_fmt_size(st['suggested_bytes'])}).")
        print(f"  Report saved in: {out_dir}")
    return root, out_dir, results


def run_review(root: Path, out_dir: Path, results: dict, port: int = 8765, open_browser: bool = True):
    from .server import ReviewApp

    app = ReviewApp(root, out_dir, results)
    url = app.serve(port=port, open_browser=open_browser)
    print(f"\nReview screen: {url}")
    print("  It opens in your web browser. Keep this window open while you review.")
    print("  Click 'Finish' in the browser (or press Ctrl+C here) when you are done.\n", flush=True)
    try:
        app.run_forever()
    except KeyboardInterrupt:
        pass
    print("Review closed. Your choices are saved.")


def _load_existing(folder: str, out: Optional[str]):
    from . import report

    root = Path(_clean_path(folder)).resolve()
    out_dir = resolve_out_dir(root, out, create=False)
    results = report.load_json(out_dir)
    if results is None:
        raise SystemExit(f"No scan results found for {root}. Run a scan first.")
    return root, out_dir, results


def cmd_move(args):
    from . import actions, report

    root, out_dir, results = _load_existing(args.folder, args.out)
    actions.apply_status(results, root)
    fids = set()
    for g in results["groups"]:
        for m in g["files"]:
            if m["status"] != "present":
                continue
            if m["selected"] or (args.include_similar and m["role"] == "similar"):
                fids.add(m["fid"])
    allowed, blocked = actions.guard_selection(results, fids, root)
    total = sum(m["size"] for m in allowed)
    print(f"Ready to move {len(allowed):,} files ({_fmt_size(total)}) to {out_dir / REVIEW_BIN_NAME}")
    if not allowed:
        return
    if not args.yes:
        if input("Type YES to continue: ").strip().upper() != "YES":
            print("Cancelled. Nothing was moved.")
            return
    res = actions.move_to_bin(root, out_dir, allowed)
    dest = {x["rel"]: x["dst"] for x in res["moved_to"]}
    moved = set(dest)
    for m in allowed:
        if m["rel"] in moved:
            m["status"], m["selected"], m["bin_rel"] = "moved", False, dest[m["rel"]]
    results["batches"] = actions.batches(out_dir)
    report.save_all(out_dir, results)
    print(f"Moved {len(moved):,} files (+{res['companions']} linked files). Skipped {len(res['skipped'])}.")
    if blocked:
        print(f"  {len(blocked)} group(s) left alone: no unchanged file would have stayed in place.")
    for s in res["skipped"][:10]:
        print(f"  skipped {s['rel']}: {s['why']}")
    if res["batch"]:
        print("To undo: run this tool with 'undo' and the same folder.")


def cmd_undo(args):
    from . import actions, report

    root = Path(_clean_path(args.folder)).resolve()
    out_dir = resolve_out_dir(root, args.out, create=False)
    res = actions.undo(root, out_dir, all_batches=args.all)
    print(
        f"Put back {len(res['restored']):,} files."
        + (f" {len(res['failed'])} could not be put back." if res["failed"] else "")
    )
    for f in res["failed"][:10]:
        print(f"  {f['rel']}: {f['why']}")
    results = report.load_json(out_dir)
    if results:
        restored = set(res["restored"])
        for g in results["groups"]:
            for m in g["files"]:
                if m["rel"] in restored:
                    m["selected"] = True
        actions.apply_status(results, root)
        results["batches"] = actions.batches(out_dir)
        report.save_all(out_dir, results)


def cmd_check(_args=None) -> int:
    import numpy
    import PIL

    from .ai import ai_status
    from .loader import HEIF_OK, RAW_OK, avif_supported
    from .video import video_support

    ok_ai, msg_ai = ai_status()
    ok_v, msg_v = video_support()
    rows = [
        ("Python", platform.python_version(), True),
        ("Computer", f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPU threads", True),
        ("JPEG, PNG, GIF, BMP, TIFF, WEBP", f"yes (Pillow {PIL.__version__})", True),
        ("iPhone HEIC / HEIF", "yes" if HEIF_OK else "NO: run  pip install pillow-heif", HEIF_OK),
        ("AVIF", "yes" if avif_supported() else "NO: update Pillow / pillow-heif", avif_supported()),
        (
            "Camera RAW (DNG, CR2, CR3, NEF, ARW, RAF, ORF, RW2 ...)",
            "yes" if RAW_OK else "NO: run  pip install rawpy",
            RAW_OK,
        ),
        ("Videos (MP4, MOV, MKV, AVI, WMV, 3GP, WEBM ...)", msg_v if ok_v else "NO: run  pip install av", ok_v),
        ("AI matching", msg_ai if ok_ai else f"NO: {msg_ai}", ok_ai),
        ("numpy", numpy.__version__, True),
    ]
    print(f"{APP_NAME} {__version__}: system check\n")
    for name, val, ok in rows:
        print(f"  [{'ok' if ok else '!!'}] {name}: {val}")
    print(f"\n  Help and updates: {HOMEPAGE}")
    return 0 if all(ok for _, _, ok in rows) else 1


def _prepare_ai():
    from .ai import ensure_model

    ok, msg = ensure_model()
    if msg != "ready":
        print(("  " if ok else "  Note: ") + msg + ("." if not msg.endswith(".") else ""))


def interactive() -> int:
    print(f"{APP_NAME} {__version__}  ({HOMEPAGE})")
    _prepare_ai()
    print("Finds exact copies, smaller or converted copies, and repeat shots of the same photo,")
    print("and the same video saved again (other name, other format, smaller or re-encoded copy).")
    print("Nothing is deleted: you review everything first, and chosen files go to a Review Bin.\n")
    print("Step 1: choose your photo or video folder (a window opens)...", flush=True)
    folder = pick_folder()
    if not folder or not os.path.isdir(folder):
        print("No folder chosen. Closing.")
        return 1
    root, out_dir, results = run_scan(folder)
    print("\nStep 2: review the results.", flush=True)
    run_review(root, out_dir, results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dupefinder",
        description=f"{APP_NAME}: find duplicate photos and videos safely.",
        epilog=f"Run with no options for the guided mode (folder picker + review screen). Docs: {HOMEPAGE}",
    )
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__} ({HOMEPAGE})")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("scan", help="scan a folder, then open the review screen")
    s.add_argument("folder")
    s.add_argument(
        "--sensitivity",
        choices=["strict", "normal", "loose"],
        default="normal",
        help="strict = fewer, surer matches; loose = more matches to review (default normal)",
    )
    s.add_argument("--no-ai", action="store_true", help="skip AI matching (faster, finds fewer repeat shots)")
    s.add_argument(
        "--media",
        choices=["all", "photos", "videos"],
        default="all",
        help="what to check: photos and videos (default), only photos, or only videos",
    )
    s.add_argument("--workers", type=int, default=None, help="parallel workers (default: CPU threads - 1, max 6)")
    s.add_argument("--include-hidden", action="store_true", help="also scan hidden folders")
    s.add_argument("--out", help="where to save results (default: <folder>/_DupeFinder)")
    s.add_argument("--no-review", action="store_true", help="only scan and save the report")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")

    r = sub.add_parser("review", help="open the review screen for the last scan")
    r.add_argument("folder")
    r.add_argument("--out")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--no-browser", action="store_true")

    m = sub.add_parser("move", help="move the suggested files to the Review Bin without the browser")
    m.add_argument("folder")
    m.add_argument("--out")
    m.add_argument("--include-similar", action="store_true", help="also move similar shots (best one stays)")
    m.add_argument("--yes", action="store_true", help="do not ask for confirmation")

    u = sub.add_parser("undo", help="put moved files back")
    u.add_argument("folder")
    u.add_argument("--out")
    u.add_argument("--all", action="store_true", help="undo every move, not just the last one")

    sub.add_parser("check", help="show which photo and video formats and features work on this computer")
    return p


def main(argv=None) -> int:
    _console_setup()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        try:
            return interactive()
        except KeyboardInterrupt:
            print("\nStopped.")
            return 1
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "scan":
            root, out_dir, results = run_scan(
                args.folder,
                args.out,
                args.sensitivity,
                not args.no_ai,
                args.workers,
                args.include_hidden,
                media=args.media,
            )
            if not args.no_review:
                run_review(root, out_dir, results, args.port, not args.no_browser)
        elif args.cmd == "review":
            root, out_dir, results = _load_existing(args.folder, args.out)
            run_review(root, out_dir, results, args.port, not args.no_browser)
        elif args.cmd == "move":
            cmd_move(args)
        elif args.cmd == "undo":
            cmd_undo(args)
        elif args.cmd == "check":
            return cmd_check(args)
        else:
            build_parser().print_help()
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    return 0
