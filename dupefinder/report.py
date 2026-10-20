"""Write results: results.json (for the review screen), duplicates.csv (Excel), report.html, thumbnails."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import APP_NAME, __version__
from .features import make_thumbnail
from .grouping import fmt_size

_CTX = multiprocessing.get_context("spawn")  # safe with RAW/HEIC decoders on every OS

THUMB_SIDE = 360
WEB_DIR = Path(__file__).resolve().parent / "web"


def make_any_thumbnail(task: tuple):
    """Worker entry point: (src, dst, side, media). Videos get a picture from about a third of the way in."""
    src, dst, side, media = task
    if media == "video":
        from .video import make_video_thumbnail

        return make_video_thumbnail((src, dst, side))
    return make_thumbnail((src, dst, side))


ROLE_LABEL = {
    "keep": "KEEP (suggested)",
    "exact": "Exact copy",
    "lower": "Lower-quality or converted copy",
    "similar": "Similar shot",
    "rawpair": "RAW/JPEG pair",
}


def file_id(rel: str) -> str:
    key = rel.replace("\\", "/") if os.sep == "\\" else rel  # "\\" is a normal character on Mac/Linux
    return hashlib.sha1(key.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def thumb_name(m: dict) -> str:
    return f"{m['fid']}-{m.get('sha12', 'x')}.jpg"  # new picture under the same name = new thumbnail


def build_results(root: Path, scan, groups: List[dict], settings_name: str) -> dict:
    files_in_groups = sum(len(g["files"]) for g in groups)
    selected = [m for g in groups for m in g["files"] if m["selected"]]
    used = set()
    for g in groups:
        for m in g["files"]:
            fid = file_id(m["rel"])
            n = 1
            while fid in used:  # never let two files share an id
                n += 1
                fid = file_id(f"{m['rel']}\x00{n}")
            used.add(fid)
            m["fid"] = fid
            m["status"] = "present"
            m["name"] = os.path.basename(m["rel"])
            folder = os.path.dirname(m["rel"]).replace("\\", "/")
            m["folder"] = folder or "."
            m.pop("idx", None)
    by_kind: Dict[str, int] = {}
    for g in groups:
        by_kind[g["kind"]] = by_kind.get(g["kind"], 0) + 1
    return {
        "app": APP_NAME,
        "version": __version__,
        "root": str(root),
        "scanned_at": time.strftime("%Y-%m-%d %H:%M"),
        "sensitivity": settings_name,
        "ai_used": scan.ai_used,
        "ai_message": scan.ai_message,
        "video_message": scan.video_message,
        "stats": {
            "photos_found": len(scan.discovery.files),
            "photos_read": len(scan.records),
            "videos_found": len(scan.discovery.videos),
            "videos_read": len(scan.videos),
            "videos_from_cache": scan.video_from_cache,
            "live_clips": scan.discovery.live_clips,
            "video_groups": sum(1 for g in groups if g.get("media") == "video"),
            "unreadable": len(scan.errors),
            "other_files": scan.discovery.other_files,
            "groups": len(groups),
            "groups_by_kind": by_kind,
            "files_in_groups": files_in_groups,
            "suggested_files": len(selected),
            "suggested_bytes": int(sum(m["size"] for m in selected)),
            "seconds": round(scan.seconds, 1),
            "from_cache": scan.from_cache,
        },
        "notices": scan.discovery.notices,
        "skipped_dirs": scan.discovery.skipped_dirs[:200],
        "errors": scan.errors,
        "groups": groups,
        "batches": [],
    }


def make_thumbnails(
    root: Path,
    out_dir: Path,
    groups: List[dict],
    workers: int,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> None:
    tdir = out_dir / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    wanted = set()
    for g in groups:
        for m in g["files"]:
            src = root / m["rel"]
            name = thumb_name(m)
            wanted.add(name)
            dst = tdir / name
            if dst.exists():
                continue
            tasks.append((str(src), str(dst), THUMB_SIDE, m.get("media") or "photo"))
    for old in tdir.glob("*.jpg"):  # remove previews of files that are no longer listed
        if old.name not in wanted:
            try:
                old.unlink()
            except OSError:
                pass
    total = len(tasks)
    if not total:
        return
    progress = progress or (lambda *a: None)
    done = 0
    try:
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=_CTX) as pool:
            for _ in pool.map(make_any_thumbnail, tasks, chunksize=4):
                done += 1
                if done % 5 == 0 or done == total:
                    progress("thumbs", done, total)
    except Exception:
        # a broken file crashed a worker: redo the rest one at a time, each in its own process,
        # so the scan always finishes (a missing preview is shown as a placeholder)
        pool = None
        for t in tasks:
            if os.path.exists(t[1]):
                continue
            try:
                if pool is None:
                    pool = ProcessPoolExecutor(max_workers=1, mp_context=_CTX)
                pool.submit(make_any_thumbnail, t).result(timeout=300)
            except Exception:
                if pool is not None:
                    pool.shutdown(wait=False)
                pool = None
        if pool is not None:
            pool.shutdown()
        progress("thumbs", total, total)


def write_json(out_dir: Path, results: dict) -> Path:
    p = out_dir / "results.json"
    tmp = out_dir / f"results.json.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, p)
    return p


def load_json(out_dir: Path) -> Optional[dict]:
    p = out_dir / "results.json"
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
            raise ValueError("unexpected content")
        return data
    except (ValueError, OSError) as exc:
        raise SystemExit(f"The saved results in {out_dir} are damaged ({exc}). Please scan the folder again.") from None


def _cell(v):
    """Stop spreadsheet apps from treating a file name like '=SUM(...)' as a formula."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def write_csv(out_dir: Path, results: dict) -> Path:
    p = out_dir / "duplicates.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig opens cleanly in Excel
        w = csv.writer(f)
        w.writerow(
            [
                "Group",
                "Group type",
                "Media",
                "Suggestion",
                "Role",
                "File",
                "Folder",
                "Duplicate of (keeper)",
                "Size",
                "Size (bytes)",
                "Resolution",
                "Length",
                "Format",
                "Date taken",
                "Status",
                "Why",
            ]
        )
        kinds = {"exact": "Exact copies", "copies": "Copies (different quality)", "similar": "Similar shots"}
        from .video import fmt_duration

        for g in results["groups"]:
            keeper = next((x for x in g["files"] if x["role"] == "keep"), g["files"][0])
            kref = keeper["rel"].replace("\\", "/")
            for m in g["files"]:
                video = m.get("media") == "video"
                w.writerow(
                    [
                        _cell(v)
                        for v in [
                            g["id"],
                            kinds.get(g["kind"], g["kind"]),
                            "Video" if video else "Photo",
                            "MOVE" if m["selected"] else ("KEEP" if m["role"] == "keep" else "REVIEW"),
                            ROLE_LABEL.get(m["role"], m["role"]),
                            m["name"],
                            m["folder"],
                            "" if m is keeper else kref,
                            fmt_size(m["size"]),
                            m["size"],
                            f"{m['w']}x{m['h']}",
                            fmt_duration(m.get("dur")) if video else "",
                            (f"{m['fmt']} ({m['vcodec']})" if video and m.get("vcodec") else m["fmt"]),
                            m["dt"] or "",
                            m.get("status", "present"),
                            m["reason"],
                        ]
                    ]
                )
    return p


def render_page(results: Optional[dict], live: bool, token: str = "") -> str:
    tpl = (WEB_DIR / "review.html").read_text(encoding="utf-8")
    boot = {"live": live, "token": token}
    if not live:
        boot["data"] = results
    payload = (
        json.dumps(boot, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return tpl.replace("/*__BOOT__*/null", payload)


def write_static_report(out_dir: Path, results: dict) -> Path:
    p = out_dir / "report.html"
    p.write_text(render_page(results, live=False), encoding="utf-8")
    return p


def save_all(out_dir: Path, results: dict) -> None:
    write_json(out_dir, results)
    write_csv(out_dir, results)
    write_static_report(out_dir, results)
