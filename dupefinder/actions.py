"""Safe file actions: move duplicates into a Review Bin (never delete) and undo."""

from __future__ import annotations

import errno
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .formats import (
    COMPANION_FULLNAME_EXTS,
    COMPANION_STEM_EXTS,
    LIVE_VIDEO_EXTS,
    PHOTO_EXTS,
    REVIEW_BIN_NAME,
    VIDEO_EXTS,
    VIDEO_SIDECAR_EXTS,
)
from .paths import lp

LOG_NAME = "move_log.jsonl"


def review_bin(out_dir: Path) -> Path:
    return Path(out_dir) / REVIEW_BIN_NAME


def _unique_dest(dst: Path) -> Path:
    if not os.path.lexists(lp(dst)):
        return dst
    stem, ext = dst.stem, dst.suffix
    n = 2
    while True:
        cand = dst.with_name(f"{stem} (dupefinder {n}){ext}")
        if not os.path.lexists(lp(cand)):
            return cand
        n += 1


def _cross_device(exc: OSError) -> bool:
    return exc.errno == errno.EXDEV or getattr(exc, "winerror", None) == 17


def _move(src: Path, dst: Path) -> Path:
    """Rename when possible. Across drives: copy to a temporary name, then rename, then remove the
    original; if the original cannot be removed, the copy is removed again (nothing half-moved)."""
    dst = _unique_dest(dst)
    os.makedirs(lp(dst.parent), exist_ok=True)
    try:
        os.rename(lp(src), lp(dst))
        return dst
    except OSError as exc:
        if not _cross_device(exc):
            raise
    tmp = dst.with_name(dst.name + ".dupefinder-partial")
    try:
        shutil.copy2(lp(src), lp(tmp))
        os.replace(lp(tmp), lp(dst))
    except BaseException:
        try:
            os.remove(lp(tmp))
        except OSError:
            pass
        raise
    try:
        os.remove(lp(src))
    except OSError:
        try:
            os.remove(lp(dst))
        except OSError:
            pass
        raise
    return dst


def _is_live_video(p: Path) -> bool:
    """True for files that are not videos, and for videos small enough to be a Live Photo clip."""
    if p.suffix.lower() not in LIVE_VIDEO_EXTS:
        return True
    try:
        size = p.stat().st_size
    except OSError:
        return False
    from .video import is_live_clip

    return is_live_clip(p, size)  # a few seconds long; a longer video with the same name is its own file


def find_companions(item: Path, moving: set) -> List[Path]:
    """Files that belong to `item` and move with it.
    Photo: Live Photo video (.MOV/.MP4), Apple edit file (.AAE), sidecars (.XMP), Google Takeout .json.
    Video: Apple edit file (.AAE), sidecars (.XMP), camera thumbnail (.THM), subtitles (.SRT) - never
    another video or photo. Name-matched files are only taken when no other photo or video with the
    same name stays behind in that folder (e.g. IMG_1.HEIC kept + IMG_1.JPG moved keeps IMG_1.MOV)."""
    folder = item.parent
    try:
        siblings = list(folder.iterdir())
    except OSError:
        return []
    is_video = item.suffix.lower() in VIDEO_EXTS
    stem_exts = VIDEO_SIDECAR_EXTS if is_video else COMPANION_STEM_EXTS
    stem = item.stem.casefold()
    full = item.name.casefold()
    staying_same_stem = any(
        (s.suffix.lower() in PHOTO_EXTS or (is_video and s.suffix.lower() in VIDEO_EXTS))
        and s.stem.casefold() == stem
        and s != item
        and str(s) not in moving
        and s.is_file()
        for s in siblings
    )
    out = []
    for s in siblings:
        if s == item or not s.is_file() or s.is_symlink() or str(s) in moving:
            continue
        name = s.name.casefold()
        ext = s.suffix.lower()
        if name.startswith(full + ".") and ext in COMPANION_FULLNAME_EXTS:
            out.append(s)  # IMG_1.HEIC.json, IMG_1.HEIC.supplemental-metadata.json, IMG_1.HEIC.xmp
        elif (
            not staying_same_stem and s.stem.casefold() == stem and ext in stem_exts and (is_video or _is_live_video(s))
        ):
            out.append(s)  # IMG_1.MOV, IMG_1.AAE, IMG_1.xmp (photo) / clip.THM, clip.SRT (video)
    for clip in [c for c in out if c.suffix.lower() in VIDEO_EXTS]:
        cfull = clip.name.casefold()  # the Live Photo clip's own Takeout file: IMG_1.MOV.json
        for s in siblings:
            if (
                s not in out
                and s != item
                and str(s) not in moving
                and s.is_file()
                and not s.is_symlink()
                and s.name.casefold().startswith(cfull + ".")
                and s.suffix.lower() in COMPANION_FULLNAME_EXTS
            ):
                out.append(s)
    return out


class _Log:
    """Append-only move log; every line is written and flushed the moment its file moves."""

    def __init__(self, out_dir: Path):
        self.f = open(Path(out_dir) / LOG_NAME, "a", encoding="utf-8")

    def write(self, entry: dict):
        self.f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except OSError:
            pass

    def close(self):
        self.f.close()


def _unchanged(path: Path, it: dict) -> Optional[str]:
    try:
        st = os.stat(lp(path))
    except OSError:
        return "File not found (already moved or deleted)"
    if it.get("size") is not None and (
        st.st_size != it["size"] or st.st_mtime_ns != it.get("mtime_ns", st.st_mtime_ns)
    ):
        return "File changed after the scan; scan again first"
    return None


def move_to_bin(root: Path, out_dir: Path, items: Iterable[dict]) -> dict:
    """items: dicts with rel, size, mtime_ns (from the scan). Files changed since the scan are skipped."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    bin_dir = review_bin(out_dir)
    # pass 1: decide what will really move
    ready, skipped = [], []
    for it in items:
        why = _unchanged(root / it["rel"], it)
        if why:
            skipped.append({"rel": it["rel"], "why": why})
        else:
            ready.append(it)
    moving = {str(root / it["rel"]) for it in ready}
    # pass 2: move, logging each file immediately
    moved, companions_moved = [], 0
    log = _Log(out_dir)
    try:
        for it in ready:
            src = root / it["rel"]
            companions = find_companions(src, moving)
            try:
                dst = _move(src, bin_dir / it["rel"])
            except OSError as exc:
                skipped.append({"rel": it["rel"], "why": f"Could not move: {exc.strerror or exc}"})
                moving.discard(str(src))
                continue
            dst_rel = str(dst.relative_to(out_dir))
            log.write(
                {
                    "id": uuid.uuid4().hex,
                    "batch": batch,
                    "time": time.time(),
                    "src": it["rel"],
                    "dst": dst_rel,
                    "kind": it.get("media") or "photo",
                }
            )
            moved.append({"rel": it["rel"], "dst": dst_rel})
            for c in companions:
                try:
                    crel = str(c.relative_to(root))
                    cdst = _move(c, bin_dir / crel)
                    log.write(
                        {
                            "id": uuid.uuid4().hex,
                            "batch": batch,
                            "time": time.time(),
                            "src": crel,
                            "dst": str(cdst.relative_to(out_dir)),
                            "kind": "companion",
                            "of": it["rel"],
                        }
                    )
                    companions_moved += 1
                except (OSError, ValueError):
                    pass
    finally:
        log.close()
    return {
        "batch": batch if moved else None,
        "moved": [m["rel"] for m in moved],
        "moved_to": moved,
        "skipped": skipped,
        "companions": companions_moved,
    }


def read_log(out_dir: Path) -> List[dict]:
    p = Path(out_dir) / LOG_NAME
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    v = json.loads(line)
                    if isinstance(v, dict):
                        out.append(v)
                except ValueError:
                    pass
    return out


def _open_entries(lines: List[dict]) -> List[dict]:
    """Move entries that have not been put back yet."""
    done_ids = {l["undone"] for l in lines if "undone" in l}
    done_batches = {l["undo"] for l in lines if "undo" in l}  # older log format
    return [
        l
        for l in lines
        if "batch" in l and "src" in l and "dst" in l and l.get("id") not in done_ids and l["batch"] not in done_batches
    ]


def batches(out_dir: Path) -> List[dict]:
    """Move batches that still have files in the Review Bin, newest first."""
    seen: Dict[str, dict] = {}
    for l in _open_entries(read_log(out_dir)):
        info = seen.setdefault(
            l["batch"], {"batch": l["batch"], "time": l.get("time", 0), "photos": 0, "companions": 0}
        )
        if l.get("kind") == "companion":
            info["companions"] += 1
        else:
            info["photos"] += 1
    return sorted(seen.values(), key=lambda x: x["time"], reverse=True)


def undo(root: Path, out_dir: Path, batch: Optional[str] = None, all_batches: bool = False) -> dict:
    """Put files from the Review Bin back where they were. Files that cannot go back yet stay
    listed, so a later Undo can finish the job."""
    root, out_dir = Path(root), Path(out_dir)
    open_batches = batches(out_dir)
    if not open_batches:
        return {"restored": [], "failed": [], "batches": []}
    targets = [b["batch"] for b in open_batches] if all_batches else [batch or open_batches[0]["batch"]]
    entries = _open_entries(read_log(out_dir))
    restored, failed = [], []
    log = _Log(out_dir)
    try:
        for b in targets:
            for e in reversed([x for x in entries if x["batch"] == b]):
                src = out_dir / e["dst"]
                dst = root / e["src"]
                if not os.path.lexists(lp(src)):
                    failed.append({"rel": e["src"], "why": "Not in the Review Bin any more"})
                    log.write({"undone": e.get("id"), "missing": True, "time": time.time()})
                    continue
                if os.path.lexists(lp(dst)):
                    failed.append({"rel": e["src"], "why": "A file with that name is back in the original place"})
                    continue
                try:
                    _move_back(src, dst)
                except OSError as exc:
                    failed.append({"rel": e["src"], "why": f"Could not move back: {exc.strerror or exc}"})
                    continue
                if e.get("id"):
                    log.write({"undone": e["id"], "time": time.time()})
                else:  # pragma: no cover - entries written by older versions
                    log.write({"undo": b, "time": time.time()})
                if e.get("kind") != "companion":
                    restored.append(e["src"])
    finally:
        log.close()
    _prune_empty_dirs(review_bin(out_dir))
    return {"restored": restored, "failed": failed, "batches": targets}


def _move_back(src: Path, dst: Path):
    os.makedirs(lp(dst.parent), exist_ok=True)
    try:
        os.rename(lp(src), lp(dst))
    except OSError as exc:
        if not _cross_device(exc):
            raise
        tmp = dst.with_name(dst.name + ".dupefinder-partial")
        try:
            shutil.copy2(lp(src), lp(tmp))
            os.replace(lp(tmp), lp(dst))
        except BaseException:
            try:
                os.remove(lp(tmp))
            except OSError:
                pass
            raise
        os.remove(lp(src))


def _prune_empty_dirs(top: Path):
    if not top.is_dir():
        return
    for dirpath, _dirs, _files in os.walk(top, topdown=False):
        if dirpath != str(top) and not os.listdir(dirpath):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass


def apply_status(results: dict, root: Path) -> None:
    """Refresh each file's status (present / moved / missing) from disk."""
    for g in results.get("groups", []):
        for m in g["files"]:
            if os.path.exists(lp(os.path.join(str(root), m["rel"]))):
                m["status"] = "present"
            elif m.get("status") == "moved":
                pass
            else:
                m["status"] = "missing"


def guard_selection(results: dict, fids: set, root: Optional[Path] = None) -> tuple:
    """Never allow every file of a group to be moved: each group must keep at least one file that is
    not ticked, still present and unchanged since the scan. Returns (allowed items, blocked group ids)."""
    allowed, blocked = [], []
    for g in results.get("groups", []):
        present = [m for m in g["files"] if m.get("status", "present") == "present"]
        chosen = [m for m in present if m["fid"] in fids]
        if not chosen:
            continue
        keepers = [m for m in present if m["fid"] not in fids]
        if root is not None:
            keepers = [m for m in keepers if _unchanged(Path(root) / m["rel"], m) is None]
        if not keepers:
            blocked.append(g["id"])
            continue
        allowed.extend(chosen)
    return allowed, blocked
