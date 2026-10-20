"""Find photo and video files and fingerprint them in parallel (with cache and crash protection)."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import stat as _stat
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set

import numpy as np

from .cache import FeatureCache
from .features import FEATURE_VERSION, extract
from .formats import (
    COMPANION_STEM_EXTS,
    LIVE_VIDEO_EXTS,
    LIVE_VIDEO_MAX,
    OUTPUT_DIR_NAME,
    PHOTO_EXTS,
    SKIP_DIR_NAMES,
    SKIP_DIR_SUFFIXES,
    VIDEO_EXTS,
)
from .paths import lp

_CTX = multiprocessing.get_context("spawn")  # safe with RAW/HEIC decoders on every OS
ProgressFn = Callable[[str, int, int], None]


@dataclass
class Discovery:
    files: List[Path] = field(default_factory=list)
    other_files: int = 0
    skipped_dirs: List[str] = field(default_factory=list)
    notices: List[str] = field(default_factory=list)
    bad_names: List[str] = field(default_factory=list)
    with_companion: Set[str] = field(default_factory=set)  # photos that have a Live Photo video / sidecar
    videos: List[Path] = field(default_factory=list)
    live_clips: int = 0  # Live Photo videos: they belong to a photo and move with it


def _norm(p: Path) -> str:
    try:
        return os.path.normcase(str(p.resolve()))
    except OSError:
        return os.path.normcase(os.path.abspath(str(p)))


def is_tool_folder(path: Path) -> bool:
    """A results folder made by this tool (any name, any depth)."""
    if path.name.casefold() in (OUTPUT_DIR_NAME.casefold(), "photodupefinder_results"):
        return True
    try:
        return (path / "results.json").is_file() and (
            (path / "cache.sqlite").is_file() or (path / "move_log.jsonl").is_file()
        )
    except OSError:
        return False


def _text_ok(name: str) -> bool:
    try:
        name.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def discover(
    root: Path, out_dir: Optional[Path] = None, include_hidden: bool = False, photos: bool = True, videos: bool = True
) -> Discovery:
    root = Path(root)
    disc = Discovery()
    excluded = {_norm(out_dir)} if out_dir else set()
    excluded.add(_norm(root / OUTPUT_DIR_NAME))

    def on_error(err):
        disc.skipped_dirs.append(f"{getattr(err, 'filename', '?')}: no access")

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=on_error, followlinks=False):
        keep = []
        for d in sorted(dirnames):
            full = Path(dirpath) / d
            low = d.lower()
            if not _text_ok(d):
                disc.bad_names.append(os.path.join(dirpath, d.encode("utf-8", "replace").decode("utf-8")))
                continue
            if _norm(full) in excluded or is_tool_folder(full):
                continue
            if low in SKIP_DIR_NAMES:
                continue
            if any(low.endswith(s) for s in SKIP_DIR_SUFFIXES):
                disc.skipped_dirs.append(str(full))
                if low.endswith((".photoslibrary", ".photolibrary", ".migratedphotolibrary")):
                    disc.notices.append(
                        f"Skipped the Apple Photos library '{d}'. Use the Photos app's own "
                        "'Duplicates' album for photos inside it."
                    )
                continue
            if not include_hidden and d.startswith("."):
                continue
            if os.path.islink(full):
                continue
            keep.append(d)
        dirnames[:] = keep
        photo_stems = {
            os.path.splitext(n)[0].casefold() for n in filenames if os.path.splitext(n)[1].lower() in PHOTO_EXTS
        }
        live_names = _live_clips(dirpath, filenames, photo_stems)
        stems_with_companion = {
            os.path.splitext(n)[0].casefold()
            for n in filenames
            if os.path.splitext(n)[1].lower() in COMPANION_STEM_EXTS
            and (os.path.splitext(n)[1].lower() not in LIVE_VIDEO_EXTS or n in live_names)
        }
        for name in sorted(filenames):
            if name.startswith("._"):  # macOS metadata stubs on USB drives
                continue
            if not include_hidden and name.startswith("."):
                continue
            stem, ext = os.path.splitext(name)
            low = ext.lower()
            if low in VIDEO_EXTS:
                if videos:
                    _add_video(disc, Path(dirpath) / name, name in live_names)
                else:
                    disc.other_files += 1
                continue
            if low not in PHOTO_EXTS or not photos:
                disc.other_files += 1
                continue
            if not _text_ok(name):
                disc.bad_names.append(os.path.join(dirpath, name.encode("utf-8", "replace").decode("utf-8")))
                continue
            full = Path(dirpath) / name
            try:
                st = os.lstat(lp(full))
            except OSError:
                continue
            if not _stat.S_ISREG(st.st_mode):  # skip links, pipes, devices
                continue
            disc.files.append(full)
            if stem.casefold() in stems_with_companion:
                disc.with_companion.add(str(full))
    return disc


def _live_clips(dirpath: str, filenames: list, photo_stems: set) -> set:
    """Names of the Live Photo / motion photo clips in one folder: a short .MOV/.MP4 with the same
    name as a photo next to it (IMG_1.HEIC + IMG_1.MOV). They belong to the photo and move with it."""
    out = set()
    for n in filenames:
        stem, ext = os.path.splitext(n)
        if ext.lower() not in LIVE_VIDEO_EXTS or stem.casefold() not in photo_stems:
            continue
        full = Path(dirpath) / n
        try:
            st = os.lstat(lp(full))
        except OSError:
            continue
        if _stat.S_ISREG(st.st_mode) and st.st_size <= LIVE_VIDEO_MAX:
            from .video import is_live_clip

            if is_live_clip(full, st.st_size):
                out.add(n)
    return out


def _add_video(disc: Discovery, full: Path, live_clip: bool):
    if not _text_ok(full.name):
        disc.bad_names.append(str(full).encode("utf-8", "replace").decode("utf-8"))
        return
    try:
        st = os.lstat(lp(full))
    except OSError:
        return
    if not _stat.S_ISREG(st.st_mode):
        return
    if live_clip:
        disc.live_clips += 1  # IMG_1.HEIC + IMG_1.MOV (a few seconds): the clip moves with its photo
        return
    disc.videos.append(full)


@dataclass
class ScanResult:
    records: List[dict]  # readable photos: path, rel, size, mtime_ns, sha, feat, emb
    errors: List[dict]  # unreadable files: rel, size, error
    discovery: Discovery
    ai_used: bool
    ai_message: str
    seconds: float
    from_cache: int = 0
    videos: List[dict] = field(default_factory=list)  # readable videos: path, rel, size, mtime_ns, sha, feat, sig
    video_message: str = ""
    video_from_cache: int = 0


def default_workers() -> int:
    n = os.cpu_count() or 2
    return max(1, min(6, n - 1 if n > 2 else n))


def _rel(path: str, root: Path) -> str:
    try:
        return os.path.relpath(path, str(root))
    except ValueError:
        return path


def scan(
    root: Path,
    out_dir: Path,
    use_ai: bool = True,
    workers: Optional[int] = None,
    include_hidden: bool = False,
    progress: Optional[ProgressFn] = None,
    embedder=None,
    media: str = "all",
) -> ScanResult:
    t0 = time.time()
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda phase, done, total: None)
    from .video import VIDEO_SIG_VERSION

    progress("discover", 0, 0)
    disc = discover(root, out_dir, include_hidden, photos=media != "videos", videos=media != "photos")
    total = len(disc.files)

    ai_used, ai_message = False, "AI matching is off"
    if use_ai and embedder is None and total:
        from .ai import AIUnavailable, ClipEmbedder

        try:
            embedder = ClipEmbedder()
        except AIUnavailable as exc:
            embedder = None
            ai_message = f"AI matching unavailable: {exc}. Using classic matching only."
    if embedder is not None:
        ai_used = True
        ai_message = "AI matching is on"

    cache = FeatureCache(out_dir / "cache.sqlite", FEATURE_VERSION, VIDEO_SIG_VERSION)
    records: List[dict] = []
    errors: List[dict] = []
    todo = []
    from_cache = 0
    for b in disc.bad_names:
        errors.append(
            {
                "rel": _rel(b, root),
                "size": 0,
                "error": "The name contains characters that are not valid text. Rename it to include it.",
            }
        )
    inode_seen = {}
    hardlinks = 0
    for p in disc.files:
        ps = str(p)
        try:
            st = os.stat(lp(ps))
        except OSError as exc:
            errors.append({"rel": _rel(ps, root), "size": 0, "error": f"Cannot read file: {exc.strerror or exc}"})
            continue
        if st.st_ino and st.st_nlink > 1:  # hard links share one copy on disk: count it once
            key = (st.st_dev, st.st_ino)
            if key in inode_seen:
                hardlinks += 1
                continue
            inode_seen[key] = ps
        hit = cache.get(ps, st.st_size, st.st_mtime_ns)
        need_emb = embedder is not None
        if hit and (not need_emb or (hit["emb"] is not None and hit["emb_model"] == embedder.id)):
            records.append(
                {
                    "path": ps,
                    "rel": _rel(ps, root),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "sha": hit["sha"],
                    "feat": hit["feat"],
                    "emb": hit["emb"] if need_emb else None,
                    "comp": ps in disc.with_companion,
                }
            )
            from_cache += 1
        else:
            todo.append((ps, need_emb))

    done = from_cache
    pending_clip: List[tuple] = []
    ai_state = {"active": embedder is not None, "failed": ""}

    def flush_clip(force: bool = False):
        nonlocal pending_clip
        if not pending_clip or (not force and len(pending_clip) < 32):
            return
        batch, pending_clip = pending_clip, []
        if not ai_state["active"]:
            return
        try:
            embs = embedder.embed(np.stack([c for _, c in batch]))
        except Exception as exc:  # AI failed mid-way: finish the scan with classic matching
            ai_state["active"] = False
            ai_state["failed"] = f"AI model stopped with an error ({str(exc)[:120]})"
            return
        for (rec, _), e in zip(batch, embs):
            rec["emb"] = e
            cache.put_emb(rec["path"], e, embedder.id)

    def handle(res: dict):
        nonlocal done
        done += 1
        ps = res["path"]
        if "error" in res:
            errors.append({"rel": _rel(ps, root), "size": res.get("size", 0), "error": res["error"]})
        else:
            rec = {
                "path": ps,
                "rel": _rel(ps, root),
                "size": res["size"],
                "mtime_ns": res["mtime_ns"],
                "sha": res["sha"],
                "feat": res["feat"],
                "emb": None,
                "comp": ps in disc.with_companion,
            }
            cache.put(ps, res["size"], res["mtime_ns"], res["sha"], res["feat"])
            records.append(rec)
            if ai_state["active"] and res.get("clip") is not None:
                pending_clip.append((rec, res["clip"]))
        if done % 7 == 0 or done == total:
            progress("read", done, total)

    vrecords: List[dict] = []
    vmsg, vcached = "", 0
    try:
        if total:
            progress("read", done, total)
        _run_pool(todo, workers or default_workers(), handle, flush_clip)
        flush_clip(True)
        if disc.videos:
            vrecords, vmsg, vcached = _scan_videos(root, disc, cache, workers or default_workers(), progress, errors)
    finally:
        cache.close()

    if embedder is not None and (ai_state["failed"] or any(r.get("emb") is None for r in records)):
        ai_used = False
        ai_message = (ai_state["failed"] or "AI fingerprint missing for some photos") + ". Using classic matching only."
        for r in records:
            r["emb"] = None

    if hardlinks:
        disc.notices.append(f"{hardlinks} hard-linked file(s) skipped: they share disk space with another listed file.")
    records.sort(key=lambda r: r["rel"].lower())
    vrecords.sort(key=lambda r: r["rel"].lower())
    errors.sort(key=lambda r: r["rel"].lower())
    return ScanResult(records, errors, disc, ai_used, ai_message, time.time() - t0, from_cache, vrecords, vmsg, vcached)


def _scan_videos(root: Path, disc: Discovery, cache: FeatureCache, workers: int, progress, errors: list):
    """Videos: details of every video (cached), byte-for-byte hash only for files of identical size,
    sampled pictures and sound only for videos whose length matches another video's length."""
    from .video import VIDEO_SIG_VERSION, hash_file, lengths_match, probe_video, video_signature, video_support

    ok, msg = video_support()
    if not ok:
        disc.notices.append(
            f"{len(disc.videos):,} videos were not checked: {msg}. "
            "Run the start file again, or install it with: pip install av"
        )
        return [], "Videos skipped", 0
    recs: dict = {}
    todo = []
    seen_inodes: dict = {}
    hardlinks = 0
    for p in disc.videos:
        ps = str(p)
        try:
            st = os.stat(lp(ps))
        except OSError as exc:
            errors.append({"rel": _rel(ps, root), "size": 0, "error": f"Cannot read file: {exc.strerror or exc}"})
            continue
        if st.st_ino and st.st_nlink > 1:
            key = (st.st_dev, st.st_ino)
            if key in seen_inodes:
                hardlinks += 1
                continue
            seen_inodes[key] = ps
        hit = cache.vget(ps, st.st_size, st.st_mtime_ns)
        if hit and hit.get("meta"):
            recs[ps] = {"path": ps, "rel": _rel(ps, root), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "v": hit}
        else:
            todo.append((ps,))
    if hardlinks:
        disc.notices.append(
            f"{hardlinks} hard-linked video(s) skipped: they share disk space with another listed file."
        )
    from_cache = len(recs)
    total = len(recs) + len(todo)
    state = {"done": from_cache}
    progress("videos", from_cache, total)

    def on_probe(res):
        state["done"] += 1
        ps = res["path"]
        if "error" in res:
            errors.append({"rel": _rel(ps, root), "size": res.get("size", 0), "error": res["error"]})
        elif "skip" not in res:
            v = {"meta": res["meta"], "qhash": res["qhash"]}
            cache.vput(ps, res["size"], res["mtime_ns"], v)
            recs[ps] = {"path": ps, "rel": _rel(ps, root), "size": res["size"], "mtime_ns": res["mtime_ns"], "v": v}
        if state["done"] % 5 == 0 or state["done"] == total:
            progress("videos", state["done"], total)

    _run_pool(
        todo,
        workers,
        on_probe,
        lambda: None,
        fn=probe_video,
        crash="This file crashed the video decoder (it is probably damaged)",
    )

    # exact copies: same size and same first/last megabyte -> confirm with a full SHA-256
    by_key: dict = {}
    for r in recs.values():
        by_key.setdefault((r["size"], r["v"]["qhash"]), []).append(r)
    need = [(r["path"],) for g in by_key.values() if len(g) > 1 for r in g if not r["v"].get("sha")]
    if need:
        state["done"] = 0
        progress("vhash", 0, len(need))

        def on_hash(res):
            state["done"] += 1
            r = recs.get(res["path"])
            if r is not None and res.get("sha"):
                r["v"]["sha"] = res["sha"]
                cache.vput(r["path"], r["size"], r["mtime_ns"], r["v"])
            progress("vhash", state["done"], len(need))

        _run_pool(need, min(workers, 2), on_hash, lambda: None, fn=hash_file, crash="This file could not be read")
    hashed = {t[0] for t in need}
    unhashed = sum(1 for p in hashed if p in recs and not recs[p]["v"].get("sha"))
    if unhashed:
        disc.notices.append(
            f"{unhashed} video(s) could not be read to the end, so they were not compared "
            "byte for byte. Check that the drive is still connected, then scan again."
        )
    for r in recs.values():
        if r["v"].get("sha"):
            r["sha"] = r["v"]["sha"]
        elif r["path"] in hashed:  # the full read failed: never call it an exact copy on a partial check
            r["sha"] = "u" + hashlib.sha1(r["path"].encode("utf-8", "replace")).hexdigest()
        else:  # no other video has this size and these first/last bytes
            r["sha"] = "q" + r["v"]["qhash"][:40]

    # pictures and sound only for videos that have a partner of the same length
    reps: dict = {}
    for r in sorted(recs.values(), key=lambda r: r["rel"].lower()):
        reps.setdefault(r["sha"], r)
    order = sorted((r for r in reps.values() if r["v"]["meta"].get("dur")), key=lambda r: r["v"]["meta"]["dur"])
    wanted = set()
    for i, a in enumerate(order):
        da = a["v"]["meta"]["dur"]
        for b in order[i + 1 :]:
            if not lengths_match(da, b["v"]["meta"]["dur"]):
                break
            wanted.add(a["path"])
            wanted.add(b["path"])
    sig_todo = [
        (p, recs[p]["v"]["meta"]["dur"])
        for p in sorted(wanted)
        if (recs[p]["v"].get("sig") or {}).get("v") != VIDEO_SIG_VERSION
    ]
    if sig_todo:
        state["done"] = 0
        progress("vsig", 0, len(sig_todo))

        def on_sig(res):
            state["done"] += 1
            r = recs.get(res["path"])
            if r is not None and res.get("sig"):
                r["v"]["sig"] = res["sig"]
                cache.vput(r["path"], r["size"], r["mtime_ns"], r["v"])
            progress("vsig", state["done"], len(sig_todo))

        _run_pool(
            sig_todo,
            workers,
            on_sig,
            lambda: None,
            fn=video_signature,
            crash="This file crashed the video decoder (it is probably damaged)",
        )
    # double check: a denser look (4x more moments) at videos that matched another video, so a short
    # inserted clip, end card or title between the first moments is noticed
    from .video import DENSE_MIN_DUR, VIDEO_PRESETS, same_video, unpack_sig

    U = {}
    for p in wanted:
        sig = recs[p]["v"].get("sig")
        if sig and sig.get("v") == VIDEO_SIG_VERSION:
            try:
                U[p] = unpack_sig(sig)
            except Exception:
                pass
    order2 = sorted(U, key=lambda p: recs[p]["v"]["meta"]["dur"])
    matched = set()
    for i, a in enumerate(order2):
        ma = recs[a]["v"]["meta"]
        for b in order2[i + 1 :]:
            mb = recs[b]["v"]["meta"]
            if not lengths_match(ma["dur"], mb["dur"]):
                break
            if max(ma["dur"], mb["dur"]) > DENSE_MIN_DUR and same_video(U[a], U[b], ma, mb, VIDEO_PRESETS["loose"])[0]:
                matched.add(a)
                matched.add(b)
    dense_todo = [
        (p, recs[p]["v"]["meta"]["dur"], "dense")
        for p in sorted(matched)
        if (recs[p]["v"].get("dsig") or {}).get("v") != VIDEO_SIG_VERSION
    ]
    if dense_todo:
        state["done"] = 0
        progress("vdense", 0, len(dense_todo))

        def on_dense(res):
            state["done"] += 1
            r = recs.get(res["path"])
            if r is not None and res.get("sig"):
                r["v"]["dsig"] = res["sig"]
                cache.vput(r["path"], r["size"], r["mtime_ns"], r["v"])
            progress("vdense", state["done"], len(dense_todo))

        _run_pool(
            dense_todo,
            workers,
            on_dense,
            lambda: None,
            fn=video_signature,
            crash="This file crashed the video decoder (it is probably damaged)",
        )
    out = []
    for r in recs.values():
        feat = dict(r["v"]["meta"])
        sig = r["v"].get("sig")
        feat["damaged"] = bool(sig and sig.get("missing", 0) > max(1, len(sig.get("t", [])) // 4))
        if sig and sig.get("sh"):
            vals = [x for x in sig["sh"] if x > 0]
            feat["sharp"] = float(sorted(vals)[len(vals) // 2]) if vals else 0.0
        out.append(
            {
                "path": r["path"],
                "rel": r["rel"],
                "size": r["size"],
                "mtime_ns": r["mtime_ns"],
                "sha": r["sha"],
                "feat": feat,
                "sig": sig if r["path"] in wanted else None,
                "dsig": r["v"].get("dsig") if r["path"] in matched else None,
            }
        )
    n = len(out)
    return out, f"{n:,} video{'s' if n != 1 else ''} checked", from_cache


def _run_pool(
    tasks: list,
    workers: int,
    handle,
    after_batch,
    fn=extract,
    crash: str = "This file crashed the image decoder (it is probably damaged)",
):
    """Run `extract` over tasks. If a damaged file crashes a decoder process, recover and retry
    the affected files one by one so a single bad file can never stop the scan. If helper
    processes cannot start at all (e.g. blocked by security software), finish in this process."""
    if not tasks:
        return
    workers = max(1, workers)
    queue = list(reversed(tasks))
    suspects: list = []
    window = workers * 4
    dead_starts = 0
    while queue and dead_starts < 2:
        inflight: dict = {}
        done_here = 0
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_CTX) as pool:
                while queue or inflight:
                    while queue and len(inflight) < window:
                        t = queue.pop()
                        inflight[pool.submit(fn, t)] = t
                    finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                    for fut in finished:
                        res = fut.result()  # raises BrokenProcessPool if a decoder crashed
                        del inflight[fut]
                        handle(res)
                        done_here += 1
                    after_batch()
            dead_starts = 0
        except (BrokenProcessPool, OSError):
            if done_here == 0:
                dead_starts += 1  # the pool died before finishing anything
                queue.extend(reversed(list(inflight.values())))
            else:
                dead_starts = 0
                suspects.extend(inflight.values())
    if dead_starts >= 2:
        _in_process(list(reversed(queue)) + suspects, handle, after_batch, fn)
        return
    # retry crash suspects one at a time so only the truly broken file is reported
    pool = None
    failures_in_a_row = 0
    for n, t in enumerate(suspects):
        if failures_in_a_row >= 3:  # helpers keep dying instantly: not a bad file, a bad setup
            _in_process(suspects[n:], handle, after_batch, fn)
            break
        try:
            if pool is None:
                pool = ProcessPoolExecutor(max_workers=1, mp_context=_CTX)
            res = pool.submit(fn, t).result()
            failures_in_a_row = 0
        except (BrokenProcessPool, OSError):
            res = {"path": t[0], "error": crash}
            failures_in_a_row += 1
            if pool is not None:
                pool.shutdown(wait=False)
            pool = None
        handle(res)
        after_batch()
    if pool is not None:
        pool.shutdown()


_WARNED = {"single": False}


def _in_process(tasks: list, handle, after_batch, fn=extract):
    if not _WARNED["single"]:
        print("\n  Note: helper processes could not start; continuing in a single process (slower).", flush=True)
        _WARNED["single"] = True
    for t in tasks:
        handle(fn(t))
        after_batch()
