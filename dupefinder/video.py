"""Videos: find the same video saved again (other name, other format, smaller or re-encoded copy).

How it works:
1. Every video is opened once to read its length, picture size, rotation, codec, sound and date.
2. Files of exactly the same size are compared byte for byte (SHA-256): exact copies.
3. Only videos whose length matches another video's length are looked at more closely. Pictures
   are taken at the same moments in both videos, on a fixed time grid (every 0.125, 0.25, 0.5, 1, 2,
   4 ... seconds) so that a copy and its original are sampled at exactly the same points. The
   loudness of the sound in the second after each moment is recorded too.
4. Two videos are the same video when their lengths match, their pictures match at (nearly) every
   moment, region by region, and their sound rises and falls together.

Needs the optional package 'av' (PyAV, which bundles FFmpeg). Without it, videos are skipped.
"""

from __future__ import annotations

import base64
import calendar
import hashlib
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .paths import lp

VIDEO_SIG_VERSION = 3  # bump when video fingerprints change (invalidates the cache)
GRID_MAX = 16  # at most 16 moments per video for the first comparison
DENSE_MAX = 64  # at most 64 moments for the double check of videos that matched
DENSE_MIN_DUR = 16.0  # shorter videos are already checked every second or more often
PIX = 32  # grey detail grid per moment (32 x 32)
AUDIO_RATE = 8000
AUDIO_SECONDS = 1.0
AUDIO_BINS = 20  # loudness every 50 ms
QUICK_BYTES = 1 << 20  # quick fingerprint: size + first and last megabyte
OPEN_TIMEOUT = (60.0, 60.0)
SILENT_DB = -55.0  # quieter than this everywhere = no usable sound

CODEC_LABEL = {
    "h264": "H.264",
    "hevc": "HEVC",
    "h265": "HEVC",
    "vp8": "VP8",
    "vp9": "VP9",
    "av1": "AV1",
    "mpeg4": "MPEG-4",
    "msmpeg4v3": "DivX",
    "msmpeg4v2": "MPEG-4 v2",
    "mpeg2video": "MPEG-2",
    "mpeg1video": "MPEG-1",
    "wmv1": "WMV",
    "wmv2": "WMV",
    "wmv3": "WMV",
    "vc1": "VC-1",
    "h263": "H.263",
    "flv1": "Flash",
    "mjpeg": "Motion JPEG",
    "prores": "ProRes",
    "dvvideo": "DV",
    "theora": "Theora",
    "rv40": "RealVideo",
    "rv30": "RealVideo",
    "cinepak": "Cinepak",
}


# ----------------------------------------------------------------- availability


def _av():
    try:
        import av  # noqa: F401

        return av
    except Exception:
        return None


def video_support() -> Tuple[bool, str]:
    av = _av()
    if av is None:
        return False, "video support not installed (package 'av')"
    try:
        lv = av.library_versions.get("libavcodec")
        extra = f", FFmpeg codecs {lv[0]}.{lv[1]}" if lv else ""
    except Exception:
        extra = ""
    return True, f"yes (PyAV {av.__version__}{extra})"


# ----------------------------------------------------------------- small helpers


def grid_step(dur: float, gmax: int = GRID_MAX) -> float:
    """Sampling step: a power of two (0.03125 s ... ) so grids of similar lengths line up, and the
    denser grid of the double check contains every moment of the first grid."""
    d = 0.125 if gmax <= GRID_MAX else 0.03125
    while dur / d > gmax:
        d *= 2.0
    return d


def grid_times(dur: float, gmax: int = GRID_MAX) -> List[float]:
    d = grid_step(dur, gmax)
    out, k = [], 1
    while k * d < dur - 0.15:
        out.append(k * d)
        k += 1
    if not out and dur > 0.05:
        out.append(min(dur / 2.0, 0.1))
    return out


def fmt_label(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {"qt": "MOV", "mpeg": "MPG", "mpe": "MPG", "m2ts": "M2TS", "mts": "MTS"}.get(ext, ext.upper() or "VIDEO")


def codec_label(name: Optional[str]) -> str:
    if not name:
        return ""
    return CODEC_LABEL.get(name.lower(), name.upper())


def fmt_duration(sec: Optional[float]) -> str:
    if sec is None:
        return "?"
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


_TZ = re.compile(r"([+-])(\d{2}):?(\d{2})$")


def parse_video_time(value) -> Optional[float]:
    """'2024-05-01T10:00:00.000000Z', '2024-05-01 10:00:00', '2024-05-01T12:00:00+0530' -> seconds
    (local clock time when an offset is given, like a photo's EXIF time). None when missing or bogus."""
    if not value:
        return None
    s = str(value).strip().replace("T", " ")
    if s.endswith("Z"):
        s = s[:-1]
    else:
        m = _TZ.search(s)
        if m and len(s) > 19:
            s = s[: m.start()]  # keep the clock time as written (local time for Apple's date)
    s = s.split(".")[0].strip()
    try:
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if not (1990 <= dt.year <= 2100):
        return None
    return float(calendar.timegm(dt.timetuple()))


def quick_hash(path: str, size: int) -> str:
    h = hashlib.sha256(str(size).encode())
    with open(lp(path), "rb") as f:
        h.update(f.read(QUICK_BYTES))
        if size > 2 * QUICK_BYTES:
            f.seek(size - QUICK_BYTES)
            h.update(f.read(QUICK_BYTES))
    return h.hexdigest()


def full_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(lp(path), "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(path: str):
    av = _av()
    if av is None:
        raise RuntimeError("Video support is not installed (run: pip install av)")
    return av.open(lp(path), timeout=OPEN_TIMEOUT)


ATTACHED_PIC = 0x0400  # FFmpeg's flag for a cover image stored as a "video" track


def _pick_video(c):
    """The main picture track (not a cover image or thumbnail)."""
    best = None
    for s in c.streams.video:
        try:
            if int(s.disposition) & ATTACHED_PIC:
                continue
        except Exception:
            if "attached" in str(getattr(s, "disposition", "")).lower():
                continue
        cc = getattr(s, "codec_context", None)
        if cc is None or not (cc.width or s.width):
            continue
        if best is None or (s.frames or 0) > (best.frames or 0):
            best = s
    return best


def _meta_get(md: dict, *keys) -> Optional[str]:
    low = {str(k).lower(): v for k, v in (md or {}).items()}
    for k in keys:
        v = low.get(k)
        if v:
            return str(v)
    return None


def _rotate(arr: np.ndarray, rotation) -> np.ndarray:
    k = int(round((rotation or 0) / 90.0)) % 4
    return np.rot90(arr, k) if k else arr


# ----------------------------------------------------------------- worker: probe


def probe_video(task: tuple) -> dict:
    """Worker entry point. task = (path,). Reads length and details, decodes the first picture
    (to learn the rotation and to make sure the video can be played). Never raises."""
    path = task[0]
    rec: dict = {"path": path}
    try:
        st = os.stat(lp(path))
        rec["size"], rec["mtime_ns"] = st.st_size, st.st_mtime_ns
        if st.st_size == 0:
            rec["error"] = "Empty file (0 bytes)"
            return rec
        rec["qhash"] = quick_hash(path, st.st_size)
        with _open(path) as c:
            fmt_name = (getattr(c.format, "name", "") or "").lower()
            if fmt_name.startswith(("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "gif", "tiff_pipe")):
                rec["skip"] = "a picture, not a video"
                return rec
            vs = _pick_video(c)
            if vs is None:
                rec["skip"] = "no picture track"
                return rec
            tb = float(vs.time_base) if vs.time_base else 0.0
            dur = None
            if vs.duration and tb:
                dur = float(vs.duration) * tb
            if (not dur or dur <= 0) and c.duration:
                dur = c.duration / 1_000_000.0
            w, h = int(vs.codec_context.width or vs.width or 0), int(vs.codec_context.height or vs.height or 0)
            sar = vs.sample_aspect_ratio
            sarf = float(sar) if sar and float(sar) > 0 else 1.0
            fps = float(vs.average_rate) if vs.average_rate else (float(vs.guessed_rate) if vs.guessed_rate else 0.0)
            aus = c.streams.audio[0] if c.streams.audio else None
            md = dict(c.metadata or {})
            vmd = dict(vs.metadata or {})
            ctime = parse_video_time(_meta_get(md, "creation_time")) or parse_video_time(
                _meta_get(vmd, "creation_time")
            )
            when = (
                parse_video_time(_meta_get(md, "com.apple.quicktime.creationdate"))
                or ctime
                or parse_video_time(_meta_get(md, "date"))
            )
            make = _meta_get(md, "com.apple.quicktime.make", "make", "com.android.manufacturer")
            model = _meta_get(md, "com.apple.quicktime.model", "model", "com.android.model")
            gps = bool(_meta_get(md, "location", "com.apple.quicktime.location.iso6709", "location-eng"))
            rotation = 0
            first_ok = False
            try:
                vs.thread_type = "AUTO"
                for frame in c.decode(vs):
                    rotation = int(getattr(frame, "rotation", 0) or 0)
                    w, h = frame.width or w, frame.height or h
                    first_ok = True
                    break
            except Exception:
                first_ok = False
            if not first_ok:
                rec["error"] = "The picture could not be decoded (damaged file or unsupported codec)"
                return rec
            dw, dh = w * sarf, float(h)
            if int(round(rotation / 90.0)) % 2:
                dw, dh = dh, dw
            rec["meta"] = {
                "dur": round(dur, 3) if dur else None,
                "w": int(round(dw)),
                "h": int(round(dh)),
                "cw": w,
                "ch": h,
                "rot": rotation,
                "fps": round(fps, 3),
                "vcodec": vs.codec_context.name or "",
                "audio": aus is not None,
                "acodec": (aus.codec_context.name if aus is not None else ""),
                "bitrate": int(c.bit_rate or 0),
                "dt": when,
                "ctime": ctime,
                "make": make,
                "model": model,
                "gps": gps,
                "fmt": fmt_label(path),
            }
    except MemoryError:
        rec["error"] = "Not enough memory to open this video"
    except PermissionError:
        rec["error"] = "No permission to read this file"
    except FileNotFoundError:
        rec["error"] = "File disappeared during the scan"
    except Exception as exc:
        rec["error"] = _friendly(exc)
    return rec


def _friendly(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    low = text.lower()
    if "invalid data" in low or "moov atom" in low or "end of file" in low:
        return "The video is damaged or incomplete (could not be read)"
    if "not installed" in low:
        return text
    return text[:200]


LIVE_MAX_SECONDS = 6.0  # iPhone Live Photo and Google/Samsung motion clips are about 1.5 to 3 seconds


def quick_duration(path: str) -> Optional[float]:
    """Length in seconds from the file header (no decoding). None when unknown or unreadable."""
    av = _av()
    if av is None:
        return None
    try:
        with av.open(lp(path), timeout=OPEN_TIMEOUT) as c:
            vs = _pick_video(c)
            if vs is not None and vs.duration and vs.time_base:
                return float(vs.duration * vs.time_base)
            return c.duration / 1_000_000.0 if c.duration else None
    except Exception:
        return None


def is_live_clip(path, size: int) -> bool:
    """A small video with the same name as a photo next to it: a Live Photo / motion photo clip that
    belongs to the photo (it moves with it). Longer videos are separate videos."""
    from .formats import LIVE_VIDEO_EXTS, LIVE_VIDEO_MAX

    ext = os.path.splitext(str(path))[1].lower()
    if ext not in LIVE_VIDEO_EXTS or size > LIVE_VIDEO_MAX:
        return False
    d = quick_duration(str(path)) if _av() is not None else None
    return d is None or d <= LIVE_MAX_SECONDS  # length unknown: the small-file rule alone decides


def hash_file(task: tuple) -> dict:
    """Worker entry point: full SHA-256 of a video (only for files of identical size)."""
    path = task[0]
    try:
        return {"path": path, "sha": full_hash(path)}
    except Exception as exc:
        return {"path": path, "error": _friendly(exc)}


# ----------------------------------------------------------------- worker: signature

# pictures around each moment: absorb timing differences between formats and frame rates (one frame
# at 25 fps, B-frame delays of AVI/MPEG files)
BURST = (-0.2, -0.1, -0.04, 0.0, 0.04, 0.1, 0.2)
CENTER = 3
DENSE_BURST = (-0.04, 0.0, 0.04)  # the double check: 4x more moments, fewer pictures around each
SMALL = 16  # grey grid of the pictures around the moment (16 x 16)


def _frame_features(frame) -> dict:
    arr = frame.to_ndarray(width=128, height=128, format="rgb24", interpolation="AREA")
    arr = _rotate(arr, getattr(frame, "rotation", 0))
    img = Image.fromarray(np.ascontiguousarray(arr))
    gray = img.convert("L")
    from .features import dhash, phash

    return {
        "ph": phash(gray),
        "dh": dhash(gray),
        "col": np.asarray(img.resize((4, 4), Image.Resampling.BOX), dtype=np.uint8).ravel(),
        "p16": np.asarray(gray.resize((SMALL, SMALL), Image.Resampling.BOX), dtype=np.uint8).ravel(),
        "px": np.asarray(gray.resize((PIX, PIX), Image.Resampling.BOX), dtype=np.uint8).ravel(),
        "std": float(np.asarray(gray, dtype=np.float32).std()),
        "sh": _sharpness(gray),
    }


def _sharpness(gray: Image.Image) -> float:
    g = np.asarray(gray, dtype=np.float32)
    lap = -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def _keyframe_times(vs, tb: float, start: float) -> List[float]:
    try:
        ie = vs.index_entries
        if ie is None or len(ie) == 0:
            return []
        return sorted(e.timestamp * tb - start for e in ie if e.is_keyframe)
    except Exception:
        return []


def _video_samples(c, vs, times: List[float], offsets=BURST) -> Dict[Tuple[int, int], dict]:
    """Pictures closest to each moment and to the moments 0.1 and 0.2 s around it.
    Returns {(moment index, burst index): features}. Seeks only when that is cheaper than decoding
    on; when a seek lands in the wrong place (some MPEG and AVI files), seeks earlier or reads from
    the start of the file instead."""
    import bisect

    tb = float(vs.time_base)
    start = float(vs.start_time) * tb if vs.start_time is not None else 0.0
    keys = _keyframe_times(vs, tb, start)
    fps = float(vs.average_rate) if vs.average_rate else 30.0
    tol = max(0.12, 1.5 / max(fps, 1.0))  # a picture further away than this is not that moment
    wanted: Dict[float, List[Tuple[int, int]]] = {}
    for i, t in enumerate(times):
        for j, o in enumerate(offsets):
            if t + o >= 0:
                wanted.setdefault(round(t + o, 5), []).append((i, j))
    st = {"frames": None, "prev": None, "cur": None}

    def tm(fr):
        return fr.time - start

    def jump(sec):
        try:
            if sec is None:
                c.seek(0)
            else:
                c.seek(max(0, int((start + sec) / tb)), stream=vs, backward=True)
            st["frames"] = c.decode(vs)
        except Exception:
            st["frames"] = None
        st["prev"] = st["cur"] = None

    def step() -> bool:
        if st["frames"] is None:
            return False
        try:
            for fr in st["frames"]:
                if fr.time is None:
                    continue
                st["prev"], st["cur"] = st["cur"], fr
                return True
        except Exception:
            pass
        st["frames"] = None
        return False

    def seek_to(w) -> bool:
        for back in (0.0, 2.0, 8.0, 30.0):
            jump(max(0.0, w - back))
            if step() and tm(st["cur"]) <= w + tol:
                return True
        return False

    def closest(w):
        """The picture on screen at time w (or the one starting just after it)."""
        while st["cur"] is None or tm(st["cur"]) < w:
            if not step():
                break
        prev, cur = st["prev"], st["cur"]
        cands = []
        if cur is not None and tm(cur) > w:
            if tm(cur) - w <= tol:
                cands.append(cur)
            if prev is not None and tm(prev) <= w:
                cands.append(prev)  # still on screen: variable frame rate files hold pictures
        elif cur is not None and w - tm(cur) <= tol:
            cands.append(cur)  # end of the file: the last picture
        if not cands:
            return None
        return min(cands, key=lambda f: abs(tm(f) - w))

    out: Dict[Tuple[int, int], dict] = {}
    sequential = False
    done_pts: Dict[int, dict] = {}
    for w in sorted(wanted):
        if not sequential:
            cur_t = tm(st["cur"]) if st["cur"] is not None else None
            if st["frames"] is None and cur_t is not None and w <= cur_t:
                need = False
            elif cur_t is None or st["frames"] is None:
                need = True
            elif w <= cur_t:
                need = False
            elif keys:
                k = bisect.bisect_right(keys, w) - 1
                need = k >= 0 and keys[k] > cur_t + 1e-3  # a keyframe lies between: jump to it
            else:
                need = (w - cur_t) > 2.0
            if need and not seek_to(w):
                sequential = True  # read forward from the start instead
                jump(None)
        fr = closest(w)
        if fr is None:
            continue
        key = int(fr.pts) if fr.pts is not None else id(fr)
        feat = done_pts.get(key)
        if feat is None:
            try:
                feat = _frame_features(fr)
            except Exception:
                continue
            done_pts[key] = feat
            if len(done_pts) > 64:
                done_pts.pop(next(iter(done_pts)))
        for ij in wanted[w]:
            out[ij] = feat
    return out


def _audio_samples(c, aus, times: List[float]) -> Dict[float, np.ndarray]:
    """Loudness (dB) every 50 ms for the second after each time."""
    import av

    tb = float(aus.time_base) if aus.time_base else 0.0
    start = float(aus.start_time) * tb if (aus.start_time is not None and tb) else 0.0
    need = int(AUDIO_RATE * AUDIO_SECONDS)
    per = need // AUDIO_BINS
    out: Dict[float, np.ndarray] = {}
    if not tb:
        return out
    for t in times:
        for back in (0.2, 2.0, 8.0):
            try:
                c.seek(max(0, int((start + max(t - back, 0.0)) / tb)), stream=aus, backward=True)
                res = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_RATE)
                buf: List[np.ndarray] = []
                have = 0
                first_time = None
                for af in c.decode(aus):
                    if af.time is None:
                        continue
                    for rf in res.resample(af):
                        if rf.time is None:
                            continue
                        a = rf.to_ndarray().reshape(-1).astype(np.float32)
                        if first_time is None:
                            first_time = rf.time - start
                        buf.append(a)
                        have += len(a)
                    if first_time is not None and first_time + have / AUDIO_RATE >= t + AUDIO_SECONDS:
                        break
            except Exception:
                break
            if first_time is None or not buf:
                break
            if first_time > t + 0.03:  # the seek landed after the moment: try from further back
                continue
            x = np.concatenate(buf)
            skip = int(round((t - first_time) * AUDIO_RATE))
            x = x[skip : skip + need]
            if len(x) >= per * 4:
                nb = len(x) // per
                e = (x[: nb * per].reshape(nb, per) ** 2).mean(1)
                env = np.full(AUDIO_BINS, np.nan, np.float32)
                env[:nb] = 10.0 * np.log10(e + 1e-10)
                out[t] = env
            break
    return out


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def video_signature(task: tuple) -> dict:
    """Worker entry point. task = (path, duration) for the first comparison, or (path, duration,
    "dense") for the double check of videos that matched. Never raises."""
    path, dur = task[0], task[1]
    dense = len(task) > 2 and task[2] == "dense"
    gmax, offsets = (DENSE_MAX, DENSE_BURST) if dense else (GRID_MAX, BURST)
    center = offsets.index(0.0)
    rec: dict = {"path": path}
    try:
        times = grid_times(float(dur), gmax)
        if dense:  # only the moments between those of the first comparison
            main = {round(t, 5) for t in grid_times(float(dur), GRID_MAX)}
            times = [t for t in times if round(t, 5) not in main]
        with _open(path) as c:
            vs = _pick_video(c)
            if vs is None:
                raise RuntimeError("no picture track")
            vs.thread_type = "AUTO"
            pics = _video_samples(c, vs, times, offsets)
            aus = c.streams.audio[0] if c.streams.audio and not dense else None
            snd: Dict[float, np.ndarray] = {}
            if aus is not None:
                snd = _audio_samples(c, aus, times)
        if dense:
            k, nb = len(times), len(offsets)
            ok = np.zeros((k, nb), bool)
            ph = np.zeros((k, nb), np.uint64)
            std = np.zeros(k, np.float32)
            for (i, j), f in pics.items():
                ok[i, j] = True
                ph[i, j] = f["ph"]
                if j == center:
                    std[i] = f["std"]
            rec["sig"] = {
                "v": VIDEO_SIG_VERSION,
                "dense": True,
                "step": grid_step(float(dur), gmax),
                "t": [round(t, 5) for t in times],
                "b": list(offsets),
                "ok": _b64(np.packbits(ok)),
                "ph": _b64(ph),
                "std": [round(float(x), 2) for x in std],
                "missing": int(k - ok[:, center].sum()),
            }
            return rec
        k, nb = len(times), len(offsets)
        ok = np.zeros((k, nb), bool)
        ph = np.zeros((k, nb), np.uint64)
        dh = np.zeros((k, nb), np.uint64)
        col = np.zeros((k, nb, 48), np.uint8)
        p16 = np.zeros((k, nb, SMALL * SMALL), np.uint8)
        px = np.zeros((k, PIX * PIX), np.uint8)
        std = np.zeros(k, np.float32)
        sh = np.zeros(k, np.float32)
        for (i, j), f in pics.items():
            ok[i, j] = True
            ph[i, j], dh[i, j] = f["ph"], f["dh"]
            col[i, j], p16[i, j] = f["col"], f["p16"]
            if j == center:
                px[i], std[i], sh[i] = f["px"], f["std"], f["sh"]
        env = np.full((k, AUDIO_BINS), np.nan, np.float32)
        for i, t in enumerate(times):
            if t in snd:
                env[i] = snd[t]
        loud = float(np.nanmax(env)) if np.isfinite(env).any() else None
        rec["sig"] = {
            "v": VIDEO_SIG_VERSION,
            "step": grid_step(float(dur), gmax),
            "t": [round(t, 5) for t in times],
            "b": list(offsets),
            "ok": _b64(np.packbits(ok)),
            "ph": _b64(ph),
            "dh": _b64(dh),
            "col": _b64(col),
            "p16": _b64(p16),
            "px": _b64(px),
            "std": [round(float(x), 2) for x in std],
            "sh": [round(float(x), 1) for x in sh],
            "aud": _b64(np.nan_to_num(env, nan=-200.0).astype(np.float16)) if aus is not None and k else None,
            "loud": None if loud is None else round(loud, 1),
            "missing": int(k - ok[:, center].sum()),
        }
    except MemoryError:
        rec["error"] = "Not enough memory to read this video"
    except Exception as exc:
        rec["error"] = _friendly(exc)
    return rec


# ----------------------------------------------------------------- comparing two videos


def lengths_match(a: Optional[float], b: Optional[float]) -> bool:
    """Same length: within 0.35 s, up to 1 s for videos over 6 minutes (re-encoding can add or drop a
    frame or two; measured on converted copies: at most 0.23 s). A trimmed version is not a copy."""
    if not a or not b:
        return False
    return abs(a - b) <= max(0.35, min(1.0, 0.001 * max(a, b)))


def unpack_sig(sig: dict) -> dict:
    offsets = tuple(sig.get("b") or BURST)
    k, nb = len(sig["t"]), len(offsets)
    dec = lambda key, dt: np.frombuffer(base64.b64decode(sig[key]), dtype=dt)
    out = {
        "t": np.round(np.asarray(sig["t"], dtype=np.float64) * 1_000_000).astype(np.int64),
        "ok": np.unpackbits(dec("ok", np.uint8))[: k * nb].reshape(k, nb).astype(bool),
        "ph": dec("ph", np.uint64).reshape(k, nb),
        "dh": dec("dh", np.uint64).reshape(k, nb),
        "col": dec("col", np.uint8).reshape(k, nb, 48).astype(np.float32),
        "p16": dec("p16", np.uint8).reshape(k, nb, SMALL, SMALL).astype(np.float32),
        "px": dec("px", np.uint8).reshape(k, PIX, PIX).astype(np.float32),
        "std": np.asarray(sig["std"], dtype=np.float32),
        "sh": np.asarray(sig.get("sh") or [0.0] * k, dtype=np.float32),
        "c": offsets.index(0.0),
        "aud": None,
    }
    if sig.get("aud"):
        a = dec("aud", np.float16).reshape(k, AUDIO_BINS).astype(np.float32)
        a[a < -150] = np.nan
        out["aud"] = a
    return out


def unpack_dense(sig: dict) -> dict:
    offsets = tuple(sig["b"])
    k, nb = len(sig["t"]), len(offsets)
    return {
        "t": np.round(np.asarray(sig["t"], dtype=np.float64) * 1_000_000).astype(np.int64),
        "step": float(sig["step"]),
        "ok": np.unpackbits(np.frombuffer(base64.b64decode(sig["ok"]), np.uint8))[: k * nb].reshape(k, nb).astype(bool),
        "ph": np.frombuffer(base64.b64decode(sig["ph"]), np.uint64).reshape(k, nb),
        "std": np.asarray(sig["std"], dtype=np.float32),
        "c": offsets.index(0.0),
    }


def _audio_corr(ea: np.ndarray, eb: np.ndarray) -> Optional[float]:
    """Best correlation of two loudness curves, allowing up to 150 ms of delay between them."""
    best = None
    for s in range(-3, 4):
        if s >= 0:
            x, y = ea[:, s:], eb[:, : AUDIO_BINS - s]
        else:
            x, y = ea[:, : AUDIO_BINS + s], eb[:, -s:]
        x, y = x.ravel(), y.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 30:
            continue
        x, y = x[m], y[m]
        sx, sy = x.std(), y.std()
        if sx < 1e-6 or sy < 1e-6:
            continue
        c = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        best = c if best is None else max(best, c)
    return best


def _grid(a: np.ndarray, g: int) -> np.ndarray:
    n = a.shape[-1]
    if g == n:
        return a
    f = n // g
    return a.reshape(a.shape[:-2] + (g, f, g, f)).mean((-3, -1))


def pair_metrics(A: dict, B: dict, small_side: int) -> dict:
    """Compare two unpacked signatures at the moments both have. For every moment: the centre
    pictures, and the best match of each centre picture among the other video's nearby pictures."""
    from .grouping import popcount

    _, ia, ib = np.intersect1d(A["t"], B["t"], return_indices=True)
    ca_, cb_ = A["c"], B["c"]
    both = A["ok"][ia, ca_] & B["ok"][ib, cb_]
    ia, ib = ia[both], ib[both]
    n = len(ia)
    out = {"n": int(n)}
    if not n:
        return out
    info = (A["std"][ia] >= 4.0) & (B["std"][ib] >= 4.0)
    g = PIX if small_side >= 300 else SMALL
    # centre against centre
    dp = popcount(A["ph"][ia, ca_] ^ B["ph"][ib, cb_]).astype(np.int64)
    dc = np.abs(A["col"][ia, ca_] - B["col"][ib, cb_]).mean(-1)
    D = np.abs(_grid(A["px"][ia], g) - _grid(B["px"][ib], g)).reshape(n, -1)
    rc = np.percentile(D, 95, axis=1)
    # centre against the nearby pictures of the other video (and the other way round)
    best_r = np.full(n, np.inf, np.float32)
    best_dp = np.full(n, 64, np.int64)
    min_dp = dp.copy()  # closest picture pattern among all nearby pictures (both directions)
    best_dc = np.full(n, np.inf, np.float32)
    best_map = np.zeros((n, SMALL * SMALL), np.float32)  # signed A - B of the best-aligned pictures
    for X, Y, ix, iy, sign in ((A, B, ia, ib, 1.0), (B, A, ib, ia, -1.0)):
        xc = X["c"]
        cph, ccol, cp = X["ph"][ix, xc], X["col"][ix, xc], X["p16"][ix, xc].reshape(n, -1)
        for j in range(Y["ok"].shape[1]):
            okj = Y["ok"][iy, j]
            if not okj.any():
                continue
            dpj = popcount(cph ^ Y["ph"][iy, j]).astype(np.int64)
            dcj = np.abs(ccol - Y["col"][iy, j]).mean(-1)
            diff = cp - Y["p16"][iy, j].reshape(n, -1)
            rj = np.percentile(np.abs(diff), 95, axis=1)
            better = okj & (rj < best_r)
            best_r = np.where(better, rj, best_r)
            best_dp = np.where(better, dpj, best_dp)
            min_dp = np.where(okj, np.minimum(min_dp, dpj), min_dp)
            best_dc = np.where(better, dcj, best_dc)
            best_map = np.where(better[:, None], sign * diff, best_map)
    # an overlay (logo, subtitles, title) or a local edit differs in the same place at every moment;
    # compression noise and timing differences do not, so they average out
    persist = 0.0
    if info.sum() >= 3:
        persist = float(np.abs(best_map[info].mean(0)).max())
    # the same on the finer 32x32 picture: a small title or name that differs between two versions.
    # The median over the moments ignores moments where the pictures are a frame apart.
    fine = 0.0
    if info.sum() >= 5:
        D = A["px"][ia][info] - B["px"][ib][info]
        D = D - np.median(D.reshape(len(D), -1), axis=1)[:, None, None]
        fine = float(np.abs(np.median(D, axis=0)).max())
    # colours, brightness and contrast: a copy keeps them, an edited version shifts them everywhere
    ca = A["col"][ia, ca_].reshape(n, 16, 3)
    cb = B["col"][ib, cb_].reshape(n, 16, 3)
    bias = float((ca.mean((1, 2)) - cb.mean((1, 2)))[info].mean()) if info.any() else 0.0
    chroma_a = float((ca.max(2) - ca.min(2)).mean(1)[info].mean()) if info.any() else 0.0
    chroma_b = float((cb.max(2) - cb.min(2)).mean(1)[info].mean()) if info.any() else 0.0
    con_a = float(A["std"][ia][info].mean()) if info.any() else 0.0
    con_b = float(B["std"][ib][info].mean()) if info.any() else 0.0
    sh_a, sh_b = A["sh"][ia][info], B["sh"][ib][info]
    out.update(
        persist=persist,
        fine=fine,
        bias=bias,
        chroma=(chroma_a, chroma_b),
        contrast=(con_a, con_b),
        sharp=(float(np.median(sh_a)) if len(sh_a) else 0.0, float(np.median(sh_b)) if len(sh_b) else 0.0),
    )

    # moments where the picture changes (something moves) in both videos: different recordings of the
    # same still scene cannot agree there, so these moments are the real proof of "same recording"
    def change(X, ix):
        P = X["px"][ix].reshape(n, -1)
        d = np.zeros(n, np.float32)
        if n >= 2:
            step = np.percentile(np.abs(P[1:] - P[:-1]), 95, axis=1)
            d[1:] = np.maximum(d[1:], step)
            d[:-1] = np.maximum(d[:-1], step)
        return d

    out.update(
        info=info,
        dp=dp,
        dc=dc,
        rc=rc,
        grid=g,
        bdp=best_dp,
        mdp=min_dp,
        bdc=best_dc,
        br=best_r,
        cha=change(A, ia),
        chb=change(B, ib),
    )
    if A["aud"] is not None and B["aud"] is not None:
        ea, eb = A["aud"][ia], B["aud"][ib]
        fa, fb = np.isfinite(ea), np.isfinite(eb)
        out["aud_loud"] = (float(np.nanmax(ea)) if fa.any() else -200.0, float(np.nanmax(eb)) if fb.any() else -200.0)
        out["aud_spread"] = (float(np.nanstd(ea)) if fa.any() else 0.0, float(np.nanstd(eb)) if fb.any() else 0.0)
        out["acorr"] = _audio_corr(ea, eb)
    return out


@dataclass(frozen=True)
class VideoSettings:
    phash: int  # max picture-pattern distance at a moment (out of 64)
    colour: float  # max colour change at a moment (0-255)
    region: float  # max local difference (95th percentile of a 32x32 / 16x16 grey grid)
    misses: float  # share of moments allowed to differ (a damaged or mistimed picture)
    sound: float  # min correlation of the loudness curves when both videos have real sound
    changes: int  # without sound: matching moments where the picture changes in both
    persist: float  # max difference that stays in the same place at every moment (logo, subtitles)
    fine: float  # the same on the finer 32x32 picture (a small title that differs)


CHANGE_LEVEL = 8.0  # a moment "changes" when it differs this much from the moments next to it


# Calibrated on 299 videos from real footage (106 originals, 181 copies in 14 formats, look-alikes):
# copies: picture pattern <= 6 and local difference <= 4.6 at 99.5% of moments (allowing 0.2 s timing
# differences between formats); sound curves correlate >= 0.91. Different clips of an empty, fixed
# camera scene match at every moment, so without sound a match must also include moments where
# something moves in both videos (measured: 0 of 34 such look-alike pairs have any).
# Edited versions (blur, colour grades, watermark, subtitles, title cards, end card, inserted footage,
# vignette, crop) are told apart by: a difference that stays in one place (copies <= 6.8 at 99%,
# edits 12 to 121), brightness shift (copies <= 2.8), colourfulness (copies <= 3 apart or <= 15%),
# contrast (copies <= 7%), sharpness (copies <= 4.4x, blur 37x) and any moment showing other footage.
# Two versions with a different small title in the same place: finer 32x32 check (copies <= 12, edit 23).
VIDEO_PRESETS: Dict[str, VideoSettings] = {
    "strict": VideoSettings(10, 5.0, 8.0, 0.05, 0.8, 3, 9.0, 14.0),
    "normal": VideoSettings(12, 6.0, 10.0, 0.10, 0.7, 2, 11.0, 16.0),
    "loose": VideoSettings(14, 7.0, 12.0, 0.15, 0.6, 2, 13.0, 18.0),
}
MAX_BIAS = 4.0  # brightness shift (0-255)
MAX_CONTRAST = 1.08  # contrast ratio
MAX_BLUR = 10.0  # sharpness ratio


def same_video(A: dict, B: dict, ma: dict, mb: dict, vs: VideoSettings) -> Tuple[bool, float, str]:
    """Decide whether two videos are the same video. Returns (same, score 0-100, evidence)."""
    if not lengths_match(ma.get("dur"), mb.get("dur")):
        return False, 0.0, "length"
    if abs(math.log(max(ma["w"], 1) / max(ma["h"], 1)) - math.log(max(mb["w"], 1) / max(mb["h"], 1))) > 0.06:
        return False, 0.0, "shape"
    # quick look first: most different videos already differ in picture pattern (copies: median <= 4)
    from .grouping import popcount

    _, qa, qb = np.intersect1d(A["t"], B["t"], return_indices=True)
    q = A["ok"][qa, A["c"]] & B["ok"][qb, B["c"]]
    if q.sum() >= 3 and float(np.median(popcount(A["ph"][qa[q], A["c"]] ^ B["ph"][qb[q], B["c"]]))) > 20:
        return False, 0.0, "pictures differ"
    m = pair_metrics(A, B, min(ma["w"], ma["h"], mb["w"], mb["h"]))
    if not m["n"]:
        return False, 0.0, "no common moments"
    info = m["info"]
    n = int(info.sum())
    if n < 3:
        return False, 0.0, "blank pictures"
    okc = (m["dp"] <= vs.phash) & (m["dc"] <= vs.colour) & (m["rc"] <= vs.region)
    okb = (m["bdp"] <= vs.phash) & (m["bdc"] <= vs.colour) & (m["br"] <= vs.region)
    okc_or_b = okc | okb
    # a moment showing other footage (inserted clip, end card, swapped photo) is never a copy
    if ((m["mdp"] > 20) & info).any():
        return False, 0.0, "other footage at some moments"
    good = okc_or_b[info]
    misses = int((~good).sum())
    allowed = max(1 if n >= 6 else 0, int(vs.misses * n))
    if misses > allowed:
        return False, 0.0, "pictures differ"
    edit = _edited(m, vs)
    if edit:
        return False, 0.0, edit
    score = 100.0 * float(good.mean()) - float(np.median(np.minimum(m["rc"], m["br"])[info]))
    audible = m.get("acorr") is not None and min(m["aud_loud"]) > SILENT_DB and min(m["aud_spread"]) > 2.0
    if audible:
        if m["acorr"] < vs.sound:
            return False, 0.0, "sound differs"
        return True, score, "pictures and sound match"
    moving = int(((m["cha"] >= CHANGE_LEVEL) & (m["chb"] >= CHANGE_LEVEL) & okc_or_b & info).sum())
    if moving >= vs.changes:
        return True, score, "pictures match at every moment"
    return False, 0.0, "still scene without sound"


OTHER_RUN = 2  # dense moments in a row showing other footage: an inserted clip, title or end card


def other_footage_between(A: dict, B: dict) -> bool:
    """Double check with the dense signatures (the moments between those of the first comparison)
    of two videos that already match: True when moments in a row show completely different pictures,
    i.e. something was inserted or replaced there (a clip, a title, an end card). A single odd
    moment (a scene cut that falls between two frames of a low frame rate copy) is not enough."""
    from .grouping import popcount

    if abs(A["step"] - B["step"]) > 1e-9:
        return False
    _, ia, ib = np.intersect1d(A["t"], B["t"], return_indices=True)
    if len(ia) < OTHER_RUN:
        return False
    worst = np.full(len(ia), 64, np.int64)  # best match of each centre picture among nearby pictures
    for X, Y, ix, iy in ((A, B, ia, ib), (B, A, ib, ia)):
        cx = X["c"]
        okx = X["ok"][ix, cx]
        for j in range(Y["ok"].shape[1]):
            d = popcount(X["ph"][ix, cx] ^ Y["ph"][iy, j]).astype(np.int64)
            worst = np.where(okx & Y["ok"][iy, j], np.minimum(worst, d), worst)
    seen = A["ok"][ia, A["c"]] & B["ok"][ib, B["c"]] & (A["std"][ia] >= 4.0) & (B["std"][ib] >= 4.0)
    hard = seen & (worst > 20)
    t = A["t"][ia]
    gap = int(round(A["step"] * 1_000_000 * 1.01))
    run = 0
    for x in range(len(hard)):
        if hard[x] and run and t[x] - t[x - 1] <= gap:
            run += 1
        else:
            run = 1 if hard[x] else 0
        if run >= OTHER_RUN:
            return True
    return False


def _edited(m: dict, vs: VideoSettings) -> Optional[str]:
    """Why the second video looks like an edited version of the first (None for a plain copy)."""
    if m["persist"] > vs.persist or m.get("fine", 0.0) > vs.fine:
        return "an overlay or local change (logo, subtitles, title)"
    if abs(m["bias"]) > MAX_BIAS:
        return "brightness changed"
    ca, cb = m["chroma"]
    if abs(ca - cb) > 3.0 and abs(ca - cb) / max(ca, cb, 6.0) > 0.15:
        return "colours changed"
    na, nb = m["contrast"]
    if min(na, nb) > 0 and max(na, nb) / min(na, nb) > MAX_CONTRAST:
        return "contrast changed"
    sa, sb = m["sharp"]
    if min(sa, sb) > 0 and max(sa, sb) / min(sa, sb) > MAX_BLUR:
        return "blurred version"
    return None


# ----------------------------------------------------------------- previews


def _picture_at(path: str, frac: float, max_side: int) -> Image.Image:
    with _open(path) as c:
        vs = _pick_video(c)
        if vs is None:
            raise RuntimeError("no picture track")
        vs.thread_type = "AUTO"
        tb = float(vs.time_base)
        start = float(vs.start_time) * tb if vs.start_time is not None else 0.0
        dur = float(vs.duration) * tb if vs.duration else (c.duration / 1e6 if c.duration else 0.0)
        img = _decode_one(c, vs, start, tb, max(0.0, dur * frac))
        if img is None and dur:
            img = _decode_one(c, vs, start, tb, 0.0)
        if img is None:
            raise RuntimeError("no picture could be decoded")
        sar = vs.sample_aspect_ratio
        return _display_image(img, float(sar) if sar and float(sar) > 0 else 1.0, max_side)


def _decode_one(c, vs, start, tb, t):
    try:
        c.seek(max(0, int((start + t) / tb)), stream=vs, backward=True)
        for fr in c.decode(vs):
            if fr.time is not None and fr.time - start >= t - 0.02:
                return fr
    except Exception:
        return None
    return None


def _display_image(fr, sar: float, max_side: int) -> Image.Image:
    w, h = fr.width, fr.height
    dw = max(1, int(round(w * sar)))
    s = min(1.0, max_side / max(dw, h))
    tw, th = max(2, int(round(dw * s))), max(2, int(round(h * s)))
    arr = fr.to_ndarray(width=tw, height=th, format="rgb24", interpolation="AREA")
    arr = _rotate(arr, getattr(fr, "rotation", 0))
    return Image.fromarray(np.ascontiguousarray(arr))


def make_video_thumbnail(task: tuple) -> Optional[str]:
    """Worker entry point: (src, dst jpg, max side). A picture from about a third of the way in."""
    src, dst, side = task
    try:
        img = _picture_at(src, 0.33, side)
        tmp = dst + ".tmp"
        img.save(tmp, "JPEG", quality=82, optimize=True)
        os.replace(tmp, dst)
        return None
    except Exception as exc:
        return str(exc)[:200]


def make_video_strip(src: str, dst: str, count: int = 6, width: int = 320) -> Optional[str]:
    """Six pictures across the video in one image (shown when the browser cannot play the file)."""
    try:
        with _open(src) as c:
            vs = _pick_video(c)
            if vs is None:
                raise RuntimeError("no picture track")
            vs.thread_type = "AUTO"
            tb = float(vs.time_base)
            start = float(vs.start_time) * tb if vs.start_time is not None else 0.0
            dur = float(vs.duration) * tb if vs.duration else (c.duration / 1e6 if c.duration else 0.0)
            sar = vs.sample_aspect_ratio
            sarf = float(sar) if sar and float(sar) > 0 else 1.0
            tiles = []
            for i in range(count):
                fr = _decode_one(c, vs, start, tb, dur * (i + 0.5) / count)
                if fr is not None:
                    tiles.append(_display_image(fr, sarf, width))
        if not tiles:
            raise RuntimeError("no picture could be decoded")
        tw = max(t.width for t in tiles)
        th = max(t.height for t in tiles)
        cols = 3 if len(tiles) > 2 else len(tiles)
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * tw + (cols - 1) * 6, rows * th + (rows - 1) * 6), (16, 17, 19))
        for i, t in enumerate(tiles):
            r, cidx = divmod(i, cols)
            sheet.paste(t, (cidx * (tw + 6) + (tw - t.width) // 2, r * (th + 6) + (th - t.height) // 2))
        tmp = dst + ".tmp"
        sheet.save(tmp, "JPEG", quality=84)
        os.replace(tmp, dst)
        return None
    except Exception as exc:
        return str(exc)[:200]


# ----------------------------------------------------------------- grouping

VIDEO_FORMAT_RANK = {
    "MOV": 3,
    "MP4": 3,
    "M4V": 3,
    "MTS": 3,
    "M2TS": 3,
    "MXF": 3,
    "DV": 3,
    "MKV": 2,
    "WEBM": 2,
    "AVI": 1,
    "WMV": 1,
    "ASF": 1,
    "MPG": 1,
    "VOB": 1,
    "TS": 2,
    "3GP": 0,
    "3G2": 0,
    "FLV": 0,
    "F4V": 0,
    "RM": 0,
    "RMVB": 0,
    "OGV": 1,
    "DIVX": 1,
}


def _vpixels(r: dict) -> int:
    return int(r["feat"]["w"]) * int(r["feat"]["h"])


def _vcamera(r: dict) -> bool:
    f = r["feat"]
    return bool(f.get("make") or f.get("model") or f.get("gps"))


def _vbitrate(r: dict) -> float:
    d = r["feat"].get("dur") or 0
    return r["size"] / d if d else 0.0


def _vsharp(r: dict) -> float:
    return float(r["feat"].get("sharp") or 0.0)


def _vcompare(a: dict, b: dict) -> int:
    """>0 when `a` is the better copy of the same video."""
    from .grouping import name_penalty

    fa, fb = a["feat"], b["feat"]
    if bool(fa.get("damaged")) != bool(fb.get("damaged")):
        return 1 if fb.get("damaged") else -1
    da, db = fa.get("dur") or 0.0, fb.get("dur") or 0.0
    if abs(da - db) > 0.25:  # a few frames cut off: keep the complete one
        return 1 if da > db else -1
    ca, cb = _vcamera(a), _vcamera(b)
    pa, pb = _vpixels(a), _vpixels(b)
    if ca != cb:  # the file that still has the camera details is the original, unless the other is far larger
        cam, other = (a, b) if ca else (b, a)
        if _vpixels(other) <= 4.5 * _vpixels(cam):
            return 1 if cam is a else -1
    if pa != pb and max(pa, pb) > 1.02 * min(pa, pb):
        return 1 if pa > pb else -1
    if bool(fa.get("audio")) != bool(fb.get("audio")):
        return 1 if fa.get("audio") else -1
    sa, sb = _vsharp(a), _vsharp(b)
    if min(sa, sb) > 0 and max(sa, sb) >= 1.5 * min(sa, sb):  # blurrier = a worse copy
        return 1 if sa > sb else -1
    ra, rb = _vbitrate(a), _vbitrate(b)
    # much more compressed = a worse copy; only comparable within one codec (old codecs such as WMV,
    # FLV or MPEG-2 need 2 to 3 times the data of H.264 for the same picture)
    if fa.get("vcodec") == fb.get("vcodec") and min(ra, rb) > 0 and max(ra, rb) >= 1.5 * min(ra, rb):
        return 1 if ra > rb else -1
    ka = (
        -name_penalty(a["rel"]),
        -(a["mtime_ns"] // 1_000_000_000),
        VIDEO_FORMAT_RANK.get(fa.get("fmt"), 1),
        ra,
        -len(a["rel"]),
        a["rel"],
    )
    kb = (
        -name_penalty(b["rel"]),
        -(b["mtime_ns"] // 1_000_000_000),
        VIDEO_FORMAT_RANK.get(fb.get("fmt"), 1),
        rb,
        -len(b["rel"]),
        b["rel"],
    )
    return (ka > kb) - (ka < kb)


def video_key(r: dict):
    from functools import cmp_to_key

    return cmp_to_key(_vcompare)(r)


def build_video_groups(records: List[dict], sensitivity: str = "normal") -> List[dict]:
    """records: readable videos from the scan (path, rel, size, mtime_ns, sha, feat, sig).
    Returns review groups: exact copies, and the same video saved again (other format / quality)."""
    from .grouping import UnionFind, complete_linkage

    vs = VIDEO_PRESETS.get(sensitivity, VIDEO_PRESETS["normal"])
    if len(records) < 2:
        return []
    by_sha: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_sha.setdefault(r["sha"], []).append(i)
    shas = list(by_sha)
    reps = [by_sha[s][0] for s in shas]
    R = len(reps)
    U: Dict[int, dict] = {}
    Dn: Dict[int, dict] = {}
    for k, i in enumerate(reps):
        for key, store, unpack in (("sig", U, unpack_sig), ("dsig", Dn, unpack_dense)):
            sig = records[i].get(key)
            if sig and sig.get("v") == VIDEO_SIG_VERSION:
                try:
                    store[k] = unpack(sig)
                except Exception:
                    pass
    order = sorted((k for k in U if records[reps[k]]["feat"].get("dur")), key=lambda k: records[reps[k]]["feat"]["dur"])
    pairs = []
    evidence: Dict[Tuple[int, int], str] = {}
    for x, a in enumerate(order):
        fa = records[reps[a]]["feat"]
        for b in order[x + 1 :]:
            fb = records[reps[b]]["feat"]
            if not lengths_match(fa["dur"], fb["dur"]):
                break
            same, score, why = same_video(U[a], U[b], fa, fb, vs)
            if same and a in Dn and b in Dn and other_footage_between(Dn[a], Dn[b]):
                same, score, why = False, 0.0, "other footage between the checked moments"
            if same:
                lo, hi = min(a, b), max(a, b)
                pairs.append((lo, hi, score))
                evidence[(lo, hi)] = why
    uf = UnionFind(R)
    for a, b, _ in pairs:
        uf.union(a, b)
    by_root: Dict[int, List[tuple]] = {}
    for a, b, sv in pairs:
        by_root.setdefault(uf.find(a), []).append((a, b, sv))
    clusters: List[List[int]] = []
    for root_id, comp in uf.groups().items():
        if len(comp) <= 2:
            clusters.append(comp)
            continue
        local = {g: l for l, g in enumerate(comp)}
        edges = by_root.get(root_id, [])
        for cl in complete_linkage(
            len(comp),
            np.array([local[a] for a, _, _ in edges], dtype=np.int64),
            np.array([local[b] for _, b, _ in edges], dtype=np.int64),
            np.array([sv for _, _, sv in edges], dtype=np.float32),
        ):
            clusters.append([comp[i] for i in cl])
    groups = []
    for cl in clusters:
        idx = [i for k in cl for i in by_sha[shas[k]]]
        if len(idx) < 2:
            continue
        groups.append(_describe_video(idx, records, cl, evidence))
    return groups


def _describe_video(idx: List[int], records: List[dict], cl: List[int], evidence: dict) -> dict:
    from .grouping import PRESELECT, ROLE_EXACT, ROLE_KEEP, ROLE_LOWER, fmt_size, iso_dt

    ordered = sorted(idx, key=lambda i: video_key(records[i]), reverse=True)
    keeper = records[ordered[0]]
    kf = keeper["feat"]
    max_pix = max(_vpixels(records[i]) for i in idx)
    members = []
    for i in ordered:
        r = records[i]
        f = r["feat"]
        if i == ordered[0]:
            role = ROLE_KEEP
            parts = []
            if _vpixels(r) == max_pix and any(_vpixels(records[j]) < max_pix for j in idx):
                parts.append("highest resolution")
            if f.get("audio") and any(not records[j]["feat"].get("audio") for j in idx):
                parts.append("has sound")
            if _vcamera(r) and any(not _vcamera(records[j]) for j in idx):
                parts.append("original camera details")
            reason = "Suggested keeper: " + (", ".join(parts) if parts else "best overall version")
        elif r["sha"] == keeper["sha"]:
            role = ROLE_EXACT
            reason = f"Identical file to {os.path.basename(keeper['rel'])}"
        else:
            role = ROLE_LOWER
            if f.get("damaged"):
                reason = "Same video, damaged or incomplete copy"
            elif (kf.get("dur") or 0) - (f.get("dur") or 0) > 0.25:
                reason = (
                    f"Same video, {(kf.get('dur') or 0) - (f.get('dur') or 0):.1f} s shorter (a few frames missing)"
                )
            elif _vpixels(r) < _vpixels(keeper) and max(_vpixels(r), 1) * 1.02 < _vpixels(keeper):
                reason = f"Same video, smaller size ({f['w']}x{f['h']} vs {kf['w']}x{kf['h']})"
            elif kf.get("audio") and not f.get("audio"):
                reason = "Same video without sound"
            elif _vsharp(r) > 0 and _vsharp(keeper) >= 1.5 * _vsharp(r):
                reason = "Same video, less sharp (lower quality)"
            elif f.get("fmt") != kf.get("fmt") or f.get("vcodec") != kf.get("vcodec"):
                clabel = codec_label(f.get("vcodec"))
                reason = f"Same video, converted to {f.get('fmt')}" + (
                    f" ({clabel})" if clabel and clabel != f.get("fmt") else ""
                )
            elif r["size"] < keeper["size"]:
                reason = f"Same video, more compressed ({fmt_size(r['size'])} vs {fmt_size(keeper['size'])})"
            else:
                reason = "Same video, saved again"
        members.append(
            {
                "idx": i,
                "rel": r["rel"],
                "size": r["size"],
                "mtime_ns": r["mtime_ns"],
                "w": f["w"],
                "h": f["h"],
                "fmt": f.get("fmt", ""),
                "raw": False,
                "dt": iso_dt(f.get("dt")),
                "sharp": 0.0,
                "damaged": bool(f.get("damaged")),
                "role": role,
                "reason": reason,
                "selected": PRESELECT[role],
                "shot": 0,
                "sha12": r["sha"][:12],
                "media": "video",
                "dur": f.get("dur"),
                "fps": f.get("fps"),
                "vcodec": codec_label(f.get("vcodec")),
                "audio": bool(f.get("audio")),
            }
        )
    kind = "exact" if len({m["sha12"] for m in members}) == 1 else "copies"
    ev = sorted({evidence.get((min(a, b), max(a, b))) for a in cl for b in cl if a != b} - {None})
    return {
        "kind": kind,
        "media": "video",
        "files": members,
        "savings": int(sum(m["size"] for m in members if m["selected"])),
        "match": None,
        "ai": False,
        "evidence": ev,
    }
