"""Group photos into: exact copies, lower-quality copies, and similar shots. Pick a keeper."""

from __future__ import annotations

import base64
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cmp_to_key
from typing import Dict, List, Optional

import numpy as np

from .formats import format_rank


@dataclass(frozen=True)
class Settings:
    name: str
    copy_phash: int  # max perceptual-hash distance for "same picture" (out of 64)
    copy_dhash: int  # max difference-hash distance for "same picture"
    copy_color: float  # max colour change (0-255 scale) for "same picture"; more = edited version
    sim_ai_time: float  # min AI similarity for shots taken close together (a burst)
    sim_ai_notime: float  # min AI similarity for any other pair (all photos are compared)
    window: float  # "close together" in seconds
    sim_hash_time: int  # classic mode: max hash distance for shots taken close together
    sim_hash_notime: int  # classic mode: max hash distance for any other pair
    copy_pixel: float = 6.5  # max difference in any small region (0-255) for "same picture"


# Calibrated on 500 real photos (124,750 unrelated pairs) plus 1,560 generated copies, edits and bursts:
# copies of the same picture: hash distance <= 4, colour change <= 2.1 (edits: usually 4 to 40)
# unrelated photos: hash distance never <= 12; AI similarity >= 0.94 in 0 pairs, >= 0.86 in 0.03%
# region check (64x64 grey grid, same-size pairs): copies differ by <= 5 anywhere; date stamps,
# moved subjects, 0.3% camera shifts and one changed text line differ by >= 8 somewhere.
# Shrunken copies (1/2.5 to 1/4 size) reach 6.5 on a 32x32 grid and 7.1 on a 16x16 grid.
PRESETS: Dict[str, Settings] = {
    "strict": Settings("strict", 6, 8, 3.0, 0.90, 0.97, 120.0, 12, 8, 5.5),
    "normal": Settings("normal", 8, 10, 4.0, 0.86, 0.95, 300.0, 16, 10, 6.5),
    "loose": Settings("loose", 10, 12, 4.0, 0.82, 0.93, 1800.0, 20, 12, 6.5),
}

ROLE_KEEP, ROLE_EXACT, ROLE_LOWER, ROLE_SIMILAR, ROLE_RAWPAIR = "keep", "exact", "lower", "similar", "rawpair"
PRESELECT = {ROLE_EXACT: True, ROLE_LOWER: True, ROLE_KEEP: False, ROLE_SIMILAR: False, ROLE_RAWPAIR: False}

# ----------------------------------------------------------------- fast pair search

if hasattr(np, "bitwise_count"):

    def popcount(x: np.ndarray) -> np.ndarray:
        return np.bitwise_count(x)
else:  # numpy < 2.0
    _LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount(x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.uint64)
        return _LUT[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1, dtype=np.uint16)


def hamming_pairs(h: np.ndarray, t: int):
    """All pairs i<j with popcount(h[i]^h[j]) <= t. Returns (I, J, D)."""
    n = len(h)
    if n < 2:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e
    rows = max(1, int(1.5e7 // n))
    I, J, D = [], [], []
    for s in range(0, n, rows):
        d = popcount(h[s : s + rows, None] ^ h[None, :])
        ii, jj = np.nonzero(d <= t)
        gi = ii + s
        m = jj > gi
        I.append(gi[m])
        J.append(jj[m])
        D.append(d[ii[m], jj[m]].astype(np.int64))
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def cosine_pairs(E: np.ndarray, t: float):
    """All pairs i<j with dot(E[i], E[j]) >= t (E rows are unit vectors). Returns (I, J, S)."""
    n = len(E)
    if n < 2:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float32)
    rows = max(1, int(8e6 // n))
    I, J, S = [], [], []
    for s in range(0, n, rows):
        sim = E[s : s + rows] @ E.T
        ii, jj = np.nonzero(sim >= t)
        gi = ii + s
        m = jj > gi
        I.append(gi[m])
        J.append(jj[m])
        S.append(sim[ii[m], jj[m]])
    return np.concatenate(I), np.concatenate(J), np.concatenate(S).astype(np.float32)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.p)):
            out.setdefault(self.find(i), []).append(i)
        return out


def complete_linkage(k: int, a: np.ndarray, b: np.ndarray, s: np.ndarray) -> List[List[int]]:
    """Merge nodes into clusters where EVERY pair inside a cluster is a confirmed match
    (prevents A~B, B~C chains from lumping different photos together)."""
    M = np.full((k, k), -np.inf, dtype=np.float32)
    M[a, b] = s
    M[b, a] = s
    members = [[i] for i in range(k)]
    while True:
        idx = int(np.argmax(M))
        x, y = divmod(idx, k)
        if not np.isfinite(M[x, y]):
            break
        members[x].extend(members[y])
        members[y] = []
        new = np.minimum(M[x], M[y])
        new[x] = -np.inf
        new[y] = -np.inf
        M[x, :] = new
        M[:, x] = new
        M[y, :] = -np.inf
        M[:, y] = -np.inf
    return [m for m in members if m]


# ----------------------------------------------------------------- helpers

LOSSLESS = {"PNG", "TIFF", "BMP", "GIF", "PPM", "TGA"}
VERIFY_LIMIT = 12.0  # full-resolution 8x8 block check, see verify.py


def _px(f: dict) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(f["px"]), dtype=np.uint8)
    return a.reshape(64, 64)


def region_diff(feats: List[dict], I: np.ndarray, J: np.ndarray, base: float) -> tuple:
    """Largest difference between two photos in any small region, plus the limit for a copy.
    Same-size pairs are compared on a fine 64x64 grid. When one file is a shrunken copy, the grid is
    matched to the detail the small file can still hold (32x32 or 16x16) with a slightly higher limit."""
    diff = np.empty(len(I), dtype=np.float32)
    limit = np.empty(len(I), dtype=np.float32)
    cache: Dict[int, np.ndarray] = {}

    def get(k):
        v = cache.get(k)
        if v is None:
            v = cache[k] = _px(feats[k]).astype(np.float32)
        return v

    for n, (a, b) in enumerate(zip(I.tolist(), J.tolist())):
        sa = min(feats[a]["w"], feats[a]["h"])
        sb = min(feats[b]["w"], feats[b]["h"])
        small, large = min(sa, sb), max(sa, sb)
        if small >= 0.8 * large or small >= 700:
            grid, lim = 64, base
        elif small >= 350:
            grid, lim = 32, base + 2.5
        else:
            grid, lim = 16, base + 3.5
        A, B = get(a), get(b)
        if grid != 64:
            f = 64 // grid
            A = A.reshape(grid, f, grid, f).mean((1, 3))
            B = B.reshape(grid, f, grid, f).mean((1, 3))
        diff[n] = float(np.abs(A - B).max())
        limit[n] = lim
    return diff, limit


# "IMG_1 - Copy", "IMG_1 copy", "IMG_1_copy", "Copy of IMG_1", "IMG_1 (1)", "IMG_1~2" ... (letters-only word
# boundaries, so "_copy" and "Copy_2" count too)
_COPY_NAME = re.compile(
    r"(?i)((?<![a-z])(copy|copia|kopie|duplicate|dupe)(?![a-z])|\(\d+\)|~\d+$|\s\d{1,2}$|-\d{1,2}$)"
)
_EDIT_NAME = re.compile(r"(?i)(edit|bearbeitet|modifi|^IMG_E\d)")


def name_penalty(rel: str) -> int:
    stem = os.path.splitext(os.path.basename(rel))[0]
    return (1 if _COPY_NAME.search(stem) else 0) + (1 if _EDIT_NAME.search(stem) else 0)


def looks_edited(rel: str) -> bool:
    return bool(_EDIT_NAME.search(os.path.splitext(os.path.basename(rel))[0]))


def pixels(r: dict) -> int:
    f = r["feat"]
    return int(f["w"]) * int(f["h"])


def has_camera_info(r: dict) -> bool:
    f = r["feat"]
    return f.get("dt") is not None or bool(f.get("make"))


def _origin_rank(r: dict) -> int:
    """RAW files and camera HEIC files are originals; a HEIC or PNG made from a JPG is not."""
    f = r["feat"]
    if f["raw"]:
        return 2
    if f["fmt"] in ("HEIC", "HEIF") and has_camera_info(r):
        return 1
    return 0


def _tiebreak(r: dict):
    """Same picture, same size: prefer the original over a converted or re-saved copy."""
    f = r["feat"]
    return (
        bool(f["raw"]),  # a camera RAW file is always the original
        bool(r.get("comp")),  # keeps its Live Photo video / sidecar files
        -name_penalty(r["rel"]),  # "copy", "(1)", "edited" ...
        -(r["mtime_ns"] // 1_000_000_000),  # the older file came first (to the second)
        _origin_rank(r),  # camera HEIC over a JPG export made at the same moment
        r["size"],
        bool(f.get("exif")),
        -len(r["rel"]),
        r["rel"],
    )


def _compare(a: dict, b: dict) -> int:
    """>0 when `a` is the better copy of the same picture."""
    da, db = bool(a["feat"].get("damaged")), bool(b["feat"].get("damaged"))
    if da != db:
        return 1 if db else -1
    ca, cb = has_camera_info(a), has_camera_info(b)
    pa, pb = pixels(a), pixels(b)
    if ca != cb:
        # the file that still has camera details is the original, unless the other one is
        # far larger (an enlarged copy is at most ~2x per side; real originals are bigger)
        cam, other = (a, b) if ca else (b, a)
        if pixels(other) <= 4.5 * pixels(cam):
            return 1 if cam is a else -1
    if pa != pb:
        return 1 if pa > pb else -1
    ka, kb = _tiebreak(a), _tiebreak(b)
    return (ka > kb) - (ka < kb)


version_key = cmp_to_key(_compare)


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def fmt_gap(sec: float) -> str:
    sec = abs(sec)
    if sec < 1:
        return "under 1 second apart"
    if sec < 90:
        return f"{sec:.0f} s apart"
    if sec < 5400:
        return f"{sec / 60:.0f} min apart"
    return f"{sec / 3600:.1f} h apart"


def iso_dt(dt: Optional[float]) -> Optional[str]:
    if dt is None:
        return None
    try:
        return datetime.fromtimestamp(dt, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return None


# ----------------------------------------------------------------- main


def build_groups(records: List[dict], settings: Settings, use_ai: bool, verifier=None) -> List[dict]:
    """records: readable photos from the scan. Returns a list of group dicts for the review screen.
    verifier: optional function that re-opens same-size candidate copies and compares them at full
    resolution (see verify.py). Without it, only the compact fingerprints are used."""
    n = len(records)
    if n < 2:
        return []

    # 1) identical files (same bytes)
    by_sha: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_sha.setdefault(r["sha"], []).append(i)
    sha_list = list(by_sha)
    reps = [by_sha[s][0] for s in sha_list]
    R = len(reps)
    feats = [records[i]["feat"] for i in reps]
    ph = np.array([int(f["ph"], 16) for f in feats], dtype=np.uint64)
    dh = np.array([int(f["dh"], 16) for f in feats], dtype=np.uint64)
    log_ar = np.array([math.log(max(f["w"], 1) / max(f["h"], 1)) for f in feats], dtype=np.float64)
    col = np.array([f["col"] for f in feats], dtype=np.float32)
    _c = col.reshape(len(feats), 16, 3)
    chroma = (_c.max(2) - _c.min(2)).mean(1)  # how colourful each photo is
    lowd = np.array([f.get("std", 99.0) < 5.0 for f in feats], dtype=bool)
    unsure = np.array([bool(f.get("damaged") or f.get("anim")) for f in feats], dtype=bool)
    dts = np.array([np.nan if f.get("dt") is None else f["dt"] for f in feats], dtype=np.float64)
    subs = np.array([bool(f.get("subsec")) for f in feats], dtype=bool)
    lossless = np.array([f["fmt"] in LOSSLESS and not f["raw"] for f in feats], dtype=bool)
    dims = np.array([[f["w"], f["h"]] for f in feats], dtype=np.int64)
    E = None
    if use_ai and all(records[i].get("emb") is not None for i in reps):
        E = np.stack([records[i]["emb"] for i in reps]).astype(np.float32)

    # 2) same picture saved again (resized, recompressed, converted, renamed)
    edit_rep_pairs: List[tuple] = []  # same composition, colours changed = edited version
    near_rep_pairs: List[tuple] = []  # look alike but are separate moments or locally different
    copy_pairs: List[tuple] = []
    I, J, Dp = hamming_pairs(ph, settings.copy_phash)
    if len(I):
        Dd = popcount(dh[I] ^ dh[J]).astype(np.int64)
        cdiff = np.abs(col[I] - col[J]).mean(1)
        shape_ok = (Dd <= settings.copy_dhash) & (np.abs(log_ar[I] - log_ar[J]) <= 0.05)
        trusted = ~(lowd[I] | lowd[J] | unsure[I] | unsure[J])  # blank, damaged, multi-page: exact only
        # copies keep the capture time (or lose it); different capture times = separate shots
        di, dj = dts[I], dts[J]
        unknown = np.isnan(di) | np.isnan(dj)
        both_sub = subs[I] & subs[J]
        same_time = ~unknown & ((both_sub & (np.abs(di - dj) < 0.002)) | (~both_sub & (np.floor(di) == np.floor(dj))))
        # a copy keeps its colourfulness (measured: <= 8% change); saturation edits change it 40%+
        chroma_rel = np.abs(chroma[I] - chroma[J]) / np.maximum(np.maximum(chroma[I], chroma[J]), 6.0)
        colour_same = (cdiff <= settings.copy_color) & (chroma_rel <= 0.15)
        look_same = shape_ok & trusted & colour_same
        # Region by region check: a copy differs only by compression noise everywhere
        pix = np.full(len(I), np.inf, dtype=np.float32)
        lim = np.zeros(len(I), dtype=np.float32)
        cand = np.nonzero(look_same)[0]
        if len(cand):
            pix[cand], lim[cand] = region_diff(feats, I[cand], J[cand], settings.copy_pixel)
        pix_ok = pix <= lim
        same_dims = (dims[I] == dims[J]).all(1)
        pix_ok &= ~(lossless[I] & lossless[J] & same_dims) | (pix <= 1.0)  # lossless: pixels must match
        # Without a capture time to confirm "same moment", only the tight hash range counts.
        tight = (Dp <= min(settings.copy_phash, 4)) & (Dd <= min(settings.copy_dhash, 6))
        ok = look_same & pix_ok & (same_time | (unknown & tight))
        # same-size candidates: confirm at full resolution (a changed digit or date stamp is tiny)
        if verifier is not None and ok.any():
            raw = np.array([bool(f["raw"]) for f in feats], dtype=bool)
            near_size = (np.abs(dims[I] - dims[J]) <= np.maximum(2, 0.02 * np.maximum(dims[I], dims[J]))).all(1)
            check = np.nonzero(ok & near_size & ~raw[I] & ~raw[J])[0]
            if len(check):
                pairs = [
                    (records[reps[I[k]]]["path"], sha_list[I[k]], records[reps[J[k]]]["path"], sha_list[J[k]])
                    for k in check
                ]
                diffs = verifier(pairs)
                for k, d in zip(check, diffs):
                    if d is None or d > VERIFY_LIMIT:
                        ok[k] = False
        copy_pairs = [(int(a), int(b), float(100.0 - p)) for a, b, p in zip(I[ok], J[ok], pix[ok])]
        edit = shape_ok & trusted & ~colour_same & (cdiff <= 70.0)
        edit_rep_pairs = list(zip(I[edit].tolist(), J[edit].tolist()))
        near = look_same & ~ok
        near_rep_pairs = list(zip(I[near].tolist(), J[near].tolist()))
    # every pair inside a copy set must match (no chaining through a stripped copy)
    uf = UnionFind(R)
    for a, b, _ in copy_pairs:
        uf.union(a, b)
    by_root: Dict[int, List[tuple]] = {}
    for a, b, sv in copy_pairs:
        by_root.setdefault(uf.find(a), []).append((a, b, sv))
    shot_lists: List[List[int]] = []
    for root_id, comp in uf.groups().items():
        if len(comp) <= 2:
            shot_lists.append(comp)
            continue
        local = {g: l for l, g in enumerate(comp)}
        edges = by_root.get(root_id, [])
        for cl in complete_linkage(
            len(comp),
            np.array([local[a] for a, _, _ in edges], dtype=np.int64),
            np.array([local[b] for _, b, _ in edges], dtype=np.int64),
            np.array([sv for _, _, sv in edges], dtype=np.float32),
        ):
            shot_lists.append([comp[i] for i in cl])
    shots = [sorted(v) for v in shot_lists]  # lists of rep indices
    rep_files = {k: by_sha[sha_list[k]] for k in range(R)}  # rep index -> all record indices

    def shot_files(shot):
        out = []
        for k in shot:
            out.extend(rep_files[k])
        return out

    best_of_shot = [max(shot_files(s), key=lambda i: version_key(records[i])) for s in shots]
    shot_rep = []  # rep index of the best version (for features)
    rep_index_of_sha = {sha_list[k]: k for k in range(R)}
    for bi in best_of_shot:
        shot_rep.append(rep_index_of_sha[records[bi]["sha"]])

    # 3) similar shots (burst / taken again) and edited versions
    S = len(shots)
    clusters: List[List[int]] = [[s] for s in range(S)]
    pair_sim: Dict[tuple, float] = {}
    shot_of_rep = {}
    for si, shot in enumerate(shots):
        for k in shot:
            shot_of_rep[k] = si
    edit_pairs, near_pairs = set(), set()
    for src, dst in ((edit_rep_pairs, edit_pairs), (near_rep_pairs, near_pairs)):
        for a, b in src:
            sa, sb = shot_of_rep[a], shot_of_rep[b]
            if sa != sb:
                dst.add((min(sa, sb), max(sa, sb)))
    if S >= 2:
        sr = np.array(shot_rep, dtype=np.int64)
        sdt = dts[sr]
        slow = lowd[sr] | np.array([bool(feats[k].get("damaged")) for k in sr], dtype=bool)
        if E is not None:
            Es = E[sr]
            lo = min(settings.sim_ai_time, settings.sim_ai_notime)
            I, J, Sim = cosine_pairs(Es, lo)
            if len(I):
                both = ~np.isnan(sdt[I]) & ~np.isnan(sdt[J])
                close = both & (np.abs(sdt[I] - sdt[J]) <= settings.window)
                # every photo is compared with every other photo; being taken minutes apart only
                # lowers the bar a little (a burst), it never blocks a match
                ok = np.where(close, Sim >= settings.sim_ai_time, Sim >= settings.sim_ai_notime)
                ok &= ~(slow[I] | slow[J]) | close
                I, J, Sim = I[ok], J[ok], Sim[ok]
        else:
            sph = ph[sr]
            I, J, Dh = hamming_pairs(sph, max(settings.sim_hash_time, settings.sim_hash_notime))
            if len(I):
                both = ~np.isnan(sdt[I]) & ~np.isnan(sdt[J])
                close = both & (np.abs(sdt[I] - sdt[J]) <= settings.window)
                ok = np.where(close, Dh <= settings.sim_hash_time, Dh <= settings.sim_hash_notime)
                ok &= np.abs(log_ar[sr][I] - log_ar[sr][J]) <= 0.12
                ok &= ~(slow[I] | slow[J]) | close
                I, J = I[ok], J[ok]
                Sim = (1.0 - Dh[ok] / 64.0).astype(np.float32)
            else:
                Sim = np.zeros(0, dtype=np.float32)
        for a, b, sv in zip(I.tolist(), J.tolist(), Sim.tolist()):
            pair_sim[(a, b)] = sv
        for a, b in edit_pairs | near_pairs:
            if (a, b) not in pair_sim:
                if E is not None:
                    pair_sim[(a, b)] = float(E[sr[a]] @ E[sr[b]])
                else:
                    pair_sim[(a, b)] = (
                        1.0 - float(popcount(np.array([ph[sr[a]] ^ ph[sr[b]]], dtype=np.uint64))[0]) / 64.0
                    )
        if pair_sim:
            cuf = UnionFind(S)
            for a, b in pair_sim:
                cuf.union(a, b)
            edges_by_root: Dict[int, List[tuple]] = {}
            for (a, b), sv in pair_sim.items():
                edges_by_root.setdefault(cuf.find(a), []).append((a, b, sv))
            clusters = []
            for root_id, comp in cuf.groups().items():
                if len(comp) == 1:
                    clusters.append(comp)
                    continue
                comp = sorted(comp, key=lambda s: (np.nan_to_num(sdt[s], nan=0.0), s))
                edges = edges_by_root.get(root_id, [])
                for start in range(0, len(comp), 1200):  # very large bursts: work in time slices
                    part = comp[start : start + 1200]
                    local = {g: l for l, g in enumerate(part)}
                    la, lb, ls = [], [], []
                    for a, b, sv in edges:
                        if a in local and b in local:
                            la.append(local[a])
                            lb.append(local[b])
                            ls.append(sv)
                    for cl in complete_linkage(
                        len(part),
                        np.array(la, dtype=np.int64),
                        np.array(lb, dtype=np.int64),
                        np.array(ls, dtype=np.float32),
                    ):
                        clusters.append([part[i] for i in cl])

    # 4) build review groups
    groups: List[dict] = []
    for cl in clusters:
        files_total = sum(len(shot_files(shots[s])) for s in cl)
        if files_total < 2:
            continue
        groups.append(_describe(cl, shots, shot_files, best_of_shot, records, pair_sim, E is not None, edit_pairs))
    groups.sort(key=lambda g: (-g["savings"], -len(g["files"]), g["files"][0]["rel"].lower()))
    for gi, g in enumerate(groups, 1):
        g["id"] = gi
    return groups


def _shot_score(rows: List[dict]) -> List[float]:
    sharp = [max(r["feat"]["sharp"], 0.0) for r in rows]
    pix = [pixels(r) for r in rows]
    ms, mp = max(sharp) or 1.0, max(pix) or 1
    out = []
    for r, s, p in zip(rows, sharp, pix):
        expo = 1.0 - min(1.0, r["feat"].get("clipped", 0.0) * 2.5)
        dmg = 0.5 if r["feat"].get("damaged") else 0.0
        edited = 0.3 if looks_edited(r["rel"]) else 0.0  # prefer the original when names say "edited"
        # GIF dithering and multi-page files look "sharp" to the blur test; prefer normal photos
        weak = 0.4 if (r["feat"].get("anim") or format_rank(r["feat"]["fmt"], r["feat"]["raw"]) <= 1) else 0.0
        out.append(0.60 * s / ms + 0.25 * p / mp + 0.15 * expo - dmg - edited - weak)
    return out


def _describe(cluster, shots, shot_files, best_of_shot, records, pair_sim, ai_on, edit_pairs=frozenset()) -> dict:
    cluster = list(cluster)
    bests = [records[best_of_shot[s]] for s in cluster]
    if len(cluster) > 1:
        scores = _shot_score(bests)
        k = max(range(len(cluster)), key=lambda i: (scores[i], version_key(bests[i])))
    else:
        scores = [1.0]
        k = 0
    keep_shot = cluster[k]
    keeper_idx = best_of_shot[keep_shot]
    keeper = records[keeper_idx]
    all_idx = [i for s in cluster for i in shot_files(shots[s])]
    max_pix = max(pixels(records[i]) for i in all_idx)
    max_sharp = max(b["feat"]["sharp"] for b in bests)

    def sim_between(s1, s2):
        a, b = min(s1, s2), max(s1, s2)
        return pair_sim.get((a, b))

    members = []
    for si, s in enumerate(cluster):
        base_idx = best_of_shot[s]
        base = records[base_idx]
        for i in sorted(shot_files(shots[s]), key=lambda i: version_key(records[i]), reverse=True):
            r = records[i]
            f = r["feat"]
            if i == keeper_idx:
                role = ROLE_KEEP
                parts = []
                if pixels(r) == max_pix and len(all_idx) > 1 and any(pixels(records[j]) < max_pix for j in all_idx):
                    parts.append("highest resolution")
                if len(cluster) > 1 and f["sharp"] >= max_sharp:
                    parts.append("sharpest shot")
                if f["raw"] or f["fmt"] in ("HEIC", "HEIF"):
                    if any(not records[j]["feat"]["raw"] and records[j]["feat"]["fmt"] != f["fmt"] for j in all_idx):
                        parts.append("original camera format")
                reason = "Suggested keeper: " + (", ".join(parts) if parts else "best overall version")
            elif i == base_idx:
                role = ROLE_SIMILAR
                if (min(s, keep_shot), max(s, keep_shot)) in edit_pairs:
                    if looks_edited(r["rel"]):
                        reason = "Edited version of the same photo (colours or brightness changed)"
                    elif looks_edited(keeper["rel"]):
                        reason = "Unedited version of the same photo"
                    else:
                        reason = "Same photo with different colours or brightness (an edited version)"
                    members.append(_member(i, r, f, role, reason, si))
                    continue
                bits = []
                if f.get("dt") is not None and keeper["feat"].get("dt") is not None:
                    bits.append("taken " + fmt_gap(f["dt"] - keeper["feat"]["dt"]))
                if f["sharp"] < keeper["feat"]["sharp"] * 0.85:
                    bits.append("less sharp")
                sim = sim_between(s, keep_shot)
                if sim is not None and ai_on:
                    bits.append(f"AI match {sim * 100:.0f}%")
                reason = "Similar shot" + (": " + ", ".join(bits) if bits else "")
            else:
                if r["sha"] == base["sha"]:
                    role = ROLE_EXACT
                    reason = f"Identical file to {os.path.basename(base['rel'])}"
                elif f["raw"] or base["feat"]["raw"]:
                    role = ROLE_RAWPAIR
                    reason = "RAW and JPEG versions of the same shot (both kept unless you choose)"
                else:
                    role = ROLE_LOWER
                    bf = base["feat"]
                    if pixels(r) < pixels(base):
                        reason = f"Same photo, smaller size ({f['w']}x{f['h']} vs {bf['w']}x{bf['h']})"
                    elif pixels(r) > pixels(base):
                        reason = "Same photo, enlarged copy without the camera details of the original"
                    elif f["fmt"] != bf["fmt"]:
                        reason = f"Same photo, converted to {f['fmt']}"
                    elif r["size"] < base["size"]:
                        reason = f"Same photo, more compressed ({fmt_size(r['size'])} vs {fmt_size(base['size'])})"
                    else:
                        reason = "Same photo, saved again"
                if s != keep_shot:
                    reason += " (copy of a similar shot)"
            members.append(_member(i, r, f, role, reason, si))
    order = {ROLE_KEEP: 0, ROLE_SIMILAR: 0, ROLE_RAWPAIR: 1, ROLE_EXACT: 2, ROLE_LOWER: 2}
    members.sort(key=lambda m: (m["role"] != ROLE_KEEP, m["shot"] != k, m["shot"], order[m["role"]], m["rel"].lower()))
    shas = {m["sha12"] for m in members}
    if len(cluster) > 1:
        kind = "similar"
    elif len(shas) == 1:
        kind = "exact"
    else:
        kind = "copies"
    sims = [sim_between(s, keep_shot) for s in cluster if s != keep_shot]
    sims = [x for x in sims if x is not None]
    return {
        "kind": kind,
        "files": members,
        "savings": int(sum(m["size"] for m in members if m["selected"])),
        "match": round(float(min(sims)) * 100) if sims else None,
        "ai": bool(ai_on),
    }


def _member(i, r, f, role, reason, si) -> dict:
    return {
        "idx": i,
        "rel": r["rel"],
        "size": r["size"],
        "mtime_ns": r["mtime_ns"],
        "w": f["w"],
        "h": f["h"],
        "fmt": f["fmt"],
        "raw": f["raw"],
        "dt": iso_dt(f.get("dt")),
        "sharp": round(f["sharp"], 1),
        "damaged": bool(f.get("damaged")),
        "role": role,
        "reason": reason,
        "selected": PRESELECT[role],
        "shot": si,
        "sha12": r["sha"][:12],
    }
