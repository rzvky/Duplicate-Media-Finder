"""Full-resolution check for same-size "copies": re-open both files and compare them in 8x8 blocks.
Catches tiny local differences (one changed digit in a screenshot, a date stamp, a small movement)
that a compact fingerprint cannot see. Calibrated: real copies (even JPEG quality 20) stay <= 7.8;
date stamps and a one-digit change in a screenshot measure >= 18."""

from __future__ import annotations

import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .loader import load_image
from .paths import lp

VERIFY_SIDE = 2048
BLOCK = 8
COPY_LIMIT = 12.0
_CTX = multiprocessing.get_context("spawn")


def block_diff(task: Tuple[str, str]) -> Optional[float]:
    """Worker: largest mean difference of any 8x8 block between two images (grey, 0-255)."""
    pa, pb = task
    try:
        with open(lp(pa), "rb") as f:
            a = load_image(pa, f.read(), work_side=VERIFY_SIDE).image.convert("L")
        with open(lp(pb), "rb") as f:
            b = load_image(pb, f.read(), work_side=VERIFY_SIDE).image.convert("L")
        if b.size != a.size:
            b = b.resize(a.size, Image.Resampling.BILINEAR)
        A = np.asarray(a, dtype=np.float32)
        B = np.asarray(b, dtype=np.float32)
        h, w = (A.shape[0] // BLOCK) * BLOCK, (A.shape[1] // BLOCK) * BLOCK
        if h == 0 or w == 0:
            return float(np.abs(A - B).max()) if A.size else 0.0
        d = np.abs(A[:h, :w] - B[:h, :w]).reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3))
        return float(d.max())
    except Exception:
        return None


class _Cache:
    def __init__(self, path: Optional[Path]):
        self.db = None
        if path is None:
            return
        try:
            self.db = sqlite3.connect(str(path), timeout=30)
            self.db.execute("CREATE TABLE IF NOT EXISTS verify_v1 (a TEXT, b TEXT, diff REAL, PRIMARY KEY (a, b))")
        except sqlite3.Error:
            self.db = None

    def get(self, a: str, b: str) -> Optional[float]:
        if self.db is None:
            return None
        row = self.db.execute("SELECT diff FROM verify_v1 WHERE a=? AND b=?", (a, b)).fetchone()
        return None if row is None else row[0]

    def put(self, a: str, b: str, diff: float):
        if self.db is not None:
            self.db.execute("INSERT OR REPLACE INTO verify_v1 VALUES (?,?,?)", (a, b, diff))

    def close(self):
        if self.db is not None:
            try:
                self.db.commit()
                self.db.close()
            except sqlite3.Error:
                pass


def make_verifier(
    workers: int, cache_path: Optional[Path] = None, progress: Optional[Callable[[str, int, int], None]] = None
):
    """Returns verify(pairs) where pairs = [(path_a, sha_a, path_b, sha_b)]; result: list of diffs (None = failed)."""

    def verify(pairs: Sequence[tuple]) -> List[Optional[float]]:
        out: List[Optional[float]] = [None] * len(pairs)
        cache = _Cache(cache_path)
        todo = []
        for i, (_pa, sa, _pb, sb) in enumerate(pairs):
            k1, k2 = sorted((sa, sb))
            hit = cache.get(k1, k2)
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        total = len(todo)
        if total:
            prog = progress or (lambda *a: None)
            tasks = [(pairs[i][0], pairs[i][2]) for i in todo]
            done = 0
            try:
                with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=_CTX) as pool:
                    for i, d in zip(todo, pool.map(block_diff, tasks, chunksize=2)):
                        out[i] = d
                        done += 1
                        if done % 5 == 0 or done == total:
                            prog("verify", done, total)
            except Exception:
                pass  # unverified pairs stay None and are treated as "not a copy"
            for i in todo:
                if out[i] is not None:
                    k1, k2 = sorted((pairs[i][1], pairs[i][3]))
                    cache.put(k1, k2, out[i])
        cache.close()
        return out

    return verify
