"""Per-photo fingerprints: exact hash, visual hashes, colour signature, sharpness, AI input."""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Optional

import numpy as np
from PIL import Image

from .loader import load_image
from .paths import lp

FEATURE_VERSION = 5  # bump when fingerprints change (invalidates the cache)
CLIP_SIZE = 224


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] /= np.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def _bits_to_int(bits: np.ndarray) -> int:
    packed = np.packbits(bits.astype(np.uint8).ravel())
    return int.from_bytes(packed.tobytes(), "big")


def dhash(gray: Image.Image) -> int:
    """Difference hash: compares neighbouring pixels of a 9x8 version of the photo."""
    a = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    return _bits_to_int(a[:, 1:] > a[:, :-1])


def phash(gray: Image.Image) -> int:
    """Perceptual hash: low-frequency DCT pattern of a 32x32 version of the photo."""
    a = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    d = _DCT32 @ a @ _DCT32.T
    low = d[:8, :8].ravel()
    return _bits_to_int(low > np.median(low))


def color_signature(img: Image.Image) -> list:
    return list(np.asarray(img.resize((4, 4), Image.Resampling.BOX), dtype=np.uint8).ravel().tolist())


def sharpness_and_exposure(gray: Image.Image) -> tuple:
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0, float(g.mean()) if g.size else 0.0, 0.0, 0.0
    lap = -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    sharp = float(lap.var())
    mean = float(g.mean())
    clipped = float(((g <= 4) | (g >= 251)).mean())
    std = float(g.std())
    return sharp, mean, clipped, std


PIX_SIDE = 64


def pixel_signature(gray: Image.Image) -> str:
    """64x64 grey thumbnail (area-averaged). Used to confirm a "copy" region by region."""
    a = np.asarray(gray.resize((PIX_SIDE, PIX_SIDE), Image.Resampling.BOX), dtype=np.uint8)
    return base64.b64encode(a.tobytes()).decode("ascii")


def clip_input(img: Image.Image) -> np.ndarray:
    """Resize short side to 224 (bicubic) and centre-crop, like CLIP's own preprocessing."""
    w, h = img.size
    s = CLIP_SIZE / min(w, h)
    nw, nh = max(CLIP_SIZE, round(w * s)), max(CLIP_SIZE, round(h * s))
    r = img.resize((nw, nh), Image.Resampling.BICUBIC)
    left, top = (nw - CLIP_SIZE) // 2, (nh - CLIP_SIZE) // 2
    r = r.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    return np.asarray(r, dtype=np.uint8).copy()


def compute_from_loaded(li, need_clip: bool) -> tuple:
    img = li.image
    gray = img.convert("L")
    sharp, mean, clipped, std = sharpness_and_exposure(gray)
    feat = {
        "w": int(li.width),
        "h": int(li.height),
        "fmt": li.fmt,
        "raw": bool(li.is_raw),
        "dt": li.dt,
        "subsec": bool(li.subsec),
        "make": li.make,
        "model": li.model,
        "gps": bool(li.has_gps),
        "exif": bool(li.has_exif),
        "damaged": bool(li.damaged),
        "anim": bool(li.animated),
        "ph": format(phash(gray), "016x"),
        "dh": format(dhash(gray), "016x"),
        "col": color_signature(img),
        "px": pixel_signature(gray),
        "minside": int(min(img.size)),
        "sharp": round(sharp, 3),
        "bright": round(mean, 2),
        "clipped": round(clipped, 4),
        "std": round(std, 2),
    }
    clip = clip_input(img) if need_clip else None
    return feat, clip


def extract(task: tuple) -> dict:
    """Worker entry point. task = (absolute path, need_clip). Never raises."""
    path, need_clip = task
    rec = {"path": path}
    try:
        st = os.stat(lp(path))
        rec["size"] = st.st_size
        rec["mtime_ns"] = st.st_mtime_ns
        with open(lp(path), "rb") as f:
            data = f.read()
        rec["sha"] = hashlib.sha256(data).hexdigest()
        li = load_image(path, data)
        del data
        feat, clip = compute_from_loaded(li, need_clip)
        rec["feat"] = feat
        rec["clip"] = clip
    except MemoryError:
        rec["error"] = "Not enough memory to open this photo"
    except PermissionError:
        rec["error"] = "No permission to read this file"
    except FileNotFoundError:
        rec["error"] = "File disappeared during the scan"
    except Exception as exc:  # unreadable / damaged / unsupported
        rec["error"] = str(exc)[:200] or exc.__class__.__name__
    return rec


def make_thumbnail(task: tuple) -> Optional[str]:
    """Worker entry point: task = (src path, dst jpg path, max side). Returns error text or None."""
    src, dst, side = task
    try:
        with open(lp(src), "rb") as f:
            data = f.read()
        li = load_image(src, data, work_side=max(side, 64))
        img = li.image
        if max(img.size) > side:
            img.thumbnail((side, side), Image.Resampling.LANCZOS)
        tmp = dst + ".tmp"
        img.save(tmp, "JPEG", quality=82, optimize=True)
        os.replace(tmp, dst)
        return None
    except Exception as exc:
        return str(exc)[:200]
