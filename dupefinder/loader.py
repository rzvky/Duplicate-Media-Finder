"""Open any photo file (JPEG, PNG, HEIC, AVIF, WEBP, TIFF, GIF, RAW ...) as an upright RGB image."""

from __future__ import annotations

import calendar
import io
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFile, ImageOps

from .formats import AVIF_EXTS, HEIF_EXTS, RAW_EXTS

Image.MAX_IMAGE_PIXELS = 180_000_000  # big panoramas OK (refused above 360 MP to protect memory)
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

HEIF_OK = False
HEIF_ERROR = ""
try:  # iPhone HEIC / HEIF (and AVIF on older pillow-heif versions)
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
    try:
        pillow_heif.register_avif_opener()  # removed in pillow-heif 1.x (Pillow reads AVIF itself)
    except Exception:
        pass
except Exception as exc:  # pragma: no cover - depends on install
    HEIF_ERROR = str(exc)

try:
    import rawpy

    RAW_OK = True
except Exception:  # pragma: no cover - depends on install
    rawpy = None
    RAW_OK = False


def avif_supported() -> bool:
    try:
        from PIL import features

        if features.check("avif"):
            return True
    except Exception:
        pass
    return "AVIF" in Image.registered_extensions().values() or ".avif" in Image.registered_extensions()


class UnsupportedFormat(Exception):
    pass


@dataclass
class Loaded:
    image: Image.Image  # upright RGB working copy (long side <= work_side, may be larger for tiny inputs)
    width: int  # full size, upright
    height: int
    fmt: str
    is_raw: bool = False
    dt: Optional[float] = None  # capture time, seconds (naive local time treated as UTC)
    subsec: bool = False  # capture time includes fractions of a second
    make: str = ""
    model: str = ""
    has_gps: bool = False
    has_exif: bool = False
    damaged: bool = False
    animated: bool = False


_DT_RE = re.compile(r"(\d{4})[:\-./](\d{1,2})[:\-./](\d{1,2})[ T](\d{1,2}):(\d{2}):(\d{2})")


def parse_exif_datetime(value, subsec=None) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    m = _DT_RE.search(str(value))
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    if not (1980 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31 and h < 24 and mi < 60 and s < 61):
        return None
    try:
        ts = float(calendar.timegm((y, mo, d, h, mi, min(s, 59), 0, 0, 0)))
    except Exception:
        return None
    if subsec is not None:
        if isinstance(subsec, bytes):
            subsec = subsec.decode("ascii", "ignore")
        digits = re.sub(r"\D", "", str(subsec))[:6]
        if digits:
            ts += int(digits) / (10 ** len(digits))
    return ts


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        v = v.decode("utf-8", "ignore")
    return str(v).replace("\x00", "").strip()[:60]


def _has_digits(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bytes):
        v = v.decode("ascii", "ignore")
    return bool(re.sub(r"\D", "", str(v)))


def exif_info(exif) -> dict:
    info = {"dt": None, "subsec": False, "make": "", "model": "", "has_gps": False, "has_exif": False}
    if not exif:
        return info
    info["has_exif"] = len(exif) > 0
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:
        sub = {}
    dt, has_sub = None, False
    for tag, subtag in ((36867, 37521), (36868, 37522)):
        if sub.get(tag):
            dt = parse_exif_datetime(sub.get(tag), sub.get(subtag))
            if dt:
                has_sub = _has_digits(sub.get(subtag))
                break
    if dt is None:
        dt = parse_exif_datetime(exif.get(306), sub.get(37520))
        has_sub = dt is not None and _has_digits(sub.get(37520))
    info["dt"] = dt
    info["subsec"] = bool(has_sub)
    info["make"] = _clean(exif.get(271))
    info["model"] = _clean(exif.get(272))
    try:
        gps = exif.get_ifd(0x8825)
        info["has_gps"] = bool(gps) and any(k in gps for k in (2, 4))
    except Exception:
        pass
    return info


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow mode to 8-bit RGB; transparent areas become white."""
    mode = img.mode
    if mode == "RGB":
        return img
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        mx = float(arr.max()) if arr.size else 0.0
        if mx > 255:
            arr = arr * (255.0 / (65535.0 if mx <= 65535 else mx))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").convert("RGB")
    has_alpha = mode in ("RGBA", "LA", "PA", "RGBa", "La") or (mode in ("P", "L", "RGB") and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def _shrink(img: Image.Image, work_side: int) -> Image.Image:
    if max(img.size) > work_side:
        img = img.copy()
        img.thumbnail((work_side, work_side), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return img


def _load_pillow(data: bytes, work_side: int) -> Loaded:
    with Image.open(io.BytesIO(data)) as im:
        fmt = (im.format or "UNKNOWN").upper()
        animated = bool(getattr(im, "is_animated", False))
        try:
            exif = im.getexif()
        except Exception:
            exif = None
        w, h = im.size
        try:
            info = exif_info(exif)
        except Exception:  # unreadable EXIF block: keep going without it
            info = {"dt": None, "subsec": False, "make": "", "model": "", "has_gps": False, "has_exif": False}
        if fmt in ("JPEG", "MPO"):
            try:
                im.draft(None, (work_side, work_side))  # fast reduced-size JPEG decode
            except Exception:
                pass
        im.load()
        pre = im.size
        try:
            img = ImageOps.exif_transpose(im)  # rotates by the EXIF orientation tag (copy if upright)
        except Exception:  # malformed EXIF: rotate by hand
            img = _transpose_by_tag(im)
        if img is None:
            img = im.copy()
        post = img.size
        if pre != post and pre == (post[1], post[0]):  # rotated 90/270: report the size the viewer sees
            w, h = h, w
        img = to_rgb(img)
        img = _shrink(img, work_side)
    return Loaded(image=img, width=w, height=h, fmt=fmt, animated=animated, **info)


_ORIENT = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT],
    3: [Image.Transpose.ROTATE_180],
    4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE],
    6: [Image.Transpose.ROTATE_270],
    7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],
}


def _transpose_by_tag(im: Image.Image) -> Image.Image:
    try:
        o = int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        o = 1
    out = im.copy()
    for t in _ORIENT.get(o, []):
        out = out.transpose(t)
    return out


def _apply_flip(img: Image.Image, flip: int) -> Image.Image:
    if flip == 3:
        return img.transpose(Image.Transpose.ROTATE_180)
    if flip == 5:
        return img.transpose(Image.Transpose.ROTATE_90)
    if flip == 6:
        return img.transpose(Image.Transpose.ROTATE_270)
    return img


def _load_raw(path: str, data: bytes, work_side: int) -> Loaded:
    if not RAW_OK:
        raise UnsupportedFormat("RAW photo support is not installed (package 'rawpy')")
    info = {"dt": None, "subsec": False, "make": "", "model": "", "has_gps": False, "has_exif": False}
    try:  # TIFF-based RAW files (DNG, CR2, NEF, ARW, ...) expose EXIF through Pillow
        with Image.open(io.BytesIO(data)) as tim:
            info = exif_info(tim.getexif())
    except Exception:
        pass
    with rawpy.imread(io.BytesIO(data)) as raw:
        sizes = raw.sizes
        flip = int(getattr(sizes, "flip", 0) or 0)
        w0, h0 = int(sizes.width), int(sizes.height)
        w, h = (h0, w0) if flip in (5, 6) else (w0, h0)
        img = None
        try:  # the camera's own embedded JPEG preview is fast and matches the camera JPEG
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                tim = Image.open(io.BytesIO(thumb.data))
                tim.draft(None, (work_side, work_side))
                tim.load()
                texif = tim.getexif()
                if texif and int(texif.get(0x0112, 1) or 1) != 1:
                    tim = ImageOps.exif_transpose(tim)
                    flip_needed = False
                else:
                    flip_needed = True
            else:
                tim = Image.fromarray(thumb.data)
                flip_needed = True
            if min(tim.size) >= 160:
                if flip_needed and flip in (3, 5, 6):
                    # some previews are stored upright already: compare aspect ratios
                    t_land = tim.size[0] >= tim.size[1]
                    sensor_land = w0 >= h0
                    if flip == 3 or t_land == sensor_land:
                        tim = _apply_flip(tim, flip)
                img = to_rgb(tim)
        except Exception:
            img = None
        if img is None:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
            img = Image.fromarray(rgb)  # LibRaw already applies orientation here
        img = _shrink(img, work_side)
    ext = os.path.splitext(path)[1].upper().lstrip(".") or "RAW"
    return Loaded(image=img, width=w, height=h, fmt=ext, is_raw=True, **info)


def _friendly(exc: Exception, ext: str) -> str:
    msg = str(exc) or exc.__class__.__name__
    if ext in HEIF_EXTS and not HEIF_OK:
        return "iPhone HEIC support is not installed (package 'pillow-heif')"
    if ext in AVIF_EXTS and not avif_supported():
        return "AVIF support is not available in this install"
    low = msg.lower()
    if "cannot identify image file" in low:
        return "Not a readable image (damaged file or unsupported format)"
    if "truncated" in low or "premature end" in low or "broken data stream" in low:
        return "File is damaged or incomplete"
    if "exceeds limit" in low or "decompression bomb" in low:
        return "Image is too large to open safely"
    return msg[:160]


def load_image(path: str, data: bytes, work_side: int = 512) -> Loaded:
    """Decode `data` (the bytes of `path`). Raises on unreadable files with a plain-language message."""
    ext = os.path.splitext(path)[1].lower()
    if not data:
        raise ValueError("Empty file (0 bytes)")
    if ext in RAW_EXTS:
        try:
            return _load_raw(path, data, work_side)
        except UnsupportedFormat:
            raise
        except Exception as exc:
            # some ".raw"/".dng" files are really JPEG/TIFF; fall through to Pillow
            try:
                return _load_pillow(data, work_side)
            except Exception:
                raise ValueError(_friendly(exc, ext)) from None
    try:
        res = _load_pillow(data, work_side)
        if res.fmt == "HEIF" and ext in (".heic", ".heics"):
            res.fmt = "HEIC"  # the name people know from iPhones
        return res
    except Exception as first:
        # 1) damaged/incomplete file: try to read what is there
        prev = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            res = _load_pillow(data, work_side)
            res.damaged = True
            return res
        except Exception:
            pass
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev
        # 2) camera RAW file with an unusual extension
        if RAW_OK:
            try:
                return _load_raw(path, data, work_side)
            except Exception:
                pass
        raise ValueError(_friendly(first, ext)) from None
