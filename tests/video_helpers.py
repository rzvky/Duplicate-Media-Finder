"""Make small test videos with PyAV (no other tools needed)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    import av
except Exception:  # pragma: no cover
    av = None


def encoder(*names):
    for n in names:
        try:
            av.Codec(n, "w")
            return n
        except Exception:
            continue
    return None


def scene(seed: int, w: int, h: int, static: bool = False):
    """Frame maker: a textured background with moving shapes (or a still scene with sensor noise).
    Pictures are drawn at 640x480 and then resized, so every size shows the same content."""
    from PIL import Image

    W, H = 640, 480
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        f1, f2 = rng.uniform(0.005, 0.03, 2)
        base[..., c] = 110 + 60 * np.sin(xx * f1 + rng.uniform(0, 6)) * np.cos(yy * f2 + rng.uniform(0, 6))
    for _ in range(14):
        bx, by, r, col = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(15, 60), rng.uniform(0, 255, 3)
        base[(xx - bx) ** 2 + (yy - by) ** 2 < r * r] = col
    movers = [
        (
            rng.uniform(0, W),
            rng.uniform(0, H),
            rng.uniform(80, 200) * rng.choice([-1, 1]),
            rng.uniform(40, 120) * rng.choice([-1, 1]),
            rng.uniform(40, 80),
            rng.uniform(0, 255, 3),
        )
        for _ in range(5)
    ]
    noise_rng = np.random.default_rng(seed + 999)

    def frame(t: float) -> np.ndarray:
        img = base.copy()
        if static:
            img += noise_rng.normal(0, 2.0, img.shape)
        else:
            for x0, y0, vx, vy, r, col in movers:
                x = (x0 + vx * t) % W
                y = (y0 + vy * t) % H
                img[(np.abs(xx - x) < r) & (np.abs(yy - y) < r * 0.7)] = col
        pic = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        if (w, h) != (W, H):
            pic = pic.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(pic)

    return frame


def sound(seed: int, seconds: float, rate: int = 16000) -> np.ndarray:
    """Speech-like bursts, different for every seed (a shorter clip gets the start of the same sound)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    x = np.random.default_rng(seed + 7).normal(0, 0.003, n)
    t = 0.0
    while t < seconds:
        t += rng.uniform(0.1, 0.5)
        L = rng.uniform(0.1, 0.4)
        i0, i1 = int(t * rate), min(n, int((t + L) * rate))
        if i1 <= i0:
            break
        k = np.arange(i1 - i0) / rate
        x[i0:i1] += np.sin(2 * np.pi * rng.uniform(150, 400) * k) * rng.uniform(0.1, 0.5) * np.exp(-k * 4)
        t += L
    return np.clip(x, -1, 1).astype(np.float32)


def make_clip(
    path,
    seed: int,
    seconds: float = 3.0,
    fps: int = 15,
    size=(320, 240),
    vcodec=None,
    audio_seed=None,
    static: bool = False,
    rotate: int = 0,
    bake_rotation: bool = False,
    insert=None,
    logo: bool = False,
):
    """Write a clip. `rotate` stores a display rotation in the file; `bake_rotation` turns the pictures
    instead (what a converter does). The same seed always gives the same pictures.
    Edits: `insert=(t0, t1, seed)` shows other footage from t0 to t1 s; `logo` adds a corner logo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    base_make = scene(seed, w, h, static)
    other = scene(insert[2], w, h) if insert else None

    def make(t):
        img = other(t) if insert and insert[0] <= t < insert[1] else base_make(t)
        if logo:
            img = img.copy()
            img[h // 20 : h // 20 + h // 6, w - w // 20 - w // 4 : w - w // 20] = (250, 250, 250)
        return img

    vcodec = vcodec or encoder("libx264", "h264", "mpeg4")
    out = av.open(str(path), "w")
    vs = out.add_stream(vcodec, rate=fps)
    vw, vh = (h, w) if bake_rotation and rotate % 180 else (w, h)
    vs.width, vs.height, vs.pix_fmt = vw, vh, ("yuv420p" if vcodec != "mjpeg" else "yuvj420p")
    if rotate and not bake_rotation:
        vs.set_display_rotation(rotate)
    ast = None
    if audio_seed is not None:
        acodec = encoder("aac", "libmp3lame", "mp2") if path.suffix.lower() != ".webm" else encoder("libopus", "opus")
        ast = out.add_stream(acodec, rate=48000 if "opus" in acodec else 16000)
        ast.layout = "mono"
    for i in range(int(seconds * fps)):
        img = make(i / fps)
        if bake_rotation and rotate:
            img = np.ascontiguousarray(np.rot90(img, rotate // 90))
        fr = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in vs.encode(fr):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    if ast is not None:
        rate = ast.rate
        x = sound(audio_seed, seconds, 16000)
        if rate != 16000:
            x = np.interp(np.arange(int(seconds * rate)) * 16000 / rate, np.arange(len(x)), x).astype(np.float32)
        step = ast.codec_context.frame_size or 1024
        pts = 0
        for s in range(0, len(x), step):
            chunk = x[s : s + step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            af = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1),
                format="flt" if ast.codec_context.name in ("libmp3lame",) else "fltp",
                layout="mono",
            )
            af.sample_rate = rate
            af.pts = pts
            af.time_base = Fraction(1, rate)
            pts += step
            for pkt in ast.encode(af):
                out.mux(pkt)
        for pkt in ast.encode():
            out.mux(pkt)
    out.close()
    return path


def audio_roundtrip_ok(path, seed: int, seconds: float) -> bool:
    """True when the sound written to `path` decodes back like the source (some test machines have
    broken audio encoders; real files are only ever decoded by the tool)."""
    try:
        c = av.open(str(path))
        a = c.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=8000)
        x = np.concatenate([rf.to_ndarray().reshape(-1) for fr in c.decode(a) for rf in res.resample(fr)])
        c.close()
    except Exception:
        return False
    src = sound(seed, seconds, 16000)[::2]
    n = min(len(x), len(src))
    h = n // 2

    def rms(v):
        return float(np.sqrt((v**2).mean())) if len(v) else 0.0

    return all(abs(rms(x[a:b]) - rms(src[a:b])) <= 0.2 * max(rms(src[a:b]), 1e-6) for a, b in ((0, h), (h, n)))
