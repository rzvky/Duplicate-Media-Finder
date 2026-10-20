"""Build a test photo library with a known answer key.

Usage:  python tests/make_dataset.py OUT_DIR [ORIGINALS_DIR] [--n 40]   (--n: 28 or more)
Without ORIGINALS_DIR, synthetic pictures are generated.
Writes OUT_DIR/photos/... and OUT_DIR/truth.json
"""

from __future__ import annotations

import io
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF = True
except Exception:
    HEIF = False


def synthetic_original(seed: int, size=(1200, 900)) -> Image.Image:
    """A textured picture with random shapes (good enough for hash tests)."""
    rng = random.Random(seed)
    w, h = size
    base = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    c1 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    c2 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    t = (xx / w * rng.uniform(0.3, 1) + yy / h * rng.uniform(0.3, 1)) / 2
    base = c1 * (1 - t[..., None]) + c2 * t[..., None]
    nrng = np.random.default_rng(seed)
    base += nrng.normal(0, 6, base.shape)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 14)):
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        x1, y1 = x0 + rng.randint(60, 500), y0 + rng.randint(60, 400)
        col = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            d.ellipse((x0, y0, x1, y1), fill=col)
        else:
            d.rectangle((x0, y0, x1, y1), fill=col)
    return img.filter(ImageFilter.GaussianBlur(1.2))


def exif_bytes(dt: str | None, orientation: int = 1, subsec: str | None = None, make="Apple", model="iPhone 13"):
    ex = Image.Exif()
    ex[271] = make
    ex[272] = model
    if orientation != 1:
        ex[0x0112] = orientation
    if dt:
        sub = ex.get_ifd(0x8769)
        sub[36867] = dt
        if subsec:
            sub[37521] = subsec
    return ex.tobytes()


def save(img: Image.Image, path: Path, fmt: str, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, fmt, **kw)


def burst(img: Image.Image, rng: random.Random, blur: float = 0.0) -> Image.Image:
    w, h = img.size
    z = rng.uniform(1.02, 1.06)
    dx, dy = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
    cw, ch = w / z, h / z
    cx, cy = w / 2 + dx * w, h / 2 + dy * h
    box = (max(0, cx - cw / 2), max(0, cy - ch / 2), min(w, cx + cw / 2), min(h, cy + ch / 2))
    out = img.rotate(rng.uniform(-1, 1), resample=Image.BICUBIC).crop(box).resize((w, h), Image.LANCZOS)
    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.95, 1.05))
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def build(out: Path, originals: list, n: int = 40, seed: int = 11, raw_files: list = ()) -> dict:
    rng = random.Random(seed)
    out = Path(out)
    photos = out / "photos"
    if photos.exists():
        shutil.rmtree(photos)
    photos.mkdir(parents=True)
    truth = {"families": {}, "unrelated": [], "errors": [], "skipped": [], "companions": {}}

    def add(fam, rel, kind):
        truth["families"].setdefault(fam, []).append({"rel": rel, "kind": kind})

    imgs = []
    for i in range(n):
        src = originals[i] if i < len(originals) else None
        im = Image.open(src).convert("RGB") if src else synthetic_original(1000 + i)
        if max(im.size) < 1000:  # phone-like size
            s = 1600 / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        imgs.append(im)

    def stamp(k, sec=0):
        # each family gets its own day so unrelated photos are far apart in time
        d = 1 + (k % 27)
        return f"2024:{1 + k // 27:02d}:{d:02d} 1{k % 10}:{(sec // 60) % 60:02d}:{sec % 60:02d}"

    k = 0
    # A) exact copies with new names in other folders (incl. unicode + spaces)
    for j in range(6):
        im = imgs[k]
        fam = f"exact{j}"
        rel = f"DCIM/100APPLE/IMG_{1000 + k}.JPG"
        save(im, photos / rel, "JPEG", quality=92, exif=exif_bytes(stamp(k)))
        add(fam, rel, "original")
        copies = [
            f"Backup/IMG_{1000 + k} - Copy.JPG",
            f"WhatsApp Images/Photo ({j + 1}).jpg",
            f"Família & Friends/été {j}.jpg",
        ]
        for crel in copies[: 1 + j % 3]:
            (photos / crel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(photos / rel, photos / crel)
            add(fam, crel, "exact")
        k += 1
    # B) lower-quality copies: resized, recompressed, converted (EXIF often stripped)
    conv = [
        ("half", "JPEG", ".jpg", {"quality": 70}, 0.5),
        ("quarter", "JPEG", ".jpg", {"quality": 60}, 0.25),
        ("png", "PNG", ".png", {}, 1.0),
        ("webp", "WEBP", ".webp", {"quality": 60}, 1.0),
        ("small_webp", "WEBP", ".webp", {"quality": 50}, 0.4),
    ]
    if HEIF:
        conv.append(("heic", "HEIF", ".heic", {"quality": 55}, 1.0))
    try:
        Image.new("RGB", (8, 8)).save(io.BytesIO(), "AVIF")
        conv.append(("avif", "AVIF", ".avif", {"quality": 50}, 0.6))
    except Exception:
        pass
    for j in range(8):
        im = imgs[k]
        fam = f"lower{j}"
        rel = f"DCIM/101APPLE/IMG_{2000 + k}.HEIC" if HEIF and j % 2 == 0 else f"Camera/IMG_2024_{k}.jpg"
        fmt = "HEIF" if rel.endswith(".HEIC") else "JPEG"
        save(im, photos / rel, fmt, quality=90, exif=exif_bytes(stamp(k)))
        add(fam, rel, "original")
        for c in range(2):
            name, f2, ext, kw, scale = conv[(j * 2 + c) % len(conv)]
            v = im if scale == 1.0 else im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            crel = f"Exports/{name}/pic_{k}_{c}{ext}"
            save(v, photos / crel, f2, **kw)
            add(fam, crel, "lower")
        k += 1
    # C) bursts: 3 shots 1 s apart, the 3rd is blurry
    for j in range(6):
        im = imgs[k]
        fam = f"burst{j}"
        for b in range(3):
            v = im if b == 0 else burst(im, rng, blur=(2.2 if b == 2 else 0))
            rel = f"DCIM/102APPLE/IMG_{3000 + k * 10 + b}.JPG"
            save(v, photos / rel, "JPEG", quality=90, exif=exif_bytes(stamp(k, b), subsec=str(10 * b)))
            add(fam, rel, "burst_sharp" if b < 2 else "burst_blurry")
        k += 1
    # D) edited versions (colour/brightness changed), same capture time
    for j in range(4):
        im = imgs[k]
        fam = f"edit{j}"
        rel = f"Camera/IMG_{4000 + k}.jpg"
        save(im, photos / rel, "JPEG", quality=92, exif=exif_bytes(stamp(k)))
        add(fam, rel, "original")
        e = ImageEnhance.Color(im).enhance(1.6) if j % 2 == 0 else ImageEnhance.Brightness(im).enhance(1.35)
        erel = f"Camera/IMG_{4000 + k}_edited.jpg"
        save(e, photos / erel, "JPEG", quality=92, exif=exif_bytes(stamp(k)))
        add(fam, erel, "edit")
        k += 1
    # E) orientation: stored sideways + EXIF rotate tag vs. an upright copy without EXIF
    for j in range(3):
        im = imgs[k]
        fam = f"orient{j}"
        sideways = im.transpose(Image.Transpose.ROTATE_90)  # undone by orientation 6
        rel = f"DCIM/103APPLE/IMG_{5000 + k}.JPG"
        save(sideways, photos / rel, "JPEG", quality=92, exif=exif_bytes(stamp(k), orientation=6))
        add(fam, rel, "original")
        urel = f"Shared/upright_{k}.jpg"
        save(im, photos / urel, "JPEG", quality=80)
        add(fam, urel, "lower")
        k += 1
    # F) Live Photo: HEIC + MOV, and an exact copy of the HEIC (with its own MOV) elsewhere
    if HEIF:
        im = imgs[k]
        fam = "live0"
        rel = f"iPhone/IMG_{6000 + k}.HEIC"
        save(im, photos / rel, "HEIF", quality=90, exif=exif_bytes(stamp(k)))
        (photos / f"iPhone/IMG_{6000 + k}.MOV").write_bytes(b"fake live photo video")
        add(fam, rel, "original")
        crel = f"iPhone Backup/IMG_{6000 + k}.HEIC"
        (photos / crel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(photos / rel, photos / crel)
        (photos / f"iPhone Backup/IMG_{6000 + k}.MOV").write_bytes(b"fake live photo video")
        (photos / f"iPhone Backup/IMG_{6000 + k}.AAE").write_bytes(b"<plist/>")
        (photos / f"iPhone Backup/IMG_{6000 + k}.HEIC.supplemental-metadata.json").write_text("{}")
        add(fam, crel, "exact")
        truth["companions"][crel] = [
            f"iPhone Backup/IMG_{6000 + k}.MOV",
            f"iPhone Backup/IMG_{6000 + k}.AAE",
            f"iPhone Backup/IMG_{6000 + k}.HEIC.supplemental-metadata.json",
        ]
        k += 1
    # RAW + camera JPEG pairs (needs real RAW sample files)
    for j, rf in enumerate(raw_files):
        try:
            import rawpy

            with rawpy.imread(str(rf)) as raw:
                th = raw.extract_thumb()
            if th.format != rawpy.ThumbFormat.JPEG:
                continue
        except Exception:
            continue
        fam = f"raw{j}"
        stem = f"DSC_{7000 + j}"
        rrel = f"Camera RAW/{stem}{Path(rf).suffix.upper()}"
        (photos / rrel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rf, photos / rrel)
        (photos / f"Camera RAW/{stem}.JPG").write_bytes(th.data)
        add(fam, rrel, "original")
        add(fam, f"Camera RAW/{stem}.JPG", "rawpair")
    # G) unrelated photos (must not be grouped), spread over folders and formats, no time info for some
    while k < n:
        im = imgs[k]
        ext, fmt = [(".jpg", "JPEG"), (".png", "PNG"), (".webp", "WEBP")][k % 3]
        rel = f"Misc/{'Sub ' + str(k % 4)}/unique_{k}{ext}"
        kw = {"quality": 88} if fmt != "PNG" else {}
        if k % 2 == 0 and fmt == "JPEG":
            kw["exif"] = exif_bytes(stamp(k))
        save(im, photos / rel, fmt, **kw)
        truth["unrelated"].append(rel)
        k += 1
    # H) problem files: damaged, empty, fake, plus things that must be ignored
    trunc = photos / "Problems/half_downloaded.jpg"
    buf = io.BytesIO()
    imgs[0].rotate(180).save(buf, "JPEG", quality=90)
    trunc.parent.mkdir(parents=True, exist_ok=True)
    trunc.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])
    truth["damaged_ok"] = ["Problems/half_downloaded.jpg"]
    (photos / "Problems/empty.png").write_bytes(b"")
    (photos / "Problems/not_really.jpg").write_bytes(b"this is text, not a photo" * 20)
    (photos / "Problems/fake.heic").write_bytes(os.urandom(3000))
    truth["errors"] += ["Problems/empty.png", "Problems/not_really.jpg", "Problems/fake.heic"]
    (photos / "notes.txt").write_text("not a photo")
    th = photos / "DCIM/.thumbnails"
    th.mkdir(parents=True, exist_ok=True)
    imgs[1].resize((200, 150)).save(th / "thumb1.jpg")
    (photos / "DCIM/._IMG_1000.JPG").write_bytes(b"\x00\x05\x16\x07appledouble")
    truth["skipped"] += ["DCIM/.thumbnails/thumb1.jpg", "DCIM/._IMG_1000.JPG"]
    (out / "truth.json").write_text(json.dumps(truth, indent=1, ensure_ascii=False), encoding="utf-8")
    return truth


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build a test photo library with a known answer key.")
    ap.add_argument("out_dir", help="where to create photos/ and truth.json")
    ap.add_argument("originals_dir", nargs="?", help="folder of your own JPG photos to copy from (optional)")
    ap.add_argument("--n", type=int, default=40, help="number of original photos to use (default 40)")
    args = ap.parse_args()
    if args.n < 28:
        ap.error("--n must be at least 28 (one photo per duplicate family)")
    n = args.n
    outdir = Path(args.out_dir)
    originals = []
    if args.originals_dir:
        originals = sorted(
            str(p) for p in Path(args.originals_dir).rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
        )
        random.Random(3).shuffle(originals)
    raws = [str(p) for p in Path(os.environ["RAW_SAMPLES"]).iterdir()] if os.environ.get("RAW_SAMPLES") else []
    t = build(outdir, originals, n=n, raw_files=raws)
    print(f"built {sum(len(v) for v in t['families'].values())} family files, {len(t['unrelated'])} unrelated")
