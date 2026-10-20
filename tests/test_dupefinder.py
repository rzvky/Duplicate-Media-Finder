"""Automated checks. Run from the tool folder:  python -m pytest tests -q"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from evaluate import evaluate  # noqa: E402
from make_dataset import build, exif_bytes, synthetic_original  # noqa: E402

from dupefinder import actions, report  # noqa: E402
from dupefinder.cli import run_scan  # noqa: E402
from dupefinder.features import compute_from_loaded, extract  # noqa: E402
from dupefinder.grouping import PRESETS, build_groups, complete_linkage, hamming_pairs, popcount  # noqa: E402
from dupefinder.loader import HEIF_OK, load_image, parse_exif_datetime  # noqa: E402


def _fp(img, fmt, **kw):
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    li = load_image("x." + fmt.lower(), b.getvalue())
    f, _ = compute_from_loaded(li, False)
    return li, f


def _ham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ------------------------------------------------------------------ loading
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF"] + (["HEIF"] if HEIF_OK else []))
def test_formats_open_and_rotate(fmt):
    img = synthetic_original(1, (600, 400))
    kw = {}
    if fmt in ("JPEG", "PNG", "WEBP", "TIFF", "HEIF"):
        kw["exif"] = exif_bytes("2023:05:01 10:00:00", orientation=6)
    if fmt == "GIF":
        img = img.convert("P")
    li, f = _fp(img, fmt, **kw)
    if "exif" in kw:
        assert (li.width, li.height) == (400, 600), "EXIF rotation must be applied"
        assert li.dt is not None
    else:
        assert (li.width, li.height) == (600, 400)
    assert li.image.mode == "RGB"


def test_odd_modes_and_transparency():
    rgba = Image.new("RGBA", (300, 200), (255, 0, 0, 0))
    li, _ = _fp(rgba, "PNG")
    assert li.image.getpixel((5, 5)) == (255, 255, 255)  # transparent becomes white
    g16 = Image.fromarray((np.arange(300 * 200).reshape(200, 300) % 65535).astype(np.uint16))
    li, _ = _fp(g16, "PNG")
    assert li.image.mode == "RGB"
    cmyk = synthetic_original(2, (300, 200)).convert("CMYK")
    li, _ = _fp(cmyk, "JPEG")
    assert li.image.mode == "RGB"


def test_unreadable_files_report_errors(tmp_path):
    (tmp_path / "empty.jpg").write_bytes(b"")
    (tmp_path / "text.jpg").write_bytes(b"hello" * 100)
    for name in ("empty.jpg", "text.jpg"):
        r = extract((str(tmp_path / name), False))
        assert "error" in r and r["error"]


def test_truncated_jpeg_is_still_read(tmp_path):
    b = io.BytesIO()
    synthetic_original(3, (800, 600)).save(b, "JPEG", quality=90)
    p = tmp_path / "half.jpg"
    p.write_bytes(b.getvalue()[: len(b.getvalue()) // 2])
    r = extract((str(p), False))
    assert "error" not in r and r["feat"]["damaged"]


def test_exif_datetime_parsing():
    assert parse_exif_datetime("2024:01:02 03:04:05") == parse_exif_datetime("2024-01-02 03:04:05")
    assert parse_exif_datetime("0000:00:00 00:00:00") is None
    assert parse_exif_datetime("2024:01:02 03:04:05", "5") - parse_exif_datetime(
        "2024:01:02 03:04:05"
    ) == pytest.approx(0.5)


# ------------------------------------------------------------------ fingerprints
def test_copies_hash_close_unrelated_far():
    a = synthetic_original(10)
    b = synthetic_original(11)
    _, fa = _fp(a, "JPEG", quality=95)
    for v, fmt, kw in [(a.resize((300, 225)), "JPEG", {"quality": 40}), (a, "PNG", {}), (a, "WEBP", {"quality": 50})]:
        _, fv = _fp(v, fmt, **kw)
        assert _ham(fa["ph"], fv["ph"]) <= 4
        assert _ham(fa["dh"], fv["dh"]) <= 6
    _, fb = _fp(b, "JPEG", quality=95)
    assert _ham(fa["ph"], fb["ph"]) > 12


def test_pair_search_and_linkage():
    h = np.array([0, 1, 3, 0xFFFF], dtype=np.uint64)
    I, J, D = hamming_pairs(h, 2)
    assert set(zip(I.tolist(), J.tolist())) == {(0, 1), (0, 2), (1, 2)}
    assert popcount(np.array([0xFF], dtype=np.uint64))[0] == 8
    # chain A~B, B~C but A!~C must not merge all three
    cl = complete_linkage(3, np.array([0, 1]), np.array([1, 2]), np.array([0.9, 0.95], dtype=np.float32))
    assert sorted(len(c) for c in cl) == [1, 2]


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("lib")
    build(d, [], n=40)
    return d


@pytest.mark.parametrize("use_ai", [True, False])
def test_end_to_end_accuracy(dataset, use_ai):
    run_scan(str(dataset / "photos"), use_ai=use_ai, workers=2, quiet=True)
    r = evaluate(dataset)
    assert r["precision"] == 1.0, r["problems"][:10]
    assert r["recall"] == 1.0, r["problems"][:10]
    assert not r["problems"], r["problems"][:10]


def test_move_and_undo_are_safe(dataset):
    root = dataset / "photos"
    root_, out, results = run_scan(str(root), workers=2, quiet=True)

    def snap():
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file() and "_DupeFinder" not in p.parts
        }

    before = snap()
    fids = {m["fid"] for g in results["groups"] for m in g["files"] if m["selected"]}
    allowed, blocked = actions.guard_selection(results, fids)
    assert not blocked and len(allowed) == len(fids)
    every = {m["fid"] for m in results["groups"][0]["files"]}
    assert actions.guard_selection(results, every)[1] == [results["groups"][0]["id"]]
    res = actions.move_to_bin(root_, out, allowed)
    assert len(res["moved"]) == len(allowed) and not res["skipped"]
    # the kept photo of every group is still in place
    for g in results["groups"]:
        assert any((root / m["rel"]).exists() for m in g["files"])
    truth = json.loads((dataset / "truth.json").read_text(encoding="utf-8"))
    for photo, comps in truth["companions"].items():
        if photo in {m["rel"].replace("\\", "/") for m in allowed}:
            for c in comps:
                assert not (root / c).exists() and (out / "Review_Bin" / c).exists()
    u = actions.undo(root_, out)
    assert len(u["restored"]) == len(allowed) and not u["failed"]
    assert snap() == before


def test_changed_file_is_not_moved(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    f = root / "a.jpg"
    synthetic_original(5, (200, 150)).save(f, "JPEG")
    st = f.stat()
    f.write_bytes(f.read_bytes() + b"x")
    res = actions.move_to_bin(root, out, [{"rel": "a.jpg", "size": st.st_size, "mtime_ns": st.st_mtime_ns}])
    assert not res["moved"] and f.exists()


def test_server_requires_token(dataset):
    from dupefinder.server import ReviewApp

    root = dataset / "photos"
    out = root / "_DupeFinder"
    run_scan(str(root), workers=2, quiet=True)
    app = ReviewApp(root, out, report.load_json(out))
    url = app.serve(port=8899, open_browser=False)
    t = threading.Thread(target=app.run_forever, daemon=True)
    t.start()
    try:

        def call(path, body=None, token=None):
            req = urllib.request.Request(
                url.rstrip("/") + path,
                method="POST" if body is not None else "GET",
                data=json.dumps(body).encode() if body is not None else None,
            )
            if token:
                req.add_header("X-Token", token)
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code

        assert call("/") == 200
        assert call("/api/results") == 403
        assert call("/api/results", token=app.token) == 200
        assert call("/api/move", {"fids": []}) == 403
        assert call("/thumb/../../secret.jpg") == 403  # previews need the token too
        assert call("/thumb/../../secret.jpg", token=app.token) == 404
        assert call("/api/move", [], token=app.token) == 400  # body must be an object
    finally:
        app.httpd.shutdown()
        t.join(5)


def test_presets_are_ordered():
    s, n, l = PRESETS["strict"], PRESETS["normal"], PRESETS["loose"]
    assert s.copy_phash <= n.copy_phash <= l.copy_phash
    assert s.sim_ai_time >= n.sim_ai_time >= l.sim_ai_time
    assert build_groups([], n, False) == []


# ------------------------------------------------------------------ regression tests (edge cases found while testing)
from PIL import ImageDraw, ImageFont  # noqa: E402


def _ticked(folder):
    res = report.load_json(Path(folder) / "_DupeFinder")
    return [m["rel"].replace("\\", "/") for g in res["groups"] for m in g["files"] if m["selected"]], res


def test_same_second_burst_is_not_ticked(tmp_path):
    base = synthetic_original(21, (1600, 1067))
    for b in range(4):
        v = base.copy()
        d = ImageDraw.Draw(v)
        x = 500 + 60 * b  # subject moves between frames
        d.ellipse((x, 300, x + 160, 460), fill=(230, 180, 150))
        d.rectangle((x - 20, 460, x + 180, 900), fill=(40, 60, 160))
        v.save(
            tmp_path / f"DSC0{b}.JPG", quality=93, exif=exif_bytes("2024:06:01 15:20:10", make="NIKON", model="D750")
        )
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, _ = _ticked(tmp_path)
    assert ticked == []


def test_screenshots_with_one_changed_line_are_not_ticked(tmp_path):
    font = ImageFont.load_default()
    for k in range(3):
        im = Image.new("RGB", (1080, 2340), (245, 245, 245))
        d = ImageDraw.Draw(im)
        for i in range(25):
            y = 120 + i * 85
            d.rectangle((40, y - 10, 700, y + 55), fill=(220, 235, 255))
            d.text((60, y), "Account 5021 %d paid" % (i if i != 12 else 90000 + k), fill=(20, 20, 20), font=font)
        im.save(tmp_path / f"Screenshot_{k}.png")
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, _ = _ticked(tmp_path)
    assert ticked == []


def test_stripped_copy_does_not_chain_two_shots(tmp_path):
    base = synthetic_original(22, (1600, 1067))
    a = base
    b = base.copy()
    ImageDraw.Draw(b).ellipse((700, 400, 900, 600), fill=(250, 250, 30))
    a.save(tmp_path / "DSC_0000.JPG", quality=93, exif=exif_bytes("2024:06:01 10:00:00"))
    b.save(tmp_path / "DSC_0001.JPG", quality=93, exif=exif_bytes("2024:06:01 10:00:01"))
    a.save(tmp_path / "IMG-WA0007.jpg", quality=80)  # forwarded copy of frame 0, no EXIF
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, _ = _ticked(tmp_path)
    assert "DSC_0001.JPG" not in ticked and "DSC_0000.JPG" not in ticked


def test_nested_results_folder_is_never_scanned(tmp_path):
    trip = tmp_path / "Trip"
    trip.mkdir()
    synthetic_original(23).save(trip / "a.jpg", quality=95)
    synthetic_original(23).resize((300, 225)).save(trip / "a_small.jpg", quality=70)
    run_scan(str(trip), workers=2, quiet=True)
    assert (trip / "_DupeFinder" / "thumbs").is_dir()
    _, _, results = run_scan(str(tmp_path), workers=2, quiet=True)
    rels = [m["rel"] for g in results["groups"] for m in g["files"]]
    assert rels and not any("_DupeFinder" in r for r in rels)


def test_camera_original_beats_enlarged_copy(tmp_path):
    base = synthetic_original(24, (1600, 1067))
    base.save(tmp_path / "IMG_4512.JPG", quality=95, exif=exif_bytes("2024:01:01 10:00:00"))
    base.resize((800, 533)).resize((3200, 2134)).save(tmp_path / "enhanced.jpg", quality=92)
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, _ = _ticked(tmp_path)
    assert ticked == ["enhanced.jpg"]


def test_multipage_and_animated_files_are_not_ticked(tmp_path):
    p1 = synthetic_original(25, (800, 600))
    p1.save(tmp_path / "page1.jpg", quality=95)
    p1.save(tmp_path / "contract.tif", save_all=True, append_images=[synthetic_original(26, (800, 600))])
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, _ = _ticked(tmp_path)
    assert "contract.tif" not in ticked


def test_undo_can_finish_after_a_blocked_restore(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    synthetic_original(27, (400, 300)).save(root / "a.jpg", quality=90)
    shutil = __import__("shutil")
    shutil.copy2(root / "a.jpg", root / "a copy.jpg")
    _, out, results = run_scan(str(root), workers=2, quiet=True)
    fids = {m["fid"] for g in results["groups"] for m in g["files"] if m["selected"]}
    allowed, _ = actions.guard_selection(results, fids, root)
    actions.move_to_bin(root, out, allowed)
    moved = allowed[0]["rel"]
    (root / moved).write_bytes(b"something new")  # blocks the restore
    assert actions.undo(root, out)["failed"]
    (root / moved).unlink()  # user clears the way
    assert actions.undo(root, out)["restored"] == [moved]


def test_ai_model_is_rebuilt_from_parts(tmp_path, monkeypatch):
    from dupefinder import ai

    src = ai.MODEL_FILE
    if not src.is_file():
        pytest.skip("model not present")
    data = src.read_bytes()
    size = -(-len(data) // ai.PARTS)
    for i in range(ai.PARTS):
        (tmp_path / ai.PART_NAME.format(i + 1)).write_bytes(data[i * size : (i + 1) * size])
    target = tmp_path / "models" / src.name
    monkeypatch.setattr(ai, "MODEL_FILE", target)
    monkeypatch.setattr(ai, "MODEL_DIR", target.parent)
    ok, msg = ai.ensure_model([tmp_path])
    assert ok and target.read_bytes() == data
    assert not any(tmp_path.glob("*_AI_part*"))  # parts tidied away
    # a damaged part is refused
    target.unlink()
    for i in range(ai.PARTS):
        (tmp_path / ai.PART_NAME.format(i + 1)).write_bytes(data[i * size : (i + 1) * size])
    (tmp_path / ai.PART_NAME.format(2)).write_bytes(b"broken")
    ok, msg = ai.ensure_model([tmp_path])
    assert not ok and not target.exists()


def test_saturation_edit_of_a_grey_photo_is_not_ticked(tmp_path):
    from PIL import ImageEnhance

    base = synthetic_original(31, (1200, 900))
    grey = ImageEnhance.Color(base).enhance(0.3)  # a low-colour photo
    grey.save(tmp_path / "IMG_1.jpg", quality=92, exif=exif_bytes("2024:01:01 10:00:00"))
    ImageEnhance.Color(grey).enhance(1.6).save(
        tmp_path / "IMG_1_vivid.jpg", quality=92, exif=exif_bytes("2024:01:01 10:00:00")
    )
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, _ = _ticked(tmp_path)
    assert ticked == []


def test_original_beats_same_size_conversion(tmp_path):
    import time as _t

    base = synthetic_original(32, (900, 600))
    base.save(tmp_path / "IMG_2.jpg", quality=92)
    old = _t.time() - 3600
    os.utime(tmp_path / "IMG_2.jpg", (old, old))  # the original existed first
    base.save(tmp_path / "export.png")
    if HEIF_OK:
        base.save(tmp_path / "export.heic", quality=80)
    run_scan(str(tmp_path), workers=2, quiet=True)
    ticked, res = _ticked(tmp_path)
    keepers = [m["rel"] for g in res["groups"] for m in g["files"] if m["role"] == "keep"]
    assert keepers == ["IMG_2.jpg"] and "IMG_2.jpg" not in ticked


def test_scan_finishes_when_helper_processes_cannot_start(tmp_path, monkeypatch):
    from dupefinder import scanner

    class NoStart:
        def __init__(self, *a, **k):
            raise OSError("process creation blocked")

    monkeypatch.setattr(scanner, "ProcessPoolExecutor", NoStart)
    synthetic_original(33).save(tmp_path / "a.jpg", quality=90)
    shutil = __import__("shutil")
    shutil.copy2(tmp_path / "a.jpg", tmp_path / "a copy.jpg")
    res = scanner.scan(tmp_path, tmp_path / "_DupeFinder", use_ai=False, workers=2)
    assert len(res.records) == 2 and not res.errors


# ------------------------------------------------------------------ videos
from video_helpers import audio_roundtrip_ok, av, encoder, make_clip  # noqa: E402

needs_av = pytest.mark.skipif(av is None, reason="video support (package 'av') not installed")


def _groups(folder, media=None):
    res = report.load_json(Path(folder) / "_DupeFinder")
    gs = [g for g in res["groups"] if media is None or g.get("media") == media]
    return [sorted(m["rel"].replace("\\", "/") for m in g["files"]) for g in gs], res


@needs_av
def test_video_copies_in_other_formats_are_found(tmp_path):
    make_clip(tmp_path / "Phone/VID_0001.mp4", 1, seconds=4, size=(480, 360), audio_seed=11)
    (tmp_path / "Backup").mkdir()
    import shutil

    shutil.copy2(tmp_path / "Phone/VID_0001.mp4", tmp_path / "Backup/holiday.mp4")  # exact copy, renamed
    make_clip(tmp_path / "WhatsApp/VID-2024-WA0001.avi", 1, seconds=4, size=(240, 180), vcodec="mpeg4", audio_seed=11)
    if encoder("libvpx-vp9", "libvpx"):
        make_clip(
            tmp_path / "Shared/clip.webm",
            1,
            seconds=4,
            size=(480, 360),
            vcodec=encoder("libvpx-vp9", "libvpx"),
            audio_seed=11,
        )
        if not audio_roundtrip_ok(tmp_path / "Shared/clip.webm", 11, 4):
            (tmp_path / "Shared/clip.webm").unlink()  # this machine's Opus encoder is broken (seen under Wine)
    make_clip(tmp_path / "Shared/no sound.mkv", 1, seconds=4, size=(480, 360), vcodec="mpeg4")  # sound removed
    make_clip(
        tmp_path / "Phone/VID_0002.mp4", 2, seconds=4, size=(480, 360), audio_seed=12
    )  # another video, same length
    make_clip(
        tmp_path / "Phone/VID_0003.mp4", 1, seconds=2.5, size=(480, 360), audio_seed=11
    )  # trimmed: different length
    run_scan(str(tmp_path), workers=2, quiet=True)
    groups, res = _groups(tmp_path, "video")
    assert res["stats"]["videos_found"] == res["stats"]["videos_read"] >= 6
    expect = {"Phone/VID_0001.mp4", "Backup/holiday.mp4", "WhatsApp/VID-2024-WA0001.avi", "Shared/no sound.mkv"}
    if (tmp_path / "Shared/clip.webm").exists():
        expect.add("Shared/clip.webm")
    assert groups == [sorted(expect)]
    g = [g for g in res["groups"] if g.get("media") == "video"][0]
    roles = {m["rel"].replace("\\", "/"): (m["role"], m["selected"]) for m in g["files"]}
    keeper = [r for r, (role, _) in roles.items() if role == "keep"]
    assert keeper in (["Phone/VID_0001.mp4"], ["Backup/holiday.mp4"])
    assert all(sel for r, (role, sel) in roles.items() if role != "keep")
    assert all(m["dur"] and m["media"] == "video" for m in g["files"])
    reasons = " ".join(m["reason"] for m in g["files"])
    assert "Identical file" in reasons and "smaller size" in reasons and "without sound" in reasons
    assert g["evidence"], "the review screen says why videos were grouped"
    csv_text = (tmp_path / "_DupeFinder/duplicates.csv").read_text(encoding="utf-8-sig")
    assert "Video" in csv_text and "0:04" in csv_text


@needs_av
def test_edited_and_trimmed_versions_are_not_grouped(tmp_path):
    """A smaller copy is the same video. A version with a logo, with a second of other footage
    inserted between the checked moments, or with the end cut off is a different video."""
    kw = dict(seconds=24, fps=12, size=(480, 360), audio_seed=11)
    make_clip(tmp_path / "holiday.mp4", 11, **kw)
    make_clip(tmp_path / "small/holiday_small.mp4", 11, **{**kw, "size": (240, 180)})
    make_clip(tmp_path / "edits/holiday_logo.mp4", 11, logo=True, **kw)
    make_clip(tmp_path / "edits/holiday_insert.mp4", 11, insert=(4.4, 5.6, 99), **kw)
    make_clip(tmp_path / "edits/holiday_trim.mp4", 11, **{**kw, "seconds": 22})
    run_scan(str(tmp_path), workers=2, quiet=True, media="videos")
    groups, _ = _groups(tmp_path, "video")
    assert groups == [["holiday.mp4", "small/holiday_small.mp4"]], groups


@needs_av
def test_the_complete_copy_is_kept_over_a_slightly_shorter_one(tmp_path):
    make_clip(tmp_path / "Backup/clip (1).mp4", 12, seconds=6.0, size=(320, 240), audio_seed=12)
    make_clip(tmp_path / "clip.mp4", 12, seconds=5.7, size=(320, 240), audio_seed=12)
    os.utime(tmp_path / "Backup/clip (1).mp4", (2_000_000_000, 2_000_000_000))  # the complete one is newer
    run_scan(str(tmp_path), workers=2, quiet=True, media="videos")
    groups, res = _groups(tmp_path, "video")
    assert groups == [["Backup/clip (1).mp4", "clip.mp4"]]
    g = res["groups"][0]
    assert [m["rel"].replace("\\", "/") for m in g["files"] if m["role"] == "keep"] == ["Backup/clip (1).mp4"]
    assert "shorter" in " ".join(m["reason"] for m in g["files"])


@needs_av
def test_still_scene_without_sound_is_not_grouped(tmp_path):
    # two different recordings of the same still scene, same length, no sound: they look identical,
    # so pictures alone must not be enough
    make_clip(tmp_path / "cam/clip_0800.mp4", 5, seconds=4, static=True)
    make_clip(tmp_path / "cam/clip_0900.mp4", 5, seconds=4, static=True, vcodec="mpeg4")
    # the same still scene WITH different sound: also not grouped
    make_clip(tmp_path / "cam2/a.mp4", 6, seconds=4, static=True, audio_seed=1)
    make_clip(tmp_path / "cam2/b.mp4", 6, seconds=4, static=True, audio_seed=2)
    # a real copy of a still scene keeps its sound: grouped
    make_clip(tmp_path / "cam3/talk.mp4", 7, seconds=4, static=True, audio_seed=3)
    make_clip(
        tmp_path / "cam3/talk small.mkv", 7, seconds=4, static=True, audio_seed=3, size=(240, 180), vcodec="mpeg4"
    )
    run_scan(str(tmp_path), workers=2, quiet=True)
    groups, _ = _groups(tmp_path, "video")
    assert groups == [["cam3/talk small.mkv", "cam3/talk.mp4"]]


@needs_av
@pytest.mark.skipif(
    av is not None and not hasattr(av.video.stream.VideoStream, "set_display_rotation"),
    reason="this PyAV version cannot write a rotated test video",
)
def test_rotated_phone_video_matches_converted_copy(tmp_path):
    make_clip(tmp_path / "VID_portrait.mp4", 8, seconds=3, rotate=90, audio_seed=4)
    make_clip(
        tmp_path / "shared/portrait copy.mkv",
        8,
        seconds=3,
        rotate=90,
        bake_rotation=True,
        vcodec="mpeg4",
        audio_seed=4,
        size=(320, 240),
    )
    run_scan(str(tmp_path), workers=2, quiet=True)
    groups, res = _groups(tmp_path, "video")
    assert groups == [["VID_portrait.mp4", "shared/portrait copy.mkv"]]
    dims = {m["rel"].replace("\\", "/"): (m["w"], m["h"]) for g in res["groups"] for m in g["files"]}
    assert dims["VID_portrait.mp4"] == (240, 320), "rotation from the file is applied"


@needs_av
def test_live_photo_clip_stays_with_its_photo(tmp_path):
    img = synthetic_original(40, (800, 600))
    (tmp_path / "iPhone").mkdir()
    (tmp_path / "Backup").mkdir()
    img.save(tmp_path / "iPhone/IMG_1001.JPG", quality=92)
    make_clip(tmp_path / "iPhone/IMG_1001.MOV", 9, seconds=1.5)
    img.save(tmp_path / "Backup/IMG_1001.JPG", quality=92)
    make_clip(tmp_path / "Backup/IMG_1001.MOV", 9, seconds=1.5)
    run_scan(str(tmp_path), workers=2, quiet=True)
    res = report.load_json(tmp_path / "_DupeFinder")
    assert res["stats"]["videos_found"] == 0 and res["stats"]["live_clips"] == 2
    ticked = [m for g in res["groups"] for m in g["files"] if m["selected"]]
    assert len(ticked) == 1
    out = actions.move_to_bin(tmp_path, tmp_path / "_DupeFinder", ticked)
    assert out["companions"] == 1
    moved_photo = Path(ticked[0]["rel"])
    assert not (tmp_path / moved_photo.with_suffix(".MOV")).exists()


@needs_av
def test_moving_a_video_takes_its_sidecars_but_never_another_video(tmp_path):
    make_clip(tmp_path / "GOPR0001.MP4", 10, seconds=3, audio_seed=9)
    (tmp_path / "GOPR0001.THM").write_bytes(b"thumbnail")
    (tmp_path / "GOPR0001.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nGPS\n")
    make_clip(tmp_path / "GOPR0001.MOV", 10, seconds=3, audio_seed=9, vcodec="mpeg4", size=(160, 120))
    run_scan(str(tmp_path), workers=2, quiet=True)
    groups, res = _groups(tmp_path, "video")
    assert groups == [["GOPR0001.MOV", "GOPR0001.MP4"]]
    ticked = [m for g in res["groups"] for m in g["files"] if m["selected"]]
    assert [m["rel"] for m in ticked] == ["GOPR0001.MOV"]
    out = actions.move_to_bin(tmp_path, tmp_path / "_DupeFinder", ticked)
    assert out["companions"] == 0, "sidecars stay with the video that stays"
    assert (tmp_path / "GOPR0001.MP4").exists() and (tmp_path / "GOPR0001.THM").exists()
    # now move the kept one by hand: it takes its sidecars, and the other video is not touched
    item = [m for g in res["groups"] for m in g["files"] if m["rel"] == "GOPR0001.MP4"][0]
    out = actions.move_to_bin(tmp_path, tmp_path / "_DupeFinder", [item])
    assert out["companions"] == 2
    assert actions.undo(tmp_path, tmp_path / "_DupeFinder", all_batches=True)["failed"] == []
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == [
        "GOPR0001.MOV",
        "GOPR0001.MP4",
        "GOPR0001.THM",
        "GOPR0001.srt",
    ]


@needs_av
def test_video_rescan_uses_the_cache(tmp_path):
    make_clip(tmp_path / "a.mp4", 12, seconds=3, audio_seed=1)
    make_clip(tmp_path / "a copy.avi", 12, seconds=3, audio_seed=1, vcodec="mpeg4")
    make_clip(tmp_path / "b.mp4", 13, seconds=3, audio_seed=2)
    _, _, r1 = run_scan(str(tmp_path), workers=2, quiet=True)
    _, _, r2 = run_scan(str(tmp_path), workers=2, quiet=True)
    assert r2["stats"]["videos_from_cache"] == 3
    assert [len(g["files"]) for g in r1["groups"]] == [len(g["files"]) for g in r2["groups"]] == [2]


@needs_av
def test_damaged_and_fake_videos_are_reported_not_crashing(tmp_path):
    make_clip(tmp_path / "good.mp4", 14, seconds=3)
    data = (tmp_path / "good.mp4").read_bytes()
    (tmp_path / "half.mp4").write_bytes(data[: len(data) // 3])
    (tmp_path / "fake.mov").write_bytes(b"this is not a video" * 50)
    (tmp_path / "empty.avi").write_bytes(b"")
    run_scan(str(tmp_path), workers=2, quiet=True)
    res = report.load_json(tmp_path / "_DupeFinder")
    bad = sorted(e["rel"] for e in res["errors"])
    assert {"fake.mov", "empty.avi"} <= set(bad)


def test_videos_are_skipped_with_a_notice_without_video_support(tmp_path, monkeypatch):
    from dupefinder import video

    monkeypatch.setattr(video, "_av", lambda: None)
    synthetic_original(41).save(tmp_path / "a.jpg", quality=90)
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 100)
    run_scan(str(tmp_path), workers=1, quiet=True, use_ai=False)
    res = report.load_json(tmp_path / "_DupeFinder")
    assert res["stats"]["photos_read"] == 1
    assert any("pip install av" in n for n in res["notices"])


@needs_av
def test_review_server_streams_video_with_ranges(tmp_path):
    from dupefinder.server import ReviewApp

    make_clip(tmp_path / "a.mp4", 15, seconds=3, audio_seed=1)
    make_clip(tmp_path / "a.avi", 15, seconds=3, audio_seed=1, vcodec="mpeg4")
    run_scan(str(tmp_path), workers=2, quiet=True)
    out = tmp_path / "_DupeFinder"
    app = ReviewApp(tmp_path, out, report.load_json(out))
    url = app.serve(port=8796, open_browser=False)
    th = threading.Thread(target=app.run_forever, daemon=True)
    th.start()
    try:
        fid = {m["rel"]: m["fid"] for g in app.results["groups"] for m in g["files"]}["a.mp4"]
        size = (tmp_path / "a.mp4").stat().st_size
        req = urllib.request.Request(f"{url}video/{fid}?t={app.token}", headers={"Range": "bytes=10-109"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.headers["Content-Range"] == f"bytes 10-109/{size}"
            assert r.read() == (tmp_path / "a.mp4").read_bytes()[10:110]
        head = urllib.request.Request(f"{url}video/{fid}?t={app.token}", method="HEAD")
        with urllib.request.urlopen(head) as r:  # some players ask for the size first
            assert r.status == 200 and int(r.headers["Content-Length"]) == size and r.read() == b""
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{url}video/{fid}")
        assert e.value.code == 403
        avi = {m["rel"]: m["fid"] for g in app.results["groups"] for m in g["files"]}["a.avi"]
        with urllib.request.urlopen(f"{url}strip/{avi}.jpg?t={app.token}") as r:
            assert Image.open(io.BytesIO(r.read())).width > 300
    finally:
        app.httpd.shutdown()
