# How the duplicate photo and video detection works

This page explains how Duplicate Media Finder decides that two photos or two videos are duplicates, and why the rules are the way they are. The guiding rule: **a different photo or video must never be ticked as a copy.** Missing a duplicate is acceptable; moving a unique file is not.

## Photos

1. **Find the files.** Walk the folder and its sub-folders. Hidden folders, phone thumbnail caches (`.thumbnails`), the Recycle Bin, the tool's own `_DupeFinder` folder and the Apple Photos library package are skipped. Hard-linked files are counted once.
2. **Read each photo once** (in parallel, results cached in `cache.sqlite`):
   - SHA-256 of the file, for exact copies
   - perceptual hashes (64-bit pHash and dHash) of the picture as you see it, after EXIF rotation
   - a 4x4 colour signature, colourfulness and a 64x64 grey detail grid
   - sharpness, size, format and the capture time from EXIF (sub-second aware)
   - optionally a CLIP image embedding (OpenCLIP ViT-B/32, 8-bit ONNX, runs locally)
3. **Compare all pairs.** Every photo is compared with every other photo using NumPy matrix operations on the hashes (10,000 photos are about 50 million pairs; the search runs in blocks so memory stays low).
4. **Confirm copies.** A pair is only called "same photo, different quality" when it passes every check:
   - picture pattern (pHash and dHash) within a few bits, same shape
   - colours within limit, and colourfulness within 15% (real copies change by 8% at most; saturation edits by 40% or more)
   - same capture moment when both files carry a date (burst shots are different photos)
   - no local difference anywhere on the 64x64 grid (the grid adapts when one file is much smaller)
   - same-size files: a full-resolution comparison in 8x8 pixel blocks, which tells a one-digit change in a screenshot or a date stamp apart from compression noise
5. **Similar shots** (never ticked for you) need a high AI similarity, or a close picture pattern, and are allowed more easily when the photos were taken within a few minutes of each other.
6. **Group** with complete linkage: every pair inside a group must match, so "A looks like B" and "B looks like C" never pulls an unrelated C into A's group. Damaged, blank and multi-page files only group as exact copies.
7. **Pick the keeper:** not damaged, then camera details, resolution, RAW, Live Photo companion, a clean name (no "copy", "(1)", "edited"), the older file, the camera's own format, and size.

### Sensitivity levels (photos)

| Level | "Taken close together" window | AI similarity needed (close in time / any time) |
|---|---|---|
| `strict` | 2 minutes | 0.90 / 0.97 |
| `normal` | 5 minutes | 0.86 / 0.95 |
| `loose` | 30 minutes | 0.82 / 0.93 |

Exact copies are found the same way at every level. The copy checks are a little tighter on `strict` and a little looser on `loose`.

## Videos

Videos need the `av` package (PyAV, which bundles FFmpeg). The start files install it.

1. **Read the details** of every video from its header, without decoding: length, picture size after rotation and pixel aspect ratio, codec, frame rate, sound, bitrate and the recording date. A short clip (up to 6 seconds, 25 MB) with the same name as a photo next to it is that photo's Live Photo or motion clip and stays with the photo.
2. **Exact copies:** files of exactly the same size and the same first and last megabyte get a full SHA-256. A file that cannot be read to the end is never reported as an exact copy.
3. **Same length only:** only videos whose lengths match (within 0.35 seconds, up to 1 second for videos over 6 minutes) are compared further. A big folder of unique videos is therefore checked quickly, and a trimmed version is never a copy.
4. **Sample pictures and sound.** Pictures are taken at the same moments in both videos on a fixed time grid (every 0.125, 0.25, 0.5, 1, 2, 4... seconds, at most 16 moments), together with the pictures 0.04, 0.1 and 0.2 seconds around each moment. Different formats label frames slightly differently (an AVI or MPG copy can be a few frames out of step), so a moment counts as matching when any nearby picture matches. The loudness of the second after each moment is stored as a short curve.
5. **Decide.** Two videos are the same video when:
   - the shape matches and the pictures match at every moment (pattern, colours and a region-by-region check; one damaged or mistimed moment in ten is allowed)
   - no moment shows completely different pictures (an inserted clip, a title or an end card)
   - if both have real sound, the loudness curves rise and fall together
   - if there is no sound, the matching moments include movement in both videos. Two different recordings of an empty, fixed-camera scene look identical at every moment, so pictures alone are not proof.
6. **Keep edited versions out:** a difference that stays in the same place at every moment (a logo, subtitles, a different title; checked on a 16x16 and a finer 32x32 grid), a brightness, colour or contrast change, or a blurred version means it is a different video.
7. **Double check.** Videos over 16 seconds that match are checked again at the moments in between (up to 64 moments in total). Two moments in a row of other footage means something was inserted there.
8. **Pick the keeper:** not damaged, complete (not a few frames shorter), camera details, resolution, sound, sharpness, less compression (only compared within the same codec, because older codecs such as WMV or MPEG-2 need two to three times the data of H.264 for the same picture), then a clean name, the older file and the original container.

### Why these thresholds

The video thresholds were calibrated on 299 videos cut from real camera footage and real photos: 106 originals, 181 copies in 14 formats, and look-alikes (different clips of the same fixed camera with identical length, trimmed and colour-graded copies, black clips). Real copies differ by at most 6 bits of picture pattern at 99.5% of moments; their sound curves correlate at 0.91 or more; a difference that stays in one place is at most 6.8 for copies and 12 or more for edits. The comments in `dupefinder/video.py` list the measured values next to each setting.

## The review screen

`dupefinder/server.py` serves `web/review.html` on `127.0.0.1` with a random access key in the address. It only accepts requests for the local host name, streams videos with range requests (so seeking works), and refuses to move every file of a group. Moves are written to `move_log.jsonl` one line at a time, so Undo works even after a crash.

## Project layout

```
dupefinder/
    cli.py          commands, guided mode, progress bar
    scanner.py      finds photos and videos, parallel reading, cache
    loader.py       opens every photo format (HEIC, RAW, AVIF...), EXIF
    features.py     photo fingerprints, hashes, colour and detail signatures
    ai.py           CLIP model setup and image embeddings
    grouping.py     all-against-all photo comparison, groups, keeper choice
    verify.py       full-resolution double check for same-size photos
    video.py        video details, sampled pictures and sound, matching, keeper choice
    report.py       results.json, duplicates.csv, report.html, previews
    server.py       local review screen
    actions.py      Review Bin, undo, move log, linked files
    cache.py        SQLite cache
    formats.py      supported file types and skipped folders
    paths.py        long path support on Windows
    web/review.html the review screen
tests/
    test_dupefinder.py  automated tests (pytest)
    make_dataset.py     builds a photo library with an answer key
    evaluate.py         scores a scan against the answer key
    video_helpers.py    makes small test videos
scripts/
    make_release_assets.py  splits the AI model for a release
```
