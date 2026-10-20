# Accuracy and speed

All numbers below come from test libraries with a known answer key: every planted duplicate is listed, so both missed copies and wrong matches can be counted. Speed was measured on a 2-core machine.

## Photos

**10,000-photo run:** 8,591 different real photos (Flickr8k and Berkeley BSDS500) plus 1,409 planted duplicates. Every photo compared with all others.

| Planted duplicate type | Found |
|---|---|
| Exact copies (renamed, in other folders) | 399 of 399 pairs |
| WhatsApp-style shrunk copies (no date in the file) | 300 of 300 |
| Converted to PNG, WEBP, HEIC or AVIF | 260 of 260 |
| Burst shots 1 to 2 seconds apart | 450 of 450 |
| Colour or brightness edits | 100 of 100 (shown as similar, never ticked) |
| Same scene retaken hours or days later | 84 of 100 |
| **Wrong matches between different photos** | **0** (4 extra matches turned out to be real duplicates already inside the public photo sets) |
| **Different photos ticked to move** | **0** |

Scan time 16 minutes with AI matching, about 520 MB of memory; a second scan of the same folder takes 7 seconds.

**Hard repeat shots** (more camera movement, half without a date in the file): 94% found with AI matching, 61% without, 0 wrong matches.

**Look-alikes that must not be ticked:** burst frames from the same second, screenshots that differ by one digit, scanned forms, date-stamped edits, multi-page TIFFs, damaged files. 0 ticked.

## Videos

**299-video library:** 106 originals cut from real camera footage and Ken Burns clips of real photos, 181 copies in 14 formats, and look-alikes. Every video compared with all others of the same length (about 10,900 pairs of different videos).

| Copy type | Found |
|---|---|
| Copies of videos with movement | 224 of 226 pairs |
| Copies of still-scene videos with sound | 31 of 31 |
| Copies of still-scene videos without sound | 10 of 24 (exact copies always; re-encodes only when something moves) |
| Formats tested | exact, MOV and MKV remux, WhatsApp-style (848 px, low bitrate), HEVC, AVI (MPEG-4), WEBM (VP9), WMV, 3GP, FLV, MTS, MPG (MPEG-2), 30 to 15 fps, sound removed, low bitrate, rotated phone video |
| **Wrong matches between different videos** | **0** (including different clips from the same fixed camera with identical length) |
| Trimmed copies, colour-graded copies, black "pocket" clips of equal length | 0 grouped |
| **Different videos ticked to move** | **0** |

**Edited versions** (a separate adversarial set: blur, colour grades, a logo, subtitles, two versions with a different title, an end card, one slide swapped in a slideshow, a few seconds of inserted footage, different takes with the same soundtrack): 0 ticked.

**Long videos** (11.7 minutes): the re-encoded copy is found, versions trimmed by 3 seconds at the start or end are not grouped. 9 seconds for the folder.

**Speed:** 299 videos (8 to 20 seconds each, all with a same-length partner) in 2 minutes. 880 videos (5.4 hours, 2.5 GB, 2 to 90 seconds each, nearly all with a same-length partner, so close to the slowest case) in 13 minutes; a second scan takes 7 seconds.

**Known limits:**
- Re-encoded copies of a still scene without sound are only found when something moves in the picture (by design: different recordings of an empty room look identical).
- Two cuts of the same footage with the same length that start less than half a second apart are grouped, because copies made by different apps can be up to 0.2 seconds out of step.
- Two edited versions of one video (for example two colour grades) can be grouped with each other when the original is not in the folder.

## Platforms

The full test suite runs on every push on Windows and Linux with Python 3.10, 3.12 and 3.13 (see the badge in the README). Before release, the official Windows builds of Python 3.12 and every library (run under Wine) were also checked with a fresh setup from the start file, paths over 260 characters, and the same 10,000-photo and 299-video runs as on Linux: the same groups and the same files ticked. macOS is covered by CI only and has not been tried on a real Mac yet.

## Measure it on your own files

```bash
python -m pip install -r requirements-dev.txt
python tests/make_dataset.py test_lib                            # generated pictures
python tests/make_dataset.py test_lib path/to/your/jpgs --n 40   # or copies made from your own JPG photos
python -m dupefinder scan test_lib/photos --no-review
python tests/evaluate.py test_lib --verbose
```

`--n` is how many original photos to use (default 40, at least 28). The result shows precision (the share of reported matches that are right), recall (the share of planted duplicates found) and whether every file got the right label.
