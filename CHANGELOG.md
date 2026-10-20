# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org/).

## [1.1.0] - 2026-10-20

### Added
- Duplicate videos: exact copies, and the same recording saved again under another name, in another
  format (MP4, MOV, MKV, AVI, WMV, WEBM, 3GP, FLV, MPG, TS and more), smaller, re-encoded or without sound.
- Edited versions (blur, colour change, logo, subtitles, title text, end card, inserted clip) and trimmed
  versions are recognised and kept apart from real copies.
- Review screen: Videos tab, video playback, a six-picture strip for formats the browser cannot play.
- `--media all|photos|videos` option for `scan`.
- Live Photo and motion photo clips (up to 6 seconds) stay with their photo; longer videos with the same
  name are treated as separate videos.
- `pyproject.toml`: install with `pip install .` and run the `dupefinder` command.
- GitHub Actions: tests on Windows, Linux and macOS.

### Changed
- Project renamed from Photo Duplicate Finder to **Duplicate Media Finder**.
- The AI model release files are now called `DuplicateMediaFinder_AI_part1of4.bin` to `part4of4.bin`.

## 1.0.0 - 2026-10-12

### Added
- First version: exact copies, resized or converted copies and similar shots of photos.
- JPG, PNG, HEIC/HEIF, AVIF, WEBP, TIFF, GIF, BMP, JPEG 2000 and camera RAW files.
- Local AI matching of repeat shots (OpenCLIP ViT-B/32, runs offline).
- Review screen in the browser, Review Bin with undo, CSV and HTML reports.
- Start files for Windows, macOS and Linux that set everything up on the first run.

[1.1.0]: https://github.com/rzvky/Duplicate-Media-Finder/releases/tag/v1.1.0
