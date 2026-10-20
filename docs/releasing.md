# Publishing a release

The AI model (96 MB) is too large for git, so every release carries it as 4 files of about 24 MB. The tool looks for these exact names in Downloads, the Desktop and its own folder.

1. Update `__version__` in `dupefinder/__init__.py` and add the changes to `CHANGELOG.md`.
2. With the model in `dupefinder/models/`, make the 4 parts:
   ```bash
   python scripts/make_release_assets.py dist
   ```
3. Tag and publish with the GitHub CLI (or on github.com: **Releases**, **Draft a new release**, attach the 4 files):
   ```bash
   git tag v1.1.0 && git push origin v1.1.0
   gh release create v1.1.0 dist/DuplicateMediaFinder_AI_part*.bin -t "Duplicate Media Finder 1.1.0" -F CHANGELOG.md
   ```

Attach the 4 parts to every release: the README links to the latest release.

If the model ever changes, update `MODEL_SHA256` and `MODEL_SIZE` in `dupefinder/ai.py`.

| Model | |
|---|---|
| File | `dupefinder/models/clip_vitb32_visual_w8.onnx` |
| Size | 96,359,160 bytes |
| SHA-256 | `c2580cf93f5fe7425cdde1c42b1d3c4221cb7449decf0a0b619399423af9a257` |
| Source | OpenCLIP ViT-B/32, LAION-400M weights (MIT), image part only, ONNX opset 17, 8-bit weights |
