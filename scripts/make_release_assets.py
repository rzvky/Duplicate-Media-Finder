"""Split the AI model into the 4 files that are attached to each GitHub release.

Usage (from the repository folder, with the model in dupefinder/models/):

    python scripts/make_release_assets.py [output folder]

The tool looks for these exact file names, so keep them unchanged when uploading.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dupefinder.ai import MODEL_FILE, MODEL_SHA256, MODEL_SIZE, PART_NAME, PARTS  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
    if not MODEL_FILE.is_file():
        print(f"Model not found: {MODEL_FILE}")
        return 1
    data = MODEL_FILE.read_bytes()
    if len(data) != MODEL_SIZE or hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        print("The model file does not match MODEL_SIZE / MODEL_SHA256 in dupefinder/ai.py. Update them first.")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    size = -(-len(data) // PARTS)
    for i in range(PARTS):
        part = out / PART_NAME.format(i + 1)
        part.write_bytes(data[i * size : (i + 1) * size])
        print(f"{part}  {part.stat().st_size:,} bytes")
    print(f"\nUpload these {PARTS} files as assets of the release (gh release create vX.Y.Z {out}/*.bin).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
