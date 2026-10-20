"""AI image matching with a local CLIP model (runs offline on your computer, no upload)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = MODEL_DIR / "clip_vitb32_visual_w8.onnx"
MODEL_LABEL = "CLIP ViT-B/32 (OpenCLIP, LAION-400M), 8-bit, local"

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


MODEL_SHA256 = "c2580cf93f5fe7425cdde1c42b1d3c4221cb7449decf0a0b619399423af9a257"
MODEL_SIZE = 96359160
PARTS = 4
PART_NAME = "DuplicateMediaFinder_AI_part{}of4.bin"  # how the model is shipped (each part < 30 MB)


class AIUnavailable(Exception):
    pass


def _part_dirs():
    tool_root = MODEL_DIR.parent.parent
    home = Path.home()
    dirs = [MODEL_DIR, tool_root, tool_root.parent, home / "Downloads", home / "Desktop", Path.cwd()]
    seen, out = set(), []
    for d in dirs:
        try:
            key = str(d.resolve())
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def ensure_model(search_dirs=None) -> Tuple[bool, str]:
    """Make sure the AI model file exists. If it was shipped in parts, find the parts (tool folder,
    Downloads, Desktop), join them, check the checksum, and remove the parts. Never raises."""
    import hashlib

    if MODEL_FILE.is_file() and MODEL_FILE.stat().st_size == MODEL_SIZE:
        return True, "ready"
    dirs = list(search_dirs) if search_dirs is not None else _part_dirs()
    found = []
    for i in range(1, PARTS + 1):
        name = PART_NAME.format(i)
        hit = next((d / name for d in dirs if (d / name).is_file()), None)
        if hit is None:
            from . import RELEASES

            return False, (
                "AI model not found. Download the 4 files "
                + PART_NAME.format(1)
                + " to "
                + PART_NAME.format(PARTS)
                + f" from {RELEASES} and leave "
                "them in Downloads or put them in the tool folder"
            )
        found.append(hit)
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_FILE.with_name(MODEL_FILE.name + ".joining")
        h = hashlib.sha256()
        with open(tmp, "wb") as out:
            for part in found:
                with open(part, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                        out.write(chunk)
        if h.hexdigest() != MODEL_SHA256:
            os.remove(tmp)
            return False, "AI model parts are damaged or incomplete. Download the 4 parts again"
        os.replace(tmp, MODEL_FILE)
    except OSError as exc:
        return False, f"Could not set up the AI model: {exc}"
    for part in found:
        try:
            part.unlink()
        except OSError:
            pass
    return True, "AI model set up from the downloaded parts"


def model_id() -> str:
    try:
        return f"{MODEL_FILE.name}:{MODEL_FILE.stat().st_size}"
    except OSError:
        return MODEL_FILE.name


class ClipEmbedder:
    batch_size = 16

    def __init__(self, model_path: Optional[Path] = None, threads: Optional[int] = None):
        path = Path(model_path or MODEL_FILE)
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise AIUnavailable("AI engine not installed (package 'onnxruntime')") from exc
        if model_path is None and not path.is_file():
            ok, msg = ensure_model()
            if not ok:
                raise AIUnavailable(msg)
        if not path.is_file():
            raise AIUnavailable(f"AI model file missing: {path.name}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = int(threads or max(1, os.cpu_count() or 1))
        so.log_severity_level = 3
        try:
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise AIUnavailable(
                "AI model could not start (update with: pip install -U onnxruntime). Details: " + str(exc)[:200]
            ) from exc
        self.input_name = self.sess.get_inputs()[0].name
        outs = self.sess.get_outputs()
        names = [o.name for o in outs]
        self.output_name = "image_embeds" if "image_embeds" in names else names[0]
        self.id = model_id()
        test = self.embed(np.zeros((1, 224, 224, 3), dtype=np.uint8))
        self.dim = int(test.shape[1])

    def embed(self, batch_uint8: np.ndarray) -> np.ndarray:
        """batch_uint8: N x 224 x 224 x 3 RGB. Returns N x D unit-length float32 vectors."""
        x = batch_uint8.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
        out = []
        for i in range(0, len(x), self.batch_size):
            out.append(self.sess.run([self.output_name], {self.input_name: x[i : i + self.batch_size]})[0])
        e = np.concatenate(out, 0).astype(np.float32)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return e / n


def ai_status() -> Tuple[bool, str]:
    try:
        ClipEmbedder(threads=1)
        return True, MODEL_LABEL
    except AIUnavailable as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover
        return False, f"AI could not start: {exc}"
