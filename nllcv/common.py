"""Shared helpers: config, run directory, video I/O through ffmpeg."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- config
def load_config(path: str | Path | None) -> dict:
    path = Path(path) if path else PKG_ROOT / "configs" / "nll_2026_broadcast.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PKG_ROOT / p


# --------------------------------------------------------------------------- run dir
@dataclass
class Run:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def p(self, *parts) -> Path:
        out = self.root.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    @property
    def manifest(self) -> dict:
        mp = self.root / "manifest.json"
        return json.loads(mp.read_text()) if mp.exists() else {}

    def update_manifest(self, **kw):
        m = self.manifest
        m.update(kw)
        (self.root / "manifest.json").write_text(json.dumps(m, indent=2, default=str))

    def _abs(self, p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def video(self) -> Path:
        return self._abs(self.manifest["video"])

    @property
    def audio(self) -> Path | None:
        a = self.manifest.get("audio")
        return self._abs(a) if a else None


# --------------------------------------------------------------------------- ffmpeg
def need(binary: str):
    if shutil.which(binary) is None:
        raise SystemExit(f"'{binary}' not found on PATH (macOS: brew install {binary})")


def probe(path: str | Path) -> dict:
    need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {
        "duration": float(info["format"]["duration"]),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": num / den,
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def iter_frames(path, fps: float, size: tuple[int, int] | None = None, start: float = 0.0,
                end: float | None = None, crop: tuple[int, int, int, int] | None = None,
                ) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (broadcast_time_s, BGR frame) sampled at `fps`.

    crop=(x, y, w, h) is applied at native resolution before any resize, so a full-res
    scorebug strip can be pulled without decoding-and-copying whole 1080p frames into Python.
    """
    need("ffmpeg")
    info = probe(path)
    W, H = info["width"], info["height"]
    filters = []
    if crop:
        x, y, w, h = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
        W, H = w, h
    if size:
        filters.append(f"scale={size[0]}:{size[1]}:flags=area")
        W, H = size
    filters.append(f"fps={fps}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(end - start, 0.01):.3f}"]
    cmd += ["-vf", ",".join(filters), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    nbytes = W * H * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield start + i / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_frame(path, t: float) -> np.ndarray:
    for _, f in iter_frames(path, fps=30, start=t, end=t + 0.05):
        return f.copy()
    raise ValueError(f"no frame at t={t}")


def scale_box(box, ref_size, size):
    sx, sy = size[0] / ref_size[0], size[1] / ref_size[1]
    x, y, w, h = box
    return int(round(x * sx)), int(round(y * sy)), max(1, int(round(w * sx))), max(1, int(round(h * sy)))


def hsv_mask(bgr: np.ndarray, low, high) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))


def fmt_clock(sec) -> str:
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return ""
    sec = float(sec)
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def log(msg: str):
    print(f"[nllcv] {msg}", flush=True)
