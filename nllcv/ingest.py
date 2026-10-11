"""Stage 1 - ingest a broadcast from a URL (NLL YouTube channel etc. via yt-dlp) or a local file.

Produces in the run dir:
    source.<ext>   the video (symlinked when local, downloaded otherwise)
    audio.wav      mono PCM for the audio stage (if the source has an audio track)
    manifest.json  duration / fps / resolution / provenance
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .common import Run, log, need, probe


def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s))


def download(url: str, out_dir: Path, max_height: int = 1080) -> Path:
    """Best video <= max_height merged with best audio into an mp4."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp not installed: pip install yt-dlp")
    need("ffmpeg")
    tmpl = str(out_dir / "source.%(ext)s")
    cmd = ["yt-dlp", "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]",
           "--merge-output-format", "mp4", "-o", tmpl, "--no-playlist",
           "--write-info-json", "--no-progress", url]
    log("downloading: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    hits = sorted(out_dir.glob("source.mp4")) or sorted(p for p in out_dir.glob("source.*")
                                                       if p.suffix not in (".json", ".part"))
    if not hits:
        raise SystemExit("yt-dlp finished but no source file found")
    return hits[0]


def link_local(src: Path, out_dir: Path) -> Path:
    dst = out_dir / ("source" + src.suffix.lower())
    if dst.exists() and dst.resolve() == src.resolve():
        return dst
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:  # relative link so the run folder survives being opened from another mount point
        os.symlink(os.path.relpath(src.resolve(), out_dir.resolve()), dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def extract_audio(video: Path, out: Path, sr: int, external_audio: Path | None = None) -> Path:
    src = external_audio or video
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(out)], check=True)
    return out


def run(run: Run, cfg: dict, source: str, audio: str | None = None):
    if _is_url(source):
        video = download(source, run.root)
        provenance = {"url": source}
    else:
        src = Path(source).expanduser()
        if not src.exists():
            raise SystemExit(f"no such file: {src}")
        video = link_local(src, run.root)
        provenance = {"file": str(src.resolve())}

    info = probe(video)
    log(f"video {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
        f"{info['duration'] / 60:.1f} min, audio={'yes' if info['has_audio'] else 'no'}")
    if (info["width"], info["height"]) != tuple(cfg["scorebug"]["ref_size"]):
        log("note: resolution differs from the scorebug reference; boxes will be rescaled")

    wav = None
    ext_audio = Path(audio).expanduser() if audio else None
    if info["has_audio"] or ext_audio:
        wav = extract_audio(video, run.p("audio.wav"), cfg["audio"]["sample_rate"], ext_audio)
        log(f"audio -> {wav.name}")
    else:
        log("no audio track: goal confirmation will rely on the scoreboard only")

    # paths are stored relative to the run dir so a run folder can be moved / opened elsewhere
    run.update_manifest(video=video.name, audio=wav.name if wav else None, **info, **provenance,
                        config=cfg["_path"])
    return info
