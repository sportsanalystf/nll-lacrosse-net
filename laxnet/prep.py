"""Stage 1 - pick the live, wide shots from the nllcv run and dump them as JPEG folders.

SAM 3's video predictor takes a folder of JPEGs named 0.jpg, 1.jpg, ... ; one folder per shot keeps
every tracking session inside a single camera cut, so identities never jump across a cut.
"""
from __future__ import annotations

import shutil
import subprocess

import pandas as pd

from .core import NLL, Run, log, resolve


def select_shots(nll: NLL, cfg: dict, only=None) -> pd.DataFrame:
    pc = cfg["prep"]
    s = nll.shots
    s = s[(s.label == "live") & (s.duration >= pc["min_shot_s"]) & (s.floor + s.crease >= pc["min_floor_frac"])]
    if only:
        s = s[s.shot_id.isin(only)]
    return s.reset_index(drop=True)


def run(run: Run, cfg: dict, only=None, force=False):
    nll = NLL(cfg)
    video = resolve(cfg["video"])
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    info = nll.common.probe(video)
    pc = cfg["prep"]
    shots = select_shots(nll, cfg, only)
    rows = []
    for s in shots.itertuples():
        d = run.p("frames", f"shot_{s.shot_id:04d}", "x").parent
        n_exp = int((s.end - s.start - 0.1) * pc["track_fps"])
        have = len(list(d.glob("*.jpg")))
        if force or have < max(1, n_exp - 2):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s.start + 0.05:.3f}", "-i", str(video),
                            "-t", f"{s.end - s.start - 0.1:.3f}",
                            "-vf", f"fps={pc['track_fps']},scale={pc['frame_width']}:-2", "-q:v", "3",
                            "-start_number", "0", str(d / "%d.jpg")], check=True)
        n = len(list(d.glob("*.jpg")))
        for i in range(n):
            rows.append(dict(shot_id=s.shot_id, frame=i, t=round(s.start + 0.05 + i / pc["track_fps"], 3)))
        log(f"  shot {s.shot_id}: {n} frames ({s.start:.1f}-{s.end:.1f}s)")
    fr = pd.DataFrame(rows)
    fr.to_csv(run.p("frames.csv"), index=False)
    shots.to_csv(run.p("shots_selected.csv"), index=False)
    run.save_json("prep.json", dict(source_width=info["width"], source_height=info["height"],
                                    frame_width=pc["frame_width"], scale=info["width"] / pc["frame_width"],
                                    track_fps=pc["track_fps"], video=str(video)))
    log(f"prep: {len(shots)} shots, {len(fr)} frames at {pc['track_fps']} fps")
    return fr
