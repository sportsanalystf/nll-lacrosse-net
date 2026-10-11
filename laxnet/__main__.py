"""python -m laxnet <stage> --run runs/<game> [options]

stages: prep, track, teams, numbers, floor, ball, plays, advanced, analytics, render, run (all in order)
ball labelling + training: ball-sample, ball-prelabel, ball-review, ball-export, ball-train

  --shots 7 10 14        restrict prep/track/render to these nllcv shot ids
  --backend sam3|cpu     tracker (default from config: sam3)
  --embedder siglip|hsv  team features (default from config)
  --dry-run              numbers: write the planned Claude requests and cost, call nothing
  --budget SECONDS       cpu tracker / ball detector: stop after this long and resume on the next run
  --detector rfdetr|sam3|motion   ball detector (default from config; falls back to motion if unavailable)
  --n N                  ball-sample: frames to sample
  --dataset DIR          ball-train: COCO folder from ball-export (default datasets/ball_<run name>)
  --epochs N             ball-train: override the epochs
"""
from __future__ import annotations

import argparse
import time

from .core import Run, load_config, log

STAGES = ["prep", "track", "teams", "numbers", "floor", "ball", "plays", "advanced", "analytics", "render"]
TOOLS = ["ball-sample", "ball-prelabel", "ball-review", "ball-export", "ball-train"]


def stage(name, run, cfg, a):
    t0 = time.time()
    if name == "prep":
        from . import prep
        r = prep.run(run, cfg, a.shots, a.force)
    elif name == "track":
        from . import track
        r = track.run(run, cfg, a.backend, a.shots, a.budget)
    elif name == "teams":
        from . import teams
        r = teams.run(run, cfg, a.embedder)
    elif name == "numbers":
        from . import numbers
        r = numbers.run(run, cfg, a.dry_run, a.model)
        r = True  # optional stage: never blocks the rest
    elif name == "floor":
        from . import floor
        r = floor.run(run, cfg)
    elif name == "ball":
        from . import ball_detect, ball_track
        r = ball_detect.run(run, cfg, a.shots, a.detector, a.force, a.budget)
        r = ball_track.run(run, cfg) if r is not None else None
    elif name == "plays":
        from . import plays
        r = plays.run(run, cfg) or True
    elif name == "advanced":
        from . import advanced
        r = advanced.run(run, cfg) or True
    elif name == "ball-sample":
        from . import ball_label
        r = ball_label.sample(run, cfg, a.n, budget=a.budget)
    elif name == "ball-prelabel":
        from . import ball_label
        r = ball_label.prelabel_sam3(run, cfg)
    elif name == "ball-review":
        from . import ball_label
        r = ball_label.review(run, cfg)
    elif name == "ball-export":
        from . import ball_label
        r = ball_label.export(run, cfg, a.dataset)
    elif name == "ball-train":
        from . import ball_detect
        from .core import resolve
        if not a.dataset and run is None:
            raise SystemExit("ball-train: give --dataset (or --run to use datasets/ball_<run name>)")
        ds = a.dataset or str(resolve("datasets") / f"ball_{run.root.name}")
        r = ball_detect.train(cfg, ds, a.epochs)
    elif name == "analytics":
        from . import analytics
        r = analytics.run(run, cfg) or True
    elif name == "render":
        from . import render
        r = render.run(run, cfg, a.shots) or True
    log(f"{name} done in {time.time() - t0:.0f}s")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="laxnet")
    ap.add_argument("stage", choices=STAGES + TOOLS + ["run"])
    ap.add_argument("--run", default=None, help="run folder, e.g. runs/<game> (not needed for ball-train)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--shots", type=int, nargs="*", default=None)
    ap.add_argument("--backend", choices=["sam3", "cpu"], default=None)
    ap.add_argument("--embedder", choices=["siglip", "hsv"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=None, help="numbers: override the Claude model id")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-stage", default=None)
    ap.add_argument("--detector", choices=["rfdetr", "sam3", "motion"], default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.run is None and a.stage != "ball-train":
        ap.error("--run is required")
    run = Run(a.run) if a.run else None
    if a.stage != "run":
        stage(a.stage, run, cfg, a)
        return
    todo = STAGES[STAGES.index(a.from_stage):] if a.from_stage else STAGES
    for s in todo:
        if stage(s, run, cfg, a) is None:
            log(f"stopped after '{s}'; rerun with --from-stage {s} to continue")
            return


if __name__ == "__main__":
    main()
