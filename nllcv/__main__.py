"""Command line entry point.

    python -m nllcv run        --source <url-or-file> --run runs/<game>   # every stage
    python -m nllcv ingest     --source <url-or-file> --run runs/<game>
    python -m nllcv shots      --run runs/<game>
    python -m nllcv scoreboard --run runs/<game>
    python -m nllcv detect     --run runs/<game>
    python -m nllcv calibrate  --run runs/<game> --at 41.0              # click floor landmarks
    python -m nllcv homography --run runs/<game>
    python -m nllcv events     --run runs/<game>
    python -m nllcv outputs    --run runs/<game>
"""
from __future__ import annotations

import argparse
import time

from .common import Run, load_config, log

STAGES = ["shots", "scoreboard", "detect", "homography", "events", "outputs"]


def _stage(name, run, cfg, args):
    t0 = time.time()
    res = True
    if name == "ingest":
        from . import ingest
        ingest.run(run, cfg, args.source, getattr(args, "audio", None))
    elif name == "shots":
        from . import shots
        shots.run(run, cfg)
    elif name == "scoreboard":
        from . import scoreboard
        scoreboard.run(run, cfg)
    elif name == "detect":
        from . import detect
        res = detect.run(run, cfg)
    elif name == "homography":
        from . import homography
        res = homography.run(run, cfg)
    elif name == "events":
        from . import events
        events.run(run, cfg)
    elif name == "outputs":
        from . import outputs
        outputs.run(run, cfg)
    log(f"{name} done in {time.time() - t0:.0f}s")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nllcv")
    ap.add_argument("stage", choices=["run", "ingest", "calibrate"] + STAGES)
    ap.add_argument("--run", required=True, help="run directory, e.g. runs/hfx_at_ga_2026-05-02")
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", help="URL or local video path (ingest/run)")
    ap.add_argument("--audio", help="optional separate audio file (ingest/run)")
    ap.add_argument("--at", type=float, nargs="*", help="calibrate: broadcast times (s) of reference frames")
    ap.add_argument("--from-stage", default=None, help="run: start from this stage")
    ap.add_argument("--auto", action="store_true",
                    help="calibrate: fit the PTZ camera model automatically instead of clicking landmarks")
    ap.add_argument("--points", default=None,
                    help="calibrate: JSON of landmark -> [u, v] pixels for the single --at frame")
    ap.add_argument("--force", action="store_true", help="recompute cached intermediates")
    ap.add_argument("--budget", type=float, default=None,
                    help="detect/homography: stop after this many seconds; rerun to resume")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["_force"] = args.force
    cfg["_budget"] = args.budget
    run = Run(args.run)

    if args.stage == "calibrate":
        if args.auto or args.points:
            import json
            from . import calib, detect, homography
            from .common import read_frame, resolve
            dc = cfg["detect"]
            model = detect.Yolox(resolve(dc["model"]), dc["input"], dc["conf"], dc["nms"])
            boxes = {}
            for t in args.at:
                b, _, c = model(read_frame(run.video, t), dc["tiles"])
                boxes[t] = b[c == detect.PERSON]
            if args.points:  # scripted landmarks for one frame: '{"left_goal": [u, v], "@-57.5,27.5": [u, v]}'
                homography.add_reference(run, cfg, args.at[0], json.loads(args.points), boxes=boxes[args.at[0]])
            elif len(homography.References(cfg)):
                calib.calibrate_from_anchor(run, cfg, args.at, boxes)  # camera fixed by the existing reference
            else:
                calib.calibrate_auto(run, cfg, args.at, boxes)
        else:
            from . import homography
            homography.calibrate_interactive(run, cfg, args.at or [])
        return
    if args.stage == "run":
        todo = ["ingest"] + STAGES if args.source else STAGES
        if args.from_stage:
            todo = todo[todo.index(args.from_stage):]
        for s in todo:
            if _stage(s, run, cfg, args) is None:
                log(f"stopped after '{s}' (time budget); rerun with --from-stage {s} to resume")
                return
        return
    _stage(args.stage, run, cfg, args)


if __name__ == "__main__":
    main()
