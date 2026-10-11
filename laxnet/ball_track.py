"""Ball stage, part 2 - one ball path per shot from the per-frame candidates.

1. Camera-stabilised coordinates: candidate positions minus the accumulated camera pan
   (ball_frames.csv), so scenery stays put and the ball's own motion is what moves.
2. Tracklets: a constant-velocity predictor per tracklet, Hungarian matching of candidates inside a
   gate around each prediction (the gate widens while a tracklet coasts through missed frames).
3. Pruning: tracklets that are too short, or never move (a floor dot, a logo, a stick at rest), go.
4. One ball: a weighted interval schedule picks the set of non-overlapping tracklets with the most
   evidence, so at any moment there is at most one ball.
5. Short gaps between and inside chosen tracklets are interpolated; longer gaps stay empty (ball in a
   pocket or hidden), and the play logic handles those.

Writes ball_track.csv, one row per shot frame at ball.fps:
    shot_id, bframe, t, state (det | interp | none), x, y (source px), sx, sy (stabilised px),
    vx, vy, speed_px_s (camera-compensated), score, tracklet, plane_x, plane_y,
    cam_dx, cam_dy, crease_x, crease_y
plane_x/y is the ball's pixel projected onto the floor plane: exact on the floor, biased when airborne
(the play logic uses player positions for anything that needs floor coordinates).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .core import NLL, HomographyTrack, Run, log, project


def tracklets(c: pd.DataFrame, n_frames: int, tc: dict, fps: float, min_speed=0.0, max_per_frame=30):
    by = {f: g.nlargest(max_per_frame, "score")[["sx", "sy", "score", "x", "y"]].values for f, g in c.groupby("bframe")}
    active, done = [], []
    for f in range(n_frames):
        d = by.get(f, np.zeros((0, 5)))
        if active and len(d):
            pred = np.array([a["p"] + a["v"] * (f - a["f"]) for a in active])
            gate = np.array([tc["gate_px"] * (1 + 0.5 * (f - a["f"] - 1)) for a in active])
            cost = np.hypot(pred[:, None, 0] - d[None, :, 0], pred[:, None, 1] - d[None, :, 1])
            big = 1e6
            cm = np.where(cost <= gate[:, None], cost, big)
            r, k = linear_sum_assignment(cm)
            used = set()
            for i, j in zip(r, k):
                if cm[i, j] >= big:
                    continue
                a = active[i]
                gap = f - a["f"]
                newv = (d[j, :2] - a["p"]) / gap
                a["v"] = newv if len(a["pts"]) == 1 else 0.5 * a["v"] + 0.5 * newv
                a["p"], a["f"] = d[j, :2], f
                a["pts"].append((f, *d[j]))
                used.add(j)
            rest = [j for j in range(len(d)) if j not in used]
        else:
            rest = list(range(len(d)))
        still = []
        for a in active:
            (still if f - a["f"] <= tc["max_miss"] else done).append(a)
        active = still
        for j in rest:
            active.append(dict(p=d[j, :2].copy(), v=np.zeros(2), f=f, pts=[(f, *d[j])]))
    done += active
    out = []
    for a in done:
        p = np.array(a["pts"])           # f, sx, sy, score, x, y
        if len(p) < tc["min_len"]:
            continue
        disp = np.hypot(p[:, 1] - p[0, 1], p[:, 2] - p[0, 2]).max()
        if disp < tc["min_motion_px"]:
            continue
        f = p[:, 0]
        span = max(f[-1] - f[0], 1)
        speed = disp / span * fps
        if speed < min_speed:
            continue
        # a ball between sticks flies on a smooth path: quadratic fit residual in stabilised pixels
        straight = 1.0
        if len(p) >= 5:
            res = []
            for col in (1, 2):
                c = np.polyfit(f - f[0], p[:, col], 2)
                res.append(p[:, col] - np.polyval(c, f - f[0]))
            rms = float(np.sqrt(np.mean(np.square(res))))
            straight = 1.0 if rms <= 4 else 0.3 if rms > 12 else 0.6
        w = float(p[:, 3].sum() * min(1.0, disp / 80) * straight * (1 + min(speed / 400, 2)))
        out.append(dict(f0=int(f[0]), f1=int(f[-1]), pts=p, w=w, speed=speed))
    return out


def choose(tl):
    """Weighted interval scheduling: best set of tracklets that never overlap in time."""
    if not tl:
        return []
    tl = sorted(tl, key=lambda a: a["f1"])
    ends = [a["f1"] for a in tl]
    best = [0.0] * (len(tl) + 1)
    take = [False] * len(tl)
    prev = []
    for i, a in enumerate(tl):
        j = int(np.searchsorted(ends, a["f0"], side="left"))   # tracklets ending before a starts: [0, j)
        prev.append(j)
        incl = a["w"] + best[j]
        take[i] = incl > best[i]
        best[i + 1] = max(best[i], incl)
    out, i = [], len(tl) - 1
    while i >= 0:
        if take[i]:
            out.append(tl[i])
            i = prev[i] - 1
        else:
            i -= 1
    return out[::-1]


def run(run: Run, cfg: dict):
    nll = NLL(cfg)
    bc, tc = cfg["ball"], cfg["ball"]["track"]
    fps = bc["fps"]
    cand, frames = run.csv("ball_candidates.csv"), run.csv("ball_frames.csv")
    src = cand.source.iloc[0] if len(cand) else "motion"
    meta = json.loads(run.p("ball_meta.json").read_text()) if run.exists("ball_meta.json") else {}
    tc = dict(tc, min_len=max(2, int(np.ceil(tc["min_len"] / meta.get("stride", 1)))))
    min_speed = tc.get("min_speed_px_s", {}).get(src, 0) if isinstance(tc.get("min_speed_px_s"), dict) else 0
    ht = HomographyTrack(*nll.homographies(), cfg["homography"]["max_interp_gap_s"])
    out = []
    n_tl = n_det = 0
    for sid, fr in frames.groupby("shot_id"):
        fr = fr.sort_values("bframe").reset_index(drop=True)
        n = int(fr.bframe.max()) + 1
        C = np.zeros((n, 2))
        C[fr.bframe.values] = fr[["cam_dx", "cam_dy"]].fillna(0).values
        C = np.cumsum(C, axis=0)                          # where frame-0 scenery sits in frame f
        c = cand[cand.shot_id == sid].copy()
        c["sx"] = c.x - C[c.bframe.values, 0]
        c["sy"] = c.y - C[c.bframe.values, 1]
        tl = choose(tracklets(c, n, tc, fps, min_speed))
        n_tl += len(tl)
        S = np.full((n, 2), np.nan)
        score = np.full(n, np.nan)
        state = np.array(["none"] * n, dtype=object)
        tid = np.full(n, -1)
        for k, a in enumerate(tl):
            p = a["pts"]
            f = p[:, 0].astype(int)
            S[f] = p[:, 1:3]
            score[f] = p[:, 3]
            state[f] = "det"
            tid[a["f0"]:a["f1"] + 1] = k
        n_det += int((state == "det").sum())
        # interpolate gaps (inside tracklets, and between tracklets if short)
        have = np.where(np.isfinite(S[:, 0]))[0]
        maxgap = int(round(tc["fill_gap_s"] * fps))
        for a, b in zip(have[:-1], have[1:]):
            same = tid[a] == tid[b] and tid[a] >= 0
            near = np.hypot(*(S[b] - S[a])) <= tc["gate_px"] * (b - a)
            if 1 < b - a <= maxgap + 1 and (same or near):
                w = (np.arange(a + 1, b) - a) / (b - a)
                S[a + 1:b] = (1 - w)[:, None] * S[a] + w[:, None] * S[b]
                state[a + 1:b] = "interp"
        XY = S + C
        # camera-compensated velocity, central differences over +-2 frames, only across valid samples
        V = np.full((n, 2), np.nan)
        for i in range(n):
            a, b = max(0, i - 2), min(n - 1, i + 2)
            if b > a and np.isfinite(S[a, 0]) and np.isfinite(S[b, 0]) and np.isfinite(S[i, 0]):
                V[i] = (S[b] - S[a]) * fps / (b - a)
        t = fr.set_index("bframe").reindex(range(n))
        # crease centre is measured only on detection frames when the detector runs on a stride
        t[["crease_x", "crease_y"]] = t[["crease_x", "crease_y"]].interpolate(limit=6, limit_area="inside").ffill(limit=6).bfill(limit=6)
        tt = t.t.values
        px, py = np.full(n, np.nan), np.full(n, np.nan)
        for i in np.where(np.isfinite(XY[:, 0]))[0]:
            H = ht(tt[i])
            if H is not None:
                px[i], py[i] = project(H, XY[i:i + 1])[0]
        out.append(pd.DataFrame(dict(shot_id=sid, bframe=np.arange(n), t=tt, state=state,
                                     x=XY[:, 0].round(1), y=XY[:, 1].round(1), sx=S[:, 0].round(1), sy=S[:, 1].round(1),
                                     vx=V[:, 0].round(1), vy=V[:, 1].round(1),
                                     speed_px_s=np.hypot(V[:, 0], V[:, 1]).round(1), score=score.round(3), tracklet=tid,
                                     plane_x=px.round(2), plane_y=py.round(2),
                                     cam_dx=t.cam_dx.values, cam_dy=t.cam_dy.values,
                                     crease_x=t.crease_x.values, crease_y=t.crease_y.values)))
    bt = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    bt.to_csv(run.p("ball_track.csv"), index=False)
    if len(bt):
        vis = (bt.state != "none").mean()
        log(f"ball track: {n_tl} tracklets kept, ball located in {vis:.0%} of frames "
            f"({n_det} detections, {(bt.state == 'interp').sum()} interpolated)")
    return bt
