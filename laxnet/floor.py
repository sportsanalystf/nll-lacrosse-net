"""Stage 5 - floor coordinates and smooth trajectories.

Each track's foot point is mapped to floor feet with the nllcv homography of that moment (nearest
verified frame, or interpolated between two close ones). A constant-velocity Kalman filter then runs
over each track in floor space:

    state  [x, y, vx, vy]   (feet, feet/s)
    gating measurements with a Mahalanobis distance above `gate_chi2` are treated as outliers
           (a mis-registered frame or a foot point on a stick-check) and skipped
    gaps   frames with no measurement (occlusion, no homography) are predicted; a Rauch-Tung-Striebel
           smoother then uses the future as well as the past, so trajectories bend smoothly
           through occlusions instead of lagging. Gaps longer than `max_gap_s` split the trajectory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import NLL, HomographyTrack, Run, log, project


def kalman_smooth(t, z, accel_std, meas_std, gate_chi2):
    """t: (n,) times, z: (n, 2) measurements with NaN where missing.
    Returns smoothed states (n, 4), covariances (n, 4, 4) and an inlier mask."""
    n = len(t)
    xs, Ps, xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4)), np.zeros((n, 4)), np.zeros((n, 4, 4))
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
    R = np.eye(2) * meas_std ** 2
    first = int(np.argmax(np.isfinite(z[:, 0])))
    x = np.r_[z[first], 0, 0]
    P = np.diag([meas_std ** 2, meas_std ** 2, 15.0 ** 2, 15.0 ** 2])
    inlier = np.zeros(n, bool)
    Fs = []
    for k in range(n):
        dt = 0.0 if k == 0 else max(t[k] - t[k - 1], 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        q = accel_std ** 2
        Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0], [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                          [dt ** 3 / 2, 0, dt ** 2, 0], [0, dt ** 3 / 2, 0, dt ** 2]])
        x, P = F @ x, F @ P @ F.T + Q
        Fs.append(F)
        xp[k], Pp[k] = x, P
        if np.isfinite(z[k, 0]):
            S = Hm @ P @ Hm.T + R
            r = z[k] - Hm @ x
            d2 = float(r @ np.linalg.solve(S, r))
            if d2 <= gate_chi2 or k == first:
                K = P @ Hm.T @ np.linalg.inv(S)
                x = x + K @ r
                P = (np.eye(4) - K @ Hm) @ P
                inlier[k] = True
        xs[k], Ps[k] = x, P
    # Rauch-Tung-Striebel backward pass
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps, inlier


def run(run: Run, cfg: dict):
    nll = NLL(cfg)
    kc, hc = cfg["kalman"], cfg["homography"]
    ht = HomographyTrack(*nll.homographies(), hc["max_interp_gap_s"])
    tr = run.csv("tracks.csv")
    teams = run.csv("teams.csv") if run.exists("teams.csv") else pd.DataFrame(columns=["track_id", "team"])
    nums = run.csv("numbers.csv") if run.exists("numbers.csv") else pd.DataFrame(columns=["track_id", "number", "name"])
    fl = nll.floor
    # raw floor measurements
    mx, my = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
    for i, r in enumerate(tr.itertuples()):
        H = ht(r.t)
        if H is None:
            continue
        x, y = project(H, np.array([[r.foot_x, r.foot_y]]))[0]
        if fl.inside(x, y, margin=3):
            mx[i], my[i] = x, y
    tr["mx"], tr["my"] = mx, my
    out = []
    n_tracks = n_out = 0
    for tid, g in tr.sort_values("t").groupby("track_id"):
        g = g.drop_duplicates("t")
        z = g[["mx", "my"]].values
        ok = np.isfinite(z[:, 0])
        if ok.sum() < 3:
            continue
        t = g.t.values
        # split where measurements are missing for longer than max_gap_s
        seg = np.zeros(len(t), int)
        last = None
        sid = 0
        for k in range(len(t)):
            if ok[k]:
                if last is not None and t[k] - t[last] > kc["max_gap_s"]:
                    sid += 1
                last = k
            seg[k] = sid
        for s in np.unique(seg):
            m = seg == s
            idx = np.where(m & ok)[0]
            if len(idx) < 3:
                continue
            a, b = idx[0], idx[-1] + 1  # only between the first and last measurement
            tt, zz = t[a:b], z[a:b]
            xs, Ps, inl = kalman_smooth(tt, zz, kc["accel_std"], kc["meas_std"], kc["gate_chi2"])
            n_tracks += 1
            n_out += int((np.isfinite(zz[:, 0]) & ~inl).sum())
            sp = np.hypot(xs[:, 2], xs[:, 3])
            out.append(pd.DataFrame(dict(track_id=tid, segment=int(s), t=tt, frame=g.frame.values[a:b],
                                         shot_id=g.shot_id.values[a:b], x=xs[:, 0].round(2), y=xs[:, 1].round(2),
                                         vx=xs[:, 2].round(2), vy=xs[:, 3].round(2),
                                         speed=np.minimum(sp, kc["max_speed_fps"]).round(2),
                                         pos_std=np.sqrt(Ps[:, 0, 0] + Ps[:, 1, 1]).round(2),
                                         measured=np.isfinite(zz[:, 0]), outlier=np.isfinite(zz[:, 0]) & ~inl,
                                         raw_x=zz[:, 0].round(2), raw_y=zz[:, 1].round(2),
                                         in_crease=g.in_crease.values[a:b])))
    traj = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if not traj.empty:
        traj = traj.merge(teams[["track_id", "team"]], on="track_id", how="left")
        traj = traj.merge(nums[["track_id", "number", "name"]], on="track_id", how="left")
        traj["player"] = np.where(traj.number.notna(), traj.team.astype(str) + " #" +
                                  traj.number.astype(str).str.replace(r"\.0$", "", regex=True), None)
    traj.to_csv(run.p("trajectories.csv"), index=False)
    cov = np.isfinite(mx).mean()
    log(f"floor: {cov:.0%} of track-frames have floor coordinates; {n_tracks} trajectories, "
        f"{n_out} measurements gated as outliers")
    return traj
