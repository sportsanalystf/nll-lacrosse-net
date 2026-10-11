"""Advanced player-tracking stats that need no ball: movement load, on-floor stints, transition speed,
defensive shape and gaps, matchups, picks, and space control.

All from trajectories.csv (Kalman-smoothed floor positions), so they cover only what the broadcast
camera shows and only views with a floor calibration. Positions are normalised so each team attacks +x.

outputs/advanced/
    player_motion.csv   per player: time, distance, top speed (mph), best 1-s speed, high-speed and sprint
                        distance, sprints, accelerations
    stints.csv          identified players' on-floor stints, from sightings (experimental: the camera does
                        not show the bench, so a stint is "seen on the floor, with short gaps bridged")
    transitions.csv     a team's skaters moving end to end: time for the team centroid to go from its
                        defensive zone (past its restraining line) to the offensive zone, and its speed
    defense_shape.csv   per team while defending (>= 4 skaters in its own end): spacing, compactness,
                        widest lateral seam between neighbouring defenders, distance from the slot to the
                        nearest defender
    matchups.csv        defender-attacker pairs (optimal assignment each frame): time together and mean
                        separation; per attacker the number of switches
    picks.csv           heuristic picks: a nearly stationary attacker that a teammate brushes past, after
                        which the teammate's defender falls >= gain_ft further behind within 1 s
    space_control.csv   share of the visible floor (and of the 30-ft scoring area) each team reaches first
    space_control_<shot>.png   one example frame per shot
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .analytics import _attach_period, normalise
from .core import NLL, HomographyTrack, Run, log, project

MPH = 3600 / 5280


def _who(traj):
    return traj.player.where(traj.player.notna(), traj.team.astype(str) + " trk" + traj.track_id.astype(str))


def player_motion(traj, ac):
    rows = []
    for who, g in traj.groupby("who"):
        secs = dist = hs = spr = 0.0
        top = best1 = 0.0
        n_spr = n_acc = 0
        for _, s in g.sort_values("t").groupby(["track_id", "segment"]):
            t, v = s.t.values, s.speed.values
            if len(t) < 2:
                continue
            dt = np.clip(np.diff(t), 0, 0.5)
            vm = (v[1:] + v[:-1]) / 2
            secs += dt.sum()
            dist += (vm * dt).sum()
            hs += (vm * dt)[vm >= ac["high_speed_fps"]].sum()
            spr += (vm * dt)[vm >= ac["sprint_fps"]].sum()
            top = max(top, float(np.percentile(v, 95)))
            # best sustained 1 s: mean speed over a 1-s window
            if t[-1] - t[0] >= 1.0:
                cs = np.r_[0, np.cumsum(vm * dt)]
                j = np.searchsorted(t, t + 1.0)
                ok = j < len(t)
                if ok.any():
                    best1 = max(best1, float(((cs[j[ok]] - cs[np.where(ok)[0]]) / (t[j[ok]] - t[ok])).max()))
            # sprints: runs above sprint speed lasting min_sprint_s
            n_spr += _runs(t, v >= ac["sprint_fps"], ac["min_sprint_s"])
            a = np.gradient(v, t) if len(t) > 2 else np.zeros_like(v)
            n_acc += _runs(t, a >= ac["accel_fps2"], 0.4)
        if secs < 1:
            continue
        rows.append(dict(player=who, team=g.team.iloc[0], seconds=round(secs, 1), distance_ft=round(dist, 1),
                         ft_per_min=round(dist / secs * 60, 1), top_speed_mph=round(top * MPH, 1),
                         best_1s_speed_mph=round(best1 * MPH, 1), high_speed_distance_ft=round(hs, 1),
                         sprint_distance_ft=round(spr, 1), sprints=n_spr, accelerations=n_acc,
                         goalie=bool(g.in_crease.mean() > 0.5)))
    return pd.DataFrame(rows)


def _runs(t, flag, min_s):
    n, start = 0, None
    for k in range(len(t)):
        if flag[k] and start is None:
            start = t[k]
        if (not flag[k] or k == len(t) - 1) and start is not None:
            end = t[k]
            if end - start >= min_s:
                n += 1
            start = None
    return n


def stints(traj, ac):
    rows = []
    idf = traj[traj.player.notna()]
    for who, g in idf.groupby("player"):
        t = np.sort(g.t.unique())
        start = prev = t[0]
        for x in list(t[1:]) + [None]:
            if x is None or x - prev > ac["stint_gap_s"]:
                rows.append(dict(player=who, team=g.team.iloc[0], first_seen=round(start, 1), last_seen=round(prev, 1),
                                 seen_span_s=round(prev - start, 1)))
                if x is not None:
                    start = x
            if x is not None:
                prev = x
    return pd.DataFrame(rows)


def per_frame(traj):
    """Frames (shot, t) with each team's skaters, goalies excluded."""
    sk = traj[~traj.in_crease.astype(bool) & traj.dir_known]
    return sk.groupby(["shot_id", "t"])


def transitions(traj, fl):
    rows = []
    sk = traj[~traj.in_crease.astype(bool) & traj.dir_known]
    for (sid, team), g in sk.groupby(["shot_id", "team"]):
        c = g.groupby("t").agg(xn=("xn", "mean"), n=("xn", "size"))
        c = c[c.n >= 3]
        if len(c) < 5:
            continue
        t, x = c.index.values, c.xn.values
        k = 0
        while k < len(t):
            if x[k] < -fl.rx:                                   # centroid in own defensive zone
                j = k
                while j < len(t) and x[j] < fl.rx and t[j] - t[k] < 12:
                    j += 1
                if j < len(t) and x[j] >= fl.rx and np.all(np.diff(t[k:j + 1]) < 0.6):
                    k0 = max(i for i in range(k, j) if x[i] < -fl.rx)     # last moment still in the d-zone
                    dur = t[j] - t[k0]
                    rows.append(dict(shot_id=sid, team=team, t_start=round(t[k0], 2), t_end=round(t[j], 2),
                                     seconds=round(dur, 2), centroid_speed_fps=round((x[j] - x[k0]) / max(dur, 1e-3), 1),
                                     skaters_tracked=int(c.n.iloc[k0:j + 1].median())))
                    k = j
            k += 1
    return pd.DataFrame(rows)


def defense_shape(traj, fl):
    rows, frames = [], []
    sk = traj[~traj.in_crease.astype(bool) & traj.dir_known]
    for (sid, t), f in sk.groupby(["shot_id", "t"]):
        for team, d in f.groupby("team"):
            d = d[d.xn < -fl.rx]                                  # in its own defensive zone
            if len(d) < 4:
                continue
            p = np.c_[-d.xn.values, d.yn.values]                  # defending end at +x after the flip
            pair = np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1))[np.triu_indices(len(p), 1)]
            ys = np.sort(p[:, 1])
            hull = 0.0
            if len(p) >= 3:
                import cv2
                hull = float(cv2.contourArea(cv2.convexHull(p.astype(np.float32))))
            slot = np.array([fl.gx - 15, 0.0])
            frames.append(dict(shot_id=sid, t=t, team=team, defenders=len(p), spacing_ft=pair.mean(),
                               hull_sqft=hull, widest_seam_ft=float(np.diff(ys).max()),
                               slot_to_nearest_ft=float(np.hypot(*(p - slot).T).min())))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    fr = pd.DataFrame(frames)
    s = fr.groupby("team").agg(frames=("t", "size"), mean_defenders_tracked=("defenders", "mean"),
                               mean_spacing_ft=("spacing_ft", "mean"), mean_area_sqft=("hull_sqft", "mean"),
                               mean_widest_seam_ft=("widest_seam_ft", "mean"),
                               seam_over_12ft_share=("widest_seam_ft", lambda v: (v > 12).mean()),
                               mean_slot_to_nearest_ft=("slot_to_nearest_ft", "mean"),
                               slot_open_over_8ft_share=("slot_to_nearest_ft", lambda v: (v > 8).mean()))
    return s.round(2).reset_index(), fr


def matchups(traj, fl, fps):
    pairs, assign = [], []
    sk = traj[~traj.in_crease.astype(bool) & traj.dir_known]
    for (sid, t), f in sk.groupby(["shot_id", "t"]):
        teams = f.team.unique()
        if len(teams) != 2:
            continue
        for att in teams:
            A = f[(f.team == att) & (f.xn > fl.rx)]                # attackers in their offensive zone
            D = f[(f.team != att) & (f.xn < -fl.rx)]               # defenders in their defensive zone
            if len(A) < 2 or len(D) < 2:
                continue
            ax, ay = A.x.values, A.y.values                      # raw floor coordinates: distances only
            dx, dy = D.x.values, D.y.values
            cost = np.hypot(ax[:, None] - dx[None], ay[:, None] - dy[None])
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                pairs.append(dict(shot_id=sid, t=t, attacker=A.who.values[i], defender=D.who.values[j],
                                  team_att=att, sep=cost[i, j]))
    if not pairs:
        return pd.DataFrame()
    p = pd.DataFrame(pairs).sort_values(["attacker", "t"])
    dt = 1.0 / fps
    m = p.groupby(["team_att", "attacker", "defender"]).agg(frames=("t", "size"), mean_separation_ft=("sep", "mean"))
    m["seconds_est"] = (m.frames * dt).round(1)
    def switches(g, min_frames=max(3, int(0.5 * fps))):
        # count a switch only when the new defender stays on the attacker for >= 0.5 s
        runs = (g.defender != g.defender.shift()).cumsum()
        seq = [d for d, n in g.groupby(runs).defender.agg(["first", "size"]).values if n >= min_frames]
        return int(sum(a != b for a, b in zip(seq, seq[1:])))
    sw = {a: switches(g) for a, g in p.groupby("attacker")}
    m = m.reset_index()
    m["attacker_switches"] = m.attacker.map(sw)
    return m.round(2).sort_values("frames", ascending=False)


def picks(traj, pk):
    rows = []
    sk = traj[~traj.in_crease.astype(bool) & traj.dir_known]
    for sid, g in sk.groupby("shot_id"):
        ts = np.sort(g.t.unique())
        byt = {t: f for t, f in g.groupby("t")}
        last = {}
        for t in ts:
            f = byt[t]
            for team in f.team.unique():
                A, D = f[f.team == team], f[f.team != team]
                if len(A) < 2 or len(D) < 2:
                    continue
                for s in A.itertuples():                       # screener
                    if s.speed > pk["screener_max_speed"]:
                        continue
                    for u in A.itertuples():                   # user coming off the screen
                        if u.who == s.who or np.hypot(u.x - s.x, u.y - s.y) > pk["brush_ft"] or u.speed < 8:
                            continue
                        dd = np.hypot(D.x - u.x, D.y - u.y)
                        d0, dname = float(dd.min()), D.who.values[int(np.argmin(dd.values))]
                        t1 = t + 1.0
                        fut = g[(g.t > t + 0.6) & (g.t <= t1 + 0.05)]
                        fu, fd = fut[fut.who == u.who], fut[fut.who == dname]
                        if fu.empty or fd.empty:
                            continue
                        m = fu.merge(fd, on="t", suffixes=("_u", "_d"))
                        if m.empty:
                            continue
                        d1 = float(np.hypot(m.x_u - m.x_d, m.y_u - m.y_d).max())
                        k = (s.who, u.who)
                        if d1 - d0 >= pk["gain_ft"] and t - last.get(k, -9) > 3:
                            last[k] = t
                            rows.append(dict(shot_id=sid, t=round(t, 2), team=team, screener=s.who, user=u.who,
                                             defender=dname, separation_before_ft=round(d0, 1),
                                             separation_after_ft=round(d1, 1), x_ft=round(s.xn, 1), y_ft=round(s.yn, 1)))
    return pd.DataFrame(rows)


def visible_polygon(H, W, Hh):
    """Floor-coordinate polygon of the image border (points behind the camera horizon are dropped)."""
    if H is None:
        return None
    for top in (0.08, 0.3, 0.5):
        e = np.linspace(0, 1, 12)
        pts = np.r_[np.c_[e * W, np.full(12, top * Hh)], np.c_[np.full(12, W), top * Hh + e * (1 - top) * Hh],
                    np.c_[(1 - e) * W, np.full(12, Hh)], np.c_[np.zeros(12), Hh - e * (1 - top) * Hh]]
        q = np.c_[pts, np.ones(len(pts))] @ H.T
        if (q[:, 2] > 0).all() or (q[:, 2] < 0).all():
            return q[:, :2] / q[:, 2:3]
    return None


def space_control(run, traj, nll, ac, img_wh, colors):
    import cv2
    fl = nll.floor
    ht = HomographyTrack(*nll.homographies(), 0.7)
    g = ac["grid_ft"]
    xs = np.arange(-fl.L / 2 + g / 2, fl.L / 2, g)
    ys = np.arange(-fl.W / 2 + g / 2, fl.W / 2, g)
    GX, GY = np.meshgrid(xs, ys)
    from matplotlib.path import Path
    rink = Path(fl.outline()).contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
    rows, examples = [], {}
    sk = traj[~traj.in_crease.astype(bool)]
    for (sid, t), f in sk.groupby(["shot_id", "t"]):
        if f.team.nunique() != 2 or f.groupby("team").size().min() < 3:
            continue
        poly = visible_polygon(ht(t), *img_wh)
        if poly is None:
            continue
        gp = np.c_[(poly[:, 0] - xs[0]) / g, (poly[:, 1] - ys[0]) / g]
        gp = np.clip(gp, -1e4, 1e4).round().astype(np.int32)
        ras = np.zeros(GX.shape, np.uint8)
        cv2.fillPoly(ras, [gp.reshape(-1, 1, 2)], 1)
        inside = ras.astype(bool) & rink
        if inside.sum() < 50:
            continue
        # time to reach: drift 0.3 s along current velocity, then run at reach_speed
        px, py = f.x.values + 0.3 * f.vx.values, f.y.values + 0.3 * f.vy.values
        tt = np.hypot(GX[None] - px[:, None, None], GY[None] - py[:, None, None]) / ac["reach_speed_fps"]
        team_values = f.team.astype(str).to_numpy(dtype=str)
        owner = team_values[np.argmin(tt, axis=0)]
        teams = sorted(f.team.unique())
        for team in teams:
            share = ((owner == team) & inside).sum() / inside.sum()
            # scoring area: within 30 ft of the goal this team attacks (via its normalised direction)
            sign = np.sign(np.nanmedian((f[f.team == team].xn / f[f.team == team].x).replace([np.inf, -np.inf], np.nan)))
            sa = inside & (np.hypot(GX - sign * fl.gx, GY) <= 30) if np.isfinite(sign) and sign != 0 else None
            rows.append(dict(shot_id=sid, t=t, team=team, visible_share=share,
                             scoring_area_share=((owner == team) & sa).sum() / sa.sum() if sa is not None and sa.sum() > 20 else np.nan))
        if sid not in examples or len(f) > examples[sid][2]:
            examples[sid] = (t, (owner, inside, f), len(f))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # one example figure per shot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    pal = iter(["#e4572e", "#2e86de"])
    cols = nll_cols = {tm: colors.get(tm) or next(pal) for tm in sorted(traj.team.dropna().unique())}
    for sid, (t, (owner, inside, f), _) in examples.items():
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fl.draw(ax, face="#2b2b2b")
        rgba = np.zeros(GX.shape + (4,))
        for team, c in nll_cols.items():
            m = (owner == team) & inside
            rgba[m, :3] = to_rgb(c)
            rgba[m, 3] = 0.45
        ax.imshow(rgba, extent=(xs[0] - g / 2, xs[-1] + g / 2, ys[0] - g / 2, ys[-1] + g / 2), origin="lower", zorder=2)
        for team, c in cols.items():
            q = f[f.team == team]
            ax.scatter(q.x, q.y, s=50, color=c, edgecolor="white", zorder=4, label=team)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=4, fontsize=8, frameon=False)
        ax.set_title(f"Space control, shot {sid} at {t:.1f}s: each visible cell coloured by the team that reaches it first",
                     loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(run.p("outputs", "advanced", f"space_control_{sid:04d}.png"), dpi=120)
        plt.close(fig)
    return out


def run(run: Run, cfg: dict):
    nll = NLL(cfg)
    ac = cfg["advanced"]
    if not run.exists("trajectories.csv"):
        log("advanced: no trajectories.csv")
        return None
    traj = run.csv("trajectories.csv")
    traj = traj[traj.team.notna() & (traj.team != "official")].copy()
    if traj.empty:
        log("advanced: no trajectories")
        return None
    traj = _attach_period(traj, nll.spine)
    traj = normalise(traj, nll).reset_index(drop=True)
    traj["who"] = _who(traj)
    fl = nll.floor
    prep = json.loads(run.p("prep.json").read_text())
    od = run.p("outputs", "advanced", "x").parent
    pm = player_motion(traj, ac)
    st = stints(traj, ac)
    tr = transitions(traj, fl)
    ds, _ = defense_shape(traj, fl)
    mu = matchups(traj, fl, prep["track_fps"])
    pk = picks(traj, ac["pick"])
    sc = space_control(run, traj, nll, ac, (prep["source_width"], prep["source_height"]),
                       cfg["render"].get("team_colors", {}))
    for name, df in (("player_motion", pm), ("stints", st), ("transitions", tr), ("defense_shape", ds),
                     ("matchups", mu), ("picks", pk)):
        df.to_csv(od / f"{name}.csv", index=False)
    scs = pd.DataFrame()
    if len(sc):
        scs = sc.groupby("team").agg(frames=("t", "size"), visible_floor_share=("visible_share", "mean"),
                                     scoring_area_share=("scoring_area_share", "mean")).round(3).reset_index()
    scs.to_csv(od / "space_control.csv", index=False)
    log(f"advanced: {len(pm)} players, {len(tr)} transitions, {len(ds)} team shapes, {len(mu)} matchup pairs, "
        f"{len(pk)} picks, space control over {sc.t.nunique() if len(sc) else 0} frames")
    return dict(player_motion=pm, stints=st, transitions=tr, defense_shape=ds, matchups=mu, picks=pk, space_control=scs)
