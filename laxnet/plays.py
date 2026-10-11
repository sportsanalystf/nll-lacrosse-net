"""Plays - who has the ball, passes (who to whom), turnovers, shots with their context, pass network.

Rules layer on top of the ball track and the player tracks (as in the volleyball pipeline: detections
-> tracks -> court coordinates -> rules):

holder     the ball is on a player's stick when it sits in his "stick zone" (around and above his box)
           and is moving slowly in camera-compensated pixels, measured in player heights per second so
           the rule works at any zoom. A hidden ball (in a pocket, behind a body) stays with the last
           holder for up to max_hidden_s.
possession runs of the same holder longer than min_hold_s. Track ids that share a confirmed jersey
           number count as one player.
pass       possession moves to a teammate.            turnover  possession moves to an opponent.
shot       a release that leaves fast toward the goal the team attacks (goal mouth from the floor
           homography, or the visible crease when the view is uncalibrated), or any release followed by
           a scoreboard goal within goal_link_s. Result: goal / saved (opponent goalie gains the ball)
           / blocked (another opponent gains it within 1 s) / not seen.
shot context (floor coordinates, team attacking +x): distance and angle to goal, zone, nearest defender
           ("space" at release), defenders in the shooting lane, screens, the open gap in the defence
           around the shooter, catch-to-release time, shooter speed, assist and whether it crossed the
           crease, and whether it came within 10 s of winning the ball (transition).

Writes ball_possession_frames.csv (per ball frame: holder and phase, for rendering) and outputs/ball/:
possessions.csv, passes.csv, turnovers.csv, shots.csv, carrier_pressure.csv, pass_network_edges.csv,
pass_network_<team>.png, shot_map.png
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .core import NLL, HomographyTrack, Run, log, project


# --------------------------------------------------------------------------- helpers
def _period_at(spine, t):
    if spine is None or spine.empty:
        return np.nan
    i = int(np.clip(np.searchsorted(spine.t.values, t, side="right") - 1, 0, len(spine) - 1))
    return spine.period.values[i]


def attack_sign(ends, period, team):
    """+1 if `team` attacks the +x goal in this period, -1 if -x, 0 if unknown."""
    d = ends.get(int(period)) if np.isfinite(period) else None
    if not d:
        return 0
    for e, tm in d.items():
        if tm == team:
            return -e          # defends e -> attacks -e
    return list(d.keys())[0]   # only the opponent's end is known: attack it


def _seg_dist(p, a, b):
    ab, ap = b - a, p - a
    t = np.clip((ap @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return float(np.hypot(*(ap - t * ab)))


class Players:
    """Player boxes (source px) interpolated at arbitrary times within one shot, plus identities."""

    def __init__(self, tr_shot, ident):
        self.ids, self.boxes, self.t = [], [], []
        for tid, g in tr_shot.sort_values("t").groupby("track_id"):
            if tid not in ident:
                continue
            self.ids.append(tid)
            self.t.append(g.t.values)
            self.boxes.append(g[["x1", "y1", "x2", "y2"]].values.astype(float))
        self.ident = ident

    def at(self, tb):
        """(n_players, len(tb), 4) boxes, NaN where a track is not alive."""
        out = np.full((len(self.ids), len(tb), 4), np.nan)
        for k, (t, b) in enumerate(zip(self.t, self.boxes)):
            ok = (tb >= t[0] - 0.07) & (tb <= t[-1] + 0.07)
            if len(t) == 1:
                out[k, ok] = b[0]
                continue
            for j in range(4):
                out[k, ok, j] = np.interp(tb[ok], t, b[:, j])
        return out


def holder_raw(ball, boxes, pc, fps):
    """Per ball frame: index of the player whose stick zone holds the ball (-1 none), flight flag."""
    a, b, c = pc["stick_zone"]
    bx, by = ball.x.values, ball.y.values
    x1, y1, x2, y2 = (boxes[..., j] for j in range(4))
    w, h = x2 - x1, y2 - y1
    zx1, zx2, zy1, zy2 = x1 - a * w, x2 + a * w, y1 - b * h, y1 + c * h
    dx = np.maximum.reduce([zx1 - bx, np.zeros_like(zx1), bx - zx2])
    dy = np.maximum.reduce([zy1 - by, np.zeros_like(zy1), by - zy2])
    d = np.hypot(dx, dy) / h
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmed = np.nanmedian(h, axis=0)
    hmed = np.where(np.isfinite(hmed), hmed, np.nanmedian(h) if np.isfinite(h).any() else 150)
    speed_h = ball.speed_px_s.values / hmed
    flight = np.isfinite(speed_h) & (speed_h > pc["fast_ball"])
    centre = np.hypot((x1 + x2) / 2 - bx, y1 - by) / h
    d_rank = np.where(np.isfinite(d), d + 0.01 * centre, np.inf)
    k = np.argmin(d_rank, axis=0) if len(boxes) else np.full(len(bx), -1)
    best = d_rank[k, np.arange(len(bx))] if len(boxes) else np.full(len(bx), np.inf)
    vis = (ball.state != "none").values
    hold = np.where(vis & ~flight & (best <= pc["reach"]), k, -1)
    return hold, flight, speed_h, hmed


def segments(hold, vis, keyf, fps, pc):
    """Possession runs: same holder identity, hidden frames bridged up to max_hidden_s."""
    segs, cur = [], None
    max_hidden = int(pc["max_hidden_s"] * fps)
    for f in range(len(hold)):
        if not vis[f]:
            continue
        k = hold[f]
        if k >= 0:
            if cur and cur["key"] == keyf(k) and f - cur["f1"] <= max_hidden:
                cur["f1"] = f
                cur["frames"].append((f, k))
                continue
            if cur:
                segs.append(cur)
            cur = dict(key=keyf(k), f0=f, f1=f, frames=[(f, k)])
        else:
            if cur:
                segs.append(cur)
                cur = None
    if cur:
        segs.append(cur)
    min_n = max(2, int(round(pc["min_hold_s"] * fps)))
    segs = [s for s in segs if s["f1"] - s["f0"] + 1 >= min_n and len(s["frames"]) >= 2]
    # merge same holder separated only by a short loose/occluded stretch
    out = []
    for s in segs:
        if out and out[-1]["key"] == s["key"] and s["f0"] - out[-1]["f1"] <= fps:
            out[-1]["f1"] = s["f1"]
            out[-1]["frames"] += s["frames"]
        else:
            out.append(s)
    return out


# --------------------------------------------------------------------------- floor context
class FloorAt:
    """Floor positions of tracked players at a time (nearest smoothed trajectory sample)."""

    def __init__(self, traj):
        self.traj = traj
        self.by_shot = {s: g for s, g in traj.groupby("shot_id")} if len(traj) else {}

    def frame(self, sid, t, tol=0.1):
        g = self.by_shot.get(sid)
        if g is None:
            return pd.DataFrame(columns=self.traj.columns)
        g = g[(g.t - t).abs() <= tol]
        if g.empty:
            return g
        return g.loc[(g.t - t).abs().groupby(g.track_id).idxmin()]

    def of(self, sid, tid, t, tol=0.15):
        f = self.frame(sid, t, tol)
        r = f[f.track_id == tid]
        return (float(r.x.iloc[0]), float(r.y.iloc[0]), float(r.speed.iloc[0])) if len(r) else None


def shot_context(sh_xy, team, sign, others, fl, pc):
    """Context of a shot from (x, y) by `team` attacking `sign`. others: floor rows at release."""
    gx = fl.gx
    x, y = sh_xy[0] * sign, sh_xy[1] * sign                   # normalised: attacking +x
    dist = float(np.hypot(gx - x, y))
    angle = float(np.degrees(np.arctan2(abs(y), gx - x)))
    zone = ("behind net" if x > gx else "crease top" if dist <= 12 else
            "slot" if dist <= 25 and abs(y) <= 12 else "inside" if dist <= 25 else
            "perimeter" if dist <= 40 else "outside")
    o = others.copy()
    o["xn"], o["yn"] = o.x * sign, o.y * sign
    de = o[(o.team != team) & (o.team != "official")]
    goalie = de[de.in_crease.astype(bool)]
    de = de[~de.in_crease.astype(bool)]
    mates = o[(o.team == team) & ~o.in_crease.astype(bool)]
    S, G = np.array([x, y]), np.array([gx, 0.0])
    near = float(np.hypot(de.xn - x, de.yn - y).min()) if len(de) else None
    lane = [_seg_dist(np.array([a, b]), S, G) for a, b in zip(de.xn, de.yn)]
    in_lane = int(sum(d <= pc["lane_width_ft"] for d in lane))
    screens = int(sum(_seg_dist(np.array([a, b]), S, G) <= pc["lane_width_ft"] and np.hypot(gx - a, b) <= 12
                      for a, b in zip(mates.xn, mates.yn)))
    # open gap: defenders within 35 ft of goal, by angle around the goal; the gap the shooter stands in
    gap_deg = gap_ft = None
    d35 = de[np.hypot(gx - de.xn, de.yn) <= 35]
    if len(d35):
        phi = np.degrees(np.arctan2(d35.yn.values, gx - d35.xn.values))
        ps = np.degrees(np.arctan2(y, gx - x))
        left, right = phi[phi >= ps], phi[phi < ps]
        lo = right.max() if len(right) else -90.0
        hi = left.min() if len(left) else 90.0
        gap_deg = float(hi - lo)
        gap_ft = float(np.radians(gap_deg) * dist)
    return dict(x_ft=round(x, 1), y_ft=round(y, 1), distance_ft=round(dist, 1), angle_deg=round(angle, 1), zone=zone,
                nearest_defender_ft=round(near, 1) if near is not None else None,
                defenders_in_lane=in_lane, screens=screens,
                gap_deg=round(gap_deg, 1) if gap_deg is not None else None,
                gap_ft=round(gap_ft, 1) if gap_ft is not None else None,
                defenders_tracked=int(len(de)), goalie_tracked=bool(len(goalie)))


# --------------------------------------------------------------------------- stage
def run(run: Run, cfg: dict):
    nll = NLL(cfg)
    pc, fps, fl = cfg["plays"], cfg["ball"]["fps"], nll.floor
    if not run.exists("ball_track.csv"):
        log("plays: no ball_track.csv (run the ball stage first)")
        return None
    bt = run.csv("ball_track.csv")
    tr = run.csv("tracks.csv")
    teams = run.csv("teams.csv")
    nums = run.csv("numbers.csv") if run.exists("numbers.csv") else pd.DataFrame(columns=["track_id", "number", "name"])
    traj = run.csv("trajectories.csv") if run.exists("trajectories.csv") else pd.DataFrame()
    team_of = dict(zip(teams.track_id, teams.team))
    num_of = {r.track_id: str(r.number).replace(".0", "") for r in nums.dropna(subset=["number"]).itertuples()}
    name_of = {r.track_id: r.name for r in nums.itertuples() if isinstance(r.name, str)}
    crease_share = tr.groupby("track_id").in_crease.mean().to_dict()
    ident = {tid: tm for tid, tm in team_of.items() if tm not in ("official", "unknown") and isinstance(tm, str)}

    def key(tid):
        return f"{team_of[tid]} #{num_of[tid]}" if tid in num_of else f"{team_of[tid]} trk{tid}"

    def label(tid):
        if tid in num_of:
            nm = name_of.get(tid)
            return f"{team_of[tid]} #{num_of[tid]}" + (f" {nm.split()[-1]}" if nm else "")
        return f"{team_of[tid]} (unidentified)"

    ht = HomographyTrack(*nll.homographies(), cfg["homography"]["max_interp_gap_s"])
    ends, spine, events = nll.defended_ends(), nll.spine, nll.events
    goals = events[events.event == "goal"] if events is not None else pd.DataFrame(columns=["t", "team"])
    FA = FloorAt(traj)
    poss, passes, tos, shots, press, frames_out = [], [], [], [], [], []

    for sid, ball in bt.groupby("shot_id"):
        ball = ball.sort_values("bframe").reset_index(drop=True)
        P = Players(tr[tr.shot_id == sid], ident)
        tb = ball.t.values
        boxes = P.at(tb)
        hold, flight, speed_h, hmed = holder_raw(ball, boxes, pc, fps) if len(P.ids) else \
            (np.full(len(ball), -1), np.zeros(len(ball), bool), np.full(len(ball), np.nan), np.full(len(ball), 150.0))
        vis = (ball.state != "none").values
        segs = segments(hold, vis, lambda k: key(P.ids[k]), fps, pc)
        # segment records
        S = []
        for s in segs:
            ks = [P.ids[k] for _, k in s["frames"]]
            tid = max(set(ks), key=ks.count)
            S.append(dict(shot_id=sid, f0=s["f0"], f1=s["f1"], t0=float(tb[s["f0"]]), t1=float(tb[s["f1"]]),
                          track_id=tid, team=team_of[tid], player=label(tid), key=s["key"],
                          goalie=crease_share.get(tid, 0) >= 0.5))
        # per-frame phase for rendering
        holder_f = np.full(len(ball), -1)
        for s in S:
            holder_f[s["f0"]:s["f1"] + 1] = s["track_id"]
        phase = np.where(holder_f >= 0, "held", np.where(~vis, "hidden", np.where(flight, "flight", "loose")))
        frames_out.append(pd.DataFrame(dict(shot_id=sid, bframe=ball.bframe, t=tb, holder_track=holder_f,
                                            holder_team=[team_of.get(h, "") if h >= 0 else "" for h in holder_f],
                                            phase=phase)))
        for s in S:
            poss.append(dict(shot_id=sid, t_start=round(s["t0"], 2), t_end=round(s["t1"], 2),
                             seconds=round(s["t1"] - s["t0"] + 1 / fps, 2), team=s["team"], player=s["player"],
                             track_id=s["track_id"], goalie=s["goalie"]))
            # pressure on the carrier, sampled at the trajectory rate
            for t in np.arange(s["t0"], s["t1"] + 1e-6, 0.1):
                me = FA.of(sid, s["track_id"], t)
                if me is None:
                    continue
                fr = FA.frame(sid, t)
                opp = fr[(fr.team != s["team"]) & (fr.team != "official") & ~fr.in_crease.astype(bool)]
                if len(opp):
                    press.append(dict(player=s["player"], team=s["team"], t=round(t, 2),
                                      nearest_defender_ft=float(np.hypot(opp.x - me[0], opp.y - me[1]).min())))
        # transitions between possessions + shots at releases
        last_gain = {}
        for i, a in enumerate(S):
            if i > 0 and S[i - 1]["team"] != a["team"]:
                last_gain[a["team"]] = a["t0"]
            b = S[i + 1] if i + 1 < len(S) else None
            t_rel, f_rel = a["t1"], a["f1"]
            gap = (b["t0"] - t_rel) if b else None
            # release velocity: median camera-compensated velocity just after release
            w = ball.iloc[f_rel + 1:f_rel + 1 + int(0.3 * fps)]
            w = w[w.state != "none"].dropna(subset=["vx"])
            v = np.array([w.vx.median(), w.vy.median()]) if len(w) else np.array([np.nan, np.nan])
            rel_speed = float(np.hypot(*v) / hmed[f_rel]) if np.isfinite(v).all() else np.nan
            per = _period_at(spine, t_rel)
            sign = attack_sign(ends, per, a["team"])
            bxy = ball.loc[f_rel, ["x", "y"]].values.astype(float)
            H = ht(t_rel)
            me = FA.of(sid, a["track_id"], t_rel)
            if me is None and H is not None:
                bx = boxes[P.ids.index(a["track_id"]), f_rel]
                if np.isfinite(bx).all():
                    xy = project(H, np.array([[(bx[0] + bx[2]) / 2, bx[3]]]))[0]
                    me = (float(xy[0]), float(xy[1]), np.nan)
            if sign == 0 and me is not None:
                sign = 1 if me[0] >= 0 else -1                 # unknown ends: the nearer goal
            goal_px = None
            if H is not None and sign != 0:
                goal_px = project(np.linalg.inv(H), np.array([[sign * fl.gx, 0.0]]))[0]
            elif np.isfinite(ball.loc[f_rel, "crease_x"]):
                goal_px = ball.loc[f_rel, ["crease_x", "crease_y"]].values.astype(float)
            heading = None
            if goal_px is not None and np.isfinite(v).all() and np.hypot(*v) > 0:
                g = goal_px - bxy
                heading = float(np.degrees(np.arccos(np.clip(v @ g / (np.hypot(*v) * np.hypot(*g) + 1e-9), -1, 1))))
            gl = goals[(goals.team == a["team"]) & (goals.t >= t_rel - 1.0) & (goals.t <= t_rel + pc["goal_link_s"])]
            is_shot = len(gl) > 0 or (heading is not None and heading <= pc["shot_cone_deg"]
                                      and np.isfinite(rel_speed) and rel_speed >= pc["shot_speed"])
            if is_shot:
                if len(gl):
                    result = "goal"
                elif b and gap <= 2.0 and b["team"] != a["team"] and b["goalie"]:
                    result = "saved"
                elif b and gap <= 1.0 and b["team"] != a["team"]:
                    result = "blocked"
                elif b and gap <= 3.0 and b["team"] == a["team"]:
                    result = "missed, rebound kept"
                else:
                    result = "not seen"
                rec = dict(shot_id=sid, t=round(t_rel, 2), period=per, team=a["team"], shooter=a["player"],
                           track_id=a["track_id"], result=result,
                           release_speed_hps=round(rel_speed, 1) if np.isfinite(rel_speed) else None,
                           heading_deg=round(heading, 1) if heading is not None else None,
                           catch_to_release_s=round(a["t1"] - a["t0"] + 1 / fps, 2),
                           shooter_speed_fps=round(me[2], 1) if me is not None and np.isfinite(me[2]) else None,
                           ball_px=bxy.round(1).tolist(), goal_px=goal_px.round(1).tolist() if goal_px is not None else None,
                           after_gain_s=round(t_rel - last_gain[a["team"]], 1) if a["team"] in last_gain else None)
                rec["transition"] = rec["after_gain_s"] is not None and rec["after_gain_s"] <= 10
                if me is not None and sign != 0:
                    rec.update(shot_context(me[:2], a["team"], sign, FA.frame(sid, t_rel), fl, pc))
                # assist: the pass that gave the shooter the ball
                pa = [p for p in passes if p["shot_id"] == sid and p["receiver_track"] == a["track_id"]
                      and 0 <= a["t1"] - p["t_catch"] <= pc["assist_window_s"]]
                if pa:
                    p = pa[-1]
                    rec["assist"] = p["passer"]
                    rec["assist_cross_crease"] = p.get("cross_crease", False)
                shots.append(rec)
                continue
            if not b or gap > 3.0:
                continue
            if b["team"] == a["team"]:
                if b["key"] == a["key"]:
                    continue
                if gap < 0.1 and not flight[f_rel:b["f0"] + 1].any():
                    continue          # holder flicked between two close teammates: not a pass
                rec = dict(shot_id=sid, t_release=round(t_rel, 2), t_catch=round(b["t0"], 2), team=a["team"],
                           passer=a["player"], receiver=b["player"], passer_track=a["track_id"],
                           receiver_track=b["track_id"], flight_s=round(gap, 2),
                           release_speed_hps=round(rel_speed, 1) if np.isfinite(rel_speed) else None,
                           from_px=bxy.round(1).tolist(), to_px=ball.loc[b["f0"], ["x", "y"]].values.astype(float).round(1).tolist())
                fa, fb = me, FA.of(sid, b["track_id"], b["t0"])
                if fa is not None and fb is not None and sign != 0:
                    ax, ay, bx_, by_ = fa[0] * sign, fa[1] * sign, fb[0] * sign, fb[1] * sign
                    fr = FA.frame(sid, t_rel)
                    de = fr[(fr.team != a["team"]) & (fr.team != "official") & ~fr.in_crease.astype(bool)]
                    dxn = de.x * sign
                    rec.update(from_x=round(ax, 1), from_y=round(ay, 1), to_x=round(bx_, 1), to_y=round(by_, 1),
                               distance_ft=round(float(np.hypot(bx_ - ax, by_ - ay)), 1),
                               direction="forward" if bx_ - ax > 5 else "back" if bx_ - ax < -5 else "lateral",
                               defenders_bypassed=int(((dxn > min(ax, bx_)) & (dxn < max(ax, bx_))).sum()) if bx_ > ax else 0,
                               cross_crease=bool(ay * by_ < 0 and abs(ay) > 5 and abs(by_) > 5
                                                 and np.hypot(fl.gx - ax, ay) <= 30 and np.hypot(fl.gx - bx_, by_) <= 30),
                               into_slot=bool(np.hypot(fl.gx - bx_, by_) <= 25 and abs(by_) <= 12))
                passes.append(rec)
            else:
                kind = "interception" if gap <= 1.5 and (flight[f_rel:b["f0"]].any()) else "loose ball"
                tos.append(dict(shot_id=sid, t=round(t_rel, 2), lost_by=a["team"], player_lost=a["player"],
                                gained_by=b["team"], player_gained=b["player"], kind=kind, gap_s=round(gap, 2),
                                x_ft=round(me[0], 1) if me is not None else None,
                                y_ft=round(me[1], 1) if me is not None else None))

    od = run.p("outputs", "ball", "x").parent
    pd.concat(frames_out, ignore_index=True).to_csv(run.p("ball_possession_frames.csv"), index=False)
    P_, Pa, To, Sh = pd.DataFrame(poss), pd.DataFrame(passes), pd.DataFrame(tos), pd.DataFrame(shots)
    P_.to_csv(od / "possessions.csv", index=False)
    Pa.to_csv(od / "passes.csv", index=False)
    To.to_csv(od / "turnovers.csv", index=False)
    Sh.to_csv(od / "shots.csv", index=False)
    pr = pd.DataFrame(press)
    if len(pr):
        pr = pr.groupby(["player", "team"]).agg(samples=("t", "size"), mean_nearest_defender_ft=("nearest_defender_ft", "mean"),
                                                under_pressure_share=("nearest_defender_ft", lambda s: (s <= pc["pressure_ft"]).mean()))
        pr = pr.round(2).reset_index()
    pr.to_csv(od / "carrier_pressure.csv", index=False)
    edges = pass_network(run, Pa, P_, traj, nll, team_of)
    shot_map(run, Sh, nll)
    summary = dict(possessions=len(P_), passes=len(Pa), turnovers=len(To), shots=len(Sh),
                   possession_seconds=P_.groupby("team").seconds.sum().round(1).to_dict() if len(P_) else {},
                   edges=len(edges))
    run.save_json("outputs/ball/summary.json", summary)
    log(f"plays: {len(P_)} possessions, {len(Pa)} passes, {len(To)} turnovers, {len(Sh)} shots "
        f"({', '.join(f'{k} {v}' for k, v in Sh.result.value_counts().items()) if len(Sh) else 'none'})")
    return summary


# --------------------------------------------------------------------------- figures
def pass_network(run: Run, passes, poss, traj, nll: NLL, team_of):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    od = run.root / "outputs" / "ball"
    if passes.empty:
        pd.DataFrame(columns=["team", "passer", "receiver", "passes"]).to_csv(od / "pass_network_edges.csv", index=False)
        return pd.DataFrame()
    e = passes.groupby(["team", "passer", "receiver"]).size().rename("passes").reset_index()
    for c in ("distance_ft", "from_x", "to_x"):
        if c in passes:
            e = e.merge(passes.groupby(["team", "passer", "receiver"])[c].mean().round(1).rename(f"mean_{c}").reset_index(),
                        on=["team", "passer", "receiver"], how="left")
    e.to_csv(od / "pass_network_edges.csv", index=False)
    fl = nll.floor
    for team, g in e.groupby("team"):
        pts = {}
        tp = passes[passes.team == team]
        if "from_x" in tp:
            for who, xs, ys in [(r.passer, r.from_x, r.from_y) for r in tp.itertuples()] + \
                               [(r.receiver, r.to_x, r.to_y) for r in tp.itertuples()]:
                if pd.notna(xs):
                    pts.setdefault(who, []).append((xs, ys))
        names = sorted(set(g.passer) | set(g.receiver))
        # players without floor positions are placed on a ring in the attacking half
        for k, n in enumerate(n for n in names if n not in pts):
            a = np.pi * (0.25 + 0.5 * k / max(1, len(names)))
            pts[n] = [(fl.gx - 35 * np.sin(a), 30 * np.cos(a))]
        pos = {n: np.mean(v, axis=0) for n, v in pts.items()}
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fl.draw(ax, face="#2b2b2b")
        mx = g.passes.max()
        for r in g.itertuples():
            a, b = pos[r.passer], pos[r.receiver]
            ax.annotate("", xy=b, xytext=a, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", lw=1 + 4 * r.passes / mx, color="#f6c344", alpha=0.85,
                                        shrinkA=10, shrinkB=10, connectionstyle="arc3,rad=0.12"))
        touches = pd.concat([g.groupby("passer").passes.sum(), g.groupby("receiver").passes.sum()]).groupby(level=0).sum()
        for n, p in pos.items():
            ax.scatter(*p, s=120 + 60 * touches.get(n, 0), color="white", edgecolor="black", zorder=5)
            ax.text(p[0], p[1] - 4.5, n.replace(f"{team} ", ""), ha="center", va="top", fontsize=8, color="white", zorder=6)
        ax.set_title(f"{team} pass network, attacking right ({int(g.passes.sum())} passes seen)", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(od / f"pass_network_{team}.png", dpi=130)
        plt.close(fig)
    return e


def shot_map(run: Run, shots, nll: NLL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    od = run.root / "outputs" / "ball"
    fl = nll.floor
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fl.draw(ax, face="#2b2b2b")
    col = {"goal": "#2ecc71", "saved": "#f1c40f", "blocked": "#e67e22", "not seen": "#bbbbbb",
           "missed, rebound kept": "#9b59b6"}
    n = 0
    if len(shots) and "x_ft" in shots:
        for r in shots.dropna(subset=["x_ft"]).itertuples():
            ax.plot([r.x_ft, fl.gx], [r.y_ft, 0], color=col.get(r.result, "white"), lw=0.8, alpha=0.6, zorder=3)
            size = 60 + 12 * (r.gap_ft if isinstance(r.gap_ft, (int, float)) and np.isfinite(r.gap_ft) else 0)
            ax.scatter(r.x_ft, r.y_ft, s=size, color=col.get(r.result, "white"), edgecolor="black", zorder=4)
            ax.text(r.x_ft, r.y_ft + 3, r.team, fontsize=7, color="white", ha="center", zorder=5)
            n += 1
    for k, c in col.items():
        ax.scatter([], [], color=c, label=k, edgecolor="black")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=5, fontsize=8, frameon=False)
    ax.set_title(f"Shots with floor positions ({n}); marker size = open gap around the shooter, both teams attacking right",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(od / "shot_map.png", dpi=130)
    plt.close(fig)
