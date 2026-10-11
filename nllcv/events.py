"""Stage 6 - event detection on the clock-indexed spine.

Goals        score increments in the cleaned scoreboard. The goal moment is placed inside the live
             footage that last showed the old score, at the onset of the strongest audio spike
             (horn + crowd); `audio_confirmed` says whether such a spike was found.
Possession   per sampled live frame: the play is at the end where the players are; the team
             defending that end is learned per period from which team's goalie stands in that
             crease, so the other team has the ball. Where no goalie vote exists, the team whose
             players sit closer to the goal is taken as defending. Smoothed within each shot.
Transition   the all-player centroid crossing centre at speed, and possession changes.
Shots        NLL resets the 30 s shot clock when a shot hits the goalie or post. A reset with the
             same team still attacking afterwards is logged as a shot (offensive rebound); a reset
             followed by the other team attacking is logged as a possession change after a shot
             or turnover.
Locations    for goals and shots: the attacking player nearest the target goal in the ~1.5 s before
             the event (an estimate, the ball itself is not tracked).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Run, fmt_clock, log
from .floor import Floor
from . import audio as audio_mod
from . import shots as shots_mod


def _teams(cfg):
    return cfg["game"]["away"], cfg["game"]["home"]


def _spine_at(spine: pd.DataFrame, t: float) -> pd.Series:
    i = int(np.clip(np.searchsorted(spine.t.values, t, side="right") - 1, 0, len(spine) - 1))
    return spine.iloc[i]


# --------------------------------------------------------------------------- possession
def possession(det: pd.DataFrame, spine: pd.DataFrame, cfg: dict, floor: Floor) -> pd.DataFrame:
    """Per sampled live frame: which team has the ball.

    floor   (frame has floor coordinates) the play is at the end where the skaters' median x lies; the
            team defending that end (learned per period from goalie colours in that crease) is on
            defence. Without a goalie vote, the team whose skaters sit nearer that goal defends.
    image   (no floor coordinates) a goalie standing in the visible crease defends it, so the other
            team is attacking.
    """
    A, B = _teams(cfg)
    P = det[(det.cls == "person") & det.team.isin([A, B])].copy()
    if P.empty:
        return pd.DataFrame()
    per_t = {t: _spine_at(spine, t).period for t in P.t.unique()}
    P["period"] = P.t.map(per_t)
    has_xy = P.fx.notna() if "fx" in P else pd.Series(False, index=P.index)
    # goalie votes per period and end (floor frames only)
    gk = P[has_xy & P.in_crease & (np.abs(np.abs(P.fx) - floor.gx) < floor.cr + 3)]
    votes = {}
    for (per, team), g in gk.groupby(["period", "team"]):
        for side in np.sign(g.fx):
            votes.setdefault(per, {}).setdefault((team, int(side)), 0)
            votes[per][(team, int(side))] += 1
    defends = {}
    for per, v in votes.items():
        ends = {}
        for side in (-1, 1):
            cand = {tm: n for (tm, s_), n in v.items() if s_ == side}
            if cand:
                tm = max(cand, key=cand.get)
                if cand[tm] >= 3 and cand[tm] >= 0.7 * sum(cand.values()):
                    ends[side] = tm
        if len(ends) == 1:
            side, tm = next(iter(ends.items()))
            ends[-side] = B if tm == A else A
        if len(ends) == 2 and ends[1] != ends[-1]:
            defends[per] = ends
    rows = []
    for t, g in P.groupby("t"):
        row = dict(t=t, shot_id=int(g.shot_id.iloc[0]), period=g.period.iloc[0],
                   n_A=int(((g.team == A) & ~g.in_crease).sum()), n_B=int(((g.team == B) & ~g.in_crease).sum()))
        team, method, margin, end, cx = None, None, np.nan, 0, np.nan
        xy = g[g.fx.notna()] if "fx" in g else g.iloc[:0]
        sk = xy[~xy.in_crease]
        if len(sk) >= 4:
            cx = float(np.median(sk.fx))
            end = int(np.sign(cx)) if abs(cx) > 12 else 0
            if end != 0:
                goal_x = end * floor.gx
                dA = np.abs(sk[sk.team == A].fx - goal_x).mean() if (sk.team == A).any() else np.nan
                dB = np.abs(sk[sk.team == B].fx - goal_x).mean() if (sk.team == B).any() else np.nan
                if np.isfinite(dA) and np.isfinite(dB):
                    margin = abs(dA - dB)
                ends = defends.get(row["period"])
                if ends:
                    team, method = (B if ends[end] == A else A), "floor_end"
                elif np.isfinite(margin) and (sk.team == A).sum() >= 2 and (sk.team == B).sum() >= 2:
                    team, method = (B if dA < dB else A), "floor_depth"
        if team is None:
            goalies = g[g.in_crease]
            if len(goalies) and goalies.team.nunique() == 1:
                gt = goalies.team.iloc[0]
                team, method = (B if gt == A else A), "goalie_image"
        row.update(team=team, method=method, end=end, centroid_x=round(cx, 1) if np.isfinite(cx) else np.nan,
                   margin=round(margin, 1) if np.isfinite(margin) else np.nan)
        rows.append(row)
    pos = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    sm = []
    for _, g in pos.groupby("shot_id", sort=False):
        vals = g.team.tolist()
        for i in range(len(vals)):
            win = [v for v in vals[max(0, i - 2):i + 3] if v]
            sm.append(max(set(win), key=win.count) if win else None)
    pos["team_smooth"] = sm
    # drop possession runs shorter than min_possession_s (flicker), filling from the surrounding run
    min_run = cfg["events"].get("min_possession_s", 2.0)
    for _, g in pos.groupby("shot_id", sort=False):
        idx = list(g.index)
        runs, start = [], 0
        for k in range(1, len(idx) + 1):
            if k == len(idx) or pos.at[idx[k], "team_smooth"] != pos.at[idx[start], "team_smooth"]:
                runs.append((start, k - 1))
                start = k
        for a_, b_ in runs:
            dur = pos.at[idx[b_], "t"] - pos.at[idx[a_], "t"]
            if dur < min_run and len(runs) > 1:
                fill = pos.at[idx[a_ - 1], "team_smooth"] if a_ > 0 else pos.at[idx[b_ + 1], "team_smooth"]
                for k in range(a_, b_ + 1):
                    pos.at[idx[k], "team_smooth"] = fill
    pos["game_s"] = [_spine_at(spine, t).game_s for t in pos.t]
    pos.attrs["defends"] = defends
    return pos


# --------------------------------------------------------------------------- locations
def attacker_location(det, t, team, floor: Floor, end_hint=None, window=(-1.5, 0.3)):
    g = det[(det.cls == "person") & (det.team == team) & det.fx.notna() & ~det.in_crease &
            (det.t >= t + window[0]) & (det.t <= t + window[1])]
    if g.empty:
        return np.nan, np.nan, None
    end = end_hint if end_hint else int(np.sign(np.median(g.fx))) or 1
    goal = np.array([end * floor.gx, 0.0])
    d = np.hypot(g.fx - goal[0], g.fy - goal[1])
    i = int(np.argmin(d.values))
    return float(g.fx.iloc[i]), float(g.fy.iloc[i]), end


# --------------------------------------------------------------------------- goals
def goals(sb: pd.DataFrame, shots: pd.DataFrame, audio: pd.DataFrame | None, cfg: dict) -> list[dict]:
    """Score increments -> goal events.

    The scorebug updates a few seconds after the ball goes in (often after a replay), so the goal lies
    in the live footage that last showed the old score: [last old read - back, last old read + 1 s].
    Inside that window the goal moment is the onset of the strongest sustained loudness rise (horn +
    crowd), measured relative to the window itself so it also works on compressed highlight audio."""
    A, B = _teams(cfg)
    ec, ac = cfg["events"], cfg["audio"]
    live_like = shots[shots.label.isin(["live", "closeup"])]
    hop = ac["hop_s"]
    zs = None
    if audio is not None and len(audio):
        zs = audio.z.rolling(int(round(1.0 / hop)), center=True, min_periods=1).mean().values
    out = []
    for team, col in ((A, "away_score"), (B, "home_score")):
        s = sb[col + "_clean"].values
        raw = sb[col].values
        for i in range(1, len(s)):
            if not (np.isfinite(s[i]) and np.isfinite(s[i - 1]) and s[i] > s[i - 1]):
                continue
            old, new = s[i - 1], s[i]
            upto = sb.iloc[:i + 1]
            first_new = upto[(upto[col] == new)]
            # first read of the new value in the run that got accepted
            j = i
            while j - 1 >= 0 and raw[j - 1] == new:
                j -= 1
            t_new = float(sb.t.iloc[j])
            olds = sb[(sb.t < t_new) & (sb[col] == old)]
            t_old = float(olds.t.iloc[-1]) if len(olds) else t_new
            lo, hi = t_old - ec["goal_search_back_s"], min(t_new, t_old + 1.0)
            cand = live_like[(live_like.end > lo) & (live_like.start < hi)]
            best, rise, confirmed = None, np.nan, False
            if zs is not None and not cand.empty:
                m = np.zeros(len(zs), bool)
                tt = audio.t.values
                for c in cand.itertuples():
                    m |= (tt >= max(lo, c.start)) & (tt < min(hi, c.end))
                if m.sum() >= 10:
                    idx = np.where(m)[0]
                    k = idx[np.argmax(zs[idx])]
                    base = np.median(zs[idx])
                    rise = float(zs[k] - base)
                    # onset: walk back from the peak until the curve is below half the rise
                    on = k
                    while on - 1 in set(idx) and zs[on - 1] > base + 0.5 * rise:
                        on -= 1
                    if rise >= ac.get("goal_rise", 1.0):
                        best, confirmed = float(tt[on]) - 0.5, True
            if best is None:
                sh = live_like[(live_like.start <= t_old) & (live_like.end > t_old)]
                best = float(min(sh.end.iloc[0], t_old + 1.0)) - 1.0 if not sh.empty else t_old
            for n in range(int(new - old)):  # a jump of 2 = goals the broadcast never showed
                out.append(dict(t=round(best, 2), event="goal", team=team, score_after=int(old + n + 1),
                                audio_confirmed=confirmed, audio_z=round(rise, 2) if np.isfinite(rise) else np.nan,
                                bug_last_old=t_old, bug_first_new=t_new,
                                note="" if n == int(new - old) - 1 else "earlier goal not shown on air"))
    return sorted(out, key=lambda e: e["t"])


# --------------------------------------------------------------------------- shot clock resets
def shot_clock_resets(sb: pd.DataFrame, goal_ts: list[float]) -> list[float]:
    v = sb.dropna(subset=["shot_clock_clean"])
    out = []
    for (_, g) in v.groupby("shot_id"):
        tt, sc = g.t.values, g.shot_clock_clean.values
        for i in range(1, len(sc)):
            if sc[i] >= 28 and sc[i] - sc[i - 1] >= 5 and tt[i] - tt[i - 1] <= 3:
                t = (tt[i] + tt[i - 1]) / 2
                if not any(-2 <= t - gt <= 15 for gt in goal_ts):
                    out.append(float(t))
    return out


# --------------------------------------------------------------------------- main
def run(run: Run, cfg: dict) -> pd.DataFrame:
    from . import scoreboard as sb_mod, homography as h_mod
    floor = Floor(cfg)
    A, B = _teams(cfg)
    shots = shots_mod.load(run)
    sb = sb_mod.load(run)
    spine = pd.read_csv(run.root / "spine.csv")
    det = h_mod.load(run) if (run.root / "detections_floor.csv").exists() else pd.DataFrame()
    audio = audio_mod.analyse(run, cfg)

    pos = possession(det, spine, cfg, floor) if not det.empty else pd.DataFrame()
    if not pos.empty:
        pos.to_csv(run.p("possession.csv"), index=False)
        log(f"  possession: {pos.team_smooth.notna().mean():.0%} of sampled frames assigned; "
            f"methods {pos.method.value_counts().to_dict()}; defended ends per period {pos.attrs.get('defends')}")

    def pos_team(t0, t1):
        if pos.empty:
            return None
        g = pos[(pos.t >= t0) & (pos.t <= t1)].team_smooth.dropna()
        return g.mode().iloc[0] if len(g) else None

    events = []
    gl = goals(sb, shots, audio, cfg)
    for e in gl:
        x, y, end = attacker_location(det, e["t"], e["team"], floor) if not det.empty else (np.nan, np.nan, None)
        e.update(x_ft=x, y_ft=y, attack_end=end, confidence="high" if e["audio_confirmed"] else "medium")
        events.append(e)

    for t in shot_clock_resets(sb, [e["t"] for e in gl]):
        before, after = pos_team(t - 3, t - 0.2), pos_team(t + 0.5, t + 5)
        if before and after and before == after:
            kind, team = "shot", before
        elif before and after:
            kind, team = "possession_change", after
        else:
            kind, team = "shot_clock_reset", before or after
        e = dict(t=round(t, 2), event=kind, team=team, confidence="medium" if before and after else "low",
                 note="shot clock reset; shot with offensive rebound" if kind == "shot" else
                 "shot clock reset; shot saved or turnover" if kind == "possession_change" else "shot clock reset")
        if kind == "shot" and not det.empty:
            e["x_ft"], e["y_ft"], e["attack_end"] = attacker_location(det, t, team, floor, window=(-2.0, 0.0))
        events.append(e)
        if kind == "possession_change" and before:
            events.append(dict(t=round(t - 0.1, 2), event="shot_or_turnover", team=before, confidence="low",
                               note="attack ended at shot-clock reset"))

    # possession segments + transitions
    if not pos.empty:
        p = pos.dropna(subset=["team_smooth"])
        prev = None
        for r in p.itertuples():
            if prev is not None and r.shot_id == prev.shot_id and r.team_smooth != prev.team_smooth \
                    and r.t - prev.t <= 3:
                events.append(dict(t=round((r.t + prev.t) / 2, 2), event="possession_change", team=r.team_smooth,
                                   confidence="medium", note=f"from {prev.team_smooth} (player positions)"))
            if prev is not None and r.shot_id == prev.shot_id and np.isfinite(r.centroid_x) \
                    and np.isfinite(prev.centroid_x) and np.sign(r.centroid_x) != np.sign(prev.centroid_x) \
                    and abs(r.centroid_x - prev.centroid_x) / max(r.t - prev.t, 0.1) > 12:
                events.append(dict(t=round((r.t + prev.t) / 2, 2), event="transition",
                                   team=r.team_smooth, confidence="medium",
                                   note=f"play crossed centre toward {'right' if r.centroid_x > 0 else 'left'} end "
                                        f"at {abs(r.centroid_x - prev.centroid_x) / max(r.t - prev.t, .1):.0f} ft/s"))
            prev = r

    ev = pd.DataFrame(events)
    if ev.empty:
        log("events: none found")
        ev.to_csv(run.p("events.csv"), index=False)
        return ev
    ev = ev.sort_values("t").reset_index(drop=True)
    # attach game state from the spine / scoreboard at each event
    PL = cfg["game"]["period_length_s"]
    rows = []
    for r in ev.itertuples():
        s = _spine_at(spine, r.t)
        here = shots[(shots.start <= r.t) & (shots.end > r.t)]
        sid = int(here.shot_id.iloc[0]) if not here.empty else -1
        # game clock: last clean read at/before the event in the same shot (the clock stops at a goal,
        # so this is the goal time); otherwise the spine's interpolated value
        prior = sb[(sb.t <= r.t + 0.01) & (sb.shot_id == sid)]
        last = prior.dropna(subset=["clock_clean"]).tail(1)
        if not last.empty:
            clock, period = float(last.clock_clean.iloc[0]), float(last.period_clean.iloc[0])
        else:
            clock, period = s.clock_s, s.period
        sc_last = prior.dropna(subset=["shot_clock_clean"]).tail(1)
        rows.append(dict(period=period, clock=fmt_clock(clock),
                         game_s=round((period - 1) * PL + (PL - clock), 1) if np.isfinite(clock) else np.nan,
                         shot_clock=float(sc_last.shot_clock_clean.iloc[0]) if not sc_last.empty else np.nan,
                         away_score=s.away_score, home_score=s.home_score))
    ev = pd.concat([ev, pd.DataFrame(rows)], axis=1)
    # score columns after goals should include the goal itself
    for r in ev[ev.event == "goal"].itertuples():
        col = "away_score" if r.team == A else "home_score"
        ev.loc[r.Index, col] = r.score_after
    ev.insert(1, "broadcast_time", [fmt_clock(x) + f".{int((x % 1) * 10)}" for x in ev.t])
    cols = ["t", "broadcast_time", "period", "clock", "game_s", "shot_clock", "away_score", "home_score",
            "event", "team", "x_ft", "y_ft", "attack_end", "confidence", "audio_confirmed", "audio_z", "note"]
    ev = ev[[c for c in cols if c in ev.columns]]
    ev.to_csv(run.p("events.csv"), index=False)
    log("events: " + ", ".join(f"{k}={v}" for k, v in ev.event.value_counts().items()))
    return ev


def load(run: Run) -> pd.DataFrame:
    return pd.read_csv(run.root / "events.csv")
