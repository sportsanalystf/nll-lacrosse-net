"""Stage 6 - analytics and the LLM-readable game report.

outputs/heatmap_<team>.png   where each team's skaters were, normalised so the team attacks right
outputs/player_stats.csv     per identified player (or anonymous track): time, distance, speeds, zones
outputs/team_stats.csv       per team: spacing, depth, offensive-half share
outputs/goals_on_floor.csv   who was on the floor in the seconds before each goal
outputs/game_report.json     everything above in one file, with definitions, for a chat assistant
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .core import NLL, Run, log

TEAM_COLORS = ["#e4572e", "#2e86de"]


def _attach_period(traj, spine):
    i = np.clip(np.searchsorted(spine.t.values, traj.t.values, side="right") - 1, 0, len(spine) - 1)
    traj["period"] = spine.period.values[i]
    traj["game_s"] = spine.game_s.values[i]
    return traj


def normalise(traj, nll: NLL):
    """Mirror positions so every team attacks the right-hand (+x) goal."""
    ends = nll.defended_ends()
    traj["xn"], traj["yn"], traj["dir_known"] = traj.x, traj.y, False
    for (per, team), g in traj.groupby(["period", "team"]):
        d = ends.get(int(per)) if np.isfinite(per) else None
        if not d:
            continue
        if team in d.values():
            own_end = [e for e, tm in d.items() if tm == team][0]   # end this team defends
        else:
            own_end = -list(d.keys())[0]                          # the opponent defends the other end
        flip = own_end > 0                                       # defends right -> attacks left -> mirror
        idx = g.index
        traj.loc[idx, "xn"] = -g.x if flip else g.x
        traj.loc[idx, "yn"] = -g.y if flip else g.y
        traj.loc[idx, "dir_known"] = True
    return traj


def player_stats(traj):
    rows = []
    traj = traj.copy()
    traj["who"] = traj.player.fillna(traj.team.astype(str) + " track " + traj.track_id.astype(str))
    for who, g in traj.groupby("who"):
        g = g.sort_values("t")
        dt = g.groupby(["track_id", "segment"]).t.diff().fillna(0).clip(upper=0.5)
        rows.append(dict(player=who, team=g.team.iloc[0], number=g.number.iloc[0], name=g.name.iloc[0] if "name" in g else None,
                         tracks=g.track_id.nunique(), seconds=round(float(dt.sum()), 1),
                         distance_ft=round(float((g.speed * dt).sum()), 1),
                         avg_speed_fps=round(float(g.speed.mean()), 2),
                         top_speed_fps=round(float(g.speed.quantile(0.95)), 2),
                         top_speed_mph=round(float(g.speed.quantile(0.95)) * 3600 / 5280, 1),
                         mean_x_attacking=round(float(g.xn[g.dir_known].mean()), 1) if g.dir_known.any() else None,
                         offensive_half_share=round(float((g.xn[g.dir_known] > 0).mean()), 2) if g.dir_known.any() else None,
                         goalie=bool(g.in_crease.mean() > 0.5)))
    return pd.DataFrame(rows).sort_values(["team", "seconds"], ascending=[True, False])


def team_stats(traj):
    rows = []
    sk = traj[~traj.in_crease]
    for team, g in sk.groupby("team"):
        spacing, depth = [], []
        for _, f in g.groupby("t"):
            if len(f) >= 3:
                p = f[["x", "y"]].values
                d = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))
                spacing.append(d[np.triu_indices(len(p), 1)].mean())
                depth.append(np.ptp(f.xn.values))
        rows.append(dict(team=team, frames=g.t.nunique(), skater_frames=len(g),
                         mean_spacing_ft=round(float(np.mean(spacing)), 1) if spacing else None,
                         mean_depth_ft=round(float(np.mean(depth)), 1) if depth else None,
                         offensive_half_share=round(float((g.xn[g.dir_known] > 0).mean()), 2) if g.dir_known.any() else None))
    return pd.DataFrame(rows)


def goals_on_floor(traj, tracks, events, window=3.0):
    rows = []
    if events is None:
        return pd.DataFrame()
    for e in events[events.event == "goal"].itertuples():
        near = tracks[(tracks.t >= e.t - window) & (tracks.t <= e.t + 0.5)]
        ids = near.track_id.unique()
        names = traj[traj.track_id.isin(ids)].drop_duplicates("track_id")
        listed = sorted(set(names.player.dropna()))
        rows.append(dict(t=e.t, period=e.period, clock=e.clock, scoring_team=e.team,
                         tracks_visible=len(ids), identified=", ".join(listed),
                         unidentified=int(len(ids) - len(listed))))
    return pd.DataFrame(rows)


def heatmaps(run: Run, traj, nll: NLL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fl = nll.floor
    for i, team in enumerate(nll.teams):
        g = traj[(traj.team == team) & ~traj.in_crease & traj.dir_known]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        fl.draw(ax, face="#2b2b2b")
        if len(g):
            ax.hexbin(g.xn, g.yn, gridsize=(30, 11), extent=(-fl.L / 2, fl.L / 2, -fl.W / 2, fl.W / 2),
                      cmap="Oranges" if i == 0 else "Blues", mincnt=1, alpha=0.85, zorder=3)
        ax.set_title(f"{team} skaters, attacking right ({len(g)} smoothed positions)", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(run.p("outputs", f"heatmap_{team}.png"), dpi=130)
        plt.close(fig)


ADV_DEFS = dict(
    top_speed_mph="95th percentile of smoothed speed, in mph (broadcast-tracking estimate, +-10-15%)",
    best_1s_speed_mph="fastest average speed over any 1-second stretch",
    high_speed_distance_ft="distance covered above 14 ft/s (~9.5 mph); sprint_distance_ft above 20 ft/s (~13.6 mph)",
    stints="identified players' sightings joined when less than 25 s apart (camera-limited, experimental)",
    transitions="team centroid moving from its defensive zone to its offensive zone: seconds and centroid speed",
    widest_seam_ft="largest lateral gap between neighbouring defenders in their own zone",
    slot_to_nearest_ft="distance from the slot (15 ft in front of goal) to the nearest defender",
    matchups="each frame, attackers are paired with defenders by optimal assignment; time and mean separation",
    picks="heuristic: a near-stationary attacker that a teammate brushes past, after which the teammate's "
          "defender falls 3+ ft further behind within 1 s",
    space_control="share of the visible floor each team would reach first (0.3 s drift + 20 ft/s)",
    shot_gap_ft="the open gap in the defence around the shooter at release (angular gap between the nearest "
                "defenders on either side, in feet at the shooter's distance)",
    nearest_defender_ft="distance from the shooter (or ball carrier) to the closest defender",
    defenders_in_lane="defenders within 3 ft of the line from shooter to goal",
    carrier_pressure="for each ball carrier: mean distance to the nearest defender and share of time within 5 ft",
)


def _load_tables(run: Run, sub, names):
    out = {}
    for n in names:
        p = run.root / "outputs" / sub / f"{n}.csv"
        if p.exists() and p.stat().st_size > 5:
            try:
                out[n] = pd.read_csv(p).to_dict("records")
            except pd.errors.EmptyDataError:
                out[n] = []
        else:
            out[n] = []
    return out


def run(run: Run, cfg: dict):
    nll = NLL(cfg)
    traj = run.csv("trajectories.csv")
    tracks = run.csv("tracks.csv")
    if traj.empty:
        log("analytics: no trajectories with floor coordinates")
        return
    traj = _attach_period(traj, nll.spine)
    traj = normalise(traj, nll)
    traj = traj[traj.team != "official"]
    ps, ts = player_stats(traj), team_stats(traj)
    gf = goals_on_floor(traj, tracks.merge(run.csv("teams.csv"), on="track_id"), nll.events)
    ps.to_csv(run.p("outputs", "player_stats.csv"), index=False)
    ts.to_csv(run.p("outputs", "team_stats.csv"), index=False)
    gf.to_csv(run.p("outputs", "goals_on_floor.csv"), index=False)
    heatmaps(run, traj, nll)
    sb = nll.table("scoreboard.csv")
    last = sb.dropna(subset=["away_score_clean"]).iloc[-1] if sb is not None else None
    ev = nll.events
    report = dict(
        about="LacrosseNet output for one NLL broadcast. Positions are in feet from the centre spot: x along "
              "the floor (-100..100), y across it (-42.5..42.5). 'Attacking right' figures are mirrored so the "
              "team always attacks +x. Times are broadcast seconds; game clock where known.",
        teams=dict(away=nll.teams[0], home=nll.teams[1]),
        final_score_seen=dict(away=int(last.away_score_clean), home=int(last.home_score_clean),
                              period=int(last.period_clean)) if last is not None else None,
        coverage=dict(tracks=int(tracks.track_id.nunique()), trajectories=int(traj.groupby(["track_id", "segment"]).ngroups),
                      track_frames_with_floor_xy=float(round(traj.measured.sum() / max(len(tracks), 1), 3)),
                      identified_players=int(ps.number.notna().sum())),
        definitions=dict(distance_ft="sum of Kalman-smoothed speed x time while tracked on the floor",
                         top_speed_fps="95th percentile of smoothed speed (ft/s)",
                         offensive_half_share="share of tracked time in the half the team attacks",
                         mean_spacing_ft="mean distance between a team's tracked skaters in the same frame",
                         mean_depth_ft="mean front-to-back spread of a team's tracked skaters"),
        team_stats=ts.to_dict("records"), players=ps.to_dict("records"),
        goals=gf.to_dict("records"),
        events=ev[["t", "period", "clock", "event", "team", "x_ft", "y_ft", "note"]].to_dict("records") if ev is not None else [],
    )
    report["definitions"].update(ADV_DEFS)
    report["advanced"] = _load_tables(run, "advanced", ["player_motion", "stints", "transitions", "defense_shape",
                                                        "matchups", "picks", "space_control"])
    meta = json.loads(run.p("ball_meta.json").read_text()) if run.exists("ball_meta.json") else None
    if meta:
        ball = _load_tables(run, "ball", ["possessions", "passes", "turnovers", "shots", "carrier_pressure",
                                          "pass_network_edges"])
        ball["detector"] = meta["detector"]
        ball["reliable"] = meta["detector"] != "motion"
        if not ball["reliable"]:
            ball["note"] = ("ball positions come from the untrained motion detector (a development stand-in); "
                            "passes, shots and possession here are not trustworthy until the RF-DETR ball "
                            "detector is trained or SAM 3 is used")
        report["ball"] = ball
    run.save_json("outputs/game_report.json", report)
    log(f"analytics: {len(ps)} players/tracks, {ps.number.notna().sum()} identified, "
        f"{len(gf)} goals with on-floor lists")
