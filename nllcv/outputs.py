"""Stage 7 - deliverables in <run>/outputs/:

    clips/goal_XX_*.mp4    goal highlights with pre-roll (cut accurately, re-encoded)
    shot_map.png           goals + detected shots per team, both teams attacking right
    possession_map.png     where each team's skaters stand while it has the ball (attacking right)
    possession_timeline.png
    events.csv             clock-indexed event table
    report.html            one page that ties it together
"""
from __future__ import annotations

import html
import subprocess

import numpy as np
import pandas as pd

from .common import Run, fmt_clock, log
from .floor import Floor
from . import shots as shots_mod

TEAM_COLORS = ["#e4572e", "#2e86de"]  # away, home (plot colours, not jersey colours)


def _clip_window(t, shots, pre, post):
    """Pre-roll stops at the start of the run of live footage that contains the goal, so a clip
    never opens on the previous replay or graphic."""
    start = t - pre
    live = shots[shots.label.isin(["live", "closeup"])].sort_values("start")
    cur = shots[(shots.start <= t) & (shots.end > t)]
    if not cur.empty and cur.label.iloc[0] in ("live", "closeup"):
        s0 = float(cur.start.iloc[0])
        # walk back over contiguous live/closeup shots
        prev = live[live.end <= s0 + 0.01].sort_values("end")
        while not prev.empty and abs(prev.end.iloc[-1] - s0) < 0.2:
            s0 = float(prev.start.iloc[-1])
            prev = prev.iloc[:-1]
        start = max(start, s0)
    return max(0.0, start), t + post


def clips(run: Run, cfg: dict, ev: pd.DataFrame, shots: pd.DataFrame):
    g = ev[ev.event == "goal"]
    # one clip per goal moment; goals the broadcast never showed (score jumps) share a moment
    g = g.sort_values("t").copy()
    g["moment"] = (g.t.diff().fillna(1e9) > 2.0).cumsum()  # goals < 2 s apart = one on-air moment
    teams_at = g.groupby("moment").team.agg(lambda x: "+".join(sorted(set(x))))
    g = g.groupby("moment").tail(1).reset_index(drop=True)
    g["away_score"] = [ev[(ev.event == "goal") & (ev.t <= t)].away_score.max() for t in g.t]
    g["home_score"] = [ev[(ev.event == "goal") & (ev.t <= t)].home_score.max() for t in g.t]
    g["team"] = g.moment.map(teams_at)
    out = []
    A, B = cfg["game"]["away"], cfg["game"]["home"]
    for i, r in g.iterrows():
        a, b = _clip_window(r.t, shots, cfg["events"]["preroll_s"], cfg["events"]["postroll_s"])
        clock = str(r.clock).replace(":", "-") if isinstance(r.clock, str) and r.clock else "na"
        per = int(r.period) if pd.notna(r.period) else 0
        name = f"goal_{i + 1:02d}_P{per}_{clock}_{r.team}_{A}{int(r.away_score)}-{B}{int(r.home_score)}.mp4"
        path = run.p("outputs", "clips", name)
        out.append(dict(goal=i + 1, file=name, start=round(a, 2), end=round(b, 2), team=r.team,
                        audio_confirmed=bool(r.audio_confirmed)))
        if path.exists() and path.stat().st_size > 0 and not cfg.get("_force"):
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-i", str(run.video), "-t", f"{b - a:.2f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "128k",
                        "-movflags", "+faststart", str(path)], check=True)
    pd.DataFrame(out).to_csv(run.p("outputs", "clips", "clips.csv"), index=False)
    keep = {o["file"] for o in out}
    for f in run.p("outputs", "clips", "x").parent.glob("goal_*.mp4"):  # clips from an older event table
        if f.name not in keep:
            f.unlink()
    log(f"  {len(out)} goal clips")
    return out


def _normalise(x, y, end):
    """Mirror so the attacking team always shoots at the right-hand goal."""
    if end is None or not np.isfinite(end) or end > 0:
        return x, y
    return -x, -y


def shot_map(run: Run, cfg: dict, ev: pd.DataFrame, floor: Floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A, B = cfg["game"]["away"], cfg["game"]["home"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.5))
    for ax, team, col in zip(axes, (A, B), TEAM_COLORS):
        floor.draw(ax)
        e = ev[(ev.team == team) & ev.event.isin(["goal", "shot"]) & ev.x_ft.notna()]
        for kind, marker, size in (("shot", "o", 70), ("goal", "*", 260)):
            k = e[e.event == kind]
            if len(k):
                xy = np.array([_normalise(r.x_ft, r.y_ft, r.attack_end) for r in k.itertuples()])
                ax.scatter(xy[:, 0], xy[:, 1], s=size, marker=marker, c=col if kind == "goal" else "none",
                           edgecolors=col if kind == "shot" else "white", linewidths=1.6 if kind == "shot" else 0.8,
                           zorder=5, label=f"{kind}s ({len(k)})")
        n_goals = int(((ev.team == team) & (ev.event == "goal")).sum())
        ax.set_title(f"{team} - attacking right   goals: {n_goals}, located: {int((e.event == 'goal').sum())}",
                     fontsize=12, loc="left")
        ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    fig.text(0.01, 0.005, "Location = attacking player nearest the goal in the 1.5 s before the event "
                          "(estimate; ball not tracked).", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(run.p("outputs", "shot_map.png"), dpi=130)
    plt.close(fig)


def possession_map(run: Run, cfg: dict, floor: Floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pp = run.root / "possession.csv"
    if not pp.exists():
        return False
    pos = pd.read_csv(pp)
    det = pd.read_csv(run.root / "detections_floor.csv")
    A, B = cfg["game"]["away"], cfg["game"]["home"]
    lab = pos.dropna(subset=["team_smooth"])[["t", "team_smooth", "end"]]
    d = det[(det.cls == "person") & det.fx.notna()].merge(lab, on="t")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9.5))
    for ax, team, cmap in zip(axes, (A, B), ("Oranges", "Blues")):
        floor.draw(ax, face="#2b2b2b")
        g = d[(d.team_smooth == team) & (d.team == team) & (d.end != 0)]
        if len(g):
            x = np.where(g.end > 0, g.fx, -g.fx)
            y = np.where(g.end > 0, g.fy, -g.fy)
            ax.hexbin(x, y, gridsize=(28, 10), extent=(-floor.L / 2, floor.L / 2, -floor.W / 2, floor.W / 2),
                      cmap=cmap, mincnt=1, alpha=0.85, zorder=3)
        share = (lab.team_smooth == team).mean() if len(lab) else np.nan
        ax.set_title(f"{team} skaters while {team} has the ball - attacking right "
                     f"(share of sampled live frames: {share:.0%})", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(run.p("outputs", "possession_map.png"), dpi=130)
    plt.close(fig)

    # timeline
    spine = pd.read_csv(run.root / "spine.csv")
    fig, ax = plt.subplots(figsize=(12, 2.6))
    shots = shots_mod.load(run)
    shade = {"live": "#e9f5ec", "replay": "#fff3cd", "closeup": "#eef", "graphics": "#eee", "crowd": "#f8e1e1",
             "studio": "#e1ecf8"}
    for s in shots.itertuples():
        ax.axvspan(s.start, s.end, color=shade.get(s.label, "#fff"), lw=0)
    for team, col, yv in ((A, TEAM_COLORS[0], 1), (B, TEAM_COLORS[1], 0)):
        g = lab[lab.team_smooth == team]
        ax.scatter(g.t, np.full(len(g), yv), s=9, c=col, marker="|")
    ev = pd.read_csv(run.p("events.csv"))
    for r in ev[ev.event == "goal"].itertuples():
        ax.axvline(r.t, color=TEAM_COLORS[0] if r.team == A else TEAM_COLORS[1], lw=1.4, ls="--")
    ax.set_yticks([0, 1], [B, A])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, spine.t.max())
    ax.set_xlabel("broadcast time (s)   background: live / replay / close-up / graphics / crowd / studio")
    ax.set_title("Possession by sampled live frame; dashed = goals", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(run.p("outputs", "possession_timeline.png"), dpi=130)
    plt.close(fig)
    return True


def report(run: Run, cfg: dict, ev: pd.DataFrame, clip_rows, shots: pd.DataFrame):
    A, B = cfg["game"]["away"], cfg["game"]["home"]
    sb = pd.read_csv(run.root / "scoreboard.csv")
    last = sb.dropna(subset=["away_score_clean"]).iloc[-1]
    seg = shots.groupby("label").duration.agg(["count", "sum"])
    seg_html = "".join(f"<tr><td>{html.escape(k)}</td><td>{int(v['count'])}</td><td>{v['sum'] / 60:.1f}</td></tr>"
                       for k, v in seg.iterrows())
    show = ev.drop(columns=[c for c in ("game_s", "audio_z") if c in ev.columns])
    table = show.to_html(index=False, na_rep="", float_format=lambda v: f"{v:.1f}", classes="ev")
    clips_html = "".join(f'<li><a href="clips/{html.escape(c["file"])}">{html.escape(c["file"])}</a></li>'
                         for c in clip_rows)
    imgs = "".join(f'<figure><img src="{f}"></figure>' for f in ("shot_map.png", "possession_map.png",
                                                                  "possession_timeline.png")
                   if (run.root / "outputs" / f).exists())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{A} at {B} - nllcv</title><style>
body{{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0 auto;max-width:1100px;padding:16px;color:#1d1d1f;background:#fff}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#666;margin-bottom:18px}} h2{{font-size:16px;margin:26px 0 8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}} td,th{{border-bottom:1px solid #e5e5e5;padding:4px 6px;text-align:left;white-space:nowrap}}
th{{background:#f5f5f7;position:sticky;top:0}} .wrap{{overflow-x:auto;max-height:520px}} img{{max-width:100%}} figure{{margin:10px 0}}
.k{{display:inline-block;margin-right:22px}} .k b{{font-size:20px;display:block}}</style></head><body>
<h1>{A} {int(last.away_score_clean)} - {int(last.home_score_clean)} {B}</h1>
<div class="sub">Source: {html.escape(str(run.manifest.get('url') or run.manifest.get('file', '')))}</div>
<div><span class="k"><b>{int((ev.event == 'goal').sum())}</b>goals detected</span>
<span class="k"><b>{int(ev[ev.event == 'goal'].get('audio_confirmed', pd.Series(dtype=bool)).map(lambda v: bool(v) if pd.notna(v) else False).sum())}</b>audio-confirmed</span>
<span class="k"><b>{int((ev.event == 'shot').sum())}</b>shots (shot-clock)</span>
<span class="k"><b>{int((ev.event == 'possession_change').sum())}</b>possession changes</span></div>
<h2>Broadcast segmentation</h2><table style="width:auto"><tr><th>segment</th><th>shots</th><th>minutes</th></tr>{seg_html}</table>
<h2>Goal clips</h2><ol>{clips_html}</ol>
<h2>Maps</h2>{imgs}
<h2>Event table</h2><div class="wrap">{table}</div>
</body></html>"""
    run.p("outputs", "report.html").write_text(page)


def run(run: Run, cfg: dict):
    floor = Floor(cfg)
    ev = pd.read_csv(run.root / "events.csv")
    shots = shots_mod.load(run)
    ev.to_csv(run.p("outputs", "events.csv"), index=False)
    clip_rows = clips(run, cfg, ev, shots)
    shot_map(run, cfg, ev, floor)
    possession_map(run, cfg, floor)
    report(run, cfg, ev, clip_rows, shots)
    log(f"outputs -> {run.root / 'outputs'}")
