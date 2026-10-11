"""Stage 7 - annotated broadcast video in the HockeyNet style:

* thin mask outlines in team colour (officials not drawn)
* "#23 Connell" tags on a translucent white label, only for players whose number is confirmed
* a translucent, rounded top-down floor in the top-right corner with Kalman-smoothed player dots;
  identified players carry their number inside the dot
* optional watermark bottom-right
* with a ball track: the ball with a short fading trail, a ring under the player who has it, an arrow
  for each pass (passer -> receiver) and a line to goal with the result for each shot; on the radar the
  ball carrier's dot gets a yellow ring

Writes outputs/video/shot_XXXX.mp4 per shot and outputs/video/laxnet_render.mp4 joining them in order.
"""
from __future__ import annotations

import json
import subprocess

import cv2
import numpy as np
import pandas as pd

from .core import NLL, Run, log


def _bgr(hexstr):
    h = hexstr.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _rounded_rect(img, x0, y0, x1, y1, r, color, thickness=-1):
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
    for cx, cy, a in ((x0 + r, y0 + r, 180), (x1 - r, y0 + r, 270), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)
    if thickness > 0:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)


class Radar:
    """Top-down box-lacrosse floor drawn with OpenCV (crisp at any size)."""

    def __init__(self, floor, width):
        self.f = floor
        self.pad = 10
        self.w = width
        self.s = (width - 2 * self.pad) / floor.L
        self.h = int(floor.W * self.s + 2 * self.pad)
        self.bg = self._draw()

    def px(self, x, y):
        return (int(self.pad + (x + self.f.L / 2) * self.s), int(self.pad + (self.f.W / 2 - y) * self.s))

    def _draw(self):
        f, img = self.f, np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = (246, 246, 246)
        o = np.array([self.px(x, y) for x, y in f.outline(2.0)], np.int32)
        cv2.fillPoly(img, [o], (228, 240, 230))                    # pale turf
        line = (120, 120, 120)
        for x in (-f.rx, f.rx):
            cv2.line(img, self.px(x, -f.W / 2), self.px(x, f.W / 2), (60, 60, 200), 2, cv2.LINE_AA)
        cv2.line(img, self.px(0, -f.W / 2), self.px(0, f.W / 2), line, 1, cv2.LINE_AA)
        cv2.circle(img, self.px(0, 0), int(f.fo * self.s), line, 1, cv2.LINE_AA)
        for sgn in (-1, 1):
            c = self.px(sgn * f.gx, 0)
            r = int(f.cr * self.s)
            cv2.circle(img, c, r, (235, 200, 160), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (190, 120, 40), 1, cv2.LINE_AA)
            cv2.line(img, self.px(sgn * f.gx, -2.4), self.px(sgn * f.gx, 2.4), (40, 40, 210), 3)
            for yy in (-27.5, 27.5):
                cv2.circle(img, self.px(sgn * (f.rx + 15), yy), 2, line, -1)
        cv2.polylines(img, [o], True, (90, 90, 90), 2, cv2.LINE_AA)
        return img

    def render(self, dots):
        img = self.bg.copy()
        for x, y, col, label, *has_ball in dots:
            p = self.px(x, y)
            r = 9 if label else 6
            if has_ball and has_ball[0]:
                cv2.circle(img, p, r + 4, (0, 220, 255), 3, cv2.LINE_AA)
            cv2.circle(img, p, r, col, -1, cv2.LINE_AA)
            cv2.circle(img, p, r, (255, 255, 255), 1, cv2.LINE_AA)
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                cv2.putText(img, label, (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (255, 255, 255), 1, cv2.LINE_AA)
        return img


def _paste_panel(frame, panel, x0, y0, alpha, radius=14):
    h, w = panel.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    _rounded_rect(mask, 0, 0, w - 1, h - 1, radius, 255)
    roi = frame[y0:y0 + h, x0:x0 + w]
    blend = cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0)
    roi[mask > 0] = blend[mask > 0]


def _tag(img, text, x, y, col, scale=0.62):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x0, y0 = int(x - tw / 2 - 5), int(y - th - 8)
    x0 = max(2, min(img.shape[1] - tw - 12, x0))
    y0 = max(2, y0)
    roi = img[y0:y0 + th + 10, x0:x0 + tw + 10]
    white = np.full_like(roi, 255)
    img[y0:y0 + th + 10, x0:x0 + tw + 10] = cv2.addWeighted(white, 0.72, roi, 0.28, 0)
    cv2.putText(img, text, (x0 + 5, y0 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


class BallLayer:
    """Ball, holder, passes and shots for the render, from the ball and plays stages."""

    def __init__(self, run: Run, rc: dict):
        self.ok = False
        mode = rc.get("ball", "auto")
        if mode is False or not run.exists("ball_track.csv"):
            return
        meta = json.loads(run.p("ball_meta.json").read_text()) if run.exists("ball_meta.json") else {}
        if mode == "auto" and meta.get("detector") == "motion":
            log("  render: ball layer skipped (motion detector is a dev stand-in; set render.ball: true to draw it)")
            return
        self.ok = True
        self.bt = run.csv("ball_track.csv")
        self.bt = self.bt[self.bt.state != "none"]
        self.pf = run.csv("ball_possession_frames.csv") if run.exists("ball_possession_frames.csv") else None
        ob = run.root / "outputs" / "ball"
        self.passes = pd.read_csv(ob / "passes.csv") if (ob / "passes.csv").exists() and (ob / "passes.csv").stat().st_size > 5 else pd.DataFrame()
        self.shots = pd.read_csv(ob / "shots.csv") if (ob / "shots.csv").exists() and (ob / "shots.csv").stat().st_size > 5 else pd.DataFrame()

    def holder(self, sid, t):
        if self.pf is None:
            return -1
        g = self.pf[(self.pf.shot_id == sid) & ((self.pf.t - t).abs() < 0.02)]
        return int(g.holder_track.iloc[0]) if len(g) else -1

    def draw(self, img, sid, t, scale, boxes, tcol, team_of):
        if not self.ok:
            return -1
        h = self.holder(sid, t)
        if h >= 0 and h in boxes:
            x1, y1, x2, y2 = boxes[h]
            col = tcol.get(team_of.get(h), (255, 255, 255))
            cx, cy, rx = int((x1 + x2) / 2), int(y2), int(max(14, (x2 - x1) * 0.6))
            cv2.ellipse(img, (cx, cy), (rx, rx // 3), 0, 0, 360, (0, 230, 255), 3, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (rx, rx // 3), 0, 0, 360, col, 1, cv2.LINE_AA)
        g = self.bt[(self.bt.shot_id == sid) & (self.bt.t > t - 0.4) & (self.bt.t <= t + 0.017)].sort_values("t")
        pts = (g[["x", "y"]].values / scale).astype(int)
        for k in range(1, len(pts)):
            if np.hypot(*(pts[k] - pts[k - 1])) > 60:      # a jump between tracklets, not a ball path
                continue
            a = k / len(pts)
            cv2.line(img, tuple(pts[k - 1]), tuple(pts[k]), (0, int(200 * a) + 55, 255), max(1, int(3 * a)), cv2.LINE_AA)
        if len(g) and abs(g.t.iloc[-1] - t) < 0.05:
            p = tuple(pts[-1])
            cv2.circle(img, p, 6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, p, 5, (0, 240, 255), -1, cv2.LINE_AA)
        if len(self.passes):
            for r in self.passes[(self.passes.shot_id == sid) & (self.passes.t_release <= t) & (self.passes.t_catch + 0.6 >= t)].itertuples():
                a = (np.array(json.loads(r.from_px)) / scale).astype(int)
                b = (np.array(json.loads(r.to_px)) / scale).astype(int)
                cv2.arrowedLine(img, tuple(a), tuple(b), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.08)
                _tag(img, "PASS", (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (40, 40, 40), 0.5)
        if len(self.shots):
            for r in self.shots[(self.shots.shot_id == sid) & (self.shots.t <= t) & (self.shots.t + 1.5 >= t)].itertuples():
                a = (np.array(json.loads(r.ball_px)) / scale).astype(int)
                if isinstance(r.goal_px, str):
                    b = (np.array(json.loads(r.goal_px)) / scale).astype(int)
                    cv2.line(img, tuple(a), tuple(b), (60, 60, 255), 2, cv2.LINE_AA)
                extra = f" {r.distance_ft:.0f} ft" if hasattr(r, "distance_ft") and pd.notna(r.distance_ft) else ""
                _tag(img, f"SHOT - {r.result}{extra}", a[0], a[1] - 30, (30, 30, 200), 0.55)
        return h


def run(run: Run, cfg: dict, shots=None):
    nll = NLL(cfg)
    rc = cfg["render"]
    prep = json.loads(run.p("prep.json").read_text())
    scale, fps = prep["scale"], prep["track_fps"]
    tr = run.csv("tracks.csv")
    teams = dict(run.csv("teams.csv")[["track_id", "team"]].values) if run.exists("teams.csv") else {}
    labels = {}
    if run.exists("numbers.csv"):
        n = run.csv("numbers.csv").dropna(subset=["number"])
        for r in n.itertuples():
            num = str(r.number).replace(".0", "")
            last = str(r.name).split()[-1] if isinstance(r.name, str) and r.name else ""
            labels[r.track_id] = (num, f"#{num} {last}".strip())
    traj = run.csv("trajectories.csv") if run.exists("trajectories.csv") else pd.DataFrame()
    tcol = {k: _bgr(v) for k, v in rc.get("team_colors", {}).items()}
    for i, t in enumerate(nll.teams):
        tcol.setdefault(t, [(46, 87, 228), (222, 134, 46)][i])
    shots = shots or rc.get("shots")
    if not shots:
        cnt = traj.groupby("shot_id").size() if len(traj) else tr.groupby("shot_id").size()
        shots = list(cnt.sort_values(ascending=False).index[: rc["max_shots"]])
    shots = sorted(int(s) for s in shots)
    fr_csv = run.csv("frames.csv") if run.exists("frames.csv") else pd.DataFrame(columns=["shot_id", "frame", "t"])
    prep_t0 = fr_csv[fr_csv.frame == 0].set_index("shot_id").t.to_dict()
    radar = Radar(nll.floor, rc["radar_width"])
    ball = BallLayer(run, rc)
    made = []
    for sid in shots:
        d = run.root / "frames" / f"shot_{sid:04d}"
        cp = run.root / "contours" / f"shot_{sid:04d}.json"
        contours = json.loads(cp.read_text()) if cp.exists() else {}
        g = tr[tr.shot_id == sid]
        tj = traj[traj.shot_id == sid] if len(traj) else traj

        # Only annotate speeds from measured, non-outlier floor positions.
        speed_rows = tj
        if len(speed_rows) and {"measured", "outlier", "speed"}.issubset(speed_rows.columns):
            speed_rows = speed_rows[
                speed_rows["measured"].eq(True)
                & speed_rows["outlier"].eq(False)
                & speed_rows["speed"].between(0, 22)
            ]
        else:
            speed_rows = pd.DataFrame()

        frames = sorted(int(p.stem) for p in d.glob("*.jpg"))
        if not frames:
            continue
        H, W = cv2.imread(str(d / f"{frames[0]}.jpg")).shape[:2]
        out = run.p("outputs", "video", f"shot_{sid:04d}.mp4")
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        tmap = dict(zip(g.frame, g.t))
        for fi in frames:
            img = cv2.imread(str(d / f"{fi}.jpg"))
            tags = []
            gf = g[g.frame == fi]
            sf = speed_rows[speed_rows.frame == fi] if len(speed_rows) else speed_rows
            speed_by_track = {
                int(r.track_id): float(r.speed) * 3600 / 5280
                for r in sf.itertuples()
            }
            t_now = float(gf.t.iloc[0]) if len(gf) else tmap.get(fi, prep_t0.get(sid, 0) + fi / fps)
            boxes = {r.track_id: (r.x1 / scale, r.y1 / scale, r.x2 / scale, r.y2 / scale) for r in gf.itertuples()}
            holder = ball.draw(img, sid, t_now, scale, boxes, tcol, teams)
            for r in gf.itertuples():
                team = teams.get(r.track_id, "unknown")
                if team == "official":
                    continue
                col = tcol.get(team, (230, 230, 230))
                poly = contours.get(str(fi), {}).get(str(r.track_id))
                if poly:
                    cv2.polylines(img, [(np.array(poly) / scale).astype(np.int32)], True, col, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(img, (int(r.x1 / scale), int(r.y1 / scale)), (int(r.x2 / scale), int(r.y2 / scale)), col, 1)
                base_label = labels[r.track_id][1] if r.track_id in labels else ""
                speed_mph = speed_by_track.get(int(r.track_id))
                if speed_mph is not None:
                    tag_text = f"{base_label} | {speed_mph:.1f} mph" if base_label else f"{speed_mph:.1f} mph"
                    tags.append((tag_text, (r.x1 + r.x2) / 2 / scale, r.y1 / scale - 4, col))
                elif base_label:
                    tags.append((base_label, (r.x1 + r.x2) / 2 / scale, r.y1 / scale - 4, col))
            for text, x, y, col in tags:  # tags last so outlines never cover them
                _tag(img, text, x, y, col)
            # radar
            dots = []
            if len(tj):
                for r in tj[tj.frame == fi].itertuples():
                    team = r.team if isinstance(r.team, str) else "unknown"
                    if team == "official":
                        continue
                    dots.append((r.x, r.y, tcol.get(team, (200, 200, 200)), labels.get(r.track_id, ("",))[0],
                                 r.track_id == holder))
            panel = radar.render(dots)
            if not len(tj):
                cv2.putText(panel, "no floor calibration for this view", (radar.pad + 6, radar.h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
            ph, pw = panel.shape[:2]
            if pw < W and ph < H:
                _paste_panel(img, panel, W - pw - 12, 12, rc.get("radar_alpha", 0.82))
            if rc.get("watermark"):
                wm = rc["watermark"]
                (tw, _), _ = cv2.getTextSize(wm, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1)
                cv2.putText(img, wm, (W - tw - 12, H - 12), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (235, 235, 235), 1,
                            cv2.LINE_AA)
            ff.stdin.write(img.tobytes())
        ff.stdin.close()
        ff.wait()
        made.append(out)
        log(f"  rendered {out.name} ({len(frames)} frames)")
    if len(made) > 1:
        lst = run.p("outputs", "video", "concat.txt")
        lst.write_text("".join(f"file '{p.name}'\n" for p in made))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(run.p("outputs", "video", "laxnet_render.mp4"))], check=True)
        log("  joined -> laxnet_render.mp4")
