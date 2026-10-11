"""Stage 5 - floor homography: image pixels -> floor feet.

Per arena, a handful of reference frames from the main camera are calibrated once: floor landmarks
(crease points, restraining-line/board intersections, centre spot ...) are clicked, a homography is
fitted, then refined by aligning the projected floor model (restraining lines, face-off circle,
crease circles, board line) to the white-line / crease-edge / board-edge pixels in the image
(chamfer alignment).

Every live frame is then registered to the best-matching reference with ORB features restricted to
the floor (players masked out); NLL floors carry many painted sponsor decals, which give plenty of
stable features. H_frame = H_ref @ H_frame->ref. Frames that match no reference fall back to
frame-to-frame chaining inside the same shot. A final chamfer refinement tightens each frame.

Calibration lives in calibration/<arena>/refs.json (+ the reference images) and is reused for
every game in that arena.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from .common import Run, iter_frames, log, read_frame, resolve
from .floor import Floor
from . import shots as shots_mod


# --------------------------------------------------------------------------- image cues
def masks(frame: np.ndarray, cfg: dict, boxes=None):
    sc = cfg["shots"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    floor = cv2.inRange(hsv, np.array(sc["floor_hsv_low"], np.uint8), np.array(sc["floor_hsv_high"], np.uint8))
    crease = cv2.inRange(hsv, np.array(sc["crease_hsv_low"], np.uint8), np.array(sc["crease_hsv_high"], np.uint8))
    surface = cv2.bitwise_or(floor, crease)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    region = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, k)
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    white = cv2.inRange(hsv, np.array([0, 0, 165], np.uint8), np.array([180, 70, 255], np.uint8))
    white = cv2.bitwise_and(white, cv2.erode(region, np.ones((9, 9), np.uint8)))
    player = np.zeros(floor.shape, np.uint8)
    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(player, (int(x1 - 0.15 * w), int(y1 - 0.05 * h)), (int(x2 + 0.15 * w), int(y2 + 0.08 * h)), 255, -1)
    white[player > 0] = 0
    return dict(floor=floor, crease=crease, region=region, white=white, player=player)


def cue_maps(frame, cfg, boxes=None):
    """Distance transforms of the three line families the floor model is aligned to."""
    m = masks(frame, cfg, boxes)
    # thin painted lines only: remove big white blobs (decals, logos) by keeping elongated components
    n, lab, st, _ = cv2.connectedComponentsWithStats(m["white"], 8)
    lines = np.zeros_like(m["white"])
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if max(w, h) > 60 and a < 0.35 * w * h:
            lines[lab == k] = 255
    crease_edge = cv2.Canny(cv2.medianBlur(m["crease"], 5), 50, 150)
    crease_edge[m["player"] > 0] = 0
    board_edge = cv2.Canny(m["region"], 50, 150)
    board_edge[m["player"] > 0] = 0
    out = {}
    for name, e in (("white", lines), ("crease", crease_edge), ("boards", board_edge)):
        out[name] = cv2.distanceTransform(255 - e, cv2.DIST_L2, 3) if e.any() else None
    return out, m


# --------------------------------------------------------------------------- chamfer refinement
def _project(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def refine(H_img2floor, frame, cfg, floor: Floor, boxes=None, tau=20.0, iters=2):
    """Refine a homography by chamfer-aligning the projected floor model to image line cues.
    Returns (H, cost_before, cost_after); cost is the mean truncated distance in pixels."""
    dts, _ = cue_maps(frame, cfg, boxes)
    curves = floor.curves(step=1.0)
    h, w = frame.shape[:2]
    weights = {"white": 1.0, "crease": 1.0, "boards": 0.5}
    Hf2i = np.linalg.inv(H_img2floor)
    Hf2i = Hf2i / Hf2i[2, 2]

    def pts_for(Hm):
        sel = []
        for fam, polys in curves.items():
            if dts.get(fam) is None:
                continue
            P = np.vstack(polys)
            uv = _project(Hm, P)
            ok = (uv[:, 0] > 5) & (uv[:, 0] < w - 5) & (uv[:, 1] > 5) & (uv[:, 1] < h - 5)
            if ok.sum() >= 10:
                sel.append((fam, P[ok]))
        return sel

    def cost(Hm, sel):
        r = []
        for fam, P in sel:
            uv = _project(Hm, P)
            d = map_coordinates(dts[fam], [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=tau)
            r.append(weights[fam] * np.minimum(d, tau))
        return np.concatenate(r) if r else np.zeros(1)

    sel = pts_for(Hf2i)
    if not sel:
        return H_img2floor, np.nan, np.nan
    c0 = float(cost(Hf2i, sel).mean())
    x = Hf2i.ravel()[:8]
    for _ in range(iters):
        sel = pts_for(np.r_[x, 1].reshape(3, 3))
        if not sel:
            break
        res = least_squares(lambda v: cost(np.r_[v, 1].reshape(3, 3), sel), x, loss="soft_l1", f_scale=4.0,
                            x_scale="jac", max_nfev=150)
        x = res.x
    Hn = np.r_[x, 1].reshape(3, 3)
    sel = pts_for(Hn)
    c1 = float(cost(Hn, sel).mean()) if sel else np.nan
    if not np.isfinite(c1) or c1 > c0:
        return H_img2floor, c0, c0
    Hi = np.linalg.inv(Hn)
    return Hi / Hi[2, 2], c0, c1


# --------------------------------------------------------------------------- references
class References:
    def __init__(self, cfg: dict):
        self.dir = resolve(cfg["homography"]["refs_dir"])
        self.cfg = cfg
        self.orb = cv2.ORB_create(cfg["homography"]["orb_features"], fastThreshold=10)
        self.refs = []
        path = self.dir / "refs.json"
        if path.exists():
            for r in json.loads(path.read_text()):
                img = cv2.imread(str(self.dir / r["image"]))
                if img is None:
                    continue
                kp, des = self._features(img)
                self.refs.append(dict(r, img=img, kp=kp, des=des, H=np.array(r["H_img2floor"])))
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __len__(self):
        return len(self.refs)

    def _features(self, img, boxes=None):
        m = masks(img, self.cfg, boxes)
        mask = cv2.bitwise_and(m["region"], 255 - m["player"])
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, mask)
        return kp, des

    def match(self, kp1, des1, kp2, des2):
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, 0
        good = []
        for pair in self.bf.knnMatch(des1, des2, k=2):
            if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 8:
            return None, 0
        a = np.float32([kp1[m.queryIdx].pt for m in good])
        b = np.float32([kp2[m.trainIdx].pt for m in good])
        H, inl = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
        if H is None:
            return None, 0
        return H, int(inl.sum())

    def register(self, frame, boxes=None):
        kp, des = self._features(frame, boxes)
        best = (None, 0, None)
        for i, r in enumerate(self.refs):
            H, n = self.match(kp, des, r["kp"], r["des"])
            if H is not None and n > best[1]:
                best = (r["H"] @ H, n, i)
        return best[0], best[1], best[2], (kp, des)

    def save(self, entries):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "refs.json").write_text(json.dumps(entries, indent=1, default=float))


def fit_points(points: dict, floor: Floor):
    """points: {landmark_name or '@x,y' (floor feet): [u, v] pixel}"""
    lm = dict(floor.landmarks())
    for n in points:
        if n.startswith("@"):
            x, y = (float(v) for v in n[1:].split(","))
            lm[n] = (x, y)
    names = [n for n in points if n in lm]
    if len(names) < 4:
        raise ValueError(f"need >=4 known landmarks, got {names}")
    src = np.float32([points[n] for n in names])
    dst = np.float32([lm[n] for n in names])
    H, _ = cv2.findHomography(src, dst, 0)
    err = np.linalg.norm(_project(H, src) - dst, axis=1)
    return H, dict(zip(names, err.round(2).tolist()))


def add_reference(run: Run, cfg: dict, t: float, points: dict, do_refine=True, boxes=None):
    """Calibrate one reference frame from clicked/typed landmark pixels (at native resolution)."""
    floor = Floor(cfg)
    refs = References(cfg)
    frame = read_frame(run.video, t)
    H, err = fit_points(points, floor)
    c0 = c1 = np.nan
    if do_refine:
        from .calib import refine_two_sided
        H, c0, c1 = refine_two_sided(H, frame, cfg, floor, boxes)
    name = f"ref_{t:07.2f}.png"
    refs.dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(refs.dir / name), frame)
    entries = [{k: v for k, v in r.items() if k not in ("img", "kp", "des", "H")} for r in refs.refs]
    entries = [e for e in entries if e["image"] != name]
    entries.append(dict(image=name, t=t, points=points, fit_error_ft=err, H_img2floor=H.tolist(),
                        chamfer_px=[c0, c1]))
    refs.save(entries)
    cv2.imwrite(str(refs.dir / name.replace(".png", "_overlay.jpg")), overlay(frame, H, floor))
    log(f"reference {name}: landmark fit error (ft) {err}; chamfer {c0:.1f} -> {c1:.1f} px")
    return H


def calibrate_interactive(run: Run, cfg: dict, times):
    """Click landmarks on reference frames. In the window: left-click a point, then type its
    landmark name (or number) in the terminal. Press 'q' in the window when done with a frame."""
    floor = Floor(cfg)
    names = list(floor.landmarks())
    print("\nLandmarks:\n" + "\n".join(f"  {i:2d}  {n}" for i, n in enumerate(names)))
    for t in times:
        frame = read_frame(run.video, t)
        pts, clicks = {}, []
        win = f"t={t:.1f}s  click a landmark, name it in the terminal, q to finish"
        disp = frame.copy()

        def on_mouse(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicks.append((x, y))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            cv2.imshow(win, disp)
            k = cv2.waitKey(30) & 0xFF
            if clicks:
                x, y = clicks.pop(0)
                ans = input(f"({x},{y}) -> landmark # or name (blank = skip): ").strip()
                n = names[int(ans)] if ans.isdigit() and int(ans) < len(names) else ans
                if n in names:
                    pts[n] = [float(x), float(y)]
                    cv2.circle(disp, (x, y), 6, (0, 0, 255), 2)
                    cv2.putText(disp, n, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if k == ord("q"):
                break
        cv2.destroyWindow(win)
        if len(pts) >= 4:
            add_reference(run, cfg, t, pts)
        else:
            log(f"t={t}: fewer than 4 landmarks, skipped")


# --------------------------------------------------------------------------- drawing
def overlay(frame, H_img2floor, floor: Floor, color=(0, 0, 255)):
    vis = frame.copy()
    Hf2i = np.linalg.inv(H_img2floor)
    h, w = frame.shape[:2]
    for fam, polys in floor.curves(step=0.5).items():
        for P in polys:
            uv = _project(Hf2i, P)
            ok = (np.abs(uv) < 5 * max(h, w)).all(1)
            pts = uv[ok].astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, color if fam != "boards" else (0, 200, 255), 2)
    return vis


# --------------------------------------------------------------------------- PTZ tracking
def _main_camera(refs: "References"):
    """Camera(s) from the landmark-calibrated anchor references (pan-tilt-zoom model). One anchor per
    end/camera is allowed; the tracker picks whichever explains a shot's first frame best."""
    from .calib import decompose
    anchors = [r for r in refs.refs if "points" in r] or refs.refs
    cams = []
    for a in anchors:
        Hf = np.linalg.inv(a["H"])
        C, f, pan, tilt, roll = decompose(Hf / Hf[2, 2], (a["img"].shape[1], a["img"].shape[0]))
        cams.append(dict(C=C, f=f, tilt=tilt, t=a["t"]))
    return cams or None


def _track_shot_ptz(run, cfg, floor, s, want, times, cams):
    """Fit pan/tilt/roll/zoom on every other sampled frame of a shot (grid at the first frame or after
    a loss, local refinement from the previous fit otherwise); interpolate the frames in between."""
    from scipy.optimize import minimize
    from .calib import Objective, cue_points, SCALE, H_from_ptz, _ptz_cost, fit_ptz_fixed_camera
    hc, dc = cfg["homography"], cfg["detect"]
    max_cost = hc.get("max_line_cost_px", 24.0)
    C = cams[0]["C"]
    fitted = {}  # t -> (pan, tilt, roll, f_quarter, cost_px, how)
    state, last_grid, n_grid = None, -1e9, 0
    frames = [(round(t, 3), fr) for t, fr in iter_frames(run.video, fps=dc["fps"], start=s.start + 0.05,
                                                              end=s.end - 0.05)]
    frames = [(t, fr) for t, fr in frames if t in times]
    size = None
    for i, (t, fr) in enumerate(frames):
        if i % 2 and i != len(frames) - 1:
            continue  # interpolated later
        boxes = want.get_group(t)[["x1", "y1", "x2", "y2"]].values
        size = (int(fr.shape[1] * SCALE), int(fr.shape[0] * SCALE))
        obj = Objective(cue_points(fr, cfg, boxes), floor, size, tau=12.0)
        v, how = None, None
        if state is not None:
            def cost(q):
                return 1e3 if abs(q[2]) > 0.15 else _ptz_cost(obj, C, (q[0], q[1], q[3]), q[2])
            q = state
            for tau in (12.0, 6.0):
                obj.tau = tau
                q = minimize(cost, q, method="Powell", options={"maxiter": 400, "xtol": 1e-4, "ftol": 1e-5}).x
            v, how = q, "track"
        elif t - last_grid >= 3.0 and n_grid < 2:
            best = None
            for cm in cams:
                vv, cc, _ = fit_ptz_fixed_camera(obj, cm["C"], cm["f"], cm["tilt"])
                if best is None or cc < best[1]:
                    best = (vv, cc, cm["C"])
            v, C = best[0], best[2]
            how, last_grid, n_grid = "grid", t, n_grid + 1
        if v is None:
            continue
        obj.tau = 12.0
        c = _ptz_cost(obj, C, (v[0], v[1], v[3]), v[2]) / SCALE
        if c <= max_cost:
            fitted[t] = (*v, c, how, tuple(np.round(C, 4)))
            state = np.asarray(v, float)
        else:
            state = None
    # interpolate skipped frames between two fits less than 1 s apart
    ft = sorted(fitted)
    for t, _ in frames:
        if t in fitted or len(ft) < 2:
            continue
        j = np.searchsorted(ft, t)
        if 0 < j < len(ft) and ft[j] - ft[j - 1] <= 1.0 and fitted[ft[j - 1]][6] == fitted[ft[j]][6]:
            a, b = fitted[ft[j - 1]], fitted[ft[j]]
            w = (t - ft[j - 1]) / (ft[j] - ft[j - 1])
            fitted[t] = (*[a[k] * (1 - w) + b[k] * w for k in range(4)], max(a[4], b[4]), "interp", a[6])
    Ts, Hs, Q = [], [], []
    W, Hh = (int(round(size[0] / SCALE)), int(round(size[1] / SCALE))) if size else (1920, 1080)
    for t in sorted(fitted):
        p, tl, rl, f, c, how, Ct = fitted[t]
        Hf2i = H_from_ptz(np.array(Ct), p, tl, f / SCALE, (W, Hh), rl)
        Hi = np.linalg.inv(Hf2i)
        Ts.append(t); Hs.append(Hi / Hi[2, 2]); Q.append((how, c))
    # a couple of debug overlays per shot
    dbg = [t for t in Ts[:: max(1, len(Ts) // 2)]][:2]
    for t, fr in frames:
        if t in dbg:
            vis = cv2.resize(overlay(fr, Hs[Ts.index(t)], floor), (960, 540))
            cv2.putText(vis, f"t={t:.1f} {fitted[t][5]} line={fitted[t][4]:.1f}px", (10, 530),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(run.p("debug", f"homography_{t:07.1f}.jpg")), vis)
    return Ts, Hs, Q


# --------------------------------------------------------------------------- main
def run(run: Run, cfg: dict) -> pd.DataFrame:
    from . import detect as detect_mod
    floor = Floor(cfg)
    refs = References(cfg)
    det = detect_mod.load(run)
    if len(refs) == 0:
        log(f"homography: no references in {refs.dir} - run `python -m nllcv calibrate --at <t>` first. "
            "Writing detections without floor coordinates.")
        det["fx"] = det["fy"] = np.nan
        det.to_csv(run.p("detections_floor.csv"), index=False)
        return det
    import shutil
    import time
    shots = shots_mod.load(run)
    dc, hc = cfg["detect"], cfg["homography"]
    want = det.groupby("t")
    times = set(want.groups)
    cache = run.root / "cache" / "homography"
    if cfg.get("_force") and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    t0, budget = time.time(), cfg.get("_budget")
    n_dbg = len(list((run.root / "debug").glob("homography_*.jpg"))) if (run.root / "debug").exists() else 0
    cam = _main_camera(refs) if hc.get("method", "ptz") == "ptz" else None
    if cam is not None:
        log("homography: PTZ tracking; camera(s) " + ", ".join(
            f"({c['C'][0]:.0f}, {c['C'][1]:.0f}, {c['C'][2]:.0f}) ft from anchor t={c['t']}" for c in cam))
    for sid in sorted(det.shot_id.unique()):
        out = cache / f"shot_{sid:04d}.npz"
        if out.exists():
            continue
        if budget and time.time() - t0 > budget:
            log(f"homography: time budget reached - rerun to continue")
            return None
        s = shots[shots.shot_id == sid].iloc[0]
        if cam is not None:
            Ts, Hs, Q = _track_shot_ptz(run, cfg, floor, s, want, times, cam)
            np.savez_compressed(out, t=np.array(Ts, float), H=np.array(Hs, float).reshape(-1, 3, 3),
                                inliers=np.zeros(len(Ts), int), source=np.array([q[0] for q in Q], str),
                                chamfer=np.array([q[1] for q in Q], float))
            log(f"  shot {sid}: {len(Ts)}/{sum(1 for x in times if s.start <= x < s.end)} frames registered")
            continue
        prev = None  # (H, kp, des) of the last registered frame in this shot
        Ts, Hs, Q = [], [], []
        for t, fr in iter_frames(run.video, fps=dc["fps"], start=s.start + 0.05, end=s.end - 0.05):
            t = round(t, 3)
            if t not in times:
                continue
            boxes = want.get_group(t)[["x1", "y1", "x2", "y2"]].values
            from .calib import Objective, cue_points, SCALE
            obj = Objective(cue_points(fr, cfg, boxes), floor,
                            (int(fr.shape[1] * SCALE), int(fr.shape[0] * SCALE)), tau=12.0)

            def line_check(Hc):
                # projected floor lines must sit on the image's painted lines / crease / boards; ORB alone
                # aliases onto repeated board ads or the mirrored decals at the other end
                Hf = np.diag([SCALE, SCALE, 1.0]) @ np.linalg.inv(Hc)
                return obj(Hf / Hf[2, 2]) / SCALE

            max_cost = hc.get("max_line_cost_px", 24.0)
            H, n, ref_i, (kp, des) = refs.register(fr, boxes)
            src, line_cost = f"ref{ref_i}", np.nan
            if H is not None and n >= hc["min_inliers"]:
                line_cost = line_check(H)
            if not (np.isfinite(line_cost) and line_cost <= max_cost):
                H, line_cost = None, np.nan
                if prev is not None:  # chain from the last verified frame in this shot (small motion)
                    Hp, n2 = refs.match(kp, des, prev[1], prev[2])
                    if Hp is not None and n2 >= hc["min_inliers"]:
                        lc = line_check(prev[0] @ Hp)
                        if lc <= max_cost:
                            H, n, src, line_cost = prev[0] @ Hp, n2, "chain", lc
            if H is not None and hc.get("refine", False):
                H, _, _ = refine(H, fr, cfg, floor, boxes, iters=1)
            if H is not None:
                # sanity: the lower image should map onto (roughly) the floor
                corners = _project(H, np.float32([[0, fr.shape[0]], [fr.shape[1], fr.shape[0]],
                                                  [fr.shape[1] / 2, fr.shape[0] * 0.6]]))
                if not floor.inside(corners[:, 0], corners[:, 1], margin=60).all():
                    H = None
            if H is not None:
                prev = (H, kp, des)
                Ts.append(t); Hs.append(H); Q.append((n, src, line_cost))
                if n_dbg < 10 and len(Ts) % 12 == 1:
                    vis = cv2.resize(overlay(fr, H, floor), (960, 540))
                    cv2.putText(vis, f"t={t:.1f} {src} inl={n} line={line_cost:.1f}px", (10, 530),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imwrite(str(run.p("debug", f"homography_{t:07.1f}.jpg")), vis)
                    n_dbg += 1
        np.savez_compressed(out, t=np.array(Ts, float), H=np.array(Hs, float).reshape(-1, 3, 3),
                            inliers=np.array([q[0] for q in Q], int), source=np.array([q[1] for q in Q], str),
                            chamfer=np.array([q[2] for q in Q], float))
        log(f"  shot {sid}: {len(Ts)}/{sum(1 for x in times if s.start <= x < s.end)} frames registered")
    parts = [dict(np.load(f)) for f in sorted(cache.glob("shot_*.npz"))]
    max_cost = hc.get("max_line_cost_px", 18.0)
    for p in parts:  # the acceptance threshold is applied here, so changing it needs no recompute
        keep = ~(p["chamfer"] > max_cost)
        for k in ("t", "H", "inliers", "source", "chamfer"):
            p[k] = p[k][keep]
    Ts = np.concatenate([p["t"] for p in parts]) if parts else np.zeros(0)
    Hs = np.concatenate([p["H"] for p in parts]) if parts else np.zeros((0, 3, 3))
    np.savez_compressed(run.p("homographies.npz"), t=Ts, H=Hs,
                        source=np.concatenate([p["source"] for p in parts]) if parts else np.zeros(0, str),
                        line_cost_px=np.concatenate([p["chamfer"] for p in parts]) if parts else np.zeros(0))
    Ts = [round(float(x), 3) for x in Ts]
    Hmap = dict(zip(Ts, Hs))
    fx, fy = np.full(len(det), np.nan), np.full(len(det), np.nan)
    for i, r in enumerate(det.itertuples()):
        H = Hmap.get(round(r.t, 3))
        if H is not None:
            x, y = _project(H, np.array([[r.foot_x, r.foot_y]]))[0]
            if floor.inside(x, y, margin=4):
                fx[i], fy[i] = x, y
    det["fx"], det["fy"] = np.round(fx, 2), np.round(fy, 2)
    det.to_csv(run.p("detections_floor.csv"), index=False)
    log(f"homography: {len(Ts)}/{len(times)} frames registered; "
        f"{np.isfinite(fx).mean():.0%} of detections on the floor map")
    return det


def load(run: Run) -> pd.DataFrame:
    return pd.read_csv(run.root / "detections_floor.csv")
