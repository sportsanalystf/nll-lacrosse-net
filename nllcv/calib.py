"""Automatic floor calibration with a pan-tilt-zoom broadcast camera model.

The main broadcast camera sits at a fixed point C in the arena and only pans, tilts and zooms, so
every main-camera frame is described by 3 numbers (pan, tilt, focal) plus the shared C. For a
floor point (x, y, 0):

    forward f = (sin(pan) cos(tilt), cos(pan) cos(tilt), -sin(tilt))   pan 0 looks at the far boards
    right   r = (f_y, -f_x, 0) / |.|,   down d = f x r
    H_floor->img = K [r d f]^T [e1 e2 -C]  (columns for x, y and the translation)

Fitting uses a two-sided chamfer cost between the projected floor model (restraining, centre and
goal lines, crease circles, board line) and image cues (thin white floor lines, crease edges, the
floor/board boundary): model->image distances say every projected line is supported, image->model
distances stop the fit from pushing the model out of frame. A coarse grid search over pan/tilt/zoom
(and camera position) is followed by Powell refinement, jointly across all reference frames so the
camera position is shared.

    python -m nllcv calibrate --run runs/<game> --at 40 12 148 --auto
"""
from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import minimize

from .common import Run, log, read_frame, resolve
from .floor import Floor

SCALE = 0.25  # work at 480x270
_DEBUG = False


def H_from_ptz(C, pan, tilt, f, size, roll=0.0):
    w, h = size
    fx, fy = np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt)
    fwd = np.array([fx, fy, -np.sin(tilt)])
    r = np.array([fy, -fx, 0.0])
    r /= np.linalg.norm(r)
    d = np.cross(fwd, r)
    if roll:
        r, d = np.cos(roll) * r + np.sin(roll) * d, -np.sin(roll) * r + np.cos(roll) * d
    R = np.stack([r, d, fwd])
    t = -R @ np.asarray(C, float)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    H = K @ np.c_[R[:, 0], R[:, 1], t]
    return H / H[2, 2]


def model_polylines(floor: Floor, step=1.0):
    """Floor model lines that are actually painted/visible in NLL arenas."""
    L2, W2 = floor.L / 2, floor.W / 2
    ys = np.arange(-W2, W2 + step, step)
    th = np.linspace(0, 2 * np.pi, 90)
    gy = np.linspace(-floor.cr, floor.cr, 12)
    white = [np.c_[np.full_like(ys, x), ys] for x in (-floor.rx, 0.0, floor.rx)]
    white += [np.c_[np.full_like(gy, s * floor.gx), gy] for s in (-1, 1)]
    crease = [np.c_[s * floor.gx + floor.cr * np.cos(th), floor.cr * np.sin(th)] for s in (-1, 1)]
    return {"white": white, "crease": crease, "boards": [floor.outline(step)]}


def cue_points(frame, cfg, boxes=None):
    """Image cue pixels (at SCALE) for each line family, plus distance transforms."""
    from .homography import masks
    m = masks(frame, cfg, boxes)
    region = cv2.erode(m["region"], np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m["white"], 8)
    lines = np.zeros_like(m["white"])
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if max(w, h) > 80 and a < 0.3 * w * h:
            lines[lab == k] = 255
    H, W = region.shape
    # crease: solid, bright royal-blue blobs (dasher boards are darker navy; the centre logo is
    # hollowed out by its yellow lettering, so its blue fills little of its convex hull)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([100, 140, 90], np.uint8), np.array([125, 255, 255], np.uint8))
    crease = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    crease = cv2.morphologyEx(crease, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(crease, 8)
    keep = np.zeros_like(crease)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 3000 or y < 0.12 * H or w < 1.3 * h:
            continue
        comp = (lab == k).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
        # must sit on the floor: turf right below its bottom edge
        below = m["floor"][min(H - 1, y + h + 5):min(H, y + h + 25), x:x + w]
        if hull > 0 and a / hull > 0.6 and below.size and (below > 0).mean() > 0.2:
            keep[comp > 0] = 255
        elif _DEBUG:
            print("crease reject", (x, y, w, h, a), round(a / max(hull, 1), 2),
                  round(float((below > 0).mean()) if below.size else -1, 2))
    crease_e = cv2.Canny(keep, 50, 150)
    crease_e[m["player"] > 0] = 0
    # board line: the gold kick-plate strip at the base of the far boards, directly above turf
    gold = cv2.inRange(hsv, np.array([15, 90, 110], np.uint8), np.array([35, 255, 255], np.uint8))
    floor_up = np.zeros_like(m["floor"])
    floor_up[:-12] = m["floor"][12:]
    boards = cv2.bitwise_and(gold, floor_up)
    boards = cv2.bitwise_and(boards, 255 - m["floor"])
    floor_dn = np.zeros_like(m["floor"])  # turf above as well -> a floor decal, not the boards
    floor_dn[12:] = m["floor"][:-12]
    boards[floor_dn > 0] = 0
    boards[: int(0.1 * H)] = 0
    boards[m["player"] > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(boards, 8)
    bl = np.zeros_like(boards)
    for k in range(1, n):
        if st[k, 2] > 40:  # long horizontal-ish runs only
            bl[lab == k] = 255
    boards = bl
    out = {}
    for name, img in (("white", lines), ("crease", crease_e), ("boards", boards)):
        small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        small = (small > 20).astype(np.uint8) * 255
        ys, xs = np.nonzero(small)
        if len(xs) > 2500:
            sel = np.random.default_rng(0).choice(len(xs), 2500, replace=False)
            xs, ys = xs[sel], ys[sel]
        dt = cv2.distanceTransform(255 - small, cv2.DIST_L2, 3) if small.any() else None
        out[name] = (np.c_[xs, ys].astype(np.float32), dt)
        if name == "white" and small.any():
            segs = cv2.HoughLinesP(small, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=8)
            if segs is not None:
                x1, y1, x2, y2 = max(np.asarray(segs).reshape(-1, 4), key=lambda q: np.hypot(q[2] - q[0], q[3] - q[1]))
                if abs(x2 - x1) > 5:
                    # "/" (y falls as x grows) = camera looking at the left end, "\\" = right end
                    out["_end"] = -1 if (y2 - y1) / (x2 - x1) < 0 else 1
    return out


class Objective:
    def __init__(self, cues, floor: Floor, size, tau=12.0):
        self.cues, self.floor, self.size, self.tau = cues, floor, size, tau
        self.L = floor.L
        self.model = model_polylines(floor, step=1.5)
        self.weights = {"white": 1.0, "crease": 1.0, "boards": 0.6}

    def set_length(self, L):
        if abs(L - self.L) > 1e-6:
            import copy
            fl = copy.copy(self.floor)
            fl.gx = self.floor.gx + (L - self.floor.L) / 2
            fl.L = L
            self.model = model_polylines(fl, step=1.5)
            self.L = L

    def __call__(self, H):
        w, h = self.size
        tau = self.tau
        total, wsum = 0.0, 0.0
        for fam, polys in self.model.items():
            if fam.startswith("_"):
                continue
            pts, dt = self.cues.get(fam, (None, None))
            if dt is None:
                continue
            canvas = np.zeros((h, w), np.uint8)
            vis = []
            for P in polys:
                p = np.c_[P, np.ones(len(P))] @ H.T
                good = p[:, 2] > 1e-6
                uv = p[good, :2] / p[good, 2:3]
                inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                vis.append(uv[inside])
                if good.sum() > 1:
                    q = uv[(np.abs(uv) < 4 * w).all(1)].astype(np.int32)
                    if len(q) > 1:
                        cv2.polylines(canvas, [q], False, 255, 1)
            vis = np.vstack(vis) if vis else np.zeros((0, 2))
            wt = self.weights[fam]
            if len(vis):
                d = dt[vis[:, 1].astype(int), vis[:, 0].astype(int)]
                total += wt * np.minimum(d, tau).mean() * min(1.0, len(vis) / 40)
                wsum += wt * min(1.0, len(vis) / 40)
            if len(pts):
                mdt = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
                d2 = mdt[pts[:, 1].astype(int), pts[:, 0].astype(int)]
                total += wt * np.minimum(d2, tau).mean()
                wsum += wt
        return total / max(wsum, 1e-6)


def _ptz_cost(obj, C, x, roll=0.0):
    pan, tilt, f = x
    if not (0.05 < tilt < 1.2 and 150 < f < 4000):
        return 1e3
    return obj(H_from_ptz(C, pan, tilt, f, obj.size, roll))


def fit_frame(obj: Objective, W2: float):
    """Grid search + Powell for one frame: camera position (x, y, z), pan, tilt, roll, focal."""
    end = obj.cues.get("_end")
    pans = np.deg2rad(np.arange(-80, 81, 7.5))
    if end is not None:
        pans = pans[np.sign(pans) == end]
    tilts = np.deg2rad([10, 17, 25, 35])
    focals = np.array([350, 600, 950, 1500]) * (obj.size[0] / 480)
    best = (1e9, None)
    for cx in (-40.0, 0.0, 40.0):
        for dist in (20, 40, 70):
            for height in (25, 45, 70):
                C = np.array([cx, -W2 - dist, height])
                for p in pans:
                    for tl in tilts:
                        for f in focals:
                            c = _ptz_cost(obj, C, (p, tl, f))
                            if c < best[0]:
                                best = (c, np.r_[C, p, tl, 0.0, f])

    def cost(v):
        C, (p, tl, rl, f) = v[:3], v[3:]
        if C[2] < 5 or C[1] > -W2 + 2 or abs(rl) > 0.2:
            return 1e3
        return _ptz_cost(obj, C, (p, tl, f), rl)

    v = best[1]
    for tau in (12.0, 6.0):
        obj.tau = tau
        res = minimize(cost, v, method="Powell", options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
        v = res.x
    return v, float(res.fun)


def calibrate_auto(run: Run, cfg: dict, times, boxes_by_t=None):
    from .homography import References, overlay
    floor = Floor(cfg)
    W2 = floor.W / 2
    refs = References(cfg)
    refs.dir.mkdir(parents=True, exist_ok=True)
    entries = [{k: val for k, val in r.items() if k not in ("img", "kp", "des", "H")} for r in refs.refs]
    for t in times:
        frame = read_frame(run.video, t)
        W0, H0 = frame.shape[1], frame.shape[0]
        size = (int(W0 * SCALE), int(H0 * SCALE))
        obj = Objective(cue_points(frame, cfg, (boxes_by_t or {}).get(t)), floor, size)
        v, c = fit_frame(obj, W2)
        C, (pan, tilt, roll, f) = v[:3], v[3:]
        Hf2i = H_from_ptz(C, pan, tilt, f / SCALE, (W0, H0), roll)
        Hi2f = np.linalg.inv(Hf2i)
        Hi2f /= Hi2f[2, 2]
        name = f"ref_{t:07.2f}.png"
        cv2.imwrite(str(refs.dir / name), frame)
        cv2.imwrite(str(refs.dir / name.replace(".png", "_overlay.jpg")),
                    cv2.resize(overlay(frame, Hi2f, floor), (1280, 720)))
        entries = [e for e in entries if e["image"] != name]
        entries.append(dict(image=name, t=t, method="auto_ptz", camera_ft=[float(x) for x in C],
                            pan_deg=float(np.rad2deg(pan)), tilt_deg=float(np.rad2deg(tilt)),
                            roll_deg=float(np.rad2deg(roll)), focal_px=float(f / SCALE),
                            cost_px=c / SCALE, H_img2floor=Hi2f.tolist()))
        refs.save(entries)
        log(f"  ref t={t}: camera ({C[0]:.0f}, {C[1]:.0f}, {C[2]:.0f}) ft, pan {np.rad2deg(pan):.0f}, "
            f"tilt {np.rad2deg(tilt):.0f}, roll {np.rad2deg(roll):.1f}, f {f / SCALE:.0f}px, "
            f"residual {c / SCALE:.1f}px")
    return entries


def refine_two_sided(H_img2floor, frame, cfg, floor: Floor, boxes=None):
    """Polish a homography (e.g. from clicked landmarks) with the two-sided chamfer objective.
    Works on the 8 free entries of H_floor->img at 1/4 resolution."""
    obj = Objective(cue_points(frame, cfg, boxes), floor, (int(frame.shape[1] * SCALE), int(frame.shape[0] * SCALE)))
    S = np.diag([SCALE, SCALE, 1.0])
    Hf = S @ np.linalg.inv(H_img2floor)
    Hf /= Hf[2, 2]
    c0 = obj(Hf)
    # parameterise as a perturbation of the start so all 8 parameters are well scaled
    # four image points in the lower (floor) part of the frame; their floor positions are the parameters
    w, h = obj.size
    corners = np.float32([[0.1 * w, 0.45 * h], [0.9 * w, 0.45 * h], [0.9 * w, 0.95 * h], [0.1 * w, 0.95 * h]])
    base = cv2.perspectiveTransform(corners[None], np.linalg.inv(Hf))[0]

    def H_of(v):
        moved = base + v.reshape(4, 2)
        Hn = cv2.getPerspectiveTransform(np.float32(moved), corners)
        return Hn

    v = np.zeros(8)
    for tau in (12.0, 6.0, 4.0):
        obj.tau = tau
        res = minimize(lambda q: obj(H_of(q)), v, method="Powell",
                       options={"maxiter": 4000, "xtol": 1e-2, "ftol": 1e-4})
        v = res.x
    obj.tau = 6.0
    Hn = H_of(v)
    c0, c1 = obj(Hf), obj(Hn)
    if c1 > c0:
        return H_img2floor, c0 / SCALE, c0 / SCALE
    Hi = np.linalg.inv(np.linalg.inv(S) @ Hn)
    return Hi / Hi[2, 2], c0 / SCALE, c1 / SCALE


# --------------------------------------------------------------------------- fixed-camera PTZ fits
def decompose(H_floor2img, size):
    """Recover camera centre C (ft), focal (px), pan/tilt/roll from a floor->image homography,
    assuming square pixels and a centred principal point."""
    w, h = size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    G = T @ H_floor2img
    g1, g2 = G[:, 0], G[:, 1]
    # orthogonality of r1, r2 under K = diag(f, f, 1):  (g1x g2x + g1y g2y)/f^2 + g1z g2z = 0
    num = -(g1[0] * g2[0] + g1[1] * g2[1])
    den = g1[2] * g2[2]
    f2 = num / den if abs(den) > 1e-12 else np.nan
    if not np.isfinite(f2) or f2 <= 0:  # fall back to equal-norm constraint
        f2 = (g2[0] ** 2 + g2[1] ** 2 - g1[0] ** 2 - g1[1] ** 2) / (g1[2] ** 2 - g2[2] ** 2)
    f = float(np.sqrt(abs(f2)))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    a1, a2, a3 = Kinv @ G[:, 0], Kinv @ G[:, 1], Kinv @ G[:, 2]
    lam = 2.0 / (np.linalg.norm(a1) + np.linalg.norm(a2))
    r1, r2 = a1 * lam, a2 * lam
    t = a3 * lam
    if t[2] < 0:  # camera must have the floor in front of it
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], 1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    if C[2] < 0:  # mirror solution
        R[:, :2] *= -1
        C = -R.T @ t
    fwd = R[2]          # camera z axis in world coords
    pan = np.arctan2(fwd[0], fwd[1])
    tilt = np.arcsin(-fwd[2])
    r_nom = np.array([np.cos(pan), -np.sin(pan), 0.0])
    roll = np.arctan2(np.dot(np.cross(r_nom, R[0]), fwd), np.dot(r_nom, R[0]))
    return C, f, pan, tilt, roll


def fit_ptz_fixed_camera(obj: Objective, C, f_ref, tilt_ref):
    """Coarse-to-fine: a wide truncation makes the grid cost smooth, then anneal it down."""
    end = obj.cues.get("_end")
    pans = np.deg2rad(np.arange(-80, 81, 2.0))
    if end is not None:
        pans = pans[np.sign(pans) == end]
    tilts = tilt_ref + np.deg2rad(np.arange(-9, 13, 3))
    fs = f_ref * obj.size[0] / 1920 * np.geomspace(0.45, 2.6, 12)
    obj.tau = 40.0
    scored = []
    for p in pans:
        for tl in tilts:
            for f in fs:
                scored.append((_ptz_cost(obj, C, (p, tl, f)), p, tl, f))
    scored.sort(key=lambda r: r[0])

    def cost(v):
        p, tl, rl, f = v
        if abs(rl) > 0.15:
            return 1e3
        return _ptz_cost(obj, C, (p, tl, f), rl)

    best = (1e9, None)
    for c0, p, tl, f in scored[:4]:  # refine the best few grid cells
        v = np.array([p, tl, 0.0, f])
        for tau in (25.0, 12.0, 6.0):
            obj.tau = tau
            v = minimize(cost, v, method="Powell", options={"maxiter": 1500, "xtol": 1e-4, "ftol": 1e-5}).x
        obj.tau = 6.0
        c = cost(v)
        if c < best[0]:
            best = (c, v)
    return best[1], float(best[0]), float(scored[0][0])


def calibrate_from_anchor(run: Run, cfg: dict, times, boxes_by_t=None, anchor=None):
    """Use an accurately calibrated reference (e.g. from landmarks) to fix the main camera position,
    then fit pan/tilt/roll/zoom for each new reference frame."""
    from .homography import References, overlay
    floor = Floor(cfg)
    refs = References(cfg)
    if not refs.refs:
        raise SystemExit("no anchor reference yet - calibrate one frame with landmarks first")
    a = next((r for r in refs.refs if anchor is None or abs(r["t"] - anchor) < 1e-3), refs.refs[0])
    H0 = np.linalg.inv(a["H"])
    size0 = (a["img"].shape[1], a["img"].shape[0])
    C, f0, pan0, tilt0, roll0 = decompose(H0 / H0[2, 2], size0)
    log(f"  anchor t={a['t']}: camera ({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f}) ft, pan {np.rad2deg(pan0):.1f}, "
        f"tilt {np.rad2deg(tilt0):.1f}, roll {np.rad2deg(roll0):.1f}, f {f0:.0f}px")
    entries = [{k: val for k, val in r.items() if k not in ("img", "kp", "des", "H")} for r in refs.refs]
    for t in times:
        frame = read_frame(run.video, t)
        W0, H0_ = frame.shape[1], frame.shape[0]
        size = (int(W0 * SCALE), int(H0_ * SCALE))
        obj = Objective(cue_points(frame, cfg, (boxes_by_t or {}).get(t)), floor, size)
        (pan, tilt, roll, f), c, cg = fit_ptz_fixed_camera(obj, C, f0, tilt0)
        Cf = np.array(C, float)
        # polish: let the camera position move a little (one anchor frame pins it only approximately)

        def full(v):
            dc, (p, tl, rl, ff) = v[:3], v[3:]
            if np.abs(dc).max() > 15 or abs(rl) > 0.15:
                return 1e3
            return _ptz_cost(obj, C + dc, (p, tl, ff), rl)

        v = np.r_[0.0, 0.0, 0.0, pan, tilt, roll, f]
        for tau in (8.0, 5.0):
            obj.tau = tau
            v = minimize(full, v, method="Powell", options={"maxiter": 3000, "xtol": 1e-4, "ftol": 1e-6}).x
        obj.tau = 6.0
        c2 = full(v)
        if c2 < c:
            Cf, (pan, tilt, roll, f), c = C + v[:3], v[3:], c2
        Hf2i = H_from_ptz(Cf, pan, tilt, f / SCALE, (W0, H0_), roll)
        Hi2f = np.linalg.inv(Hf2i)
        Hi2f /= Hi2f[2, 2]
        name = f"ref_{t:07.2f}.png"
        cv2.imwrite(str(refs.dir / name), frame)
        cv2.imwrite(str(refs.dir / name.replace(".png", "_overlay.jpg")),
                    cv2.resize(overlay(frame, Hi2f, floor), (1280, 720)))
        entries = [e for e in entries if e["image"] != name]
        entries.append(dict(image=name, t=t, method="ptz_fixed_camera", anchor_t=a["t"],
                            camera_ft=[float(x) for x in Cf], pan_deg=float(np.rad2deg(pan)),
                            tilt_deg=float(np.rad2deg(tilt)), roll_deg=float(np.rad2deg(roll)),
                            focal_px=float(f / SCALE), cost_px=c / SCALE, H_img2floor=Hi2f.tolist()))
        refs.save(entries)
        log(f"  ref t={t}: camera ({Cf[0]:.1f}, {Cf[1]:.1f}, {Cf[2]:.1f}), pan {np.rad2deg(pan):.1f}, tilt {np.rad2deg(tilt):.1f}, roll {np.rad2deg(roll):.1f}, "
            f"f {f / SCALE:.0f}px, residual {c / SCALE:.1f}px (grid {cg / SCALE:.1f})")
    return entries


def joint_refine(run: Run, cfg: dict, times, boxes_by_t=None, anchor=None):
    """Jointly refine the shared camera position with (a) the anchor reference's clicked landmarks
    (reprojection error) and (b) chamfer alignment on the other references. Each frame keeps its own
    pan/tilt/roll/zoom. Rewrites the H of every reference involved."""
    from .homography import References, overlay
    floor = Floor(cfg)
    refs = References(cfg)
    by_t = {round(r["t"], 2): r for r in refs.refs}
    a = by_t[round(anchor, 2)] if anchor is not None else next(r for r in refs.refs if "points" in r)
    lm = dict(floor.landmarks())
    for n in a["points"]:
        if n.startswith("@"):
            lm[n] = tuple(float(v) for v in n[1:].split(","))
    names = [n for n in a["points"] if n in lm]
    P_floor = np.array([lm[n] for n in names], float)
    P_img = np.array([a["points"][n] for n in names], float) * SCALE
    size_a = (int(a["img"].shape[1] * SCALE), int(a["img"].shape[0] * SCALE))
    Ha = np.linalg.inv(a["H"])
    C, fa, pa, ta, ra = decompose(Ha / Ha[2, 2], (a["img"].shape[1], a["img"].shape[0]))
    others = [t for t in times if round(t, 2) in by_t and round(t, 2) != round(a["t"], 2)]
    objs, init = {}, []
    for t in others:
        r = by_t[round(t, 2)]
        frame = r["img"]
        objs[t] = Objective(cue_points(frame, cfg, (boxes_by_t or {}).get(t)), floor,
                            (int(frame.shape[1] * SCALE), int(frame.shape[0] * SCALE)))
        Hi = np.linalg.inv(r["H"])
        _, f_, p_, t_, r_ = decompose(Hi / Hi[2, 2], (frame.shape[1], frame.shape[0]))
        init.append([p_, t_, r_, f_ * SCALE])
    v0 = np.r_[C, pa, ta, ra, fa * SCALE, np.ravel(init)]

    def unpack(v):
        return v[:3], v[3:7], {t: v[7 + 4 * i:11 + 4 * i] for i, t in enumerate(others)}

    def cost(v):
        Cc, (p, tl, rl, f), per = unpack(v)
        if Cc[2] < 5 or Cc[1] > -floor.W / 2:
            return 1e3
        H = H_from_ptz(Cc, p, tl, f, size_a, rl)
        q = np.c_[P_floor, np.ones(len(P_floor))] @ H.T
        reproj = np.linalg.norm(q[:, :2] / q[:, 2:3] - P_img, axis=1).mean()
        ch = [_ptz_cost(objs[t], Cc, (pp[0], pp[1], pp[3]), pp[2]) for t, pp in per.items()]
        return reproj + (np.mean(ch) if ch else 0.0)

    c0 = cost(v0)
    v = v0
    for tau in (12.0, 6.0):
        for o in objs.values():
            o.tau = tau
        v = minimize(cost, v, method="Powell", options={"maxiter": 6000, "xtol": 1e-4, "ftol": 1e-6}).x
    c1 = cost(v)
    Cc, pa_, per = unpack(v)
    log(f"  joint: camera ({C[0]:.1f},{C[1]:.1f},{C[2]:.1f}) -> ({Cc[0]:.1f},{Cc[1]:.1f},{Cc[2]:.1f}) ft, "
        f"cost {c0 / SCALE:.1f} -> {c1 / SCALE:.1f}px; anchor roll {np.rad2deg(ra):.1f} -> {np.rad2deg(pa_[2]):.1f}")
    entries = [{k: val for k, val in r.items() if k not in ("img", "kp", "des", "H")} for r in refs.refs]
    upd = {round(a["t"], 2): (a["img"], pa_)}
    upd.update({round(t, 2): (by_t[round(t, 2)]["img"], pp) for t, pp in per.items()})
    for e in entries:
        k = round(e["t"], 2)
        if k not in upd:
            continue
        img, (p, tl, rl, f) = upd[k]
        Hf2i = H_from_ptz(Cc, p, tl, f / SCALE, (img.shape[1], img.shape[0]), rl)
        Hi2f = np.linalg.inv(Hf2i)
        e["H_img2floor"] = (Hi2f / Hi2f[2, 2]).tolist()
        e.update(camera_ft=[float(x) for x in Cc], pan_deg=float(np.rad2deg(p)), tilt_deg=float(np.rad2deg(tl)),
                 roll_deg=float(np.rad2deg(rl)), focal_px=float(f / SCALE), joint_refined=True)
        cv2.imwrite(str(refs.dir / e["image"].replace(".png", "_overlay.jpg")),
                    cv2.resize(overlay(img, Hi2f / Hi2f[2, 2], floor), (1280, 720)))
    refs.save(entries)
    return Cc
