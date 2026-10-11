"""Stage 4 - player detection on live-play frames + team assignment by jersey colour.

Detector: YOLOX (Megvii, Apache-2.0) exported to ONNX, run through OpenCV's DNN module, so no
PyTorch is needed. With `tiles: 2` each 16:9 frame is run as two overlapping squares, which roughly
doubles the pixels per player and recovers the small far-side players.

Filtering: a person box is kept only if the floor (turf green or crease blue) is under its feet.
That drops the crowd, benches and people behind the glass.

Teams: a 15-bin jersey descriptor (12 hue bins + white/grey/dark) from the torso of every kept box,
floor pixels masked out, is clustered with k-means (k=3: two teams + officials). The cluster with
the most black-and-white stripe content is "official"; the other two are named from the configured
team colours. Nobody is identified individually.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .common import Run, iter_frames, log, resolve
from . import shots as shots_mod

PERSON, BALL = 0, 32


# --------------------------------------------------------------------------- YOLOX
class Yolox:
    def __init__(self, path, size=640, conf=0.3, nms=0.45):
        # ONNX Runtime with CoreML (Apple silicon) when available, else OpenCV DNN on CPU
        self.sess = None
        try:
            import onnxruntime as ort
            if "CoreMLExecutionProvider" in ort.get_available_providers():
                self.sess = ort.InferenceSession(str(path), providers=["CoreMLExecutionProvider",
                                                                       "CPUExecutionProvider"])
                self.inp = self.sess.get_inputs()[0].name
        except Exception:
            self.sess = None
        if self.sess is None:
            self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size, self.conf, self.nms = size, conf, nms
        grids, strides = [], []
        for s in (8, 16, 32):
            g = size // s
            xv, yv = np.meshgrid(np.arange(g), np.arange(g))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def _infer(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        rs = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        blob = pad.transpose(2, 0, 1)[None].astype(np.float32)
        if self.sess is not None:
            out = self.sess.run(None, {self.inp: blob})[0][0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()[0]
        xy = (out[:, :2] + self.grid) * self.stride
        wh = np.exp(out[:, 2:4]) * self.stride
        boxes = np.c_[xy - wh / 2, xy + wh / 2] / r
        return boxes, out[:, 4:5] * out[:, 5:]

    def __call__(self, frame, tiles=1):
        H, W = frame.shape[:2]
        if tiles == 1 or W <= H:
            regions = [(0, 0, W, H)]
        else:
            s = H
            xs = np.linspace(0, W - s, tiles).astype(int)
            regions = [(x, 0, s, s) for x in xs]
        allb, alls, allc = [], [], []
        for x0, y0, w, h in regions:
            b, sc = self._infer(frame[y0:y0 + h, x0:x0 + w])
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            for cls, thr in ((PERSON, self.conf), (BALL, 0.10)):
                m = sc[:, cls] >= thr
                allb.append(b[m]); alls.append(sc[m, cls]); allc.append(np.full(m.sum(), cls))
        b, s, c = np.concatenate(allb), np.concatenate(alls), np.concatenate(allc)
        keep = []
        for cls in (PERSON, BALL):
            idx = np.where(c == cls)[0]
            if len(idx) == 0:
                continue
            bx = b[idx]
            xywh = np.c_[bx[:, :2], bx[:, 2:] - bx[:, :2]].tolist()
            k = cv2.dnn.NMSBoxes(xywh, s[idx].tolist(), 0.0, self.nms)
            keep.extend(idx[np.array(k).ravel()].tolist())
        b, s, c = b[keep], s[keep], c[keep]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
        return b, s, c


# --------------------------------------------------------------------------- colour features
def jersey_feature(hsv, floor_mask, box):
    x1, y1, x2, y2 = box.astype(int)
    w, h = x2 - x1, y2 - y1
    ys, ye = y1 + int(0.18 * h), y1 + int(0.50 * h)
    xs, xe = x1 + int(0.22 * w), x2 - int(0.22 * w)
    if ye - ys < 3 or xe - xs < 3:
        return None
    p = hsv[ys:ye, xs:xe].reshape(-1, 3)
    m = floor_mask[ys:ye, xs:xe].reshape(-1) == 0
    p = p[m]
    if len(p) < 15:
        return None
    hh, ss, vv = p[:, 0].astype(int), p[:, 1].astype(int), p[:, 2].astype(int)
    chrom = (ss > 70) & (vv > 60)
    f = np.zeros(15, np.float32)
    np.add.at(f, (hh[chrom] * 12) // 180, 1)
    f[12] = ((ss <= 70) & (vv > 170)).sum()
    f[13] = ((ss <= 70) & (vv > 70) & (vv <= 170)).sum()
    f[14] = (vv <= 70).sum()
    return f / f.sum()


def _hex_lab(hexstr):
    h = hexstr.lstrip("#")
    rgb = np.array([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]], np.uint8)  # as BGR
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def assign_teams(det: pd.DataFrame, feats: np.ndarray, mean_bgr: np.ndarray, cfg: dict) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    ok = ~np.isnan(feats).any(1)
    X = np.sqrt(feats[ok])
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    lab = np.full(len(det), -1)
    lab[ok] = km.labels_
    d = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    centers = km.cluster_centers_ ** 2
    # officials: stripes -> strong white AND dark content, little hue
    stripe = np.minimum(centers[:, 12] + centers[:, 13], centers[:, 14]) - centers[:, :12].sum(1) * 0.5
    ref = int(np.argmax(stripe))
    teams = [k for k in range(3) if k != ref]
    lab_means = []
    for k in teams:
        bgr = mean_bgr[ok][km.labels_ == k].mean(0).astype(np.uint8)[None, None]
        lab_means.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float))
    g = cfg["game"]
    names = [g["away"], g["home"]]
    cols = g.get("colors", {})
    if all(n in cols for n in names):
        tl = [_hex_lab(cols[n]) for n in names]
        c0 = np.linalg.norm(lab_means[0] - tl[0]) + np.linalg.norm(lab_means[1] - tl[1])
        c1 = np.linalg.norm(lab_means[0] - tl[1]) + np.linalg.norm(lab_means[1] - tl[0])
        mapping = {teams[0]: names[0], teams[1]: names[1]} if c0 <= c1 else \
            {teams[0]: names[1], teams[1]: names[0]}
    else:
        mapping = {teams[0]: "team_A", teams[1]: "team_B"}
    mapping[ref] = "official"
    det = det.copy()
    det["cluster"] = lab
    det["team"] = [mapping.get(k, "unknown") for k in lab]
    # margin between nearest and second-nearest centre = assignment confidence
    dist_all = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None], axis=2)
    srt = np.sort(dist_all, 1)
    conf = np.full(len(det), np.nan)
    conf[ok] = (srt[:, 1] - srt[:, 0]) / (srt[:, 1] + 1e-6)
    det["team_conf"] = conf
    det.loc[det.cls != "person", ["team", "team_conf"]] = ["ball", np.nan]
    log("  team clusters: " + ", ".join(f"{mapping[k]}={int((lab == k).sum())}" for k in range(3)))
    return det


# --------------------------------------------------------------------------- main
def run(run: Run, cfg: dict) -> pd.DataFrame | None:
    import shutil
    import time
    dc = cfg["detect"]
    model = Yolox(resolve(dc["model"]), dc["input"], dc["conf"], dc["nms"])
    shots = shots_mod.load(run)
    live = shots[(shots.label == "live") & (shots.floor + shots.crease >= dc["min_floor_frac"])]
    log(f"detect: {len(live)} live shots, {live.duration.sum():.0f} s at {dc['fps']} fps "
        f"({'onnxruntime/CoreML' if model.sess is not None else 'OpenCV DNN'})")
    cache = run.root / "cache" / "detect"
    if cfg.get("_force") and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    t0, budget = time.time(), cfg.get("_budget")
    for s in live.itertuples():
        stem = cache / f"shot_{s.shot_id:04d}"
        if stem.with_suffix(".csv").exists():
            continue
        if budget and time.time() - t0 > budget:
            done = len(list(cache.glob("shot_*.csv")))
            log(f"detect: time budget reached, {done}/{len(live)} shots done - rerun to continue")
            return None
        rows, feats, bgrs = _detect_shot(run, cfg, model, s)
        pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
        np.save(stem.with_suffix(".feat.npy"), np.array(feats, np.float32).reshape(-1, 15))
        np.save(stem.with_suffix(".bgr.npy"), np.array(bgrs, np.float32).reshape(-1, 3))
        log(f"  shot {s.shot_id} ({s.start:.0f}-{s.end:.0f}s): {len(rows)} detections")
    parts, feats, bgrs = [], [], []
    for s in live.itertuples():
        stem = cache / f"shot_{s.shot_id:04d}"
        try:
            parts.append(pd.read_csv(stem.with_suffix(".csv")))
        except pd.errors.EmptyDataError:
            continue
        feats.append(np.load(stem.with_suffix(".feat.npy")))
        bgrs.append(np.load(stem.with_suffix(".bgr.npy")))
    det = pd.concat(parts, ignore_index=True)
    feats, bgrs = np.concatenate(feats), np.concatenate(bgrs)
    np.save(run.p("jersey_features.npy"), feats)
    det = assign_teams(det, feats, bgrs, cfg)
    det.to_csv(run.p("detections.csv"), index=False)
    _team_debug(run, det)
    log(f"detect: {len(det)} detections ({(det.cls == 'ball').sum()} ball)")
    return det


def _detect_shot(run: Run, cfg: dict, model: Yolox, s):
    dc, sc = cfg["detect"], cfg["shots"]
    rows, feats, bgrs = [], [], []
    dbg = len(list((run.root / "debug").glob("det_*.jpg"))) if (run.root / "debug").exists() else 0
    if s.end - s.start > 0.2:
        for t, fr in iter_frames(run.video, fps=dc["fps"], start=s.start + 0.05, end=s.end - 0.05):
            boxes, scores, classes = model(fr, dc["tiles"])
            hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
            floor = cv2.inRange(hsv, np.array(sc["floor_hsv_low"], np.uint8), np.array(sc["floor_hsv_high"], np.uint8))
            crease = cv2.inRange(hsv, np.array(sc["crease_hsv_low"], np.uint8), np.array(sc["crease_hsv_high"], np.uint8))
            surface = cv2.bitwise_or(floor, crease)
            H = fr.shape[0]
            kept = []
            for b, scv, c in zip(boxes, scores, classes):
                x1, y1, x2, y2 = b
                fx, fy = (x1 + x2) / 2, y2
                if c == PERSON:
                    bh = y2 - y1
                    if bh < 25:
                        continue
                    pad = max(3, int(0.08 * bh))
                    py1, py2 = int(min(H - 1, fy - pad)), int(min(H, fy + pad))
                    px1, px2 = int(x1 + 0.2 * (x2 - x1)), int(x2 - 0.2 * (x2 - x1)) + 1
                    patch = surface[py1:py2, px1:px2]
                    on_floor = patch.size > 0 and (patch > 0).mean() > 0.25
                    if not on_floor:
                        continue
                    cp = crease[py1:py2, px1:px2]
                    in_crease = cp.size > 0 and (cp > 0).mean() > 0.35
                    f = jersey_feature(hsv, floor, b)
                    x1i, y1i, x2i, y2i = b.astype(int)
                    torso = fr[y1i + int(0.18 * (y2i - y1i)):y1i + int(0.5 * (y2i - y1i)),
                               x1i + int(0.22 * (x2i - x1i)):x2i - int(0.22 * (x2i - x1i))]
                    mb = torso.reshape(-1, 3).mean(0) if torso.size else np.full(3, np.nan)
                else:
                    in_crease, f, mb = False, None, np.full(3, np.nan)
                rows.append(dict(t=round(t, 3), shot_id=s.shot_id, cls="person" if c == PERSON else "ball",
                                 conf=round(float(scv), 3), x1=round(float(x1), 1), y1=round(float(y1), 1),
                                 x2=round(float(x2), 1), y2=round(float(y2), 1), foot_x=round(float(fx), 1),
                                 foot_y=round(float(fy), 1), in_crease=bool(in_crease)))
                feats.append(f if f is not None else np.full(15, np.nan, np.float32))
                bgrs.append(mb)
                kept.append((b, c))
            if dbg < 6 and len(kept) >= 6:
                vis = fr.copy()
                for b, c in kept:
                    cv2.rectangle(vis, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)),
                                  (0, 255, 255) if c == PERSON else (0, 0, 255), 2)
                cv2.imwrite(str(run.p("debug", f"det_{t:07.1f}.jpg")), cv2.resize(vis, (960, 540)))
                dbg += 1
    return rows, feats, bgrs


def _team_debug(run: Run, det: pd.DataFrame):
    """Contact sheet of torso crops per team so the colour split can be eyeballed."""
    try:
        from .common import read_frame
        rows = []
        for team, grp in det[det.cls == "person"].groupby("team"):
            smp = grp.sample(min(16, len(grp)), random_state=0).sort_values("t")
            tiles = []
            for r in smp.itertuples():
                fr = read_frame(run.video, r.t)
                crop = fr[int(r.y1):int(r.y2), int(r.x1):int(r.x2)]
                if crop.size == 0:
                    continue
                tiles.append(cv2.resize(crop, (48, 96)))
            if tiles:
                strip = np.hstack(tiles + [np.zeros((96, 48, 3), np.uint8)] * (16 - len(tiles)))
                label = np.zeros((96, 140, 3), np.uint8)
                cv2.putText(label, team, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                rows.append(np.hstack([label, strip]))
        if rows:
            cv2.imwrite(str(run.p("debug", "teams.jpg")), np.vstack(rows))
    except Exception as e:  # debug output only
        log(f"  (team debug sheet skipped: {e})")


def load(run: Run) -> pd.DataFrame:
    return pd.read_csv(run.root / "detections.csv")
