"""Stage 2 - detect and track every person per live shot.

Backends write the same tables, so everything downstream is backend-agnostic:

    tracks.csv     one row per (track, frame): box, foot point, mask area, prompt, on-floor flag
    contours/      per shot: simplified mask outlines for rendering (SAM 3 only)
    crops/         per track: torso crops (background greyed out with the mask) for teams + numbers

sam3  Meta's SAM 3 video predictor, text-prompted ("lacrosse player", "referee"); detects, segments
      and tracks in one model (as in HockeyNet). Needs a CUDA GPU; see slurm/.
cpu   YOLOX person detector + constant-velocity Kalman / IoU (SORT-style) tracker. Runs anywhere;
      used to develop and test the rest of the pipeline.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pandas as pd

from .core import NLL, Run, log, resolve


# --------------------------------------------------------------------------- shared helpers
def floor_masks(img, ncfg):
    sc = ncfg["shots"]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    floor = cv2.inRange(hsv, np.array(sc["floor_hsv_low"], np.uint8), np.array(sc["floor_hsv_high"], np.uint8))
    crease = cv2.inRange(hsv, np.array(sc["crease_hsv_low"], np.uint8), np.array(sc["crease_hsv_high"], np.uint8))
    return cv2.bitwise_or(floor, crease), crease


def foot_on_floor(surface, crease, box):
    x1, y1, x2, y2 = box
    H = surface.shape[0]
    h = y2 - y1
    pad = max(2, int(0.08 * h))
    py1, py2 = int(min(H - 1, y2 - pad)), int(min(H, y2 + pad))
    px1, px2 = int(x1 + 0.2 * (x2 - x1)), int(x2 - 0.2 * (x2 - x1)) + 1
    patch = surface[py1:py2, max(0, px1):px2]
    cp = crease[py1:py2, max(0, px1):px2]
    on = patch.size > 0 and (patch > 0).mean() > 0.25
    inc = cp.size > 0 and (cp > 0).mean() > 0.35
    return bool(on), bool(inc)


def torso_crop(img, box, mask=None, top=0.12, bottom=0.62):
    """Upper body crop (numbers are on the chest/back), background greyed out when a mask exists."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h = y2 - y1
    ys, ye = max(0, y1 + int(top * h)), min(img.shape[0], y1 + int(bottom * h))
    xs, xe = max(0, x1), min(img.shape[1], x2)
    if ye - ys < 8 or xe - xs < 6:
        return None
    crop = img[ys:ye, xs:xe].copy()
    if mask is not None:
        m = mask[ys:ye, xs:xe]
        if m.shape[:2] == crop.shape[:2] and m.any():
            crop[~m] = (crop[~m] * 0.25 + 96).astype(np.uint8)
    return crop


def turf_mask(surface, box, shape):
    """Player silhouette for the CPU backend: everything inside the box that is not turf/crease.
    On a green floor this gives an outline close to a segmentation mask."""
    H, W = shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    sub = (surface[y1:y2, x1:x2] == 0).astype(np.uint8)
    sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 0.12 * sub.size:
        return None
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = lab == k
    return m


def mask_contour(m, scale):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    big = max(cs, key=cv2.contourArea)
    return (cv2.approxPolyDP(big, 1.2, True)[:, 0] * scale).round().astype(int).tolist()


class CropWriter:
    """Saves at most `per_s` crops per track per second."""

    def __init__(self, run: Run, per_s: float, fps: float):
        self.run, self.every = run, max(1, int(round(fps / max(per_s, 0.1))))
        self.count = {}

    def maybe(self, track_id, frame_idx, img, box, mask=None):
        k = self.count.get(track_id, 0)
        self.count[track_id] = k + 1
        if k % self.every:
            return None
        c = torso_crop(img, box, mask)
        if c is None:
            return None
        p = self.run.p("crops", f"{track_id}", f"{frame_idx}.jpg")
        cv2.imwrite(str(p), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return p.name


def finalize(run: Run, cfg: dict, rows: list[dict], fps: float):
    tc = cfg["track"]
    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(run.p("tracks.csv"), index=False)
        return df
    g = df.groupby("track_id")
    keep = (g.frame.transform("count") / fps >= tc["min_track_s"]) & (g.on_floor.transform("mean") >= tc["min_on_floor"])
    dropped = df.loc[~keep, "track_id"].nunique()
    df = df[keep].copy()
    df.to_csv(run.p("tracks.csv"), index=False)
    log(f"track: {df.track_id.nunique()} tracks kept ({dropped} dropped: too short or off the floor), "
        f"{len(df)} track-frames")
    return df


# --------------------------------------------------------------------------- SAM 3 backend
def _sam3_outputs(out, W, H, scale):
    """SAM 3 per-frame outputs -> list of (obj_id, prob, box_xyxy_src, mask_frame)."""
    ids = np.asarray(out.get("out_obj_ids", []))
    if not len(ids):
        return []
    probs = np.asarray(out.get("out_probs", np.ones(len(ids))))
    masks = np.asarray(out.get("out_binary_masks"))
    boxes = np.asarray(out.get("out_boxes_xywh", np.zeros((len(ids), 4))), float)
    res = []
    for k, oid in enumerate(ids.tolist()):
        m = masks[k].astype(bool) if masks is not None and len(masks) > k else None
        if m is not None and m.any():
            ys, xs = np.nonzero(m)
            bx = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
        else:
            x, y, w, h = boxes[k]
            if max(x, y, w, h) <= 1.5:  # relative coordinates
                x, y, w, h = x * W, y * H, w * W, h * H
            bx = np.array([x, y, x + w, y + h])
        res.append((int(oid), float(probs[k]), bx * scale, m))
    return res


def run_sam3(run: Run, cfg: dict, only=None):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("SAM 3 needs a CUDA GPU. Run this stage on Rivanna (slurm/laxnet_gpu.slurm) "
                         "or use --backend cpu for a CPU stand-in.")
    from sam3.model_builder import build_sam3_video_predictor
    sc = cfg["track"]["sam3"]
    nll = NLL(cfg)
    ncfg = nll.cfg
    prep = json.loads(run.p("prep.json").read_text())
    scale, fps = prep["scale"], prep["track_fps"]
    frames = run.csv("frames.csv")
    if only:
        frames = frames[frames.shot_id.isin(only)]
    kwargs = {}
    if sc.get("version", "sam3") != "sam3":
        from sam3.model_builder import download_ckpt_from_hf
        kwargs["checkpoint_path"] = download_ckpt_from_hf(version=sc["version"])
    predictor = build_sam3_video_predictor(**kwargs)
    crops = CropWriter(run, cfg["track"]["crops_per_s"], fps)
    rows = []
    for sid, fr in frames.groupby("shot_id"):
        d = run.root / "frames" / f"shot_{sid:04d}"
        t0 = time.time()
        tmap = dict(zip(fr.frame, fr.t))
        contours = {}
        sess = predictor.handle_request(request=dict(type="start_session", resource_path=str(d)))["session_id"]
        for pi, prompt in enumerate(sc["prompts"]):
            if pi:
                predictor.handle_request(request=dict(type="reset_session", session_id=sess))
            predictor.handle_request(request=dict(type="add_prompt", session_id=sess, frame_index=0, text=prompt))
            for resp in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=sess)):
                fi = int(resp["frame_index"])
                img = cv2.imread(str(d / f"{fi}.jpg"))
                if img is None:
                    continue
                Hh, Ww = img.shape[:2]
                surface, crease = floor_masks(img, ncfg)
                for oid, prob, box, m in _sam3_outputs(resp["outputs"], Ww, Hh, scale):
                    if prob < sc["min_prob"]:
                        continue
                    tid = int(sid) * 10000 + pi * 1000 + oid
                    bf = box / scale  # back to prep-frame pixels for floor checks and crops
                    on, inc = foot_on_floor(surface, crease, bf)
                    if m is not None and m.any():
                        ys = np.nonzero(m)[0]
                        foot_y = (ys.max() + 1) * scale
                        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        big = max(cs, key=cv2.contourArea)
                        contours.setdefault(fi, {})[tid] = (cv2.approxPolyDP(big, 1.5, True)[:, 0] * scale).round().astype(int).tolist()
                        area = float(m.sum()) * scale * scale
                    else:
                        foot_y, area = box[3], float((box[2] - box[0]) * (box[3] - box[1]))
                    rows.append(dict(shot_id=int(sid), frame=fi, t=tmap.get(fi, np.nan), track_id=tid,
                                     prompt=prompt, prob=round(prob, 3), x1=round(box[0], 1), y1=round(box[1], 1),
                                     x2=round(box[2], 1), y2=round(box[3], 1), foot_x=round((box[0] + box[2]) / 2, 1),
                                     foot_y=round(foot_y, 1), area=round(area), on_floor=on, in_crease=inc,
                                     crop=crops.maybe(tid, fi, img, bf, m)))
        predictor.handle_request(request=dict(type="close_session", session_id=sess))
        run.p("contours", f"shot_{sid:04d}.json").write_text(json.dumps(contours))
        log(f"  shot {sid}: {len(fr)} frames in {time.time() - t0:.0f}s")
    return finalize(run, cfg, rows, fps)


# --------------------------------------------------------------------------- CPU backend
class _KF:
    """Constant-velocity Kalman filter on box centre, width, height (image pixels)."""

    def __init__(self, box):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        self.P = np.diag([10, 10, 10, 10, 100, 100, 25, 25.0])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.Hm = np.eye(4, 8)
        self.Q = np.diag([1, 1, 1, 1, 4, 4, 1, 1.0])
        self.R = np.diag([16, 16, 25, 25.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box()

    def update(self, box):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.Hm @ self.x)
        self.P = (np.eye(8) - K @ self.Hm) @ self.P

    def box(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-6)


def run_cpu(run: Run, cfg: dict, only=None, budget=None):
    from scipy.optimize import linear_sum_assignment
    nll = NLL(cfg)
    from nllcv import detect as ndet
    ncfg = nll.cfg
    dc = ncfg["detect"]
    model = ndet.Yolox(nll.common.resolve(dc["model"]), dc["input"], dc["conf"], dc["nms"])
    tc = cfg["track"]["cpu"]
    prep = json.loads(run.p("prep.json").read_text())
    scale, fps = prep["scale"], prep["track_fps"]
    frames = run.csv("frames.csv")
    if only:
        frames = frames[frames.shot_id.isin(only)]
    cache = run.p("cache", "track_cpu", "x").parent
    crops = CropWriter(run, cfg["track"]["crops_per_s"], fps)
    t_start = time.time()
    for sid, fr in frames.groupby("shot_id"):
        out = cache / f"shot_{sid:04d}.csv"
        if out.exists():
            continue
        if budget and time.time() - t_start > budget:
            log("track: time budget reached - rerun to continue")
            return None
        d = run.root / "frames" / f"shot_{sid:04d}"
        tracks, next_id, rows, contours = {}, 1, [], {}
        max_age = int(tc["max_age_s"] * fps)
        for fi, t in zip(fr.frame, fr.t):
            img = cv2.imread(str(d / f"{fi}.jpg"))
            if img is None:
                continue
            surface, crease = floor_masks(img, ncfg)
            # silhouette background: turf + only the bright, saturated crease paint (purple-blue
            # jerseys share the crease hue but are darker and less saturated)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            paint = cv2.inRange(hsv, np.array([106, 150, 95], np.uint8), np.array([119, 255, 255], np.uint8))
            background = cv2.bitwise_or(cv2.bitwise_and(surface, 255 - crease), paint)
            b, s, c = model(img, dc["tiles"])
            keep = (c == ndet.PERSON) & ((b[:, 3] - b[:, 1]) >= 18)
            b, s = b[keep], s[keep]
            # only people standing on the floor enter the tracker (drops crowd and benches)
            onf = np.array([foot_on_floor(surface, crease, bb)[0] for bb in b], bool) if len(b) else np.zeros(0, bool)
            b, s = b[onf], s[onf]
            ids = list(tracks)
            pred = np.array([tracks[i]["kf"].predict() for i in ids]) if ids else np.zeros((0, 4))
            matched = set()
            if len(ids) and len(b):
                iou = _iou(pred, b)
                r, cidx = linear_sum_assignment(-iou)
                for i, j in zip(r, cidx):
                    if iou[i, j] >= tc["iou_match"]:
                        tracks[ids[i]]["kf"].update(b[j])
                        tracks[ids[i]]["age"] = 0
                        tracks[ids[i]]["det"] = (b[j], s[j])
                        matched.add(j)
            for j in range(len(b)):
                if j not in matched:
                    tracks[next_id] = dict(kf=_KF(b[j]), age=0, det=(b[j], s[j]))
                    next_id += 1
            for i in list(tracks):
                tr = tracks[i]
                if tr["age"] == 0 and tr.get("det") is not None:
                    box, sc_ = tr["det"]
                    tid = int(sid) * 10000 + i
                    on, inc = foot_on_floor(surface, crease, box)
                    bs = box * scale
                    m = turf_mask(background, box, img.shape)
                    if m is not None:
                        poly = mask_contour(m, scale)
                        if poly:
                            contours.setdefault(int(fi), {})[tid] = poly
                    rows.append(dict(shot_id=int(sid), frame=int(fi), t=t, track_id=tid, prompt="person",
                                     prob=round(float(sc_), 3), x1=round(bs[0], 1), y1=round(bs[1], 1),
                                     x2=round(bs[2], 1), y2=round(bs[3], 1), foot_x=round((bs[0] + bs[2]) / 2, 1),
                                     foot_y=round(bs[3], 1), area=round(float((bs[2] - bs[0]) * (bs[3] - bs[1]))),
                                     on_floor=on, in_crease=inc, crop=crops.maybe(tid, fi, img, box, m)))
                    tr["det"] = None
                else:
                    tr["age"] += 1
                if tr["age"] > max_age:
                    del tracks[i]
        run.p("contours", f"shot_{sid:04d}.json").write_text(json.dumps(contours))
        pd.DataFrame(rows).to_csv(out, index=False)
        log(f"  shot {sid}: {len(fr)} frames, {next_id - 1} raw tracks")
    parts = []
    for f in sorted(cache.glob("shot_*.csv")):
        try:
            parts.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass
    rows = pd.concat(parts).to_dict("records") if parts else []
    return finalize(run, cfg, rows, fps)


def run(run: Run, cfg: dict, backend=None, only=None, budget=None):
    backend = backend or cfg["track"]["backend"]
    log(f"track: backend={backend}")
    if backend == "sam3":
        return run_sam3(run, cfg, only)
    return run_cpu(run, cfg, only, budget)
