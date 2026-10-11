"""Ball stage, part 1 - find ball candidates in every frame of every live shot, at full resolution.

The NLL ball is 4-10 px wide in a 1080p broadcast and usually motion-blurred, so detection runs on the
source frames (not the 1280 px tracking JPEGs) at `ball.fps` (30 = every broadcast frame), streamed
straight from the video with ffmpeg so nothing large is written to disk.

Three detectors write the same table:

sam3    zero-shot: SAM 3 with the text prompt "lacrosse ball" on full-resolution tiles. No labels needed,
        but several model calls per frame - good for highlights and pre-labels, slow for a full game.
rfdetr  an RF-DETR model fine-tuned on our own labelled frames (see ball_label.py and `ball-train`).
        Frames are cut into native-resolution tiles (640 px, overlapping) so the ball keeps all its
        pixels; tiles with no turf in them are skipped.
motion  no training needed: the camera pan between frames is measured by phase correlation, the
        neighbouring frames are shifted onto the current one, and pixels that changed in both
        (double difference) form blobs. Small, pale blobs with turf around them are ball candidates.
        Sticks, gloves and white socks also move, so this is noisy; the tracker and the rules
        downstream clean it up. Used for CPU development and for pre-labelling.

Writes
    ball_candidates.csv  shot_id, bframe, t, x, y, w, h, score, source        (source pixels)
    ball_frames.csv      shot_id, bframe, t, cam_dx, cam_dy, crease_x, crease_y
                         cam_dx/dy: image shift of static scenery since the previous frame (camera pan)
                         crease_x/y: centre of the largest visible crease (the goal being attacked)
"""
from __future__ import annotations

import json
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

from .core import NLL, Run, log, resolve
from .track import floor_masks


# --------------------------------------------------------------------------- frames
def iter_frames(video, start, dur, fps, W, H):
    """Yield BGR frames of [start, start+dur) at `fps`, full resolution."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
           "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    n = W * H * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        p.stdout.close()
        p.wait()


def shot_windows(run: Run, cfg: dict, only=None):
    """(shot_id, t0, dur) for the live shots chosen by prep; frame i of a shot is at t0 + i / fps."""
    s = run.csv("shots_selected.csv")
    if only:
        s = s[s.shot_id.isin(only)]
    return [(int(r.shot_id), r.start + 0.05, r.end - r.start - 0.1) for r in s.itertuples()]


class CameraMotion:
    """Global image translation between consecutive frames by phase correlation on a small grey copy.
    Box-lacrosse broadcast cameras mostly pan; zoom is slow, so translation is a good local model."""

    def __init__(self, scale=0.25):
        self.s = scale
        self.prev = None
        self.win = None

    def __call__(self, img):
        g = cv2.cvtColor(cv2.resize(img, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.win is None:
            self.win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if self.prev is None:
            self.prev = g
            return 0.0, 0.0, 1.0
        (dx, dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
        self.prev = g
        return dx / self.s, dy / self.s, float(resp)


def crease_centre(crease_small, scale):
    n, lab, st, cen = cv2.connectedComponentsWithStats(crease_small, 8)
    if n <= 1:
        return np.nan, np.nan
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 150:
        return np.nan, np.nan
    return cen[k][0] / scale, cen[k][1] / scale


# --------------------------------------------------------------------------- detectors
class MotionDetector:
    def __init__(self, mc: dict, ncfg: dict):
        self.c, self.ncfg = mc, ncfg

    def detect(self, prev, cur, nxt, d_prev, d_next, surface, people=None):
        """prev/cur/nxt: BGR frames; d_prev: shift prev->cur, d_next: shift cur->next.
        people: optional (n, 4) person boxes (source px); blobs on a body (below the head) are skipped,
        since moving hands, socks and jersey trim look like a ball to frame differencing."""
        c = self.c
        g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur, nxt)]
        H, W = g[1].shape
        Mp = np.float32([[1, 0, d_prev[0]], [0, 1, d_prev[1]]])        # move prev onto cur
        Mn = np.float32([[1, 0, -d_next[0]], [0, 1, -d_next[1]]])      # move next onto cur
        wp = cv2.warpAffine(g[0], Mp, (W, H), borderMode=cv2.BORDER_REPLICATE)
        wn = cv2.warpAffine(g[2], Mn, (W, H), borderMode=cv2.BORDER_REPLICATE)
        # a 1-px max filter on the neighbours absorbs sub-pixel misregistration of static edges (lines, ads)
        k = np.ones((3, 3), np.uint8)
        d1 = cv2.subtract(g[1], cv2.dilate(wp, k))
        d2 = cv2.subtract(g[1], cv2.dilate(wn, k))
        diff = cv2.min(d1, d2)                     # brighter than both neighbours here: a pale moving object
        # also changed in raw (uncompensated) frames: drops screen-fixed graphics (scorebug, bugs) that
        # the camera-compensation would otherwise turn into "motion"
        k3 = np.ones((3, 3), np.uint8)
        raw = cv2.min(cv2.subtract(g[1], cv2.dilate(g[0], k3)), cv2.subtract(g[1], cv2.dilate(g[2], k3)))
        m = ((diff > c["diff_thresh"]) & (raw > c["diff_thresh"] // 2)).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        out = []
        if n <= 1:
            return out
        hsv = None
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < c["min_area"] or a > c["max_area"]:
                continue
            x, y, w, h = st[i, :4]
            if max(w, h) > 40 or min(w, h) == 0:
                continue
            if people is not None and len(people):
                cx_, cy_ = x + w / 2, y + h / 2
                bw, bh = people[:, 2] - people[:, 0], people[:, 3] - people[:, 1]
                on_body = ((cx_ > people[:, 0] + 0.12 * bw) & (cx_ < people[:, 2] - 0.12 * bw) &
                           (cy_ > people[:, 1] + 0.18 * bh) & (cy_ < people[:, 3] + 0.02 * bh))
                if on_body.any():
                    continue
            if hsv is None:
                hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
            pix = hsv[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]
            top = pix[np.argsort(pix[:, 2])[-max(1, len(pix) // 3):]]      # brightest third of the blob
            if top[:, 2].mean() < c["min_val"] or top[:, 1].mean() > c["max_sat"]:
                continue
            r = 7
            x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(W, x + w + r), min(H, y + h + r)
            ring = surface[y0:y1, x0:x1].astype(bool).copy()
            ring[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
            area_ring = (x1 - x0) * (y1 - y0) - w * h
            turf = ring.sum() / max(area_ring, 1)
            if turf < c["min_turf_ring"]:
                continue
            contrast = float(diff[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].mean())
            score = float(np.clip(contrast / 80, 0, 1) * (0.5 + 0.5 * turf))
            out.append((cen[i][0], cen[i][1], max(w, 4), max(h, 4), score))
        return out


class RFDETRDetector:
    def __init__(self, bc: dict):
        import rfdetr
        from PIL import Image
        self.Image = Image
        cls = getattr(rfdetr, "RFDETR" + bc["rfdetr_size"].capitalize())
        w = resolve(bc["weights"])
        if not w.exists():
            raise FileNotFoundError(w)
        self.model = cls(pretrain_weights=str(w), resolution=bc["tile"])
        # rfdetr >= 1.11: inference(); 1.9-1.10: optimize_for_inference(). compile=False keeps batched predict working
        opt = getattr(self.model, "inference", None) or getattr(self.model, "optimize_for_inference", None)
        if callable(opt):
            try:
                opt(compile=False)
            except Exception:  # not supported in this version: plain inference
                pass
        self.bc = bc

    @staticmethod
    def tiles(W, H, tile, overlap):
        def starts(n):
            if n <= tile:
                return [0]
            s = list(range(0, n - tile, tile - overlap))
            return s + [n - tile]
        return [(x, y) for y in starts(H) for x in starts(W)]

    def detect(self, img, surface):
        bc = self.bc
        H, W = img.shape[:2]
        T = bc["tile"]
        keep, ims = [], []
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for x, y in self.tiles(W, H, T, bc["overlap"]):
            if surface[y:y + T, x:x + T].mean() < 255 * 0.03:      # crowd / scorebug only
                continue
            keep.append((x, y))
            ims.append(self.Image.fromarray(np.ascontiguousarray(rgb[y:y + T, x:x + T])))
        dets = []
        for i in range(0, len(ims), bc["batch_tiles"]):
            batch = ims[i:i + bc["batch_tiles"]]
            try:
                res = self.model.predict(batch, threshold=bc["min_score"])
                res = res if isinstance(res, list) else [res]
            except Exception:
                res = [self.model.predict(im, threshold=bc["min_score"]) for im in batch]
            for (x, y), d in zip(keep[i:i + len(batch)], res):
                for (x1, y1, x2, y2), s in zip(d.xyxy, d.confidence):
                    dets.append(((x1 + x2) / 2 + x, (y1 + y2) / 2 + y, x2 - x1, y2 - y1, float(s)))
        # merge duplicates from overlapping tiles
        dets.sort(key=lambda d: -d[4])
        out = []
        for d in dets:
            if all(np.hypot(d[0] - o[0], d[1] - o[1]) > 8 for o in out):
                out.append(d)
        return out


class SAM3Detector:
    """Zero-shot: SAM 3 image model with a text prompt on overlapping full-resolution tiles. Needs no labels
    but is slow (a few tiles per frame) - fine for highlights and for pre-labelling, too slow for a full
    game at 30 fps; fine-tune RF-DETR for that."""

    def __init__(self, bc: dict, prompt="lacrosse ball", tile=640, overlap=128, min_score=None):
        import torch
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.torch, self.Image = torch, Image
        self.min_score = bc["min_score"] if min_score is None else min_score
        # boxes come back as xyxy in tile pixels, already filtered by confidence_threshold
        self.proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=self.min_score)
        self.prompt, self.tile, self.overlap = prompt, tile, overlap

    def detect(self, img, surface=None):
        H, W = img.shape[:2]
        T = self.tile
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        found = []
        for x0, y0 in RFDETRDetector.tiles(W, H, T, self.overlap):
            if surface is not None and surface[y0:y0 + T, x0:x0 + T].mean() < 255 * 0.03:
                continue
            tile = self.Image.fromarray(np.ascontiguousarray(rgb[y0:y0 + T, x0:x0 + T]))
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                st = self.proc.set_image(tile)
                o = self.proc.set_text_prompt(state=st, prompt=self.prompt)
            boxes = o["boxes"].float().cpu().numpy() if hasattr(o["boxes"], "cpu") else np.asarray(o["boxes"])
            scores = o["scores"].float().cpu().numpy() if hasattr(o["scores"], "cpu") else np.asarray(o["scores"])
            for (x1, y1, x2, y2), sc in zip(boxes.reshape(-1, 4), scores.reshape(-1)):
                if max(x2 - x1, y2 - y1) > 40 or sc < self.min_score:     # not a ball at broadcast scale
                    continue
                found.append(((x1 + x2) / 2 + x0, (y1 + y2) / 2 + y0, x2 - x1, y2 - y1, float(sc)))
        found.sort(key=lambda d: -d[4])
        out = []
        for d in found:
            if all(np.hypot(d[0] - o[0], d[1] - o[1]) > 8 for o in out):
                out.append(d)
        return out


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def make_detector(cfg: dict, ncfg: dict, which=None):
    """The configured detector, falling back rfdetr -> sam3 (only on a CUDA GPU) -> motion."""
    bc = cfg["ball"]
    which = which or bc["detector"]
    if which == "rfdetr":
        try:
            return "rfdetr", RFDETRDetector(bc)
        except FileNotFoundError as e:
            log(f"ball: no trained weights at {e} (label frames and run `ball-train`)")
        except ImportError:
            log("ball: rfdetr not installed")
        which = "sam3" if _cuda() else "motion"
    if which == "sam3":
        try:
            det = SAM3Detector(bc, cfg["ball_label"]["sam3_prompt"])
            log("ball: using SAM 3 zero-shot")
            return "sam3", det
        except Exception as e:  # missing package, no GPU, gated weights not downloaded
            log(f"ball: SAM 3 unavailable ({type(e).__name__}: {e})")
    log("ball: using the motion detector (no training; noisy - for development and pre-labels)")
    return "motion", MotionDetector(bc["motion"], ncfg)


# --------------------------------------------------------------------------- stage
def people_index(tracks: pd.DataFrame | None, sid):
    """t -> person boxes of one shot (from tracks.csv, every tracked person incl. officials)."""
    if tracks is None:
        return None
    g = tracks[tracks.shot_id == sid]
    if g.empty:
        return None
    ts = np.sort(g.t.unique())
    by = {t: f[["x1", "y1", "x2", "y2"]].values for t, f in g.groupby("t")}

    def at(t):
        j = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)) if len(ts) > 1 else 0
        j = j if abs(ts[j] - t) < abs(ts[j - 1] - t) else j - 1
        return by[ts[j]] if abs(ts[j] - t) <= 0.1 else None
    return at


def detect_shot(video, sid, t0, dur, fps, W, H, kind, det, ncfg, want=None, on_frame=None, people=None):
    """Run a detector over one shot. Returns (candidate rows, frame rows). `want`: optional set of frame
    indices to detect on (others only feed the camera model); `on_frame(i, img, dets)` is a callback;
    `people(t)` gives person boxes (motion detector only)."""
    cam = CameraMotion()
    buf = []   # (i, img, shift_from_prev, surface, crease_xy)
    cands, frames = [], []

    def handle(k):
        i, img, dprev, surf, _ = buf[k]
        if want is not None and i not in want:
            return
        if kind == "motion":
            if k == 0 or k + 1 >= len(buf):
                return
            dets = det.detect(buf[k - 1][1], img, buf[k + 1][1], dprev, buf[k + 1][2], surf,
                              people(t0 + i / fps) if people else None)
        else:
            dets = det.detect(img, surf)
        for x, y, w, h, s in dets:
            cands.append(dict(shot_id=sid, bframe=i, t=round(t0 + i / fps, 4), x=round(x, 1), y=round(y, 1),
                              w=round(w, 1), h=round(h, 1), score=round(s, 3), source=kind))
        if on_frame:
            on_frame(i, img, dets)

    for i, img in enumerate(iter_frames(video, t0, dur, fps, W, H)):
        dx, dy, _ = cam(img)
        if want is None or i in want:
            small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            surf_s, crease_s = floor_masks(small, ncfg)
            surf = cv2.resize(surf_s, (W, H), interpolation=cv2.INTER_NEAREST)
            cx, cy = crease_centre(crease_s, 0.5)
        else:                       # only feeds the camera model / neighbour frames
            surf, cx, cy = None, np.nan, np.nan
        frames.append(dict(shot_id=sid, bframe=i, t=round(t0 + i / fps, 4), cam_dx=round(dx, 2), cam_dy=round(dy, 2),
                           crease_x=round(cx, 1) if np.isfinite(cx) else np.nan,
                           crease_y=round(cy, 1) if np.isfinite(cy) else np.nan))
        buf.append((i, img, (dx, dy), surf, (cx, cy)))
        if kind == "motion":
            if len(buf) >= 3:
                handle(len(buf) - 2)
            if len(buf) > 3:
                buf.pop(0)
        else:
            handle(len(buf) - 1)
            buf.pop(0)
    return cands, frames


def run(run: Run, cfg: dict, only=None, detector=None, force=False, budget=None):
    nll = NLL(cfg)
    bc = cfg["ball"]
    prep = json.loads(run.p("prep.json").read_text())
    video = resolve(cfg["video"]) if resolve(cfg["video"]).exists() else prep["video"]
    W, H = prep["source_width"], prep["source_height"]
    kind, det = make_detector(cfg, nll.cfg, detector)
    fps = bc["fps"]
    stride = max(1, int(round(fps / bc.get("sam3_fps", fps)))) if kind == "sam3" else 1
    tracks = run.csv("tracks.csv") if run.exists("tracks.csv") else None
    t_start = time.time()
    allc, allf = [], []
    for sid, t0, dur in shot_windows(run, cfg, only):
        cp = run.p("cache", "ball", f"shot_{sid:04d}_{kind}.csv")
        fp = run.p("cache", "ball", f"shot_{sid:04d}_frames.csv")
        if cp.exists() and fp.exists() and not force:
            allc.append(pd.read_csv(cp))
            allf.append(pd.read_csv(fp))
            continue
        if budget and time.time() - t_start > budget:
            log(f"ball: time budget reached before shot {sid}; run again to continue")
            return None
        t1 = time.time()
        want = set(range(0, int(dur * fps) + 2, stride)) if stride > 1 else None
        c, f = detect_shot(video, sid, t0, dur, fps, W, H, kind, det, nll.cfg, want=want,
                           people=people_index(tracks, sid))
        c = pd.DataFrame(c, columns=["shot_id", "bframe", "t", "x", "y", "w", "h", "score", "source"])
        f = pd.DataFrame(f)
        c.to_csv(cp, index=False)
        f.to_csv(fp, index=False)
        allc.append(c)
        allf.append(f)
        log(f"  shot {sid}: {len(f)} frames, {len(c)} candidates ({kind}, {time.time() - t1:.0f}s)")
    cand = pd.concat(allc, ignore_index=True) if allc else pd.DataFrame()
    frames = pd.concat(allf, ignore_index=True) if allf else pd.DataFrame()
    cand.to_csv(run.p("ball_candidates.csv"), index=False)
    frames.to_csv(run.p("ball_frames.csv"), index=False)
    run.save_json("ball_meta.json", dict(detector=kind, fps=fps, stride=stride,
                                         weights=bc["weights"] if kind == "rfdetr" else None))
    log(f"ball: {len(cand)} candidates over {len(frames)} frames ({kind} detector)")
    return cand


# --------------------------------------------------------------------------- training
def train(cfg: dict, dataset_dir, epochs=None, output_dir=None):
    import rfdetr
    bc, tc = cfg["ball"], cfg["ball_train"]
    cls = getattr(rfdetr, "RFDETR" + bc["rfdetr_size"].capitalize())
    out = resolve(output_dir or tc["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model = cls()
    log(f"ball-train: {cls.__name__} at {bc['tile']} px on {dataset_dir} -> {out}")
    model.train(dataset_dir=str(resolve(dataset_dir)), epochs=epochs or tc["epochs"], batch_size=tc["batch_size"],
                grad_accum_steps=tc["grad_accum_steps"], lr=tc["lr"], resolution=bc["tile"],
                output_dir=str(out), early_stopping=tc["early_stopping"])
    best = out / "checkpoint_best_total.pth"
    log(f"ball-train: done; weights {best} ({'found' if best.exists() else 'NOT found - check the log'})")
