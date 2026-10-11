"""Stage 2 - shot-boundary detection and shot classification.

One decode pass at `shots.analysis_fps` on a downscaled stream. Per sampled frame we keep:
    colour histogram (cut detection), floor fraction (turf green + blue crease), scorebug presence,
    edge density, flat-area fraction, saturation, motion.
Boundaries = hard cuts (frame-to-frame histogram jump) + gradual transitions (wipes/dissolves
seen across a 1 s window). Each shot is then labelled:

    live      scorebug on, floor dominant                 -> game action, used downstream
    replay    floor dominant, scorebug off                -> NLL drops the bug for replays
    closeup   scorebug on, little floor (players, bench)  -> game on, not trackable
    graphics  large flat synthetic regions, little floor
    crowd     little floor, dense texture, no bug
    studio    little floor, faces, low motion, no bug

If runs/<game>/shot_labels.csv exists (columns shot_id,label) the hand labels override the
rules and a random forest trained on them relabels every other shot.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .common import Run, hsv_mask, iter_frames, log, scale_box

ANALYSIS_SIZE = (480, 270)
LABELS = ["live", "replay", "closeup", "graphics", "crowd", "studio"]


def _hist(hsv: np.ndarray) -> np.ndarray:
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return h / max(h.sum(), 1)


def _hdist(a, b) -> float:
    return float(1.0 - np.sum(np.sqrt(a * b)))


def _flat_frac(gray: np.ndarray, block: int = 8) -> float:
    h, w = (gray.shape[0] // block) * block, (gray.shape[1] // block) * block
    g = gray[:h, :w].astype(np.float32).reshape(h // block, block, w // block, block)
    return float((g.std(axis=(1, 3)) < 2.5).mean())


def _learn_bug_template(crops: np.ndarray) -> np.ndarray | None:
    """The static scorebug logo is the most frequently repeated crop; find that mode."""
    if len(crops) < 10:
        return None
    idx = np.linspace(0, len(crops) - 1, min(len(crops), 500)).astype(int)
    X = crops[idx].reshape(len(idx), -1).astype(np.float32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-3)
    ncc = (X @ X.T) / X.shape[1]
    support = (ncc > 0.92).sum(1)
    best = int(np.argmax(support))
    if support[best] < max(5, 0.05 * len(idx)):
        return None
    members = crops[idx][ncc[best] > 0.92]
    return np.median(members, axis=0).astype(np.float32)


def _ncc(crops: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    X = crops.reshape(len(crops), -1).astype(np.float32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-3)
    t = tmpl.ravel()
    t = (t - t.mean()) / (t.std() + 1e-3)
    return X @ t / len(t)


def extract_features(run: Run, cfg: dict) -> dict:
    sc = cfg["shots"]
    fps = sc["analysis_fps"]
    pres = scale_box(cfg["scorebug"]["presence"], cfg["scorebug"]["ref_size"], ANALYSIS_SIZE)
    face = None  # Haar cascades are absent from some OpenCV builds; studio detection degrades gracefully
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    T, H, floor, crease, edges, flat, sat, motion, faces, bugcrops, small = ([] for _ in range(11))
    prev = None
    for i, (t, fr) in enumerate(iter_frames(run.video, fps=fps, size=ANALYSIS_SIZE)):
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.resize(gray, (240, 135), interpolation=cv2.INTER_AREA)
        fm = cv2.inRange(hsv, np.array(sc["floor_hsv_low"], np.uint8), np.array(sc["floor_hsv_high"], np.uint8))
        cm = cv2.inRange(hsv, np.array(sc["crease_hsv_low"], np.uint8), np.array(sc["crease_hsv_high"], np.uint8))
        x, y, w, h = pres
        T.append(t)
        H.append(_hist(hsv))
        floor.append((fm > 0).mean())
        crease.append((cm > 0).mean())
        edges.append((cv2.Canny(g2, 80, 160) > 0).mean())
        flat.append(_flat_frac(g2))
        sat.append(hsv[..., 1].mean() / 255)
        motion.append(0.0 if prev is None else float(np.abs(g2.astype(np.int16) - prev).mean()) / 255)
        prev = g2.astype(np.int16)
        bugcrops.append(gray[y:y + h, x:x + w].copy())
        small.append(cv2.resize(fr, (64, 36), interpolation=cv2.INTER_AREA))
        # faces are only useful for studio detection; run them sparsely
        if face is not None and i % 5 == 0:
            fs = face.detectMultiScale(gray, 1.2, 5, minSize=(24, 24))
            faces.append(max((fw * fh for (_, _, fw, fh) in fs), default=0) / (ANALYSIS_SIZE[0] * ANALYSIS_SIZE[1]))
        else:
            faces.append(np.nan)
        if i and i % (fps * 60) == 0:
            log(f"  shots: {t / 60:.1f} min decoded")

    bugcrops = np.stack(bugcrops)
    tmpl = _learn_bug_template(bugcrops)
    bug_ncc = _ncc(bugcrops, tmpl) if tmpl is not None else np.zeros(len(T))
    feats = dict(t=np.array(T), hist=np.stack(H), floor=np.array(floor), crease=np.array(crease),
                 edges=np.array(edges), flat=np.array(flat), sat=np.array(sat), motion=np.array(motion),
                 face=np.array(faces), bug_ncc=bug_ncc, bug=bug_ncc >= sc["bug_ncc_min"],
                 thumbs=np.stack(small))
    if tmpl is not None:
        cv2.imwrite(str(run.p("debug", "bug_template.png")), cv2.resize(tmpl.astype(np.uint8), None, fx=4, fy=4,
                                                                        interpolation=cv2.INTER_NEAREST))
    np.savez_compressed(run.p("frame_features.npz"), **feats)
    return feats


def find_boundaries(f: dict, cfg: dict) -> list[int]:
    sc = cfg["shots"]
    fps = sc["analysis_fps"]
    H = f["hist"]
    n = len(H)
    # colour change: Hellinger distance between consecutive HSV histograms
    d1 = np.array([0.0] + [_hdist(H[i], H[i - 1]) for i in range(1, n)])
    # structural change: 1 - NCC of consecutive 64x36 thumbnails. Most cuts in a lacrosse broadcast
    # go from one wide floor shot to another, which barely moves the colour histogram.
    g = f["thumbs"].astype(np.float32).mean(-1)
    g = (g - g.mean((1, 2), keepdims=True)) / (g.std((1, 2), keepdims=True) + 1e-3)
    s = np.r_[0.0, 1 - (g[1:] * g[:-1]).mean((1, 2))]
    cuts = set()
    for i in range(1, n):
        lo, hi = max(1, i - fps), min(n, i + fps + 1)
        loc_s = np.median(np.r_[s[lo:i], s[i + 1:hi]]) if hi - lo > 2 else 0
        loc_d = np.median(np.r_[d1[lo:i], d1[i + 1:hi]]) if hi - lo > 2 else 0
        r_s, r_d = s[i] / (loc_s + 0.02), d1[i] / (loc_d + 0.003)
        hard = (s[i] > sc["cut_struct"] and r_s > 3) or (d1[i] > sc["cut_threshold"] and r_d > 8)
        soft = d1[i] > sc["soft_hist"] and r_d > 10 and s[i] > sc["soft_struct"]  # dissolves / wipes
        if hard or soft:
            cuts.add(i)
    # gradual transitions: big change across a 1 s window that no single step explains
    w = fps
    for i in range(w, n):
        dw = _hdist(H[i], H[i - w])
        if dw > sc["gradual_threshold"] and not any(j in cuts for j in range(i - w, i + 1)):
            j = i - w + int(np.argmax(d1[i - w + 1:i + 1])) + 1
            cuts.add(j)
    min_gap = max(1, int(round(sc["min_shot_s"] * fps)))
    merged = []
    for c in sorted(cuts):
        if not merged or c - merged[-1] >= min_gap:
            merged.append(c)
        elif d1[c] > d1[merged[-1]]:
            merged[-1] = c
    return merged


def _rule_label(r) -> str:
    floorish = r.floor + r.crease
    if floorish >= 0.25:
        return "live" if r.bug >= 0.5 else "replay"
    if r.bug >= 0.5:
        return "closeup"
    if r.flat >= 0.45 and r.edges < 0.10:
        return "graphics"
    if (r.face or 0) > 0.004 and r.motion < 0.04:
        return "studio"
    if r.edges >= 0.10:
        return "crowd"
    return "graphics" if r.flat >= 0.3 else "crowd"


FEATURE_COLS = ["floor", "crease", "bug", "edges", "flat", "sat", "motion", "face", "duration"]


def build_shots(f: dict, bounds: list[int], cfg: dict) -> pd.DataFrame:
    fps = cfg["shots"]["analysis_fps"]
    t = f["t"]
    edges = [0] + bounds + [len(t)]
    rows = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        sl = slice(a, b)
        rows.append(dict(
            shot_id=len(rows), start=float(t[a]), end=float(t[b - 1] + 1 / fps), duration=(b - a) / fps,
            floor=float(np.median(f["floor"][sl])), crease=float(np.median(f["crease"][sl])),
            bug=float(f["bug"][sl].mean()), edges=float(np.median(f["edges"][sl])),
            flat=float(np.median(f["flat"][sl])), sat=float(np.median(f["sat"][sl])),
            motion=float(np.median(f["motion"][sl][1:])) if b - a > 1 else 0.0,
            face=float(np.nanmax(f["face"][sl])) if np.isfinite(f["face"][sl]).any() else 0.0,
        ))
    df = pd.DataFrame(rows)
    df["label"] = [_rule_label(r) for r in df.itertuples()]
    df["label_source"] = "rules"
    return df


def apply_manual_labels(run: Run, df: pd.DataFrame) -> pd.DataFrame:
    path = run.root / "shot_labels.csv"
    if not path.exists():
        return df
    man = pd.read_csv(path)
    man = man[man["label"].isin(LABELS)]
    if man.empty:
        return df
    df = df.copy()
    lab = dict(zip(man.shot_id, man.label))
    if man.label.nunique() >= 2 and len(man) >= 10:
        from sklearn.ensemble import RandomForestClassifier
        X = df[FEATURE_COLS].fillna(0).values
        idx = df.shot_id.isin(lab)
        rf = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
        rf.fit(X[idx], df.loc[idx, "shot_id"].map(lab))
        df.loc[~idx, "label"] = rf.predict(X[~idx])
        df.loc[~idx, "label_source"] = "model"
    for sid, l in lab.items():
        df.loc[df.shot_id == sid, ["label", "label_source"]] = [l, "manual"]
    log(f"  applied {len(lab)} manual shot labels")
    return df


def run(run: Run, cfg: dict) -> pd.DataFrame:
    cache = run.root / "frame_features.npz"
    if cache.exists() and not cfg.get("_force"):
        log("shots: reusing frame_features.npz (use --force to re-decode)")
        f = dict(np.load(cache))
    else:
        log("shots: decoding")
        f = extract_features(run, cfg)
    bounds = find_boundaries(f, cfg)
    df = build_shots(f, bounds, cfg)
    df = apply_manual_labels(run, df)
    df.to_csv(run.p("shots.csv"), index=False, float_format="%.4f")
    summ = df.groupby("label").duration.agg(["count", "sum"]).round(1)
    log(f"shots: {len(df)} shots\n{summ.to_string()}")
    return df


def load(run: Run) -> pd.DataFrame:
    return pd.read_csv(run.root / "shots.csv")


def label_at(shots: pd.DataFrame, t: float) -> str | None:
    hit = shots[(shots.start <= t) & (shots.end > t)]
    return None if hit.empty else hit.iloc[0].label
