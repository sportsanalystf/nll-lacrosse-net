"""Stage 3 - team assignment from jersey appearance, one label per track.

siglip  Roboflow's recipe: SigLIP image embeddings of torso crops -> UMAP (3-D) -> KMeans. We use
        k = 3 (two teams + officials) unless the tracker already separated referees by prompt.
hsv     the nllcv 15-bin jersey colour histogram (no torch needed).

Cluster names come from the configured team colours in the nllcv config; officials are the
cluster with the most black-and-white stripe content. Each track takes the majority label of its
crops, which also absorbs single-crop mistakes.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .core import NLL, Run, log


def _hsv_feature(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    grey = (np.abs(hsv[:, 2] - 120) < 6) & (hsv[:, 1] < 12)  # masked background
    floor = (hsv[:, 0] > 35) & (hsv[:, 0] < 90) & (hsv[:, 1] > 40)
    p = hsv[~grey & ~floor]
    if len(p) < 15:
        return None
    hh, ss, vv = p[:, 0], p[:, 1], p[:, 2]
    chrom = (ss > 70) & (vv > 60)
    f = np.zeros(15, np.float32)
    np.add.at(f, (hh[chrom] * 12) // 180, 1)
    f[12] = ((ss <= 70) & (vv > 170)).sum()
    f[13] = ((ss <= 70) & (vv > 70) & (vv <= 170)).sum()
    f[14] = (vv <= 70).sum()
    return f / f.sum()


def _siglip(crops, model_name):
    import torch
    from transformers import AutoProcessor, SiglipVisionModel
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SiglipVisionModel.from_pretrained(model_name).to(dev).eval()
    proc = AutoProcessor.from_pretrained(model_name)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + 64]]
            inp = proc(images=batch, return_tensors="pt").to(dev)
            out.append(model(**inp).pooler_output.float().cpu().numpy())
    return np.concatenate(out)


def _hex_lab(h):
    h = h.lstrip("#")
    bgr = np.array([[[int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)]]], np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def run(run: Run, cfg: dict, embedder=None):
    from sklearn.cluster import KMeans
    nll = NLL(cfg)
    tr = run.csv("tracks.csv")
    has_crop = tr.dropna(subset=["crop"])
    crops, owners, hsvf, means, crop_names = [], [], [], [], []
    for r in has_crop.itertuples():
        img = cv2.imread(str(run.root / "crops" / str(r.track_id) / r.crop))
        if img is None:
            continue
        f = _hsv_feature(img)
        if f is None:
            continue
        crops.append(img)
        owners.append(r.track_id)
        crop_names.append(r.crop)
        hsvf.append(f)
        means.append(img.reshape(-1, 3).mean(0))
    owners, hsvf, means = np.array(owners), np.array(hsvf), np.array(means)
    emb = embedder or cfg["teams"]["embedder"]
    X = np.sqrt(hsvf)
    if emb == "siglip":
        try:
            import umap
            E = _siglip(crops, cfg["teams"]["siglip_model"])
            X = umap.UMAP(n_components=cfg["teams"]["umap_dims"], random_state=0).fit_transform(E)
        except ImportError as e:
            log(f"teams: SigLIP/UMAP unavailable ({e.name}); falling back to colour histograms")
            emb = "hsv"
    # referees prompted separately by SAM 3 are taken out before clustering
    ref_tracks = set(tr[tr.prompt.str.contains("referee", case=False, na=False)].track_id)
    is_ref_prompt = np.isin(owners, list(ref_tracks))
    k = 2 if is_ref_prompt.any() else 3
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X[~is_ref_prompt])
    lab = np.full(len(owners), -1)
    lab[~is_ref_prompt] = km.labels_
    # name clusters
    A, B = nll.teams
    cols = nll.cfg["game"].get("colors", {})
    clusters = list(range(k))
    names = {}
    if k == 3:
        cen = np.array([hsvf[lab == c].mean(0) for c in clusters])
        stripe = np.minimum(cen[:, 12] + cen[:, 13], cen[:, 14]) - cen[:, :12].sum(1) * 0.5
        ref = int(np.argmax(stripe))
        names[ref] = "official"
        clusters = [c for c in clusters if c != ref]
    labm = [cv2.cvtColor(means[lab == c].mean(0).astype(np.uint8)[None, None], cv2.COLOR_BGR2LAB)[0, 0].astype(float)
            for c in clusters]
    if A in cols and B in cols:
        ta, tb = _hex_lab(cols[A]), _hex_lab(cols[B])
        d0 = np.linalg.norm(labm[0] - ta) + np.linalg.norm(labm[1] - tb)
        d1 = np.linalg.norm(labm[0] - tb) + np.linalg.norm(labm[1] - ta)
        names[clusters[0]], names[clusters[1]] = (A, B) if d0 <= d1 else (B, A)
    else:
        names[clusters[0]], names[clusters[1]] = "team_A", "team_B"
    crop_team = np.array([names.get(l, "official") for l in lab])
    crop_team[is_ref_prompt] = "official"
    votes = pd.DataFrame({"track_id": owners, "team": crop_team})
    # per-crop labels: the number reader only trusts crops that look like the track's own team
    pd.DataFrame({"track_id": owners, "crop": crop_names, "crop_team": crop_team}).to_csv(
        run.p("crop_teams.csv"), index=False)
    agg = votes.groupby("track_id").team.agg(lambda s: s.value_counts().index[0]).rename("team")
    share = votes.groupby("track_id").team.agg(lambda s: s.value_counts().iloc[0] / len(s)).rename("team_share")
    n = votes.groupby("track_id").size().rename("team_crops")
    out = pd.concat([agg, share, n], axis=1).reset_index()
    out.to_csv(run.p("teams.csv"), index=False)
    log(f"teams ({emb}): " + ", ".join(f"{t}={c}" for t, c in out.team.value_counts().items()) +
        f" tracks; mean vote share {out.team_share.mean():.2f}")
    _sheet(run, out)
    return out


def _sheet(run: Run, teams: pd.DataFrame):
    rows = []
    for team, g in teams.groupby("team"):
        tiles = []
        for tid in g.sort_values("team_crops", ascending=False).track_id.head(14):
            fs = sorted((run.root / "crops" / str(tid)).glob("*.jpg"))
            if fs:
                tiles.append(cv2.resize(cv2.imread(str(fs[len(fs) // 2])), (56, 84)))
        if tiles:
            tiles += [np.zeros((84, 56, 3), np.uint8)] * (14 - len(tiles))
            lab = np.zeros((84, 130, 3), np.uint8)
            cv2.putText(lab, team, (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            rows.append(np.hstack([lab] + tiles))
    if rows:
        cv2.imwrite(str(run.p("debug", "teams.jpg")), np.vstack(rows))
