"""Ball labels for training the RF-DETR detector - sample, pre-label, review, export.

    python -m laxnet ball-sample   --run runs/<game>   # frames + motion/track proposals   (any machine)
    python -m laxnet ball-prelabel --run runs/<game>   # + SAM 3 "lacrosse ball" proposals (GPU)
    python -m laxnet ball-review   --run runs/<game>   # click through and fix               (your Mac)
    python -m laxnet ball-export   --run runs/<game>   # COCO tiles for RF-DETR (Roboflow layout)

Files under runs/<game>/labels/
    images/s0007_f00123.jpg   full-resolution frames
    index.csv                 image, shot_id, bframe, t
    proposals.json            image -> [[x, y, w, h, score, source], ...]   (centre-based, source px)
    labels.json               image -> {"status": "ball" | "none" | "skip", "balls": [[x, y, w, h], ...]}

Export writes datasets/ball_<game>/{train,valid,test}/ with _annotations.coco.json: 640 px tiles cut
at full resolution around each labelled ball plus ball-free tiles, split by camera shot. The same
folder can be uploaded to Roboflow instead of using the review tool.
"""
from __future__ import annotations

import json
import random
import time

import cv2
import numpy as np
import pandas as pd

from .core import NLL, Run, log, resolve


def _paths(run: Run):
    d = run.root / "labels"
    return dict(dir=d, images=d / "images", index=d / "index.csv", proposals=d / "proposals.json",
                labels=d / "labels.json")


def _load(p, default):
    return json.loads(p.read_text()) if p.exists() else default


# --------------------------------------------------------------------------- 1. sample (+ motion proposals)
def sample(run: Run, cfg: dict, n=None, seed=0, budget=None):
    from .ball_detect import MotionDetector, detect_shot, people_index
    nll = NLL(cfg)
    lc, bc = cfg["ball_label"], cfg["ball"]
    P = _paths(run)
    P["images"].mkdir(parents=True, exist_ok=True)
    prep = json.loads(run.p("prep.json").read_text())
    video = resolve(cfg["video"]) if resolve(cfg["video"]).exists() else prep["video"]
    W, H = prep["source_width"], prep["source_height"]
    fps = bc["fps"]
    # every live wide shot of the game (not only the ones prepped for tracking)
    from .prep import select_shots
    shots = [(int(r.shot_id), r.start + 0.05, r.end - r.start - 0.1) for r in select_shots(nll, cfg).itertuples()]
    n = n or lc["frames"]
    rng = random.Random(seed)
    # spread the picks over all live time, at least min_gap_s apart
    total = sum(d for _, _, d in shots)
    step = max(lc["min_gap_s"], total / max(n, 1))
    want = {}
    for sid, t0, dur in shots:
        k, t = [], rng.uniform(0.1, step)
        while t < dur - 0.1:
            k.append(int(t * fps))
            t += step * rng.uniform(0.8, 1.2)
        want[sid] = set(k)
    props = _load(P["proposals"], {})
    track = run.csv("ball_track.csv") if run.exists("ball_track.csv") else None
    det = MotionDetector(bc["motion"], nll.cfg)
    tracks = run.csv("tracks.csv") if run.exists("tracks.csv") else None
    old = pd.read_csv(P["index"]) if P["index"].exists() else pd.DataFrame(columns=["image", "shot_id", "bframe", "t"])
    rows = old.to_dict("records")
    have = set(old.image)
    t_start = time.time()
    for sid, t0, dur in shots:
        names = {i: f"s{sid:04d}_f{i:05d}.jpg" for i in want[sid]}
        todo = {i for i, nm in names.items() if nm not in have or not (P["images"] / nm).exists()}
        if not todo:
            continue
        if budget and time.time() - t_start > budget:
            log("ball-sample: time budget reached; run again to continue")
            break

        def keep(i, img, dets, sid=sid, t0=t0):
            name = names[i]
            cv2.imwrite(str(P["images"] / name), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            pr = [[round(float(x), 1), round(float(y), 1), round(float(w), 1), round(float(h), 1), round(float(s), 3),
                   "motion"] for x, y, w, h, s in dets]
            if track is not None:
                r = track[(track.shot_id == sid) & (track.bframe == i) & (track.state != "none")]
                for q in r.itertuples():
                    pr.insert(0, [float(q.x), float(q.y), 10, 10, 1.0, f"track:{q.state}"])
            props[name] = pr
            rows.append(dict(image=name, shot_id=int(sid), bframe=int(i), t=round(float(t0 + i / fps), 4)))

        detect_shot(video, sid, t0, dur, fps, W, H, "motion", det, nll.cfg, want=todo, on_frame=keep,
                    people=people_index(tracks, sid))
        idx = pd.DataFrame(rows).drop_duplicates("image").sort_values("image")
        idx.to_csv(P["index"], index=False)
        P["proposals"].write_text(json.dumps(props))
        have = set(idx.image)
        log(f"  shot {sid}: {len(todo)} frames sampled ({len(idx)} total)")
    idx = pd.DataFrame(rows).drop_duplicates("image").sort_values("image") if rows else pd.DataFrame()
    log(f"ball-sample: {len(idx)} frames in {P['images']}, with motion proposals")
    return idx


# --------------------------------------------------------------------------- 2. SAM 3 proposals (GPU)
def prelabel_sam3(run: Run, cfg: dict):
    """Text-prompted SAM 3 on overlapping full-resolution tiles (the ball is too small for one 1008 px pass)."""
    from .ball_detect import SAM3Detector
    lc = cfg["ball_label"]
    P = _paths(run)
    props = _load(P["proposals"], {})
    idx = pd.read_csv(P["index"])
    det = SAM3Detector(cfg["ball"], lc["sam3_prompt"], overlap=128, min_score=0.1)
    n_found = 0
    for k, name in enumerate(idx.image):
        img = cv2.imread(str(P["images"] / name))
        found = [[round(float(x), 1), round(float(y), 1), round(float(w), 1), round(float(h), 1), round(float(s), 3), "sam3"]
                 for x, y, w, h, s in det.detect(img)]
        props[name] = found + [p for p in props.get(name, []) if p[5] != "sam3"]
        n_found += bool(found)
        if k % 50 == 0:
            P["proposals"].write_text(json.dumps(props))
            log(f"  {k + 1}/{len(idx)} frames")
    P["proposals"].write_text(json.dumps(props))
    log(f"ball-prelabel: SAM 3 proposed a ball in {n_found}/{len(idx)} frames")


# --------------------------------------------------------------------------- 3. review GUI (OpenCV)
HELP = ["ENTER/SPACE accept highlighted proposal    TAB next proposal",
        "LEFT CLICK  ball here (adds; click again for a 2nd ball)    0  no ball visible",
        "U undo last ball   S skip frame (unsure)   D/RIGHT next   A/LEFT back   Q save + quit",
        "Mouse over the frame shows the 4x zoom (top-left)"]


def review(run: Run, cfg: dict, only_unlabelled=True):
    lc = cfg["ball_label"]
    P = _paths(run)
    idx = pd.read_csv(P["index"])
    props = _load(P["proposals"], {})
    labels = _load(P["labels"], {})
    names = list(idx.image)
    order = [n for n in names if n not in labels] + [n for n in names if n in labels] if only_unlabelled else names
    if not order:
        log("ball-review: nothing to review")
        return
    win = "LacrosseNet ball review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    st = dict(i=0, sel=0, mouse=(0, 0), click=None)
    disp_w = 1600

    def on_mouse(ev, x, y, flags, _):
        st["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["click"] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    box = lc["box_px"]
    while 0 <= st["i"] < len(order):
        name = order[st["i"]]
        img = cv2.imread(str(P["images"] / name))
        if img is None:
            st["i"] += 1
            continue
        H, W = img.shape[:2]
        s = disp_w / W
        pr = props.get(name, [])
        lab = labels.get(name, {"status": None, "balls": []})
        while True:
            view = cv2.resize(img, (disp_w, int(H * s)), interpolation=cv2.INTER_AREA)
            for k, p in enumerate(pr[:9]):
                c = (0, 255, 255) if k == st["sel"] else (0, 160, 255)
                cx, cy = int(p[0] * s), int(p[1] * s)
                cv2.circle(view, (cx, cy), 12 if k == st["sel"] else 9, c, 1, cv2.LINE_AA)
                cv2.putText(view, str(k + 1), (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            for b in lab["balls"]:
                cx, cy = int(b[0] * s), int(b[1] * s)
                cv2.rectangle(view, (cx - 7, cy - 7), (cx + 7, cy + 7), (0, 255, 0), 2)
            # zoom inset around the mouse (or the selected proposal)
            mx, my = st["mouse"]
            if not (0 < mx < view.shape[1] and 0 < my < view.shape[0]) and pr:
                mx, my = int(pr[min(st["sel"], len(pr) - 1)][0] * s), int(pr[min(st["sel"], len(pr) - 1)][1] * s)
            fx, fy = int(mx / s), int(my / s)
            r = 60
            x0, y0 = int(np.clip(fx - r, 0, W - 2 * r)), int(np.clip(fy - r, 0, H - 2 * r))
            z = cv2.resize(img[y0:y0 + 2 * r, x0:x0 + 2 * r], (4 * 2 * r, 4 * 2 * r), interpolation=cv2.INTER_CUBIC)
            for b in lab["balls"]:
                bx, by = int((b[0] - x0) * 4), int((b[1] - y0) * 4)
                cv2.rectangle(z, (bx - b[2] * 2, by - b[3] * 2), (bx + b[2] * 2, by + b[3] * 2), (0, 255, 0), 1)
            cv2.drawMarker(z, (int((fx - x0) * 4), int((fy - y0) * 4)), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
            view[8:8 + z.shape[0], 8:8 + z.shape[1]] = z
            cv2.rectangle(view, (8, 8), (8 + z.shape[1], 8 + z.shape[0]), (255, 255, 255), 1)
            status = lab["status"] or "unlabelled"
            done = sum(1 for n in names if n in labels)
            cv2.rectangle(view, (0, view.shape[0] - 92), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
            cv2.putText(view, f"{st['i'] + 1}/{len(order)}  {name}  [{status}]  labelled {done}/{len(names)}",
                        (10, view.shape[0] - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            for j, h in enumerate(HELP[:3]):
                cv2.putText(view, h, (10, view.shape[0] - 50 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1, cv2.LINE_AA)
            cv2.imshow(win, view)
            key = cv2.waitKeyEx(30)
            if st["click"] is not None:
                cx, cy = st["click"][0] / s, st["click"][1] / s
                st["click"] = None
                # snap to a proposal under the click when there is one
                near = [p for p in pr if np.hypot(p[0] - cx, p[1] - cy) < 10 / s]
                b = [round(near[0][0], 1), round(near[0][1], 1), max(6, near[0][2]), max(6, near[0][3])] if near else \
                    [round(cx, 1), round(cy, 1), box, box]
                lab = {"status": "ball", "balls": lab["balls"] + [b]}
                labels[name] = lab
                P["labels"].write_text(json.dumps(labels))
                continue
            if key in (-1, 255):
                continue
            ch = key & 0xFF
            if ch in (13, 32) and pr:                       # accept highlighted
                p = pr[min(st["sel"], len(pr) - 1)]
                lab = {"status": "ball", "balls": lab["balls"] + [[p[0], p[1], max(6, p[2]), max(6, p[3])]]}
                labels[name] = lab
            elif ch == 9 and pr:                            # TAB
                st["sel"] = (st["sel"] + 1) % min(len(pr), 9)
                continue
            elif ord("1") <= ch <= ord("9") and ch - ord("1") < len(pr):
                st["sel"] = ch - ord("1")
                continue
            elif ch == ord("0"):
                lab = {"status": "none", "balls": []}
                labels[name] = lab
            elif ch in (ord("u"), ord("U")):
                lab = {"status": "ball" if len(lab["balls"]) > 1 else None, "balls": lab["balls"][:-1]}
                if lab["status"] is None:
                    labels.pop(name, None)
                else:
                    labels[name] = lab
                P["labels"].write_text(json.dumps(labels))
                continue
            elif ch in (ord("s"), ord("S")):
                labels[name] = {"status": "skip", "balls": []}
            elif ch in (ord("a"), ord("A")) or key in (2424832, 65361, 63234):
                st["i"] = max(0, st["i"] - 1)
                st["sel"] = 0
                break
            elif ch in (ord("d"), ord("D")) or key in (2555904, 65363, 63235):
                pass
            elif ch in (ord("q"), ord("Q"), 27):
                P["labels"].write_text(json.dumps(labels))
                cv2.destroyAllWindows()
                log(f"ball-review: saved {len(labels)} labels to {P['labels']}")
                return
            else:
                continue
            P["labels"].write_text(json.dumps(labels))
            if ch in (13, 32) and len(pr) > 1:
                continue      # stay: there may be a second ball / correction; D moves on
            st["i"] += 1
            st["sel"] = 0
            break
    P["labels"].write_text(json.dumps(labels))
    cv2.destroyAllWindows()
    log(f"ball-review: all frames visited; {len(labels)} labels in {P['labels']}")


# --------------------------------------------------------------------------- 4. export COCO tiles
def _split_groups(idx: pd.DataFrame, split, seed=0):
    """Assign each image to train/valid/test by camera shot (or by time blocks when shots are few)."""
    rng = random.Random(seed)
    if idx.shot_id.nunique() >= 6:
        groups = idx.shot_id.astype(str)
    else:   # few shots: 3-second blocks inside each shot
        groups = idx.shot_id.astype(str) + "_" + (idx.t // 3).astype(int).astype(str)
    g = sorted(groups.unique())
    rng.shuffle(g)
    sizes = groups.value_counts()
    total, acc, part = len(idx), 0, {}
    for k in g:
        acc += sizes[k]
        part[k] = "train" if acc <= split[0] * total else ("valid" if acc <= (split[0] + split[1]) * total else "test")
    out = groups.map(part)
    for need in ("valid", "train"):         # never leave a required split empty
        if not (out == need).any():
            out.iloc[-1 if need == "valid" else 0] = need
    return out.values


def export(run: Run, cfg: dict, out=None, seed=0):
    lc, T = cfg["ball_label"], cfg["ball"]["tile"]
    P = _paths(run)
    labels = _load(P["labels"], {})
    idx = pd.read_csv(P["index"])
    idx = idx[idx.image.map(lambda n: labels.get(n, {}).get("status") in ("ball", "none"))].reset_index(drop=True)
    if idx.empty:
        raise SystemExit("ball-export: no reviewed frames yet (run ball-review, or label in Roboflow)")
    out = resolve(out) if out else resolve("datasets") / f"ball_{run.root.name}"
    idx["split"] = _split_groups(idx, lc["split"], seed)
    rng = random.Random(seed)
    coco = {s: dict(info={"description": "LacrosseNet ball tiles"}, licenses=[],
                    categories=[{"id": 0, "name": "balls", "supercategory": "none"},
                                {"id": 1, "name": "ball", "supercategory": "balls"}],
                    images=[], annotations=[]) for s in ("train", "valid", "test")}
    for s in coco:
        (out / s).mkdir(parents=True, exist_ok=True)
    n_tiles = n_balls = 0
    for r in idx.itertuples():
        img = cv2.imread(str(P["images"] / r.image))
        H, W = img.shape[:2]
        balls = labels[r.image]["balls"]
        origins = []
        for b in balls:                             # one tile per ball, ball at a random place inside
            m = 24
            x0 = int(np.clip(b[0] - rng.uniform(m, T - m), 0, W - T))
            y0 = int(np.clip(b[1] - rng.uniform(m, T - m), 0, H - T))
            origins.append((x0, y0))
        for _ in range(lc["negatives_per_image"] + (1 if not balls else 0)):
            for _try in range(20):
                x0, y0 = rng.randint(0, W - T), rng.randint(int(0.2 * (H - T)), H - T)
                if all(not (x0 - 8 <= b[0] <= x0 + T + 8 and y0 - 8 <= b[1] <= y0 + T + 8) for b in balls):
                    origins.append((x0, y0))
                    break
        c = coco[r.split]
        for x0, y0 in origins:
            iid = len(c["images"])
            fn = f"{r.image[:-4]}_{x0}_{y0}.jpg"
            cv2.imwrite(str(out / r.split / fn), img[y0:y0 + T, x0:x0 + T], [cv2.IMWRITE_JPEG_QUALITY, 95])
            c["images"].append(dict(id=iid, file_name=fn, width=T, height=T))
            for b in balls:
                bx, by = b[0] - b[2] / 2 - x0, b[1] - b[3] / 2 - y0
                if 0 <= bx and 0 <= by and bx + b[2] <= T and by + b[3] <= T:
                    c["annotations"].append(dict(id=len(c["annotations"]) + 1, image_id=iid, category_id=1,
                                                 bbox=[round(bx, 1), round(by, 1), b[2], b[3]],
                                                 area=round(b[2] * b[3], 1), iscrowd=0, segmentation=[]))
                    n_balls += 1
            n_tiles += 1
    for s, c in coco.items():
        (out / s / "_annotations.coco.json").write_text(json.dumps(c))
    summ = {s: f"{len(c['images'])} tiles / {len(c['annotations'])} balls" for s, c in coco.items()}
    log(f"ball-export: {len(idx)} reviewed frames -> {n_tiles} tiles, {n_balls} ball boxes in {out}: {summ}")
    return out
