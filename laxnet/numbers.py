"""Stage 4 - jersey numbers with a vision-language model (Claude), voted per track.

For each player track the clearest few torso crops are picked (largest, sharpest, spread over the
track's life), resized small (~100 image tokens each) and sent in batches of `images_per_request`
with a strict JSON answer format. A track gets a number only when at least `min_votes` reads agree
and they make up `min_share` of its legible reads, the same "confirm after repeated agreement"
idea as Roboflow's basketball pipeline. An optional roster limits numbers to each team's and
attaches names.

Needs ANTHROPIC_API_KEY. `--dry-run` writes the planned requests and a cost estimate instead.
Answers are cached per crop, so re-runs only pay for new crops.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re

import cv2
import numpy as np
import pandas as pd

from .core import Run, log, resolve

PROMPT = """You are reading jersey numbers from crops of box-lacrosse players in a broadcast.
Each image is labelled "Image N". For every image, report the number printed on the jersey (front or back),
or null if no number is clearly legible (blurred, turned sideways, covered by a stick or another player).
Do not guess from partial digits. Ignore sponsor text, helmet numbers and numbers on the boards.
Answer with JSON only: a list of objects {"image": N, "number": "23" or null, "confidence": "high"|"medium"|"low"}."""


def _sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pick_crops(run: Run, tracks: pd.DataFrame, teams: pd.DataFrame, k: int):
    players = set(teams[teams.team != "official"].track_id)
    out = {}
    tc = tracks.dropna(subset=["crop"])
    ct = run.root / "crop_teams.csv"
    if ct.exists():  # drop crops that show a player of another colour (tracker drift, overlaps)
        ctab = pd.read_csv(ct).merge(teams[["track_id", "team"]], on="track_id")
        ok = set(zip(ctab[ctab.crop_team == ctab.team].track_id, ctab[ctab.crop_team == ctab.team].crop))
        tc = tc[[(a, b) in ok for a, b in zip(tc.track_id, tc.crop)]]
    for tid, g in tc[tc.track_id.isin(players)].groupby("track_id"):
        cands = []
        for r in g.itertuples():
            p = run.root / "crops" / str(tid) / r.crop
            img = cv2.imread(str(p))
            if img is None or img.shape[0] < 24:
                continue
            cands.append((r.frame, img.shape[0] * img.shape[1], _sharpness(img), p))
        if not cands:
            continue
        c = pd.DataFrame(cands, columns=["frame", "area", "sharp", "path"])
        c["score"] = c.area.rank(pct=True) + c.sharp.rank(pct=True)
        # spread picks over the track: best crop in each of k time bins
        c["bin"] = pd.qcut(c.frame.rank(method="first"), q=min(k, len(c)), labels=False)
        out[tid] = list(c.sort_values("score", ascending=False).groupby("bin").head(1).path)
    return out


def _encode(path, long_edge):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode(), img.shape[:2]


def _parse(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


def read_batch(client, model, items, long_edge):
    content = []
    for i, (_, path) in enumerate(items, 1):
        data, _ = _encode(path, long_edge)
        content += [{"type": "text", "text": f"Image {i}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}]
    content.append({"type": "text", "text": PROMPT})
    msg = client.messages.create(model=model, max_tokens=60 * len(items) + 100,
                                 messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    res = {int(d.get("image", 0)): d for d in _parse(text) if isinstance(d, dict)}
    usage = (msg.usage.input_tokens, msg.usage.output_tokens)
    return [res.get(i, {}) for i in range(1, len(items) + 1)], usage


def vote(reads, min_votes, min_share):
    nums = [str(r["number"]).lstrip("#") for r in reads if r.get("number") not in (None, "", "null")]
    nums = [n for n in nums if re.fullmatch(r"\d{1,2}", n)]
    if not nums:
        return None, 0, 0.0
    s = pd.Series(nums).value_counts()
    top, cnt = s.index[0], int(s.iloc[0])
    share = cnt / len(nums)
    hi = sum(1 for r in reads if str(r.get("number")).lstrip("#") == top and r.get("confidence") == "high")
    if (cnt >= min_votes and share >= min_share) or (cnt == 1 and len(nums) == 1 and hi == 1):
        return top, cnt, share
    return None, cnt, share


def run(run: Run, cfg: dict, dry_run=False, model=None):
    nc = dict(cfg["numbers"])
    if model:
        nc["model"] = model
    tracks, teams = run.csv("tracks.csv"), run.csv("teams.csv")
    picks = pick_crops(run, tracks, teams, nc["crops_per_track"])
    items = [(tid, p) for tid, ps in picks.items() for p in ps]
    cache_p = run.p("cache", "numbers.json")
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    key = lambda p: hashlib.md5((nc["model"] + str(nc["crop_long_edge"])).encode() + open(p, "rb").read()).hexdigest()
    todo = [(tid, p) for tid, p in items if key(p) not in cache]
    est_tokens = len(todo) * 110 + (len(todo) // nc["images_per_request"] + 1) * 250
    log(f"numbers: {len(picks)} player tracks, {len(items)} crops, {len(todo)} uncached "
        f"(~{est_tokens / 1000:.0f}k input tokens with {nc['model']})")
    if nc["provider"] == "none":
        return None
    offline = dry_run or not os.environ.get("ANTHROPIC_API_KEY")
    if offline:
        log("numbers: " + ("dry run" if dry_run else "ANTHROPIC_API_KEY not set") +
            " - no API calls; voting with cached reads only")
        pd.DataFrame(todo, columns=["track_id", "crop"]).to_csv(run.p("numbers_dryrun.csv"), index=False)
        sheet = [cv2.resize(cv2.imread(str(p)), (72, 108)) for _, p in todo[:60]]
        if sheet:
            sheet += [np.zeros((108, 72, 3), np.uint8)] * (-len(sheet) % 12)
            cv2.imwrite(str(run.p("debug", "number_crops.jpg")),
                        np.vstack([np.hstack(sheet[i:i + 12]) for i in range(0, len(sheet), 12)]))
        if len(todo) == len(items):
            return None
        todo = []
    tin = tout = 0
    client = None
    if todo:
        import anthropic
        # optional: route requests to a specific Anthropic workspace
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    for i in range(0, len(todo), nc["images_per_request"]):
        batch = todo[i:i + nc["images_per_request"]]
        reads, (a, b) = read_batch(client, nc["model"], batch, nc["crop_long_edge"])
        tin, tout = tin + a, tout + b
        for (tid, p), r in zip(batch, reads):
            cache[key(p)] = r
        cache_p.write_text(json.dumps(cache))
        log(f"  read {min(i + len(batch), len(todo))}/{len(todo)} crops")
    roster = None
    if nc.get("roster"):
        roster = pd.read_csv(resolve(nc["roster"]), dtype={"number": str})
    team_of = dict(zip(teams.track_id, teams.team))
    rows = []
    for tid, ps in picks.items():
        reads = [cache.get(key(p), {}) for p in ps]
        num, cnt, share = vote(reads, nc["min_votes"], nc["min_share"])
        name, in_roster = None, None
        if num and roster is not None:
            hit = roster[(roster.team == team_of.get(tid)) & (roster.number == num)]
            in_roster = not hit.empty
            if in_roster:
                name = hit.name.iloc[0]
            # a number missing from the roster is kept (rosters change mid-season) but flagged
        rows.append(dict(track_id=tid, team=team_of.get(tid), number=num, name=name, in_roster=in_roster,
                         votes=cnt, share=round(share, 2), reads=json.dumps([r.get("number") for r in reads])))
    out = pd.DataFrame(rows)
    out.to_csv(run.p("numbers.csv"), index=False)
    log(f"numbers: {out.number.notna().sum()}/{len(out)} tracks identified; "
        f"this run used {tin} input / {tout} output tokens")
    return out
