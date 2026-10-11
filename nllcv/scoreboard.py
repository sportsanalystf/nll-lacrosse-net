"""Stage 3 - scorebug OCR -> game state and a clock-indexed spine.

Decoding: only the full-resolution scorebug strip is pulled out of ffmpeg, at `ocr_fps`, and only
samples where stage 2 saw the bug are read.

Reading: the bug font never changes, so digits are read by a self-trained glyph matcher:
    1. binarise each field, split it into glyphs with connected components
    2. the first time a field image is seen, Tesseract reads it; if its digit count matches the glyph
       count, each glyph is added to a per-field exemplar bank
    3. once every digit 0-9 has exemplars, glyphs are classified by normalised cross-correlation,
       which is exact for a fixed font and ~1000x faster than spawning Tesseract.
Low-similarity glyphs fall back to Tesseract, so unseen digits keep the bank growing.

Cleaning:
    period, scores  - a value is accepted only after `stable_samples` identical consecutive reads
    game clock      - a read must agree with a neighbouring read in the same shot under
                      "running (-1 s/s) or stopped" dynamics
    shot clock      - must agree with a neighbour, resets (jumps up) are allowed

Writes scoreboard.csv (one row per OCR sample) and spine.csv (one row per broadcast second, with
the elapsed game time where the game clock is known).
"""
from __future__ import annotations

import hashlib
import re

import cv2
import numpy as np
import pandas as pd

from .common import Run, iter_frames, log, scale_box
from . import shots as shots_mod

GLYPH = (20, 32)  # w, h of normalised glyph


# --------------------------------------------------------------------------- image prep
def binarise(crop: np.ndarray, up: int) -> np.ndarray:
    """Return a text-dark-on-white binary image, upscaled, with a white border."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # text is the minority colour: make it black
    if (b > 0).mean() < 0.5:
        b = 255 - b
    # drop components touching the border (panel edges, logo slivers)
    inv = 255 - b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    H, W = b.shape
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            b[lab == k] = 255
    return cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _split_touching(g: np.ndarray, h: int, digit_ratio: float = 0.62):
    """Bold scorebug digits sometimes touch ("13"). Split components wider than one digit at the
    weakest ink columns near the expected digit boundaries."""
    w = g.shape[1]
    k = int(round(w / (digit_ratio * h)))
    if w < 0.95 * h or k < 2:
        return [(0, g)]
    ink = (g > 0).sum(0).astype(float)
    cuts = [0]
    for j in range(1, k):
        c = int(j * w / k)
        lo, hi = max(1, c - w // (3 * k)), min(w - 1, c + w // (3 * k))
        cuts.append(lo + int(np.argmin(ink[lo:hi])) if hi > lo else c)
    cuts.append(w)
    return [(cuts[i], g[:, cuts[i]:cuts[i + 1]]) for i in range(k) if cuts[i + 1] - cuts[i] > 2]


def glyphs(b: np.ndarray):
    """Split a binary field into (digit glyphs, separator string) ordered left to right."""
    inv = 255 - b
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    if n <= 1:
        return [], []
    hs = stats[1:, 3]
    tall = hs.max()
    items = []
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a < 6:
            continue
        if h >= 0.55 * tall:
            g = (lab[y:y + h, x:x + w] == k).astype(np.uint8) * 255
            for xo, piece in _split_touching(g, h):
                items.append(("g", x + xo, cv2.resize(piece, GLYPH, interpolation=cv2.INTER_AREA).astype(np.float32)))
        elif h < 0.35 * tall and w < 0.6 * tall:
            items.append(("dot", x, (y + h / 2) / b.shape[0]))
    items.sort(key=lambda it: it[1])
    # merge dots into separators: two dots at the same x -> ':', one low dot -> '.'
    out, seps = [], []
    i = 0
    while i < len(items):
        kind, x, payload = items[i]
        if kind == "g":
            out.append(payload)
            seps.append("")
        else:
            same = [it for it in items[i:i + 2] if it[0] == "dot" and abs(it[1] - x) < 0.3 * tall]
            sym = ":" if len(same) == 2 else ("." if payload > 0.55 else ":")
            if seps:
                seps[-1] = sym
            i += len(same) - 1
        i += 1
    return out, seps


def _norm(g):
    v = g.ravel() - g.mean()
    return v / (np.linalg.norm(v) + 1e-6)


class GlyphBank:
    def __init__(self, max_per_digit=25):
        self.ex = {str(d): [] for d in range(10)}
        self.max = max_per_digit

    def ready(self, need=None):
        need = need or [str(d) for d in range(10)]
        return all(len(self.ex[d]) >= 1 for d in need)

    def add(self, ch, g):
        if ch in self.ex and len(self.ex[ch]) < self.max:
            self.ex[ch].append(_norm(g))

    def classify(self, g):
        v = _norm(g)
        best, bs = None, -1.0
        for ch, lst in self.ex.items():
            for e in lst:
                s = float(v @ e)
                if s > bs:
                    best, bs = ch, s
        return best, bs


# --------------------------------------------------------------------------- tesseract
def _tess(b: np.ndarray, whitelist: str) -> tuple[str, float]:
    import pytesseract
    d = pytesseract.image_to_data(b, config=f"--psm 7 -c tessedit_char_whitelist={whitelist}",
                                  output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if str(w).strip()]
    if not words:
        return "", 0.0
    return "".join(w for w, _ in words), float(np.mean([c for _, c in words]))


class FieldReader:
    """Reads one scorebug field, caching by image hash and learning glyphs."""

    def __init__(self, kind: str, bank: GlyphBank | None):
        self.kind = kind
        self.bank = bank
        self.cache: dict[str, tuple[str, float, str]] = {}
        self.near_vecs: list[np.ndarray] = []   # near-duplicate cache (compression noise breaks hashing)
        self.near_res: list[tuple[str, float, str]] = []
        self.calls = 0

    def read(self, b: np.ndarray) -> tuple[str, float, str]:
        key = hashlib.md5(np.packbits(cv2.resize(b, (96, 32)) > 127).tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        v = _norm(cv2.resize(b, (72, 24), interpolation=cv2.INTER_AREA).astype(np.float32))
        if self.near_vecs:
            sims = np.stack(self.near_vecs) @ v
            j = int(np.argmax(sims))
            if sims[j] >= 0.985:
                self.cache[key] = self.near_res[j]
                return self.near_res[j]
        res = self._read(b)
        self.cache[key] = res
        if res[1] >= 60:
            self.near_vecs.append(v)
            self.near_res.append(res)
            if len(self.near_vecs) > 400:
                self.near_vecs.pop(0)
                self.near_res.pop(0)
        return res

    def _read(self, b):
        if self.kind == "period":
            self.calls += 1
            txt, conf = _tess(b, "1234STNDRHO")
            return txt, conf, "tess"
        gl, seps = glyphs(b)
        if not gl:
            return "", 0.0, "empty"
        if self.bank is not None:
            chars, sims = [], []
            for g in gl:
                ch, s = self.bank.classify(g)
                chars.append(ch)
                sims.append(s)
            # a fixed font gives ~0.95+ for the right digit and well under 0.85 for the others
            if chars and None not in chars and min(sims) >= 0.88:
                return "".join(c + s for c, s in zip(chars, seps)), 100 * min(sims), "glyph"
        self.calls += 1
        txt, conf = _tess(b, "0123456789:.")
        digits = re.sub(r"\D", "", txt)
        if self.bank is not None and len(digits) == len(gl) and conf >= 60:
            for ch, g in zip(digits, gl):
                self.bank.add(ch, g)
        return txt, conf, "tess"


# --------------------------------------------------------------------------- parsing
def parse(kind: str, txt: str):
    if not txt:
        return np.nan
    if kind == "int":
        d = re.sub(r"\D", "", txt)
        return float(int(d)) if d and len(d) <= 2 else np.nan
    if kind == "period":
        # the ordinal suffix is far more legible than the small digit in front of it
        t = txt.upper()
        if "OT" in t or t.startswith("O"):
            m = re.match(r"(\d)", t)
            return 4.0 + (int(m.group(1)) if m else 1)
        for suf, p in (("ST", 1), ("ND", 2), ("RD", 3), ("TH", 4)):
            if t.endswith(suf):
                return float(p)
        m = re.match(r"([1-4])", t)
        return float(m.group(1)) if m else np.nan
    if kind == "clock":
        t = txt.strip()
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", t)
        if m:
            mm, ss = int(m.group(1)), int(m.group(2))
            return float(mm * 60 + ss) if ss < 60 else np.nan
        m = re.fullmatch(r"(\d{1,2})\.(\d)", t)
        if m:
            return int(m.group(1)) + int(m.group(2)) / 10
        d = re.sub(r"\D", "", t)
        if len(d) in (3, 4):  # colon lost
            mm, ss = int(d[:-2]), int(d[-2:])
            return float(mm * 60 + ss) if ss < 60 else np.nan
        return np.nan
    return np.nan


# --------------------------------------------------------------------------- cleaning
def stable_steps(vals: np.ndarray, k: int, monotone: bool) -> np.ndarray:
    """Accept a new value only after k identical consecutive reads; forward fill."""
    out = np.full(len(vals), np.nan)
    cur, cand, run = np.nan, np.nan, 0
    for i, v in enumerate(vals):
        if np.isnan(v):
            out[i] = cur
            continue
        if v == cand:
            run += 1
        else:
            cand, run = v, 1
        if run >= k and v != cur:
            if not monotone or np.isnan(cur) or v > cur or run >= 4 * k:
                cur = v
        out[i] = cur
    # backfill the first accepted value to the start
    first = np.where(~np.isnan(out))[0]
    if len(first):
        out[:first[0]] = out[first[0]]
    return out


def consistent_clock(t, c, shot_ids, horizon=4.0):
    """Keep a read if some neighbour in the same shot agrees with it under 'stopped or running down
    at 1 s/s'. Shot-clock resets need no special case: the post-reset values agree with each other."""
    keep = np.zeros(len(c), bool)
    for i in range(len(c)):
        if np.isnan(c[i]):
            continue
        for j in range(max(0, i - 10), min(len(c), i + 11)):
            if j == i or np.isnan(c[j]) or shot_ids[j] != shot_ids[i]:
                continue
            dt = t[j] - t[i]
            if abs(dt) > horizon:
                continue
            dc = c[j] - c[i]
            later, earlier = (dc, dt) if dt > 0 else (-dc, -dt)
            # clock either stopped or ran down at 1 s/s (with read jitter)
            if -earlier - 1.2 <= later <= 0.6:
                keep[i] = True
                break
    return keep


# --------------------------------------------------------------------------- main
def run(run: Run, cfg: dict) -> pd.DataFrame:
    sb, sc = cfg["scorebug"], cfg["scoreboard"]
    man = run.manifest
    size = (man["width"], man["height"])
    crop = scale_box(sb["crop"], sb["ref_size"], size)
    fields = {}
    for name, spec in sb["fields"].items():
        x, y, w, h = scale_box(spec["box"], sb["ref_size"], size)
        fields[name] = (x - crop[0], y - crop[1], w, h, spec["kind"])

    ff = np.load(run.root / "frame_features.npz")
    ft, fbug = ff["t"], ff["bug"]
    shots = shots_mod.load(run)

    banks = {"score": GlyphBank(), "clock": GlyphBank(), "shot_clock": GlyphBank()}
    readers = {}
    for name, (_, _, _, _, kind) in fields.items():
        bank = None if kind == "period" else banks["score" if name.endswith("score") else name]
        readers[name] = FieldReader(kind, bank)

    rows = []
    fps = sc["ocr_fps"]
    up = sc["upscale"]
    dbg_saved = False
    for t, strip in iter_frames(run.video, fps=fps, crop=crop):
        i = int(np.clip(np.searchsorted(ft, t), 0, len(ft) - 1))
        if not fbug[max(0, i - 1):i + 2].any():
            continue
        row = {"t": round(t, 3)}
        for name, (x, y, w, h, kind) in fields.items():
            b = binarise(strip[y:y + h, x:x + w], up)
            txt, conf, src = readers[name].read(b)
            row[name + "_txt"], row[name + "_conf"], row[name + "_src"] = txt, round(conf, 1), src
            row[name] = parse(kind, txt)
        if not dbg_saved:
            cv2.imwrite(str(run.p("debug", "scorebug_strip.png")), strip)
            dbg_saved = True
        rows.append(row)
        if len(rows) % 200 == 0:
            log(f"  ocr: {t / 60:.1f} min, {sum(r.calls for r in readers.values())} tesseract calls")
    df = pd.DataFrame(rows)
    if df.empty:
        log("scoreboard: no frames with the bug visible - check scorebug boxes in the config")
        return df

    # ---- clean
    t = df.t.values
    sid = np.array([int(shots[(shots.start <= x) & (shots.end > x)].shot_id.iloc[0])
                    if ((shots.start <= x) & (shots.end > x)).any() else -1 for x in t])
    df["shot_id"] = sid
    k = sc["stable_samples"]
    for col, lo, hi in [("away_score", 0, 40), ("home_score", 0, 40), ("period", 1, 7)]:
        v = df[col].where((df[col] >= lo) & (df[col] <= hi)).values.astype(float)
        df[col + "_clean"] = stable_steps(v, k, monotone=True)
    per_len = cfg["game"]["period_length_s"]
    clk = df["clock"].where((df["clock"] >= 0) & (df["clock"] <= per_len)).values.astype(float)
    df["clock_clean"] = np.where(consistent_clock(t, clk, sid), clk, np.nan)
    sclk = df["shot_clock"].where((df["shot_clock"] >= 0) & (df["shot_clock"] <= cfg["game"]["shot_clock_s"])) \
        .values.astype(float)
    df["shot_clock_clean"] = np.where(consistent_clock(t, sclk, sid), sclk, np.nan)
    df["game_s"] = (df.period_clean - 1) * per_len + (per_len - df.clock_clean)
    df.to_csv(run.p("scoreboard.csv"), index=False)

    # ---- spine: one row per broadcast second
    dur = man["duration"]
    spine = pd.DataFrame({"t": np.arange(0, int(dur) + 1, 1.0)})
    spine["shot_label"] = [shots_mod.label_at(shots, x) for x in spine.t]
    good = df.dropna(subset=["game_s"])
    gs = np.full(len(spine), np.nan)
    for s_id, grp in good.groupby("shot_id"):
        srow = shots[shots.shot_id == s_id].iloc[0]
        m = (spine.t >= srow.start) & (spine.t < srow.end)
        if len(grp) >= 2:
            gs[m] = np.interp(spine.t[m], grp.t, grp.game_s)
        else:
            gs[m] = grp.game_s.iloc[0]
    spine["game_s"] = gs
    for col in ["period_clean", "away_score_clean", "home_score_clean"]:
        spine[col.replace("_clean", "")] = np.interp(spine.t, df.t, df[col].ffill().bfill(), left=np.nan) \
            .round()
    spine["clock_s"] = np.where(np.isnan(gs), np.nan, cfg["game"]["period_length_s"] -
                                (gs - (spine.period - 1) * cfg["game"]["period_length_s"]))
    spine.to_csv(run.p("spine.csv"), index=False)

    calls = {n: r.calls for n, r in readers.items()}
    ok = df[["clock_clean", "shot_clock_clean"]].notna().mean().round(3).to_dict()
    log(f"scoreboard: {len(df)} samples, tesseract calls {calls}, valid fraction {ok}")
    last = df.iloc[-1]
    log(f"  final: {cfg['game']['away']} {last.away_score_clean:.0f} - {last.home_score_clean:.0f} "
        f"{cfg['game']['home']}  (period {last.period_clean:.0f})")
    return df


def load(run: Run) -> pd.DataFrame:
    return pd.read_csv(run.root / "scoreboard.csv")
