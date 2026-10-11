"""Config, run folder, and the bridge to the nllcv pipeline (nllcv/ in this repository)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str):
    print(f"[laxnet] {msg}", flush=True)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "laxnet.yaml"
    cfg = yaml.safe_load(open(path))
    cfg["_path"] = str(path)
    return cfg


class Run:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def p(self, *parts) -> Path:
        out = self.root.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def exists(self, *parts) -> bool:
        return self.root.joinpath(*parts).exists()

    def csv(self, name) -> pd.DataFrame:
        return pd.read_csv(self.root / name)

    def save_json(self, name, obj):
        self.p(name).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- nllcv bridge
class NLL:
    """Read-only access to an nllcv run: shots, scoreboard spine, events, possession, homographies,
    plus the nllcv floor model and helpers."""

    def __init__(self, cfg: dict):
        root = resolve(cfg["nll"]["root"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import nllcv  # noqa: F401
        from nllcv import common, floor
        self.common, self.floor_mod = common, floor
        self.cfg = common.load_config(cfg["nll"].get("config"))
        self.run = common.Run(resolve(cfg["nll"]["run"]))
        self.floor = floor.Floor(self.cfg)
        self.teams = (self.cfg["game"]["away"], self.cfg["game"]["home"])

    def table(self, name) -> pd.DataFrame | None:
        p = self.run.root / name
        return pd.read_csv(p) if p.exists() else None

    @property
    def shots(self):
        return self.table("shots.csv")

    @property
    def spine(self):
        return self.table("spine.csv")

    @property
    def events(self):
        return self.table("events.csv")

    @property
    def possession(self):
        return self.table("possession.csv")

    def homographies(self):
        """(times, H image->floor) from the nllcv homography stage (only verified frames)."""
        p = self.run.root / "homographies.npz"
        if not p.exists():
            return np.zeros(0), np.zeros((0, 3, 3))
        z = np.load(p, allow_pickle=True)
        return z["t"].astype(float), z["H"].astype(float)

    def defended_ends(self) -> dict:
        """period -> {end_sign: team defending it}, learned by nllcv from goalie positions."""
        pos = self.possession
        out = {}
        if pos is None or "end" not in pos:
            return out
        p = pos.dropna(subset=["team_smooth"])
        p = p[(p.method == "floor_end") & (p.end != 0)]
        A, B = self.teams
        for per, g in p.groupby("period"):
            # attacking team at end e -> the other team defends e
            for e in (-1, 1):
                ge = g[g.end == e]
                if len(ge):
                    att = ge.team_smooth.mode().iloc[0]
                    out.setdefault(int(per), {})[e] = B if att == A else A
        return out


class HomographyTrack:
    """Image->floor homography at any time t: the nearest verified frame, or element-wise interpolation
    between two verified frames less than max_gap apart (the broadcast camera pans smoothly)."""

    def __init__(self, times, Hs, max_gap):
        o = np.argsort(times)
        self.t, self.H = np.asarray(times)[o], np.asarray(Hs)[o]
        self.H = self.H / self.H[:, 2:3, 2:3] if len(self.H) else self.H
        self.max_gap = max_gap

    def __call__(self, t):
        if not len(self.t):
            return None
        j = int(np.searchsorted(self.t, t))
        if j < len(self.t) and abs(self.t[j] - t) < 0.06:
            return self.H[j]
        if j > 0 and abs(self.t[j - 1] - t) < 0.06:
            return self.H[j - 1]
        if 0 < j < len(self.t) and self.t[j] - self.t[j - 1] <= self.max_gap:
            w = (t - self.t[j - 1]) / (self.t[j] - self.t[j - 1])
            return (1 - w) * self.H[j - 1] + w * self.H[j]
        return None


def project(H, pts) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]
