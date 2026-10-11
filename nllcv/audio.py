"""Audio energy track for goal confirmation (goal horn + crowd roar).

Per 0.1 s hop: loudness (RMS dB), its robust z-score against a rolling 30 s baseline, and spectral
flatness (a horn is tonal, so flatness drops while loudness jumps). A 'spike' is a run where the
z-score stays above `spike_z` for at least `min_spike_s`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.ndimage import median_filter

from .common import Run, log


def analyse(run: Run, cfg: dict) -> pd.DataFrame | None:
    if run.audio is None or not run.audio.exists():
        return None
    out = run.root / "audio.csv"
    if out.exists() and out.stat().st_mtime > run.audio.stat().st_mtime and not cfg.get("_force"):
        return pd.read_csv(out)
    ac = cfg["audio"]
    sr, x = wavfile.read(run.audio)
    x = x.astype(np.float32) / 32768.0
    hop = int(ac["hop_s"] * sr)
    win = 2 * hop
    n = (len(x) - win) // hop
    frames = np.lib.stride_tricks.as_strided(x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(win, 1 / sr)
    band = (freqs > 150) & (freqs < 5000)
    sb = spec[:, band]
    flat = np.exp(np.log(sb).mean(1)) / sb.mean(1)
    k = max(3, int(ac["baseline_window_s"] / ac["hop_s"]) | 1)
    base = median_filter(db, size=k, mode="nearest")
    mad = median_filter(np.abs(db - base), size=k, mode="nearest") * 1.4826 + 0.5
    z = (db - base) / mad
    df = pd.DataFrame({"t": np.arange(n) * ac["hop_s"] + ac["hop_s"], "rms_db": db.round(2),
                       "z": z.round(3), "flatness": flat.round(4)})
    df.to_csv(out, index=False)
    log(f"audio: {len(df)} hops, {(df.z > ac['spike_z']).mean():.1%} above spike threshold")
    return df


def spikes(audio: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Contiguous runs of z > spike_z lasting at least min_spike_s."""
    ac = cfg["audio"]
    above = (audio.z > ac["spike_z"]).values
    out = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            dur = (j - i) * ac["hop_s"]
            if dur >= ac["min_spike_s"]:
                seg = audio.iloc[i:j]
                out.append(dict(start=float(seg.t.iloc[0]), end=float(seg.t.iloc[-1]), dur=dur,
                                peak_z=float(seg.z.max()), mean_z=float(seg.z.mean()),
                                min_flatness=float(seg.flatness.min())))
            i = j
        else:
            i += 1
    return pd.DataFrame(out, columns=["start", "end", "dur", "peak_z", "mean_z", "min_flatness"])
