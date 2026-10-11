"""NLL floor geometry (feet, origin at centre spot).

x runs along the floor's length (negative = left end as seen from the main camera),
y runs across it (positive = far boards, i.e. the side opposite the main camera).
"""
from __future__ import annotations

import numpy as np


class Floor:
    def __init__(self, cfg: dict):
        f = cfg["floor"]
        self.L, self.W = f["length_ft"], f["width_ft"]
        self.R = f["corner_radius_ft"]
        self.gx = self.L / 2 - f["goal_from_end_ft"]          # |x| of goal lines
        self.cr = f["crease_radius_ft"]
        self.rx = f["restraining_from_center_ft"]
        self.fo = f["faceoff_radius_ft"]

    # ------------------------------------------------------------- landmarks for clicking
    def landmarks(self) -> dict[str, tuple[float, float]]:
        L2, W2, g, r, c = self.L / 2, self.W / 2, self.gx, self.rx, self.cr
        lm = {"center_spot": (0, 0), "center_far": (0, W2), "center_near": (0, -W2),
              "faceoff_far": (0, self.fo), "faceoff_near": (0, -self.fo)}
        for side, s in (("left", -1), ("right", 1)):
            lm.update({
                f"{side}_goal": (s * g, 0),
                f"{side}_crease_front": (s * (g - c), 0),
                f"{side}_crease_back": (s * (g + c), 0),
                f"{side}_crease_far": (s * g, c),
                f"{side}_crease_near": (s * g, -c),
                f"{side}_restraining_far": (s * r, W2),
                f"{side}_restraining_near": (s * r, -W2),
                f"{side}_end_center": (s * L2, 0),
            })
        return lm

    # ------------------------------------------------------------- model curves
    def curves(self, step=1.0) -> dict[str, list[np.ndarray]]:
        """Sampled polylines grouped by how they appear in the image:
        'white' painted lines, 'crease' circle boundaries, 'boards' floor/board boundary."""
        L2, W2 = self.L / 2, self.W / 2
        ys = np.arange(-W2, W2 + step, step)
        # restraining lines, centre line, and the goal line extended across each crease
        white = [np.c_[np.full_like(ys, x), ys] for x in (-self.rx, 0.0, self.rx)]
        gy = np.linspace(-self.cr, self.cr, 20)
        white += [np.c_[np.full_like(gy, s * self.gx), gy] for s in (-1, 1)]
        th = np.linspace(0, 2 * np.pi, 120)
        crease = []
        for s in (-1, 1):
            crease.append(np.c_[s * self.gx + self.cr * np.cos(th), self.cr * np.sin(th)])
        boards = [self.outline(step)]
        return {"white": white, "crease": crease, "boards": boards}

    def outline(self, step=1.0) -> np.ndarray:
        """Board line with rounded corners, closed polyline."""
        L2, W2, R = self.L / 2, self.W / 2, self.R
        pts = []
        for cx, cy, a0 in ((L2 - R, W2 - R, 0), (-L2 + R, W2 - R, 90), (-L2 + R, -W2 + R, 180), (L2 - R, -W2 + R, 270)):
            a = np.deg2rad(np.arange(a0, a0 + 91, 3))
            pts.append(np.c_[cx + R * np.cos(a), cy + R * np.sin(a)])
        pts.append(pts[0][:1])
        return np.vstack(pts)

    def inside(self, x, y, margin=3.0):
        return (np.abs(x) <= self.L / 2 + margin) & (np.abs(y) <= self.W / 2 + margin)

    # ------------------------------------------------------------- drawing
    def draw(self, ax, face="#1f5130", line="white"):
        import matplotlib.patches as mp
        L2, W2 = self.L / 2, self.W / 2
        o = self.outline()
        ax.fill(o[:, 0], o[:, 1], color=face, zorder=0)
        ax.plot(o[:, 0], o[:, 1], color="#c9a227", lw=2.5, zorder=1)
        for x in (-self.rx, self.rx):
            ax.plot([x, x], [-W2, W2], color=line, lw=1.5, zorder=1)
        ax.plot([0, 0], [-W2, W2], color=line, lw=0.8, ls=":", zorder=1)
        ax.add_patch(mp.Circle((0, 0), self.fo, fill=False, ec=line, lw=1.2, zorder=1))
        for s in (-1, 1):
            ax.add_patch(mp.Circle((s * self.gx, 0), self.cr, fc="#24479c", ec=line, lw=1.5, zorder=1))
            ax.plot([s * self.gx, s * self.gx], [-2.4, 2.4], color="#d33", lw=3, zorder=2)
        ax.set_xlim(-L2 - 3, L2 + 3)
        ax.set_ylim(-W2 - 3, W2 + 3)
        ax.set_aspect("equal")
        ax.axis("off")
