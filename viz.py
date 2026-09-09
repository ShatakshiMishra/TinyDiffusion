"""
viz.py — plotting helpers.

The point of this whole project is to *see* the model work, so we need to draw
2D scatter plots. We use matplotlib when it is available, but fall back to a
tiny hand-written SVG writer if it is not — that way the project produces its
visual output on any Python with numpy, with zero plotting dependencies.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend: write files, never open a window
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover - only hit when matplotlib is missing
    HAVE_MPL = False


def _to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def scatter_compare(real, fake, path: str, title: str = "") -> str:
    """Side-by-side scatter of real data vs. generated samples."""
    real, fake = _to_np(real), _to_np(fake)
    if HAVE_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for ax, pts, name, color in [
            (axes[0], real, "real data", "#2a6f97"),
            (axes[1], fake, "generated", "#c1121f"),
        ]:
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, c=color)
            ax.set_title(name)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    else:
        _svg_compare(real, fake, path, title)
    return path


def loss_curve(losses, path: str) -> str:
    """Plot the training loss over steps."""
    losses = list(losses)
    if HAVE_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(losses, lw=1.2, c="#2a6f97")
        ax.set_xlabel("training step (x logging interval)")
        ax.set_ylabel("MSE noise-prediction loss")
        ax.set_title("training loss")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    else:
        _svg_line(losses, path)
    return path


# --------------------------------------------------------------------------
# Minimal dependency-free SVG fallback (only used if matplotlib is missing).
# --------------------------------------------------------------------------
def _svg_points(pts: np.ndarray, w: int, h: int, pad: int, color: str) -> str:
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    sx = (pts[:, 0] - lo[0]) / span[0] * (w - 2 * pad) + pad
    sy = h - ((pts[:, 1] - lo[1]) / span[1] * (h - 2 * pad) + pad)
    return "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="{color}" '
        f'fill-opacity="0.5"/>' for x, y in zip(sx, sy)
    )


def _svg_compare(real, fake, path, title):
    w = h = 400
    left = f'<g>{_svg_points(real, w, h, 20, "#2a6f97")}</g>'
    right = (f'<g transform="translate({w},0)">'
             f'{_svg_points(fake, w, h, 20, "#c1121f")}</g>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{2*w}" height="{h+30}">'
        f'<rect width="{2*w}" height="{h+30}" fill="white"/>'
        f'<text x="10" y="20" font-size="14">real data</text>'
        f'<text x="{w+10}" y="20" font-size="14">generated</text>'
        f'<g transform="translate(0,30)">{left}{right}</g></svg>'
    )
    with open(path, "w") as f:
        f.write(svg)


def _svg_line(vals, path):
    w, h, pad = 600, 400, 40
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        vals = np.array([vals[0], vals[0]]) if len(vals) else np.array([0.0, 0.0])
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-6)
    xs = np.linspace(pad, w - pad, len(vals))
    ys = h - ((vals - lo) / span * (h - 2 * pad) + pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" fill="white"/>'
        f'<polyline fill="none" stroke="#2a6f97" stroke-width="1.5" '
        f'points="{pts}"/></svg>'
    )
    with open(path, "w") as f:
        f.write(svg)
