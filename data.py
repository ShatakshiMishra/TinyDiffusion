"""
data.py — self-contained 2D toy datasets for the diffusion model.

WHY 2D TOY DATA?
    A diffusion model learns to turn pure noise into samples that look like the
    training data. If the data is a 2D point cloud, we can *plot* both the real
    data and the generated data on the same axes and literally see whether the
    model learned the shape. No image downloads, no internet, no GPU farm — the
    whole thing trains in seconds on an Apple-silicon MPS device.

    Everything here is generated with numpy from a seed, so the project is
    100% reproducible and has zero external data dependencies.

Each dataset function returns a float32 torch tensor of shape (N, 2), roughly
centered at 0 with a spread of order 1 (we standardize at the end). "Roughly
unit scale" matters: the noise schedule below assumes the data lives near the
same magnitude as a standard Gaussian.
"""

import numpy as np
import torch


def _standardize(points: np.ndarray) -> np.ndarray:
    """Center to mean 0 and scale to unit std, per coordinate.

    Diffusion adds standard-normal noise. If the data had, say, std 5, the
    forward process would need many more steps to drown it in noise. Rescaling
    the data to unit std lets a short, gentle noise schedule work well.
    """
    points = points - points.mean(axis=0, keepdims=True)
    points = points / (points.std(axis=0, keepdims=True) + 1e-8)
    return points.astype(np.float32)


def two_moons(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two interleaving half-circles — the classic 'moons' shape."""
    n1 = n // 2
    n2 = n - n1
    # Upper moon: a half circle from 0..pi
    t1 = rng.uniform(0, np.pi, n1)
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
    # Lower moon: shifted right and down, flipped
    t2 = rng.uniform(0, np.pi, n2)
    x2 = np.stack([1.0 - np.cos(t2), -np.sin(t2) - 0.5], axis=1)
    pts = np.concatenate([x1, x2], axis=0)
    pts += rng.normal(0, 0.06, pts.shape)  # a little jitter so it's not a razor line
    return _standardize(pts)


def spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    """A single Archimedean spiral — a harder, curved manifold."""
    t = np.sqrt(rng.uniform(0, 1, n)) * 3.0 * np.pi  # sqrt => uniform along arc length
    r = t
    x = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    x += rng.normal(0, 0.25, x.shape)
    return _standardize(x)


def eight_gaussians(n: int, rng: np.random.Generator) -> np.ndarray:
    """Eight Gaussian blobs arranged on a ring — tests multi-modal coverage.

    A common failure mode of generative models is 'mode collapse': producing
    only some of the modes. Eight separated blobs make that easy to spot.
    """
    centers = np.array(
        [[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    ) * 2.0
    idx = rng.integers(0, 8, n)
    pts = centers[idx] + rng.normal(0, 0.1, (n, 2))
    return _standardize(pts)


def checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    """A 2D checkerboard of uniform squares — sharp, disconnected support."""
    pts = []
    while len(pts) < n:
        x = rng.uniform(-2, 2, (n, 2))
        # keep a point only if it lands on a 'black' square of the board
        keep = (np.floor(x[:, 0]) + np.floor(x[:, 1])) % 2 == 0
        pts.extend(x[keep].tolist())
    pts = np.array(pts[:n])
    return _standardize(pts)


DATASETS = {
    "moons": two_moons,
    "spiral": spiral,
    "gaussians": eight_gaussians,
    "checkerboard": checkerboard,
}


def make_dataset(name: str, n: int = 4000, seed: int = 0) -> torch.Tensor:
    """Build one of the named datasets as a (N, 2) float32 tensor."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")
    rng = np.random.default_rng(seed)
    pts = DATASETS[name](n, rng)
    return torch.from_numpy(pts)


if __name__ == "__main__":
    # Quick sanity check: shapes, scale, and a text preview of the point cloud.
    for name in DATASETS:
        x = make_dataset(name, n=1000, seed=0)
        print(f"{name:12s} shape={tuple(x.shape)} "
              f"mean={x.mean(0).tolist()} std={x.std(0).tolist()}")
