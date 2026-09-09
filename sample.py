"""
sample.py — load a trained DDPM and generate samples.

Run:
    .venv/bin/python sample.py --dataset moons
    .venv/bin/python sample.py --dataset spiral --n 6000 --trajectory

Outputs go to ./outputs/<dataset>/:
    generated.png     scatter of freshly generated samples (vs. real data)
    trajectory.png    (with --trajectory) 6 snapshots of noise -> data

This script only *runs* the reverse process; all the learning happened in
train.py. It's here to show that sampling is decoupled from training — once you
have the weights, you can generate as many points as you like.
"""

from __future__ import annotations

import argparse
import os

import torch

from data import make_dataset
from diffusion import GaussianDiffusion
from model import DenoiseMLP
from viz import scatter_compare, HAVE_MPL


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DenoiseMLP(data_dim=2, hidden=cfg["hidden"], n_blocks=cfg["blocks"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def plot_trajectory(traj, path: str):
    """Show 6 evenly spaced snapshots of the reverse process (noise -> data)."""
    if not HAVE_MPL:
        print("  (matplotlib not available; skipping trajectory plot)")
        return
    import matplotlib.pyplot as plt
    idxs = [int(i) for i in torch.linspace(0, len(traj) - 1, 6).tolist()]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    for ax, i in zip(axes, idxs):
        pts = traj[i].numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, c="#5a189a")
        step = len(traj) - 1 - i  # traj[0] is t=T (pure noise), last is t=0
        ax.set_title(f"t={step}")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("reverse diffusion: pure noise (left) -> generated data (right)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Sample from a trained DDPM")
    p.add_argument("--dataset", default="moons")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--trajectory", action="store_true",
                   help="also save snapshots of the denoising process")
    p.add_argument("--device", default="auto")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    outdir = args.outdir or os.path.join("outputs", args.dataset)
    ckpt_path = os.path.join(outdir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"no checkpoint at {ckpt_path} — run train.py first "
                         f"(e.g. .venv/bin/python train.py --dataset {args.dataset})")

    model, cfg = load_model(ckpt_path, device)
    diffusion = GaussianDiffusion(T=cfg["timesteps"], device=device)
    print(f"loaded {ckpt_path}  (dataset={cfg['dataset']}, T={cfg['timesteps']})")

    real = make_dataset(cfg["dataset"], n=args.n, seed=cfg["seed"])
    if args.trajectory:
        gen, traj = diffusion.sample(model, n=args.n, dim=2, return_trajectory=True)
        plot_trajectory(traj, os.path.join(outdir, "trajectory.png"))
        print(f"saved -> {outdir}/trajectory.png")
    else:
        gen = diffusion.sample(model, n=args.n, dim=2)

    scatter_compare(real, gen, os.path.join(outdir, "generated.png"),
                    title=f"generated samples: '{cfg['dataset']}'")
    print(f"saved -> {outdir}/generated.png")


if __name__ == "__main__":
    main()
