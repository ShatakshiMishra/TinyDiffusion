"""
train.py — train the from-scratch DDPM and save proof it worked.

Run:
    .venv/bin/python train.py                      # defaults: moons, 3000 steps
    .venv/bin/python train.py --dataset spiral
    .venv/bin/python train.py --dataset gaussians --steps 5000

Outputs (written to ./outputs/<dataset>/):
    model.pt        the trained weights + config (used by sample.py)
    loss.png        the training loss curve
    samples.png     real data vs. generated samples, side by side

The whole run takes a few seconds to a couple of minutes on Apple-silicon MPS.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from data import make_dataset, DATASETS
from diffusion import GaussianDiffusion
from model import DenoiseMLP
from viz import scatter_compare, loss_curve


def pick_device(requested: str) -> torch.device:
    """Choose the best available device, honoring an explicit request."""
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser(description="Train a from-scratch DDPM on 2D data")
    p.add_argument("--dataset", default="moons", choices=list(DATASETS))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--timesteps", type=int, default=200, help="T, diffusion steps")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-data", type=int, default=4000)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    outdir = args.outdir or os.path.join("outputs", args.dataset)
    os.makedirs(outdir, exist_ok=True)
    print(f"device={device}  dataset={args.dataset}  T={args.timesteps}  "
          f"steps={args.steps}  batch={args.batch}")

    # ---- 1. Data: one big tensor of (N, 2) points on the chosen manifold. ----
    data = make_dataset(args.dataset, n=args.n_data, seed=args.seed).to(device)

    # ---- 2. The fixed forward/reverse diffusion process. ----
    diffusion = GaussianDiffusion(T=args.timesteps, device=device)

    # ---- 3. The learnable denoiser. ----
    model = DenoiseMLP(data_dim=2, hidden=args.hidden, n_blocks=args.blocks).to(device)
    print(f"model parameters: {model.num_params():,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 4. Training loop. Each step: sample a minibatch, compute the DDPM
    #         noise-prediction loss, backprop, update. That's the whole thing. ----
    model.train()
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, data.shape[0], (args.batch,), device=device)
        x0 = data[idx]

        loss = diffusion.p_losses(model, x0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            losses.append(loss.item())
        if step % 500 == 0 or step == args.steps:
            print(f"  step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                  f"({time.time()-t0:.1f}s)")

    # ---- 5. Save weights + config so sampling is fully reproducible. ----
    ckpt_path = os.path.join(outdir, "model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "dataset": args.dataset, "timesteps": args.timesteps,
            "hidden": args.hidden, "blocks": args.blocks, "seed": args.seed,
        },
    }, ckpt_path)
    print(f"saved checkpoint -> {ckpt_path}")

    # ---- 6. Proof it worked: loss curve + real-vs-generated scatter. ----
    loss_curve(losses, os.path.join(outdir, "loss.png"))
    generated = diffusion.sample(model, n=args.n_data, dim=2)
    scatter_compare(
        data, generated, os.path.join(outdir, "samples.png"),
        title=f"DDPM on '{args.dataset}'  (T={args.timesteps}, {args.steps} steps)",
    )
    print(f"saved plots -> {outdir}/loss.png, {outdir}/samples.png")

    # A quick numeric sanity check: means/stds of real vs generated should match.
    with torch.no_grad():
        rm, rs = data.mean(0).tolist(), data.std(0).tolist()
        gm, gs = generated.mean(0).tolist(), generated.std(0).tolist()
    print(f"real  mean={[f'{v:.2f}' for v in rm]} std={[f'{v:.2f}' for v in rs]}")
    print(f"gen   mean={[f'{v:.2f}' for v in gm]} std={[f'{v:.2f}' for v in gs]}")


if __name__ == "__main__":
    main()
