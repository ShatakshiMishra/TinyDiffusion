"""
diffusion.py — the DDPM math, from scratch.

This is the heart of the project. It implements a Denoising Diffusion
Probabilistic Model (Ho, Jain, Abbeel, 2020) with no library helpers — just
tensors and the equations from the paper, each one commented.

THE BIG PICTURE
    A diffusion model has two processes:

    1. FORWARD (fixed, no learning): take a real data point x0 and gradually add
       Gaussian noise over T steps until, at step T, it is indistinguishable
       from pure noise N(0, I). This is a fixed recipe — nothing to train.

    2. REVERSE (learned): start from pure noise and gradually *remove* noise,
       one step at a time, until we recover something that looks like real data.
       We train a neural network to do the denoising.

THE KEY TRICK (closed-form forward jump)
    Adding noise one step at a time is a Markov chain, but we never have to walk
    it step by step during training. Because each step adds independent Gaussian
    noise, the composition of t steps is *itself* Gaussian and available in
    closed form. If we define

        beta_t         : how much noise we inject at step t (the schedule)
        alpha_t        = 1 - beta_t
        alpha_bar_t    = product of alpha_1 .. alpha_t          (cumulative)

    then jumping straight from x0 to x_t is just:

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0,I)

    So for any random step t we can produce a noisy sample in one line, and ask
    the network: "given x_t and t, what was the noise eps I added?" That single
    regression objective is all DDPM training is.

THE REVERSE STEP (sampling)
    Given the network's noise prediction eps_theta(x_t, t), the posterior mean of
    x_{t-1} has a closed form:

        x_{t-1} = 1/sqrt(alpha_t) * ( x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta )
                  + sqrt(beta_t) * z,        z ~ N(0,I)   (z=0 at the last step)

    We just apply that T times, from t=T-1 down to t=0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02
                       ) -> torch.Tensor:
    """Linear variance schedule from Ho et al.

    beta_t controls how much variance is injected at step t. Small betas early
    (data barely perturbed) growing to larger betas late (approaching pure
    noise). A linear ramp is the original DDPM choice and works well for 2D.
    """
    return torch.linspace(beta_start, beta_end, T)


class GaussianDiffusion:
    """Holds the fixed noise schedule and implements the forward/reverse math.

    This is deliberately *not* an nn.Module — it has no learnable parameters.
    It only stores precomputed schedule constants (as plain tensors on `device`)
    and provides:
        q_sample  : forward jump  x0 -> x_t          (used in training)
        p_losses  : the training loss (predict the noise)
        p_sample  : one reverse step x_t -> x_{t-1}   (used in sampling)
        sample    : the full reverse loop  noise -> x0
    """

    def __init__(self, T: int = 200, device: str | torch.device = "cpu",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.device = torch.device(device)

        # ---- Precompute every schedule constant once, up front. ----
        betas = make_beta_schedule(T, beta_start, beta_end).to(self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)            # alpha_bar_t
        # alpha_bar_{t-1}, with alpha_bar_0 := 1 prepended (needed for posterior var)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        # Coefficients used in the forward jump  x_t = a*x0 + b*eps
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)                 # a
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars) # b

        # Coefficients used in the reverse mean
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # variance of the reverse step (the "true" posterior variance)
        self.posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    # ---- helper: gather schedule constant for a batch of timesteps t ----
    @staticmethod
    def _gather(coeffs: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        """Pick coeffs[t] for each element in the batch and reshape to broadcast.

        t has shape (B,). We index the (T,) constant vector to get (B,), then add
        trailing singleton dims so it broadcasts against x of shape (B, dim).
        """
        out = coeffs.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    # ---- FORWARD process: q(x_t | x0) in one closed-form jump ----
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """Produce the noisy x_t directly from x0 and a timestep t.

            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a = self._gather(self.sqrt_alpha_bars, t, x0.shape)
        b = self._gather(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return a * x0 + b * noise

    # ---- TRAINING objective: predict the noise that was added ----
    def p_losses(self, model, x0: torch.Tensor) -> torch.Tensor:
        """Sample a random t per example, noise x0, and regress the noise.

        This one method is the entire training signal:
            1. pick a random step t for each example
            2. draw fresh noise eps
            3. form x_t via the closed-form jump
            4. ask the model to predict eps from (x_t, t)
            5. return MSE(eps, prediction)
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)  # step 1
        noise = torch.randn_like(x0)                           # step 2
        x_t = self.q_sample(x0, t, noise)                      # step 3
        predicted_noise = model(x_t, t)                        # step 4
        return F.mse_loss(predicted_noise, noise)              # step 5

    # ---- ONE reverse step: x_t -> x_{t-1} ----
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor,
                 t_index: int) -> torch.Tensor:
        """Sample x_{t-1} from x_t using the model's noise prediction."""
        betas_t = self._gather(self.betas, t, x_t.shape)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        sqrt_recip_alphas_t = self._gather(self.sqrt_recip_alphas, t, x_t.shape)

        # Posterior mean: subtract the predicted noise, rescale by 1/sqrt(alpha_t).
        eps_theta = model(x_t, t)
        mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus * eps_theta)

        if t_index == 0:
            # At the final step we return the mean with no extra noise — this is
            # our best estimate of the clean data point x0.
            return mean
        var = self._gather(self.posterior_var, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    # ---- FULL reverse loop: pure noise -> data ----
    @torch.no_grad()
    def sample(self, model, n: int, dim: int = 2,
               return_trajectory: bool = False):
        """Generate `n` samples by denoising from pure Gaussian noise.

        If return_trajectory=True, also return a list of snapshots (one per
        step) so we can visualize noise turning into structure.
        """
        model.eval()
        x = torch.randn(n, dim, device=self.device)  # start: pure noise, x_T
        traj = [x.cpu()]
        for t_index in reversed(range(self.T)):       # walk T-1 .. 0
            t = torch.full((n,), t_index, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, t_index)
            if return_trajectory:
                traj.append(x.cpu())
        if return_trajectory:
            return x, traj
        return x
