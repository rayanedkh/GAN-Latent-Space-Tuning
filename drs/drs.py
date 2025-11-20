import os
import torch
import torchvision
from torch import Tensor
from typing import Optional

from src.device import setup_device
from drs.model import load_generator, load_discriminator
from drs.utils import soft_truncation 


@torch.no_grad()
def compute_prob_from_repo_discriminator(logits: Tensor) -> Tensor:
    """
    Discriminator returns: D(x) = 3 * tanh(raw_logit)
    So raw_logit ≈ atanh(D(x)/3).
    We approximate the effective GAN logit as raw_logit,
    and then convert prob = sigmoid(raw_logit).
    """
    scaled = logits / 3
    scaled = torch.clamp(scaled, -0.999999, 0.999999)
    raw_logit = torch.atanh(scaled)  # inverse tanh
    return torch.sigmoid(raw_logit)  # estimated probability


@torch.no_grad()
def generate_samples_drs(
    divergence: str,
    output_dir: str,
    n_samples: int = 10000,
    batch_size: int = 256,
    latent_dim: int = 100,
    burn_in: int = 20000,
    gamma_offset: float = 2.0,
    device: Optional[torch.device] = None,
    method: int = 0,  # 0 = DRS pur, 1 = soft_truncation + DRS
):
    """
    DRS (Azadi et al. 2019)

    method = 0 : DRS standard
    method = 1 : soft truncation in latent space (using D-scores) + DRS
    """
    if device is None:
        device = setup_device()

    os.makedirs(output_dir, exist_ok=True)

    G = load_generator(divergence, device)
    D = load_discriminator(divergence, device)

    # Burn-in
    max_logit = -float("inf")
    remaining = burn_in

    while remaining > 0:
        b = min(batch_size, remaining)
        z = torch.randn(b, latent_dim, device=device)

        # soft truncation
        if method == 1:
            x_pre = G(z)
            d_raw_pre = D(x_pre.view(b, -1))
            d_pre = compute_prob_from_repo_discriminator(d_raw_pre)
            z = soft_truncation(z, d_pre)

        x = G(z)  # [-1,1]

        d_raw = D(x.view(b, -1))  # shape [b,1]
        d = compute_prob_from_repo_discriminator(d_raw)  # probability approx
        logits = torch.log(d) - torch.log(1 - d)

        max_logit = max(max_logit, logits.max().item())
        remaining -= b

    gamma = max_logit + gamma_offset
    print(f"[DRS] max_logit={max_logit:.4f} | gamma={gamma:.4f}")

    # Rejection Sampling
    saved = 0
    drawn = 0
    idx = 0

    while saved < n_samples:
        b = min(batch_size, n_samples - saved)
        z = torch.randn(b, latent_dim, device=device)

        # soft truncation
        if method == 1:
            x_pre = G(z)
            d_raw_pre = D(x_pre.view(b, -1))
            d_pre = compute_prob_from_repo_discriminator(d_raw_pre)
            z = soft_truncation(z, d_pre)

        x = G(z)

        d_raw = D(x.view(b, -1))
        d = compute_prob_from_repo_discriminator(d_raw)
        logits = torch.log(d) - torch.log(1 - d)

        p_accept = torch.sigmoid(logits - max_logit + gamma)
        u = torch.rand_like(p_accept)
        mask = (u < p_accept).view(-1)  

        x_kept = x[mask]
        if x_kept.numel() > 0:
            x_kept = x_kept.view(-1, 1, 28, 28).cpu()
            for i in range(x_kept.size(0)):
                torchvision.utils.save_image(
                    x_kept[i],
                    os.path.join(output_dir, f"drs_{idx:07d}.png"),
                    normalize=True,
                    value_range=(-1, 1),
                )
                idx += 1
            saved += x_kept.size(0)

        drawn += b

    acc = saved / drawn
    print(f"[DRS] Saved {saved} images (drawn {drawn}) → acceptance = {100*acc:.2f}%")

    return output_dir
