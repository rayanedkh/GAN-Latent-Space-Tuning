from typing import List
import os
import torchvision.utils as vutils
from src.gan import Generator
import torch
import os
from torchvision import utils as vutils


def create_sample_grid_tensor(
    generator: Generator,
    grid_width: int,
    grid_height: int,
    device: torch.device,
    latent_std: float = 1.0,
):
    """
    Generates a specified number of images from a generator and saves them
    as a grid image file.
    """

    generator.eval()
    n_samples = grid_width * grid_height
    images_tensor = generator.sample(n_samples, n_samples, device, latent_std)

    grid = vutils.make_grid(
        images_tensor,
        nrow=grid_width,
        normalize=True,
        value_range=(-1, 1),
    )
    return grid


def create_truncation_grid_tensor(
    generator: Generator,
    latent_std_values: List[float],
    num_samples_per_std: int,
    device: torch.device,
):
    """
    Generates a grid demonstrating the effect of latent_std (soft truncation).
    """
    if any(std < 0 for std in latent_std_values):
        raise ValueError("latent_std_values must be non-negative.")

    generator.eval()
    all_samples = []

    for std in latent_std_values:
        images_tensor = generator.sample(
            num_samples_per_std, num_samples_per_std, device, latent_std=std
        )
        all_samples.append(images_tensor)

    all_samples = torch.cat(all_samples, dim=0)

    grid = vutils.make_grid(
        all_samples,
        nrow=num_samples_per_std,
        normalize=True,
        value_range=(-1, 1),
    )
    return grid
