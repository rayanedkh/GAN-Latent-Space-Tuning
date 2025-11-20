import os
import torch
from src.gan import Generator, Discriminator

CHECKPOINT_FOLDER = "checkpoints"


def load_generator(divergence: str, device: torch.device) -> Generator:
    """
    Loads a pretrained Generator based on the divergence name.
    divergence ∈ {"JS","KL","RKL","BCE"}.
    """
    ckpt_path = os.path.join(CHECKPOINT_FOLDER, f"{divergence}_G.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {ckpt_path}")

    G = Generator().to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    G.load_state_dict(state)
    G.eval()
    return G


def load_discriminator(divergence: str, device: torch.device) -> Discriminator:
    """
    Loads the pretrained Discriminator.
    """
    ckpt_path = os.path.join(CHECKPOINT_FOLDER, f"{divergence}_D.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Discriminator checkpoint not found: {ckpt_path}")

    D = Discriminator().to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    D.load_state_dict(state)
    D.eval()
    return D
