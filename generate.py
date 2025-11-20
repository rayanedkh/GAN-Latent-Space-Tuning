import torch
import torchvision
import os
import argparse

from src.device import setup_device
from src.gan import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Normalizing Flow.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="The batch size to use for training.",
    )
    args = parser.parse_args()

    batch_size = args.batch_size

    device = setup_device()

    print("Model Loading...")
    # Model Pipeline
    mnist_dim = 784

    model: Generator = Generator.load("checkpoints/BCE_G.pth", device=device)

    model.eval()

    print("Model loaded.")

    print("Start Generating")
    os.makedirs("samples", exist_ok=True)

    n_samples = 10000
    with torch.no_grad():
        x = model.sample(n_samples, batch_size, device)

    for i in range(x.shape[0]):
        torchvision.utils.save_image(x[i], os.path.join("samples", f"{i}.png"))
