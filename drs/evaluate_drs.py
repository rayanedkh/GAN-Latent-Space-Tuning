import os
import argparse
import torch
import torchvision
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from src.device import setup_device
from src.metrics import MetricsCalculator
from src.classifier import Classifier
from src.loading import get_mnist_data_loaders


def load_png_folder(path: str):
    to_tensor = T.ToTensor()
    out = []
    for f in sorted(os.listdir(path)):
        if f.lower().endswith(".png"):
            img = Image.open(os.path.join(path, f)).convert("L")
            x = to_tensor(img) * 2 - 1     # convert [0,1] → [-1,1]
            out.append(x)
    return torch.stack(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DRS samples.")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--nearest_k", type=int, default=5)
    parser.add_argument("--num_real", type=int, default=1000)
    args = parser.parse_args()

    device = setup_device()

    # Load fake samples
    fake_images = load_png_folder(args.samples_dir).to(device)
    print(f"Loaded {fake_images.size(0)} fake images")

    # Load real images
    train_loader, _ = get_mnist_data_loaders(batch_size=args.num_real)
    real_images, _ = next(iter(train_loader))
    real_images = real_images[:args.num_real].to(device)

    # Metrics
    classifier = Classifier.load(args.classifier_path, device)
    metrics = MetricsCalculator(classifier, device, args.batch_size, args.nearest_k)

    scores = metrics.calculate_metrics(real_images, fake_images)
    print("=== Evaluation Results ===")
    print("FID:      ", scores["fid"])
    print("Precision:", scores["precision"])
    print("Recall:   ", scores["recall"])
