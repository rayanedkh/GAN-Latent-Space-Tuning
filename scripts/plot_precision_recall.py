import os
import torch
import matplotlib.pyplot as plt
import argparse
import torchvision.utils as vutils
import argparse
from src.gan import Generator
from src.device import setup_device

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--history_path",
    type=str,
    default="histories/KL_training_history.pth",
    help="Path to training history",
)
parser.add_argument(
    "--divergence", type=str, default="KL", help="Divergence name for labeling the plot"
)
args = parser.parse_args()

device = setup_device()

# Load history
history = torch.load(args.history_path, weights_only=False)

precision_list = history.get("precision", [])
recall_list = history.get("recall", [])
eval_epochs = history.get("eval_epochs", [])


if not (precision_list and recall_list and eval_epochs):
    raise ValueError(
        "History file does not contain precision, recall, or eval_epochs lists."
    )

# --- Make folder ---
os.makedirs("precision_recall", exist_ok=True)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.plot(eval_epochs, precision_list, marker="o", label="Precision")
plt.plot(eval_epochs, recall_list, marker="s", label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title(f"Precision/Recall - Divergence: {args.divergence}")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join("precision_recall", f"precision_recall_{args.divergence}.png")
plt.savefig(save_path)
plt.close()
print(f"Precision/Recall plot saved to {save_path}")
