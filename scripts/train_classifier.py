import argparse
import torch.nn as nn
import torch.optim as optim

from src.classifier import Classifier, ClassifierTrainer
from src.loading import get_mnist_data_loaders
from src.device import setup_device


DEFAULT_MODEL_SAVE_DIR = "checkpoints/"
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MNIST Classifier")

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of epochs for training.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Size of mini-batches for SGD.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizer.",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=DEFAULT_MODEL_SAVE_DIR,
        help="Directory to save the trained model weights.",
    )

    return parser.parse_args()


def print_args_infos(args):
    print("Training MNIST Classifier with the following parameters:")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Model Save Dir: {args.model_save_dir}")


def build_trainer(args):

    device = setup_device()
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = ClassifierTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    return trainer


def main():
    args = parse_args()
    print_args_infos(args)

    train_loader, test_loader = get_mnist_data_loaders(args.batch_size)

    trainer = build_trainer(args)
    trainer.train(
        train_loader,
        test_loader,
        args.epochs,
    )
    trainer.save_model(args.model_save_dir)


if __name__ == "__main__":
    main()
