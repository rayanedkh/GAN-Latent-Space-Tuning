import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from src.gan import GANTrainer, Generator, Discriminator, weights_init_mlp
from src.loading import get_mnist_data_loaders
from src.losses import DLOSS, GLOSS
from src.metrics import MetricsCalculator
from src.classifier import Classifier
from src.device import setup_device


DEFAUL_MODEL_SAVE_DIR = "checkpoints/"
DEFAULT_HISTORY_SAVE_DIR = "histories/"
DEFAULT_EVAL_CLASSIFIER_PATH = "checkpoints/classifier.pth"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE_GENERATOR = 2e-4
DEFAULT_LEARNING_RATE_DISCRIMINATOR = 5e-5
DEFAULT_EVAL_NUM_IMAGES = 1000
DEFAULT_DIVERGENCE = "KL"
DEFAULT_EVAL_NEAREST_K = 3


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train GAN on MNIST.")

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
        "--learning_rate_generator",
        type=float,
        default=DEFAULT_LEARNING_RATE_GENERATOR,
        help="Learning rate for the generator optimizer.",
    )

    parser.add_argument(
        "--learning_rate_discriminator",
        type=float,
        default=DEFAULT_LEARNING_RATE_DISCRIMINATOR,
        help="Learning rate for the discriminator optimizer.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default=DEFAULT_DIVERGENCE,
        choices=["KL", "RKL", "JS", "BCE", "Pearson"],
        help="Type of loss to use.",
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Run evaluation every N epochs.",
    )

    parser.add_argument(
        "--eval_num_images",
        type=int,
        default=DEFAULT_EVAL_NUM_IMAGES,
        help="Number of images to generate for evaluation.",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=DEFAUL_MODEL_SAVE_DIR,
        help="Directory to save the trained model weights.",
    )

    parser.add_argument(
        "--eval_classifier_path",
        type=str,
        default=DEFAULT_EVAL_CLASSIFIER_PATH,
        help="Path to the pre-trained MNIST classifier for evaluation.",
    )

    parser.add_argument(
        "--eval_nearest_k",
        type=int,
        default=DEFAULT_EVAL_NEAREST_K,
        help="Number of nearest neighbors to use for precision/recall computation.",
    )

    parser.add_argument(
        "--history_save_dir",
        type=str,
        default=DEFAULT_HISTORY_SAVE_DIR,
        help="Directory to save training history plots.",
    )

    return parser.parse_args()


def build_trainer(args):

    device = setup_device()
    generator = Generator().to(device).apply(weights_init_mlp)
    discriminator = Discriminator().to(device).apply(weights_init_mlp)

    D_loss = DLOSS(loss_name=args.loss).to(device)
    G_loss = GLOSS(loss_name=args.loss).to(device)

    # adapt learning rates to make the training more stable - reduce through training
    G_optimizer = optim.Adam(
        generator.parameters(), lr=args.learning_rate_generator, betas=(0.5, 0.999)
    )
    D_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate_discriminator,
        betas=(0.5, 0.999),
    )

    metrics_caclulator = MetricsCalculator(
        classifier=Classifier.load(args.eval_classifier_path, device),
        device=device,
        batch_size=args.batch_size,
        nearest_k=args.eval_nearest_k,
    )

    trainer = GANTrainer(
        G=generator,
        D=discriminator,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        G_loss=G_loss,
        D_loss=D_loss,
        device=device,
        eval_num_images=args.eval_num_images,
        eval_metrics_calculator=metrics_caclulator,
    )

    return trainer


def print_args_infos(args):
    print("Training GAN with the following parameters:")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate Generator: {args.learning_rate_generator}")
    print(f"Learning Rate Discriminator: {args.learning_rate_discriminator}")
    print(f"Loss: {args.loss}")
    print(f"Number of Evaluation Images: {args.eval_num_images}")
    print(f"Model Save Dir: {args.model_save_dir}")
    print(f"History Save Dir: {args.history_save_dir}")
    print(f"Evaluation Interval: {args.eval_interval}")
    print(f"Evaluation Classifier Path: {args.eval_classifier_path}")
    print(f"Evaluation Nearest K (for Precision and Recall): {args.eval_nearest_k}")


def main():
    args = parse_args()
    print_args_infos(args)

    train_loader, test_loader = get_mnist_data_loaders(args.batch_size)

    trainer = build_trainer(args)
    trainer.train(
        train_loader,
        test_loader,
        args.epochs,
        args.eval_interval,
    )
    trainer.save_history(args.history_save_dir)
    trainer.save_models(args.model_save_dir)


if __name__ == "__main__":
    main()
