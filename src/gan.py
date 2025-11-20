import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import torchvision

from torch.utils.data import DataLoader
from src.losses import DLOSS, GLOSS
from src.loading import get_images_from_loader
from src.metrics import MetricsCalculator


LATENT_DIM = 100
MNIST_DIM = 28 * 28
IMG_SHAPE = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_dim = LATENT_DIM
        self.img_shape = IMG_SHAPE

        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, MNIST_DIM)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

    @torch.no_grad()
    def sample(self, n_samples, batch_size, device, latent_std=1.0):
        """
        Generates n_samples images for inference.
        """
        self.eval()
        self.to(device)

        generated_images = []

        for start in range(0, n_samples, batch_size):
            current_batch_size = min(start + batch_size, n_samples) - start
            z = torch.randn(current_batch_size, self.latent_dim, device=device)
            z *= latent_std
            x_flat = self(z)
            generated_images.append(x_flat.cpu())

        all_images_flat = torch.cat(generated_images, dim=0)
        return all_images_flat.view(n_samples, *self.img_shape)

    @classmethod
    def load(cls, path, device):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        return model


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(MNIST_DIM, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.fc4(x)
        # return x
        return 3 * torch.tanh(x)


def weights_init_mlp(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # mean=0.0, std=0.02
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class GANTrainer:
    def __init__(
        self,
        G: Generator,
        D: Discriminator,
        G_optimizer: torch.optim.Optimizer,
        D_optimizer: torch.optim.Optimizer,
        G_loss: GLOSS,
        D_loss: DLOSS,
        device: torch.device,
        eval_num_images: int,
        eval_metrics_calculator: MetricsCalculator,
    ):
        """
        Initializes the GANTrainer class.
        """
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.device = device
        self.eval_num_images = eval_num_images
        self.eval_metrics_calculator = eval_metrics_calculator

        self.latent_dim = self.G.latent_dim
        self.loss_name = self.D_loss.loss_name

        self.precision_history = []
        self.recall_history = []
        self.fid_history = []
        self.eval_epochs_history = []

    def _train_D_step(self, x_real):
        """Trains the Discriminator for one step."""
        self.D.train()
        self.D_optimizer.zero_grad()

        x_real = x_real.to(self.device)
        # x_real=x_real + 0.05*torch.randn_like(x_real)  # add small noise to real images
        D_real = self.D(x_real)  # * 0.95  # label smoothing

        batch_size = x_real.shape[0]
        z = torch.randn(batch_size, self.latent_dim, device=self.device)  # * 1.5

        x_fake = self.G(z).detach()  # detach: we don't want to backprop through G
        D_fake = self.D(x_fake)  # * 0.05  # label smoothing

        loss = self.D_loss(D_real, D_fake)

        loss.backward()

        # nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.)

        self.D_optimizer.step()

        return loss.item()

    def _train_G_step(self, batch_size):
        """Trains the Generator for one step."""
        self.G.train()
        self.G_optimizer.zero_grad()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        x_fake = self.G(z)
        D_fake = self.D(x_fake)

        loss = self.G_loss(D_fake)
        loss.backward()

        # nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)

        self.G_optimizer.step()

        return loss.item()

    def _evaluate(self, epoch, test_loader: DataLoader):
        """Generates samples and computes precision/recall."""

        self.G.eval()

        fake_images = self.G.sample(
            self.eval_num_images, test_loader.batch_size, device=self.device
        )

        # save fake images in a grid for visualization
        plt_dir = "evaluation_images"
        os.makedirs(plt_dir, exist_ok=True)
        plt_path = os.path.join(plt_dir, f"{self.loss_name}_epoch_{epoch}.png")
        grid = torchvision.utils.make_grid(
            fake_images, nrow=10, normalize=True, pad_value=1
        )
        torchvision.utils.save_image(grid, plt_path)

        real_images = get_images_from_loader(
            test_loader, self.device, num_images=self.eval_num_images
        )

        metrics = self.eval_metrics_calculator.calculate_metrics(
            real_images, fake_images
        )

        precision, recall, fid = metrics["precision"], metrics["recall"], metrics["fid"]

        print(
            f"Epoch {epoch}: Precision={precision:.4f}, Recall={recall:.4f}, FID={fid:.4f}"
        )

        self.G.train()

        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.fid_history.append(fid)
        self.eval_epochs_history.append(epoch)

    def train(self, train_loader, test_loader, epochs, eval_interval):
        """Main training loop."""
        for epoch in tqdm(range(1, epochs + 1), desc="Total Epochs"):
            for x, _ in train_loader:
                x = x.view(-1, MNIST_DIM).to(self.device)

                for _ in range(2):
                    self._train_G_step(batch_size=x.shape[0])
                self._train_D_step(x)

            if epoch % eval_interval == 0 or epoch == epochs:
                self._evaluate(epoch, test_loader)

    def save_history(self, save_dir):
        """Saves the training history to a file."""
        os.makedirs(save_dir, exist_ok=True)
        history_path = os.path.join(save_dir, f"{self.loss_name}_training_history.pth")
        history = {
            "precision": self.precision_history,
            "recall": self.recall_history,
            "fid": self.fid_history,
            "eval_epochs": self.eval_epochs_history,
        }
        torch.save(history, history_path)
        print(f"Training history saved to {history_path}")

    def save_models(self, save_dir):
        """Saves the Generator and Discriminator models to files."""
        os.makedirs(save_dir, exist_ok=True)
        G_path = os.path.join(save_dir, f"{self.loss_name}_G.pth")
        D_path = os.path.join(save_dir, f"{self.loss_name}_D.pth")
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        print(f"Generator model saved to {G_path}")
        print(f"Discriminator model saved to {D_path}")
