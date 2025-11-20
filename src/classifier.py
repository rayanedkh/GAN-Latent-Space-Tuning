"""
A CNN classifier for MNIST digits.
Useful for getting meaningful MNIST images embeddings for PR curve algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import Tensor
import os


PENULTIMATE_LAYER_DIM = 128


class Classifier(nn.Module):
    def __init__(self):
        """
        A standard CNN classifier for MNIST.
        Input: [batch_size, 1, 28, 28]
        """
        super().__init__()

        # --- Convolutional Feature Extractor ---

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Classifier Head ---

        self.flat_dim = 64 * 7 * 7  # = 3136

        self.fc1 = nn.Linear(self.flat_dim, PENULTIMATE_LAYER_DIM)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(PENULTIMATE_LAYER_DIM, 10)

    def _forward_up_to_embedding_layer(self, x: Tensor) -> Tensor:
        """
        Forward pass up to the embedding layer.
        """

        # --- Feature Extraction ---
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # --- Classification ---
        x = x.view(-1, self.flat_dim)

        # Get the embedding
        embedding = self.fc1(x)
        embedding = F.relu(embedding)

        return embedding

    def forward(self, x: Tensor) -> Tensor:
        """
        Runs the full network and returns the 10-class logits.
        """
        embedding = self._forward_up_to_embedding_layer(x)

        x = self.dropout(embedding)
        logits = self.fc2(x)

        return logits

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> Tensor:
        """
        Extracts 128-dim features (penulminate layer dimension) from input images for inference.
        """
        return self._forward_up_to_embedding_layer(x)

    @classmethod
    def load(cls, path, device):
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        return model


class ClassifierTrainer:
    def __init__(self, model, criterion, optimizer, device):
        """
        Initializes the trainer with all necessary components.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_loss_history = []
        self.test_accuracy_history = []

    def _evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluates the classifier on the given data loader.
        Returns accuracy.
        """

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in data_loader:

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def _train_step(self, data_loader: DataLoader, epoch: int, epochs: int) -> float:
        """
        Trains the model for one epoch. Returns avg training loss.
        """

        self.model.train()
        running_loss = 0.0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, labels in progress_bar:

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(data_loader)
        return avg_train_loss

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
    ):
        """
        The main training loop.
        """

        for epoch in range(epochs):

            avg_train_loss = self._train_step(train_loader, epoch, epochs)
            accuracy = self._evaluate(test_loader)

            self.train_loss_history.append(avg_train_loss)
            self.test_accuracy_history.append(accuracy)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.2f}%"
            )

        print("Training finished.")

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "classifier.pth")
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")
