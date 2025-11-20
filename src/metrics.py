import torch
import torch.nn.functional as F
from prdc import compute_prdc
import numpy as np
from scipy import linalg

from src.classifier import Classifier


class MetricsCalculator:
    """
    Class to compute FID, Precision, and Recall metrics using a pre-trained MNISTClassifier.
    """

    def __init__(self, classifier, device, batch_size, nearest_k):
        self.device: torch.device = device
        self.classifier: Classifier = classifier.to(device).eval()
        self.batch_size: int = batch_size
        self.nearest_k: int = nearest_k

    def _calculate_fid_from_stats(self, mu1, sigma1, mu2, sigma2):
        """Numpy implementation of the Frechet Distance."""
        diff = mu1 - mu2

        # Add a small offset for numerical stability
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

        # Handle complex numbers from numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = (
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        )
        return fid

    @torch.no_grad()
    def _get_classifier_features(self, images):
        """
        Extracts features from images using your trained MNISTClassifier.
        Expects input images to be normalized in the [-1, 1] range.
        """
        self.classifier.eval()  # Ensure model is in eval mode
        all_embeddings = []

        for start in range(0, images.size(0), self.batch_size):
            end = min(start + self.batch_size, images.size(0))
            batch = images[start:end].to(self.device)
            embeddings = self.classifier.extract_features(batch)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()

    def _compute_fid(self, real_features, fake_features):
        """Private helper to compute FID from numpy features."""
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        return self._calculate_fid_from_stats(mu_real, sigma_real, mu_fake, sigma_fake)

    def _compute_precision_recall(self, real_features, fake_features):
        """Private helper to compute PRDC from numpy features."""

        metrics = compute_prdc(
            real_features=real_features,
            fake_features=fake_features,
            nearest_k=self.nearest_k,
        )
        return metrics["precision"], metrics["recall"]

    def calculate_metrics(self, real_images, fake_images):
        """
        Calculates all metrics (FID, Precision, Recall).

        Args:
            real_images (torch.Tensor): Tensor of real images, shape [N, C, H, W]
            fake_images (torch.Tensor): Tensor of fake images, shape [N, C, H, W]

        Returns:
            dict: A dictionary containing 'fid', 'precision', and 'recall'.
        """

        real_features_np = self._get_classifier_features(real_images)
        fake_features_np = self._get_classifier_features(fake_images)

        fid = self._compute_fid(real_features_np, fake_features_np)
        precision, recall = self._compute_precision_recall(
            real_features_np, fake_features_np
        )

        return {"fid": fid, "precision": precision, "recall": recall}
