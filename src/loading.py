"""
Loading utilitary functions.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_datasets():
    """
    Loads and prepares the MNIST datasets.
    Returns train and test datasets,
    of image pixels that have been scaled in [-1,1].
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True,
    )

    return train_dataset, test_dataset


def get_mnist_data_loaders(batch_size):
    """
    Loads and prepares the MNIST DataLoader.
    Returns train and test DataLoaders,
    of image pixels that have been scaled in [-1,1].
    """

    train_dataset, test_dataset = get_mnist_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, test_loader


def get_mnist_images_scaled(num_images, device):

    train_loader, _ = get_mnist_data_loaders(batch_size=num_images)
    images, _ = next(iter(train_loader))
    images = images.to(device)
    return images


def get_images_from_loader(data_loader, device, num_images=None):
    """Fetches images from a DataLoader and returns them as a tensor on the specified device."""

    all_images = []
    total_collected = 0

    for images, _ in data_loader:
        if num_images and total_collected >= num_images:
            break

        images = images.to(device)
        all_images.append(images)
        total_collected += images.size(0)

    all_images = torch.cat(all_images, dim=0)

    if num_images:
        all_images = all_images[:num_images]

    return all_images
