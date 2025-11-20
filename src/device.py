import torch


def setup_device():
    """Determines and sets up the training device (CPU, MPS, CUDA)."""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using device: CUDA ({num_gpus} GPU(s))")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        num_gpus = 0
        print(f"Using device: MPS (Apple Metal)")

    else:
        device = torch.device("cpu")
        num_gpus = 0
        print(f"Using device: CPU")

    return device
