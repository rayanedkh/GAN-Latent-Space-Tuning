import argparse
from src.device import setup_device
from drs.drs import generate_samples_drs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using DRS.")

    parser.add_argument(
        "--divergence",
        type=str,
        required=True,
        choices=["JS", "KL", "RKL", "BCE"],
        help="Which trained GAN (divergence) to use."
    )

    parser.add_argument(
        "--method",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 = Standard DRS, 1 = Soft truncation + DRS"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples_drs",
        help="Folder where generated images will be saved."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for generation."
    )

    parser.add_argument(
        "--burn_in",
        type=int,
        default=20000,
        help="Number of samples for DRS burn-in (estimating gamma)."
    )

    parser.add_argument(
        "--gamma_offset",
        type=float,
        default=2.0,
        help="Offset added to max_logit to define acceptance gamma."
    )

    args = parser.parse_args()

    # Device
    device = setup_device()

    # Launch DRS generation
    output = generate_samples_drs(
        divergence=args.divergence,
        method=args.method,
        output_dir=args.output_dir,
        n_samples=args.num_samples,
        batch_size=args.batch_size,
        burn_in=args.burn_in,
        gamma_offset=args.gamma_offset,
        device=device,
    )

    print(f"Generated DRS samples saved in: {output}")
