# GAN Improvement Methods for Image Synthesis

This repository contains code for training Generative Adversarial Networks (GANs) with various improvement methods for image synthesis. It includes scripts for training GANs and generating images using trained models.

## Set Up Your Environment

1. Create a virtual environment for Python:

   ```bash
   python -m venv venv
   ```

2. Activate the environment:

   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install the project in editable mode (for imports in notebooks and scripts):

   ```bash
   pip install -e .
   ```

## Run Training or Generation

- Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
- These scripts will automatically activate the virtual environment (`venv`).
- If you don't use Juliet cluster, please provide a data_path argument for the training scripts. Otherwise, everything should be automated.

## train_gan.py

This script performs adversarial training for a GAN. Before running it, ensure the `DATA` variable in `scripts/train.sh` points to your dataset directory. If left unchanged, the script will create a default `data` folder in the repository root. If you use Juliet, no changes are required—the default path is already configured. You can adjust the learning rate, number of epochs, and batch size directly in `scripts/train.sh`.
