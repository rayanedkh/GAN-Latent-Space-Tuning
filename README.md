# fGAN-Precision-Recall-Tuning

[cite_start]This repository contains the implementation and analysis of **f-divergence-based Generative Adversarial Networks (f-GANs)** to investigate and improve the inherent trade-off between **Precision (fidelity)** and **Recall (diversity)** in generated images[cite: 9, 11].

[cite_start]The project explores how different training objectives (f-divergences) and post-processing techniques (Soft Truncation and Discriminator Rejection Sampling) jointly shape the generative performance on the **MNIST** dataset[cite: 16, 229].

## Key Objectives and Findings

The core focus of this work is to provide a clearer understanding of how to control GAN behavior:

* **Training Objectives:** We systematically compare the performance of GANs trained with different f-divergences:
    * [cite_start]**Jensen-Shannon (JS) Divergence (Classical GAN):** Tends to balance precision and recall[cite: 41].
    * [cite_start]**Kullback-Leibler (KL) Divergence:** Promotes broader coverage (higher **Recall**) by penalizing missing modes, though it can be unstable[cite: 47, 48].
    * [cite_start]**Reverse KL (RKL) Divergence:** Favors realism (higher **Precision**) but is prone to mode collapse[cite: 52, 53].
    * [cite_start]**Binary Cross-Entropy (BCE) / Vanilla GAN:** Used as the baseline[cite: 64].
* [cite_start]**Evaluation:** Performance is assessed using custom **Precision/Recall (PR) curves** [cite: 14, 22, 24] [cite_start]and the **Fréchet Inception Distance (FID)**, all computed in a feature space derived from a custom-trained CNN classifier on MNIST[cite: 20, 27, 28].
* **Post-Processing Techniques:**
    * [cite_start]**Soft Truncation:** A simple post-training technique to tune the quality/diversity trade-off by varying the variance of the latent space sampling ($\sigma$)[cite: 133, 134].
    * [cite_start]**Discriminator Rejection Sampling (DRS):** A post-hoc filtering method that uses the discriminator's score to selectively accept high-fidelity generated samples, significantly enhancing overall precision without retraining[cite: 15, 192].
* [cite_start]**Best Performance:** The **KL-based generator combined with DRS** achieved the best overall results (FID $\approx 24$) and a strong balance between precision and recall, confirming the robustness of post-hoc filtering[cite: 215, 233].

---

## Repository Structure

The project is organized into modular scripts and directories for training, evaluation, and storage of results.


GAN-LATENT-SPACE-TUNING/
├── checkpoints/
│   ├── BCE_D.pth, BCE_G.pth         # Trained Discriminator (D) and Generator (G) for BCE loss
│   ├── JS_D.pth, JS_G.pth           # Trained D and G for JS loss
│   ├── KL_D.pth, KL_G.pth           # Trained D and G for KL loss
│   ├── RKL_D.pth, RKL_G.pth         # Trained D and G for RKL loss
│   └── classifier.pth               # Trained CNN feature extractor for metrics
│
├── docs/
│   └── slides.pdf                   # Project presentation slides
│
├── drs/                             # Scripts for Discriminator Rejection Sampling implementation
│   ├── drs.py
│   ├── evaluate_drs.py              # Evaluate models with DRS
│   ├── generate_drs.py              # Generate samples using DRS
│   └── ...
│
├── histories/
│   ├── BCE_training_history.pth     # Training metrics (Precision, Recall, FID) for BCE
│   └── JS/KL/RKL_training_history.pth # Metrics history for f-GANs
│
├── notebooks/
│   └── models_eval.ipynb            # Evaluation, plotting, and analysis notebook
│
├── scripts/
│   ├── generate.sh                  # Script to run image generation
│   ├── plot_precision_recall.py     # Generate PR curves and trajectories
│   ├── train_classifier.py          # Train the CNN feature extractor
│   ├── train_gan.py                 # Main script for f-GAN training
│   └── train.sh                     # Shell script to launch training jobs
│
├── src/
│   ├── classifier.py                # Classifier model definition
│   ├── gan.py                       # Generator (G) and Discriminator (D) models
│   ├── losses.py                    # f-divergences (KL, RKL) and JS/BCE losses
│   ├── metrics.py                   # Precision/Recall and FID computation
│   ├── generate.py                  # Standard generation logic
│   └── ...
│
├── README.md
├── report.pdf                       # Full project report
└── requirements.txt                 # Python dependencies


---

## Set Up Your Environment

To replicate the results or run the code, you'll need to set up the environment and install dependencies.

1.  **Create a virtual environment for Python:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the environment:**

    ```bash
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode** (crucial for imports in notebooks and scripts):

    ```bash
    pip install -e .
    ```

---

## Run Training or Generation

* Use the provided shell scripts (`scripts/train.sh` or `scripts/generate.sh`) to launch jobs.
* These scripts will automatically activate the virtual environment (`venv`).
* The `train_gan.py` script performs adversarial training for a GAN. Before running, ensure the `DATA` variable in `scripts/train.sh` points to your dataset directory.
    * If you don't use the Juliet cluster (as configured by default), please provide a `data_path` argument for the training scripts or adjust the `DATA` variable.
* You can adjust hyperparameters like learning rate, number of epochs, and batch size directly in `scripts/train.sh`.

```bash
# Example to run training
scripts/train.sh

# Example to run generation (e.g., for plotting)
scripts/generate.sh


