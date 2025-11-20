# fGAN-Precision-Recall-Tuning

This repository contains the implementation and analysis of **f-divergence-based Generative Adversarial Networks (f-GANs)** to investigate and improve the inherent trade-off between **Precision (fidelity)** and **Recall (diversity)** in generated images [9, 11].

The project explores how different training objectives (f-divergences) and post-processing techniques (Soft Truncation and Discriminator Rejection Sampling) jointly shape the generative performance on the **MNIST** dataset [16, 229].

---

## Key Objectives and Findings

The core focus of this work is to provide a clearer understanding of how to control GAN behavior:

### **Training Objectives**
We systematically compare the performance of GANs trained with different f-divergences:

- **Jensen-Shannon (JS) Divergence – Classical GAN:** Tends to balance precision and recall [41].
- **Kullback-Leibler (KL) Divergence:** Promotes broader coverage (higher *Recall*) by penalizing missing modes, though it can be unstable [47, 48].
- **Reverse KL (RKL) Divergence:** Favors realism (higher *Precision*) but is prone to mode collapse [52, 53].
- **Binary Cross-Entropy (BCE) – Vanilla GAN:** Used as the baseline [64].

### **Evaluation**
Performance is assessed using custom **Precision/Recall curves** [14, 22, 24] and **Fréchet Inception Distance (FID)**, computed in a feature space derived from a custom-trained CNN classifier on MNIST [20, 27, 28].

### **Post-Processing Techniques**
- **Soft Truncation:** A simple technique to tune the quality/diversity trade-off by varying the variance of the latent sampling (σ) [133, 134].
- **Discriminator Rejection Sampling (DRS):** A post-hoc filtering method using the discriminator’s score to accept only high-fidelity samples, boosting precision without retraining [15, 192].

### **Best Performance**
The **KL-based generator combined with DRS** achieved the best overall results (FID ≈ 24) and a strong precision–recall balance [215, 233].

---

## Repository Structure

The project is organized into modular directories for training, evaluation, and result storage.

```txt
GAN-LATENT-SPACE-TUNING/
├── checkpoints/
│   ├── BCE_D.pth, BCE_G.pth           # Trained D/G for BCE loss
│   ├── JS_D.pth, JS_G.pth             # Trained D/G for JS loss
│   ├── KL_D.pth, KL_G.pth             # Trained D/G for KL loss
│   ├── RKL_D.pth, RKL_G.pth           # Trained D/G for RKL loss
│   └── classifier.pth                 # CNN classifier used for metrics
│
├── docs/
│   └── slides.pdf                     # Project slides
│
├── drs/                               # Discriminator Rejection Sampling
│   ├── drs.py
│   ├── evaluate_drs.py                # Evaluate models with DRS
│   ├── generate_drs.py                # Generate samples via DRS
│   └── ...
│
├── histories/
│   ├── BCE_training_history.pth       # BCE PR/FID histories
│   └── JS-KL-RKL_training_history.pth # f-GAN histories
│
├── notebooks/
│   └── models_eval.ipynb              # Evaluation & analysis notebook
│
├── scripts/
│   ├── generate.sh                    # Run generation
│   ├── plot_precision_recall.py       # Plot PR curves & trajectories
│   ├── train_classifier.py            # Train CNN classifier
│   ├── train_gan.py                   # f-GAN adversarial training
│   └── train.sh                       # Launch training jobs
│
├── src/
│   ├── classifier.py                  # CNN model for feature extraction
│   ├── gan.py                         # Generator & Discriminator
│   ├── losses.py                      # f-divergences + JS/BCE
│   ├── metrics.py                     # Precision/Recall + FID
│   ├── generate.py                    # Standard sample generation
│   └── ...
│
├── README.md
├── report.pdf                         # Full project report
└── requirements.txt                   # Dependencies

```
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
