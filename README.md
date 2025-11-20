# fGAN-Precision-Recall-Tuning

[cite_start]This repository contains the implementation and analysis of **f-divergence-based Generative Adversarial Networks (f-GANs)** [cite: 30] [cite_start]to investigate and improve the inherent trade-off between **Precision (fidelity)** and **Recall (diversity)** [cite: 9] in generated images.

[cite_start]The project explores how different training objectives (f-divergences) and post-processing techniques (Soft Truncation [cite: 14] [cite_start]and Discriminator Rejection Sampling [cite: 15][cite_start]) jointly shape the generative performance on the **MNIST** dataset[cite: 20, 228].

---

## Key Objectives and Findings

[cite_start]The core focus of this work is to provide a clearer understanding of how to control GAN behavior[cite: 16].

### **Training Objectives**
[cite_start]We systematically compare the performance of GANs trained with different f-divergences[cite: 31]:

- [cite_start]**Jensen-Shannon (JS) Divergence – Classical GAN:** Tends to balance precision and recall[cite: 41].
- [cite_start]**Kullback-Leibler (KL) Divergence:** Promotes broader coverage (higher *Recall*) by penalizing missing modes [cite: 47][cite_start], though it can be unstable[cite: 48].
- [cite_start]**Reverse KL (RKL) Divergence:** Favors realism (higher *Precision*) [cite: 52] [cite_start]but is prone to mode collapse[cite: 53].
- [cite_start]**Binary Cross-Entropy (BCE) – Vanilla GAN:** Used as the baseline.

### **Evaluation**
Performance is assessed using:
- [cite_start]Custom **Precision/Recall curves** [cite: 14, 22, 24] [cite_start]and the **Fréchet Inception Distance (FID)**[cite: 27].
- [cite_start]Metrics are computed in a feature space derived from a custom-trained CNN classifier on MNIST[cite: 20, 28].

### **Post-Processing Techniques**
- [cite_start]**Soft Truncation:** A simple technique to tune the quality/diversity trade-off by varying the variance of the latent sampling ($\sigma$)[cite: 133, 134].
- [cite_start]**Discriminator Rejection Sampling (DRS):** A post-hoc filtering method using the discriminator’s score to accept only high-fidelity samples, boosting precision without retraining[cite: 15, 192]. [cite_start](Introduced by **Azadi et al., 2019** [cite: 236]).

### **Best Performance**
[cite_start]The **KL-based generator combined with DRS** achieved the best overall results (FID $\approx 24$) and a strong precision–recall balance[cite: 214, 232].

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
