# TOM-CSSL: Tomato Leaf Disease Recognition with Hybrid Self-Supervised and Semi-Supervised Learning

This repository contains the implementation of **TOM-CSSL** (Tomato Contrastive Semi-Supervised Learning), a two-stage hybrid framework that combines SimCLR contrastive pretraining with MixMatch semi-supervised fine-tuning for tomato leaf disease classification under limited labelled data.

The work evaluates the proposed framework on the PlantVillage tomato subset and on the Taiwan tomato leaf disease dataset (Mendeley `ngdgg79rzb`), and benchmarks it against the recently published TOM-SSL framework [Nishankar et al., AgriEngineering 2025].

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Methods Compared](#methods-compared)
3. [Datasets](#datasets)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Setup](#setup)
7. [Dataset Preparation](#dataset-preparation)
8. [How to Run](#how-to-run)
9. [Results Summary](#results-summary)
10. [Hyperparameters](#hyperparameters)
11. [References](#references)
12. [Contact](#contact)

---

## Project Overview

The work studies how self-supervised pretraining and semi-supervised learning can be combined to recognise tomato leaf diseases when only a small fraction of the dataset is labelled. The proposed approach, TOM-CSSL, is a two-stage framework:

- **Stage 1 (Contrastive pretraining):** A ResNet-18 encoder is pretrained on the unlabelled images using the SimCLR contrastive loss, producing visual representations without using any disease labels.
- **Stage 2 (Semi-supervised fine-tuning):** The pretrained encoder is fine-tuned together with a classification head using the MixMatch algorithm, which exploits both the labelled subset and the unlabelled pool via pseudo-labelling, sharpening, and MixUp.

The framework is evaluated across four label fractions on PlantVillage (1%, 10%, 30%, 50%) and across three splitting configurations on the Taiwan dataset (original, augmented with per-image random split, augmented with source-grouped split).

---

## Methods Compared

Four methods are evaluated under identical training conditions for fair comparison:

| Code Name | Description |
|---|---|
| **B1 — Supervised** | ResNet-18 with ImageNet-initialised weights, fine-tuned on labelled data only. |
| **B3 — SimCLR + FT** | ResNet-18 with SimCLR-pretrained encoder, then fully fine-tuned on labelled data. |
| **B4 — MixMatch** | ResNet-18 with ImageNet-initialised weights, trained with the MixMatch semi-supervised objective. |
| **B5 — TOM-CSSL (proposed)** | ResNet-18 with SimCLR-pretrained encoder, then fine-tuned end-to-end with the MixMatch objective. |

All four methods share the same backbone, batch size, optimiser, number of epochs, and stratified data split — only the training algorithm differs.

---

## Datasets

The work uses two publicly available tomato disease datasets from the same Mendeley repository:

> Mendeley Dataset: https://data.mendeley.com/datasets/ngdgg79rzb/1

### 1. PlantVillage Tomato Subset
- **Total images:** 16,011
- **Classes (10):** Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mould, Septoria Leaf Spot, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Two-spotted Spider Mite
- **Characteristics:** Single-leaf images on a uniform background; heavily class-imbalanced (Yellow Leaf Curl Virus dominates; Mosaic Virus is smallest)

### 2. Taiwan Tomato Leaf Disease Dataset
Available in two variants on the same Mendeley page:

**Original (Preprocessed) variant:**
- **Total images:** 622
- **Classes (6):** Bacterial Spot (110), Black Mould (67), Gray Spot (84), Late Blight (98), Powdery Mildew (157), Healthy (106)
- **Characteristics:** Field-like images; small dataset

**Augmented variant:**
- **Total images:** 4,976 (each original has ~4–8 augmented copies through rotations, mirroring, brightness shifts)
- **Classes (6):** Same as above
- **Characteristics:** This is the variant used by the TOM-SSL paper

---

## Repository Structure

```
.
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore
├── PlantVillage_Experiments.ipynb     # Main PlantVillage notebook (1%, 10%, 30%, 50% labels)
│── Taiwan_Experiments.ipynb           # Cross-dataset Taiwan experiments
│                                      # Reuses the SimCLR checkpoint from above
├── SETUP.md                           # Detailed setup instructions
└── PROTOCOL_NOTES.md                  # Notes on the source-grouped splitting protocol
```

The two notebooks are designed to be run in sequence:

1. **`PlantVillage_Experiments.ipynb`** trains the SimCLR encoder on the unlabelled PlantVillage tomato images (this is Stage 1 of TOM-CSSL) and saves the resulting encoder checkpoint to Google Drive. It then runs all four methods (Supervised, SimCLR + FT, MixMatch, TOM-CSSL) at four different label fractions (1%, 10%, 30%, 50%) on the PlantVillage tomato subset.

2. **`Taiwan_Experiments.ipynb`** loads the SimCLR checkpoint produced by the first notebook and reuses it for the Taiwan-dataset cross-validation experiments. It evaluates the same four methods on the original (622-image) Taiwan dataset and on the augmented (4,976-image) variant under two splitting protocols.

Running the Taiwan notebook without first running the PlantVillage notebook will fail because the SimCLR checkpoint will not exist.

---

## Requirements

- Python 3.10+
- A GPU is strongly recommended (the notebook is designed for Google Colab T4 GPUs with 16 GB memory)
- Google Drive (for storing the datasets and result artifacts)

Python dependencies are listed in `requirements.txt` and are largely the default Colab environment.

---

## Setup

### Option 1: Run in Google Colab (recommended)

This is the workflow used during the development of this project.

1. Open either notebook in Colab (start with `PlantVillage_Experiments.ipynb`)
2. Switch the runtime to GPU (`Runtime → Change runtime type → T4 GPU`)
3. Mount your Google Drive when prompted by the first cell
4. Update the dataset paths near the top of the notebook to match your Google Drive structure (see [Dataset Preparation](#dataset-preparation) below)
5. Run the cells sequentially

### Option 2: Run locally

If you have a local machine with a CUDA GPU:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
jupyter notebook notebooks/PlantVillage_Experiments.ipynb
```

You will need to replace the Google Drive paths in the notebooks with local paths.

---

## Dataset Preparation

### PlantVillage Tomato Dataset

The PlantVillage notebook downloads the dataset automatically from Kaggle. To enable this, you need a Kaggle API key:

1. Go to https://www.kaggle.com/settings → API → "Create New API Token"
2. This downloads a `kaggle.json` file to your computer
3. When prompted by the PlantVillage notebook, upload this `kaggle.json` file
4. The notebook will then download and extract the tomato subset automatically

The tomato classes are filtered from the full PlantVillage dataset and saved at `/content/TomatoDataset/` on the Colab local disk.

### Taiwan Tomato Dataset

The Taiwan dataset is downloaded manually from Mendeley:

https://data.mendeley.com/datasets/ngdgg79rzb/1

Extract the archive locally — you should find a folder containing both PlantVillage and Taiwan tomato datasets. Upload the Taiwan folder to your Google Drive at the location expected by the notebook (you can adjust this path in the setup cells if needed):

```
/MyDrive/
└── taiwan/
    ├── Preprocessed data/
    │   ├── Train/
    │   │   ├── Bacterial spot/
    │   │   ├── Black mold/
    │   │   ├── Gray spot/
    │   │   ├── Late blight/
    │   │   ├── health/
    │   │   └── powdery mildew/
    │   └── Test/
    │       └── (same six class folders)
    └── data augmentation/
        ├── Train/
        │   └── (same six class folders, with augmented images)
        └── Test/
            └── (same six class folders, with augmented images)
```

The notebook automatically flattens the `Train/` and `Test/` folders into a single class-folder layout during the setup phase, so you do not need to merge them manually.

### SimCLR Checkpoint

The Taiwan notebook reuses a SimCLR encoder pretrained on the PlantVillage tomato images. This checkpoint is produced by running the SimCLR pretraining cells in `PlantVillage_Experiments.ipynb` (Section 7). After the PlantVillage notebook runs successfully, the checkpoint is saved to:

```
/MyDrive/Tomato_3regime/RESULTS/simclr/simclr_checkpoint.pth
```

The Taiwan notebook expects this checkpoint at a slightly different path:

```
/MyDrive/Tomato_3regime/RESULTS 2/simclr/simclr_checkpoint.pth
```

You can either copy the checkpoint to the second location, or update the `SIMCLR_CKPT_SRC` variable in the Taiwan notebook to point to the original `RESULTS/simclr/` path. The path difference is purely a folder-naming convention used during this project — the checkpoint itself is the same.

---

## How to Run

The two notebooks should be run in order: PlantVillage first (which trains the SimCLR encoder), then Taiwan (which reuses the SimCLR encoder for cross-dataset evaluation).

### Notebook 1: PlantVillage Experiments

`notebooks/PlantVillage_Experiments.ipynb` runs the four methods at four label fractions on the PlantVillage tomato subset, after first training the SimCLR encoder.

1. **Section 0** — Environment check.
2. **Section 1** — Mount Drive, download the PlantVillage dataset from Kaggle, extract the tomato classes, and set up the results folder structure on Google Drive.
3. **Sections 2–7** — Set up hyperparameters, define the dataset wrappers and the stratified-split function, then run **SimCLR pretraining** (Stage 1) on the unlabelled PlantVillage tomato images. The SimCLR encoder checkpoint is saved to `/MyDrive/Tomato_3regime/RESULTS/simclr/`.
4. **Section 9 (10% labels)** — Configure the labelled/unlabelled/validation split at 10% labelled fraction and run the four method cells (B1, B3, B4, B5) plus the within-experiment comparison.
5. **Section 9b (30% labels)** — Repeat for 30% labels.
6. **Section 9c (50% labels)** — Repeat for 50% labels.
7. **Section 9d (1% labels)** — Repeat for 1% labels.

Each label-fraction section saves results into a separate subfolder under `/MyDrive/Tomato_3regime/RESULTS/<pct>pct/`. After all four label fractions complete, you have the full PlantVillage results table.

### Notebook 2: Taiwan Experiments

`notebooks/Taiwan_Experiments.ipynb` runs the four methods on the Taiwan dataset under three different configurations. It assumes the SimCLR checkpoint from the PlantVillage notebook is already on Drive.

**Phase 1 — Original (Preprocessed) Taiwan**
1. Run the environment/Drive setup cells and the **Taiwan structure flattening** cell. The flattening cell merges the `Train/` and `Test/` subfolders into a single class-folder layout under `/content/Taiwan_flat/`.
2. Run the experiment cells for B1, B3, B4, and B5 in order on the 622-image preprocessed dataset.
3. Run the within-experiment comparison cell to generate Table and confusion-matrix outputs.

**Phase 2 — Augmented Taiwan, Per-Image Random Split (Experiment A)**
1. Run the **augmented dataset flattening** cell.
2. Run the **filename inspection** cell to verify the source-ID extraction pattern.
3. Run the **Experiment A setup** cell, which builds a naive random split at 10% labels and prints a leakage diagnostic.
4. Run B1, B3, B4, and B5 method cells in order.
5. Run the within-experiment comparison cell.

**Phase 3 — Augmented Taiwan, Source-Grouped Split (Experiment B)**
1. Run the **source-grouped split function** cell.
2. Run the **Experiment B setup** cell. The leakage diagnostic should report zero overlap.
3. Run B1, B3, B4, and B5 method cells in order (these are identical to Experiment A's cells — only the upstream split differs).
4. Run the within-experiment comparison cell.

**Phase 4 — Master Cross-Protocol Comparison**

Run the master comparison cell to produce the consolidated Taiwan results table and figures across all three protocols.

**Phase 5 — Multi-Seed Validation**

The Taiwan notebook includes a multi-seed sweep that repeats Experiments A and B with random seeds 123 and 456 in addition to the default seed 42. This allows reporting mean ± standard deviation across three seeds. Total runtime for the multi-seed sweep is approximately 60 minutes on a T4 GPU.

### Recovery Cells

If your Colab session disconnects mid-experiment, the Taiwan notebook includes a **Recovery section** that rebuilds the environment from scratch (mount Drive, rebuild local dataset folder, redefine helper functions and the SimCLR model class). Run these cells in order after a disconnect, then continue from where you left off. Results saved to Drive before the disconnect are preserved.

---

## Results Summary

### PlantVillage Tomato Subset

Accuracy (%) at four label fractions:

| Model | 1% | 10% | 30% | 50% |
|---|---:|---:|---:|---:|
| Supervised | 76.96 | 95.30 | 98.75 | 98.94 |
| SimCLR + FT | 84.47 | 97.68 | 99.69 | 99.37 |
| MixMatch | **86.47** | 97.75 | 99.56 | 99.44 |
| TOM-CSSL (proposed) | 82.16 | **98.00** | **99.75** | **99.56** |

### Taiwan Tomato Disease Dataset

Accuracy (%) — multi-seed mean ± std for augmented protocols (seeds 42, 123, 456); single seed for original:

| Model | Original (622) | Augmented + Naive | Augmented + Grouped |
|---|---:|---:|---:|
| Supervised | 51.5 | 81.5 ± 1.9 | 66.7 ± 2.3 |
| SimCLR + FT | 57.6 | 83.0 ± 2.6 | 64.9 ± 2.8 |
| MixMatch | **59.1** | 84.7 ± 3.1 | **72.0 ± 1.4** |
| TOM-CSSL (proposed) | 56.1 | 82.6 ± 2.3 | 65.8 ± 2.3 |
| _MixMatch [TOM-SSL paper]_ | — | _67.26_ | — |
| _TOM-SSL [paper]_ | — | _70.87_ | — |

Detailed metrics, confusion matrices, training curves, and per-class breakdowns for every method and protocol are saved as `metrics.json`, `confusion_matrix.png`, `training_curves.png`, and `per_class_metrics.png` files in the corresponding output folders under `/MyDrive/Tomato_3regime/RESULTS_TAIWAN*/`.

---

## Hyperparameters

These values are held constant across all four methods to ensure a fair comparison.

### Shared
- **Backbone:** ResNet-18 (ImageNet-pretrained)
- **Input resolution:** 224 × 224
- **Batch size:** 128
- **Optimiser:** Adam
- **Training epochs:** 30 (Stage 2 / supervised baseline)
- **Validation split:** 10% of the dataset
- **Per-class floor:** 10 images minimum per class in each split

### Stage 1 (SimCLR pretraining)
- **Epochs:** 10
- **Learning rate:** 1 × 10⁻⁴
- **Temperature:** 0.5
- **Projection head:** 512 → 256 → 128 with ReLU activation

### Stage 2 (MixMatch fine-tuning) / B4 MixMatch baseline
- **K (augmentations per unlabelled sample):** 2
- **Sharpening temperature T:** 0.5
- **MixUp α:** 0.75
- **Unsupervised loss weight λᵤ:** 1.0
- **Learning rate:** 1 × 10⁻⁴

### B1 Supervised baseline
- **Learning rate:** 1 × 10⁻³

### Multi-seed evaluation
- **Seeds:** 42 (primary), 123, 456 (for Taiwan augmented protocols)

---

## References

This work builds on the following published methods:

1. **SimCLR** — Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML 2020*.
2. **MixMatch** — Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., & Raffel, C. (2019). MixMatch: A holistic approach to semi-supervised learning. *NeurIPS 2019*.
3. **ResNet** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
4. **TOM-SSL (compared baseline)** — Nishankar, S., Mithuran, T., Thuseethan, S., Sebastian, Y., Yeo, K. C., & Shanmugam, B. (2025). TOM-SSL: Tomato disease recognition using pseudo-labelling-based semi-supervised learning. *AgriEngineering*, 7, 248.
5. **PlantVillage and Taiwan datasets** — Mendeley Data: https://data.mendeley.com/datasets/ngdgg79rzb/1

---

## Acknowledgements

We acknowledge the authors of the PlantVillage and Taiwan tomato leaf disease datasets for making their data publicly available, and the authors of SimCLR, MixMatch, and TOM-SSL for the methods we build upon and compare against.
