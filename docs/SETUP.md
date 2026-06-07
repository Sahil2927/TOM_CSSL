# Detailed Setup Instructions

This document provides step-by-step instructions for setting up the environment and running the experiments. Most users will prefer to run the notebook on Google Colab (which is the development environment used for this project), but local execution is also possible if a CUDA-enabled GPU is available.

---

## Google Colab Setup (Recommended)

This is the easiest path and matches the development environment.

### 1. Open the notebook in Colab

- Go to https://colab.research.google.com/
- File → Upload notebook → choose `notebooks/Taiwan_Experiments.ipynb`

Alternatively, if the notebook is in your GitHub repository, you can open it directly via:

```
https://colab.research.google.com/github/<username>/<repo-name>/blob/main/notebooks/Taiwan_Experiments.ipynb
```

### 2. Switch to GPU runtime

- Runtime → Change runtime type
- Hardware accelerator: **T4 GPU** (or any available CUDA GPU)
- Click Save

You can verify GPU access by running the first cell of the notebook, which prints PyTorch and CUDA information.

### 3. Mount Google Drive

The notebook stores datasets and outputs on Google Drive. The first interactive cell mounts your Drive at `/content/drive/MyDrive/`. You will be prompted to grant access.

### 4. Prepare the dataset on Drive

Download the Taiwan tomato dataset from Mendeley:

https://data.mendeley.com/datasets/ngdgg79rzb/1

Extract the archive locally on your computer, then upload the `Taiwan` (or similarly-named) folder to your Google Drive. The expected location is:

```
/MyDrive/taiwan/
├── Preprocessed data/
│   ├── Train/
│   └── Test/
└── data augmentation/
    ├── Train/
    └── Test/
```

If your folder is named differently, update the `TAIWAN_SRC` and `AUG_SRC` paths in the corresponding setup cells of the notebook.

### 5. SimCLR checkpoint

The Taiwan experiments reuse a SimCLR encoder that was pretrained earlier on the PlantVillage tomato images. The notebook expects this checkpoint at:

```
/MyDrive/Tomato_3regime/RESULTS 2/simclr/simclr_checkpoint.pth
```

If you do not have this file, you have two options:
- **Option A:** Run the SimCLR pretraining cells from the companion PlantVillage notebook first, which will generate the checkpoint at the expected location.
- **Option B:** Modify the Taiwan notebook to use SimCLR pretraining directly on the Taiwan dataset (Stage 1 would then be trained on Taiwan unlabelled images instead of PlantVillage ones). This is a one-line change in the model loading section.

### 6. Run cells sequentially

The notebook is designed to be run top-to-bottom. Each section is self-contained but assumes the earlier cells in that section have been executed.

---

## Local Setup (Alternative)

If you prefer to run the notebook on your local machine with a CUDA GPU:

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify CUDA is available

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 5. Modify paths in the notebook

The notebook uses Google Drive paths (`/content/drive/MyDrive/...`). For local execution, replace these with local paths. The key paths to update are:

- `TAIWAN_SRC` — path to the preprocessed Taiwan folder
- `AUG_SRC` — path to the augmented Taiwan folder
- `PROJECT_DIR` — where to save results
- `SIMCLR_CKPT_SRC` — path to the SimCLR checkpoint

You can also comment out the Google Drive mounting cell.

### 6. Launch Jupyter

```bash
jupyter notebook notebooks/Taiwan_Experiments.ipynb
```

---

## Expected Runtime

Approximate runtimes on a single NVIDIA T4 GPU (Google Colab free tier):

| Phase | Approximate runtime |
|---|---|
| **PlantVillage notebook** | |
| Dataset download (Kaggle) + setup | 5 min |
| SimCLR pretraining (10 epochs on PV unlabelled) | 15 min |
| 10% labels — 4 methods | 45 min |
| 30% labels — 4 methods | 60 min |
| 50% labels — 4 methods | 75 min |
| 1% labels — 4 methods | 20 min |
| **Taiwan notebook** | |
| Preprocessed Taiwan (4 methods × 1 seed) | 10 min |
| Augmented Taiwan A (4 methods × 1 seed) | 25 min |
| Augmented Taiwan B (4 methods × 1 seed) | 25 min |
| Multi-seed sweep (4 methods × 2 protocols × 2 additional seeds) | 60 min |
| **Total (full reproduction of both notebooks)** | **~6 hours** |

If a Colab session disconnects mid-run, the Taiwan notebook's **Recovery section** can be used to restore the environment without losing previously-saved results (which are persisted to Google Drive). Free-tier Colab sessions can disconnect after ~4 hours of compute, so running the full pipeline may require splitting across multiple sessions.

---

## Troubleshooting

### "Taiwan dataset not found at /content/drive/MyDrive/..."
The path in the notebook does not match where you placed the dataset on Drive. Open the cell that defines `TAIWAN_SRC` (and `AUG_SRC` for the augmented experiment) and update the path.

### "SimCLR checkpoint not found at ..."
The SimCLR checkpoint is expected at a specific path on Drive. Either run the PlantVillage SimCLR pretraining step first, or update the `SIMCLR_CKPT_SRC` path to point to your checkpoint.

### "OutOfMemoryError" during training
Reduce the batch size. The default is 128; values of 64 or 32 should fit on smaller GPUs. The MixMatch cells use roughly 3x more memory than the supervised baseline because of the K=2 augmentations.

### "OSError: Bad file descriptor" warnings during DataLoader cleanup
These are harmless multiprocessing warnings issued by PyTorch's DataLoader at the end of each epoch. They do not affect training results.

### Colab session disconnected mid-experiment
Use the **Recovery cells** in the notebook (located near the multi-seed sweep). These re-import everything, re-mount Drive, rebuild the local dataset folder, and reload the SimCLR checkpoint, so you can continue from the next experiment without losing prior results (which are saved to Drive after each method completes).

---

## Hardware Notes

- The notebook was developed and tested on **Google Colab T4 GPU** (16 GB VRAM).
- A modern desktop GPU with 8+ GB VRAM should work with the default batch size of 128.
- The MixMatch experiments are the most memory-intensive due to the K=2 augmentations per unlabelled sample.
- The whole pipeline (excluding the multi-seed sweep) fits comfortably within Colab free-tier session limits.
