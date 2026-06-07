# Notes on the Source-Grouped Splitting Protocol

This document explains the source-grouped stratified splitting protocol used in the augmented Taiwan experiments, and how it differs from the standard per-image random split.

---

## Background

The augmented Taiwan dataset contains 4,976 images derived from approximately 1,146 unique source images. Each source has roughly 4–8 augmented copies created by applying rotations (90°, 180°, 270°), horizontal and vertical mirroring, and brightness adjustments to the original image.

Filenames follow a predictable pattern that encodes the source identity:

```
Bs100.JPG                ← original source image of class "Bacterial spot"
Bs100_mirror.jpg         ← same source, horizontally mirrored
Bs100_mirror_vertical.jpg
Bs100_change_90.jpg      ← same source, rotated 90°
Bs100_change_180.jpg
Bs100_change_270.jpg
Bs100_hight.jpg          ← same source, brighter
Bs100_lower.jpg          ← same source, darker
```

These eight files are all augmented variants of the same physical leaf photographed once. Class prefixes are `Bs` (Bacterial spot), `Blm` (Black mold), `Gls` (Gray spot), `Lb` (Late blight), `h` (health), `pm` (powdery mildew).

---

## Two Splitting Protocols

The notebook supports two protocols for splitting the augmented dataset into labelled, unlabelled, and validation partitions.

### Protocol A — Per-Image Random Split (Naive)

This is the protocol described in the TOM-SSL paper. Each augmented image is treated as an independent sample, and stratified random sampling is performed at the image level.

Implementation: `stratified_split_with_floor()` in the notebook.

### Protocol B — Source-Grouped Split

In this protocol, the splitting algorithm operates at the source-image level rather than the file level. All augmented copies of a source image are assigned together to the same partition (labelled, unlabelled, or validation). This is conceptually similar to "patient-level splits" in medical imaging or "subject-level splits" in face recognition.

Implementation: `grouped_stratified_split()` in the notebook.

The source identifier is extracted from each filename using the following regular expression:

```python
import re
def source_id_from_filename(fname):
    stem = os.path.splitext(fname)[0]
    m = re.match(r"^(Train|Test)_([A-Za-z]+\d+)", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return stem
```

For example, `Test_Bs100_mirror.jpg`, `Test_Bs100_change_90.jpg`, and `Test_Bs100.JPG` all map to the source ID `Test_Bs100`. The grouping algorithm then walks through source IDs in random order and assigns each whole group to one of the three partitions until the per-class quotas are met.

---

## Leakage Diagnostic

The notebook includes a diagnostic cell that measures, for any given split, how many validation source-IDs have at least one augmented sibling in the labelled training set. This is reported as a fraction of total validation source-IDs.

- Under Protocol A: typically 35–45% of validation source-IDs have a labelled sibling.
- Under Protocol B: 0% by construction (the assertion `assert len(overlap) == 0` verifies this at runtime).

---

## When Each Protocol Is Appropriate

- **Protocol A** is the more common practice in plant disease SSL papers, and is required for direct numerical comparison with prior work that uses the same protocol.
- **Protocol B** is a more conservative evaluation that measures generalisation to unseen source images, not just unseen augmentations of seen images. Accuracies under Protocol B will be lower because the model cannot exploit augmented-sibling memorisation.

This repository reports results under both protocols for transparency. The two protocols are otherwise identical: same dataset, same code, same hyperparameters, same model, same epochs — only the split-construction step differs.

---

## Reference Implementation

For a complete reference implementation of source-aware splitting in a related context, see scikit-learn's `GroupKFold` and `GroupShuffleSplit`:

https://scikit-learn.org/stable/modules/cross_validation.html#group-cv

These library utilities accept a `groups` argument and ensure that no group appears in more than one fold. The grouping concept used in this work is conceptually equivalent.
