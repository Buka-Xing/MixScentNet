# MixScentNet — Anonymous Code Repository

> **Paper:** *MixScentNet: A Graph-Based Framework with Self-Supervised Pretraining for Olfactory Perception Prediction of Molecular Mixtures*
> **Submission:** NeurIPS 2026 — Main Track (under double-blind review)

---

## ⚠️ Anonymous Submission Notice

This repository accompanies an **anonymous** submission to NeurIPS 2026 and is shared via an anonymous link **for review purposes only**. In compliance with the double-blind reviewing policy:

- All author names, institutional identifiers, acknowledgements, and personal URLs have been removed from the codebase.
- The repository is provided **strictly for reviewer inspection of the methodology and reported results** and must not be redistributed.
- The full source code, pretrained model weights, processed datasets, and a permissive open-source license will be released **upon acceptance**.

If reviewers identify any residual de-anonymizing content, we kindly ask them to flag it in their reviews so that we can remedy it promptly.

---

## 📁 Repository Structure

```
MixScentNet/
├── MixScentNet_Label.py              # Entry point: DREAM2025 label prediction
├── MixScentNet_Similarity.py         # Entry point: DREAM2024 perceptual similarity
├── MixScentNet_OW_train&test.py      # Entry point: Olfactory White reproduction
│
├── backbones/                        # Model implementations
│   ├── dmpnn/                        # Single-molecule encoder
│   │   └── ...                       # D-MPNN instantiation (Mordred-target pretraining)
│   └── gat/                          # Mixture-level encoder
│       └── ...                       # 3-layer residual GATv2 with PNA readout
│
├── datasets/                         # Curated experimental data
│   ├── DREAM2025/                    # Mixtures + 51-d perceptual labels
│   └── DREAM2024/                    # Mixture pairs + perceptual distances
│                                     #   (Snitz, Ravia, Bushdid subsets)
│
└── results-manuscript/               # Training logs & results in the paper
    ├── DREAM2025/                    # 5-fold logs + per-fold metrics
    ├── DREAM2024/                    # 5-fold logs + per-fold metrics
    └── olfactory_white/              # 3 LOSO splits + scatter data
```

---

## 🛠️ Environment

The codebase has been tested under the following environment:

| Component        | Version                          |
| ---------------- | -------------------------------- |
| OS               | Ubuntu 16.04                     |
| CPU              | Intel Core i9-9900K              |
| GPU              | NVIDIA RTX 3090 (24 GB)          |
| Python           | 3.9                              |
| PyTorch          | 1.13                             |
| DGL              | 1.1                              |
| RDKit            | 2023.03                          |
| Mordred          | 1.2.0                            |
| Chemprop         | 2.0                              |

A complete `requirements.txt` will be released along with the de-anonymized code upon acceptance.

---

## 🚀 Quick Start

After setting up the environment, the three main experiments reported in the paper can be reproduced with single commands:

### 1. Label Prediction on DREAM2025 (Table 1, Table 2)

```bash
python MixScentNet_Label.py
```

This script:
- Loads the DREAM2025 mixture dataset from `datasets/DREAM2025/`.
- Initializes the pretrained D-MPNN encoder from `backbones/dmpnn/`.
- Builds the mixture-as-graph and applies the 3-layer residual GATv2 from `backbones/gat/`.
- Performs **5-fold cross-validation** (80% / 20% train–test) with the weighted Huber loss described in the paper.
- Reports PLCC-sam (↑) and cos-dist (↓) per fold and aggregated mean ± deviation.

### 2. Perceptual Similarity on DREAM2024 (Table 1, Table 2)

```bash
python MixScentNet_Similarity.py
```

This script:
- Loads the DREAM2024 mixture-pair dataset (Snitz + Ravia + Bushdid subsets) from `datasets/DREAM2024/`.
- Encodes each mixture in a pair independently, then computes the L1 distance between the L1-normalized mixture embeddings as the predicted perceptual distance.
- Performs **5-fold cross-validation** under the same protocol as the label task.
- Reports PLCC (↑) and KRCC (↑) per fold and aggregated mean ± deviation.

### 3. Olfactory White Reproduction (Section 5 + Appendix E)

```bash
python "MixScentNet_OW_train&test.py"
```

This script:
- Implements the **leave-one-subset-out** protocol on DREAM2024.
- Trains MixScentNet on two of {Snitz, Ravia, Bushdid} and evaluates on the held-out subset.
- For every test mixture pair, records the geometric mean √(n₁·n₂) and the predicted perceptual distance.
- Computes Pearson r, Spearman ρ, and OLS regression statistics, and exports scatter data used to render the figures in the paper.
- Cycles through all three held-out configurations to reproduce the main-text result and the two additional results in Appendix E.

> **Note:** The shell-special character `&` in the filename should be quoted (e.g. `"..."` in bash) when invoking this script.

---

## 📂 Folder Details

### `backbones/`

| Sub-folder | Contents |
| --- | --- |
| `dmpnn/` | Reproduces the **MaskedDescriptorsMPNN** pretraining described in Appendix B.1: a Bond-centric D-MPNN with hidden dim 2048, depth 6, mean aggregation, and a regression FFN that predicts standardized + Winsorized Mordred descriptors under a 15%-masked MSE objective. The module exposes a `fingerprint(bmg)` function that returns a 2048-dim per-molecule embedding used by all downstream entry points. |
| `gat/` | Implements the mixture-level encoder: a 3-layer residual GATv2 stack with multi-head attention (concat) at layers 1–2 and a single-head layer at layer 3, each followed by a residual projection, ELU, LayerNorm, and dropout. The graph-level readout uses **Principal Neighborhood Aggregation (PNA)**, concatenating mean / std / min / max statistics and projecting back to the hidden dimension via an FFN, as described in Section 3.2. |

### `datasets/`

| Sub-folder | Contents |
| --- | --- |
| `DREAM2025/` | The 650+ mixtures (2/3/5/10 components each) annotated with 51-dim RATA perceptual descriptors. Stored as preprocessed `.pt` / `.csv` splits ready for the 5-fold protocol. |
| `DREAM2024/` | The 507 mixture pairs (731 unique mixtures, 168 unique molecules) drawn from the Snitz, Ravia, and Bushdid subsets. Per-pair perceptual distances are linearly normalized to [0, 1] following the DREAM2024 organizers' protocol. Subset membership is preserved to support the leave-one-subset-out olfactory-white protocol. |

> **Pretraining data not included.** The pretraining corpus (~1M filtered SMILES strings derived from PubChem; see Appendix A.1) is large and is not bundled with this repository. Pretrained encoder weights are distributed instead, located inside `backbones/dmpnn/`.

### `results-manuscript/`

This folder contains the **training logs and result records that back every number reported in the paper**, organized to mirror the manuscript's table and figure structure:

- `DREAM2025/` — per-fold training curves, best-epoch checkpoints' metrics, and aggregated mean ± deviation that appear in Table 1 / Table 2 (label-prediction columns).
- `DREAM2024/` — same as above for the similarity-prediction columns.
- `olfactory_white/` — per-split scatter data, regression fits, and Pearson/Spearman statistics that produce Figure 4 (main paper) and Figure A.1–A.2 (Appendix E).

Reviewers can use these logs to verify that the reported numbers reflect actual training runs rather than post-hoc selection.

---

## 🔁 Reproducibility

| Aspect | Setting |
| --- | --- |
| Random seeds | Fixed per fold (`fold_idx = 0…4`); reported numbers are the mean across seeds. |
| Cross-validation | 5-fold, 80% / 20% train–test, **no separate validation set**; the best-PLCC epoch per fold is selected. |
| Hardware | Single NVIDIA RTX 3090. End-to-end training takes ~6 hours for DREAM2025 and ~4 hours for DREAM2024 (per 5-fold sweep). |
| Comparability | All baselines (XGBoost+RDKit/POM/MOLT5, CheMeleon, MolSets, POM+CheMix, POMMix) are evaluated on **the exact same fold splits** used for MixScentNet. |

All hyperparameters, optimizer settings, and loss weighting schemes follow Appendix B and are encoded as defaults in the three entry-point scripts; no per-experiment manual tuning is required to reproduce the reported numbers.

---

## 📊 Mapping from Code to Paper

| Manuscript element | Reproducing artifact |
| --- | --- |
| Table 1 (standard 5-fold CV) | `MixScentNet_Label.py`, `MixScentNet_Similarity.py` + logs in `results-manuscript/DREAM2025/`, `results-manuscript/DREAM2024/` |
| Table 2 (unseen-molecule CV) | Same scripts with the `--split unseen_molecule` flag (default protocol described in Appendix A.3) |
| Figure 4 + Appendix E (olfactory white) | `MixScentNet_OW_train&test.py` + scatter data in `results-manuscript/olfactory_white/` |
| Ablation tables (Appendix D) | Configurable variants are selectable from the entry-point scripts via CLI flags; logs for each variant are deposited under `results-manuscript/<task>/ablation_*/`. |

---

## 📝 Citation

A BibTeX entry will be provided after the double-blind review process. For the duration of the review, please refer to the submission via its OpenReview ID.

---

## 📜 License

Code and data in this anonymous repository are made available **for the sole purpose of NeurIPS 2026 peer review**. A formal open-source license (CC BY 4.0 for data, MIT for code) will be applied upon de-anonymization at the camera-ready stage.
