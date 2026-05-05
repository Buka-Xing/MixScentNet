# MixScentNet — Anonymous Code Repository

> **Paper:** *MixScentNet: A Graph-Based Framework with Self-Supervised Pretraining for Olfactory Perception Prediction of Molecular Mixtures*
> **Submission:** NeurIPS 2026 — Main Track (under double-blind review)

---

## ⚠️ Anonymous Submission Notice

This repository accompanies an **anonymous** submission to NeurIPS 2026 and is shared via an anonymous link **for review purposes only**. In compliance with the double-blind reviewing policy:

- All author names, institutional identifiers, acknowledgements, and personal URLs have been removed from the codebase.
- The repository is provided **strictly for reviewer inspection of the methodology and reported results** and must not be redistributed.
- The **pretrained model weights, processed pre-training datasets**, and a permissive open-source license will be released **upon acceptance**.

If reviewers identify any residual de-anonymizing content, we kindly ask them to flag it in their reviews so that we can remedy it promptly.

---

## 📁 Repository Structure

```
MixScentNet/
├── MixScentNet_Label.py              # Entry point: DREAM2025 label prediction
├── MixScentNet_Similarity.py         # Entry point: DREAM2024 perceptual similarity
├── MixScentNet_OW_train&test.py      # Entry point: Olfactory White reproduction
├── DMPNN_pretrained.pt               # The pretrained weight for the DMPNN (Release upon acceptance)
│
├── backbones/                        # Model implementations and dataset loader
│   ├── DMPNN.py                      # DMPNN Instantiation
│   ├── dataloader.py                 # Dataset loader
│   └── model_gat.py                  # 3-layer residual GATv2 with PNA readout
│
├── datasets/                         # Curated experimental data
│   ├── DREAM2025/                    # Mixtures + 51-d perceptual labels
│   └── DREAM2024/                    # Mixture pairs + perceptual distances
│   └── PubChem/                      # The Curated PubChem dataset for pre-training (Release upon acceptance)
│
└── results-manuscript/               # Training logs & results in the mainuscript
    ├── EXP1/                         # 5-fold logs + per-fold metrics for TABLE 1
    ├── EXP2/                         # 5-fold logs + per-fold metrics for TABLE 2
    └── olfactory_white/              # 3 LOSO splits + scatter data
```

---

## 🛠️ Environment

The codebase has been tested under the following environment:

| Component | Version |
| --- | --- |
| OS | Ubuntu 16.04 |
| CPU | Intel Core i9-9900K |
| GPU | NVIDIA RTX 3090 (24 GB) |
| CUDA | 11.8 |
| Python | 3.10 |

### Python Dependencies

| Package | Version |
| --- | --- |
| `torch` | 2.3.0+cu118 |
| `dgl` | 2.2.1+cu118 |
| `chemprop` | 2.2.3 |
| `rdkit` | 2026.3.1 |
| `scikit-learn` | 1.8.0 |
| `scipy` | 1.17.1 |
| `numpy` | 2.4.4 |
| `pandas` | 3.0.2 |
| `torchmetrics` | 1.9.0 |
| `matplotlib` | 3.10.9 |
| `tqdm` | 4.67.3 |

---

## 🚀 Quick Start

### 0. Environment settings

```bash
# 1. Create a fresh conda environment
conda create -n mixscentnet python=3.10 -y
conda activate mixscentnet

# 2. Install PyTorch with CUDA 11.8
pip install torch==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. Install DGL with matching CUDA build
pip install dgl==2.2.1+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

# 4. Install the remaining dependencies
pip install -r requirements.txt
```
### 1. Label Prediction on DREAM2025 (Table 1, Table 2)

```bash
python MixScentNet_Label.py --split random_cv --loss HuberLoss
```

This script:
- Loads the DREAM2025 mixture dataset from `datasets/DREAM2025/`.
- Performs **5-fold cross-validation** (80% / 20% train–test) with the Huber loss described in the paper.
- Reports PLCC-sam (↑) and cos-dist (↓) per fold and aggregated mean ± deviation.
- '--split' can be chosen as {'random_cv', 'random_cv_unseen'} for reproducing results in TABLE 1&2
- '--loss' can be chosen as {'HuberLoss', 'MSELoss','MAELoss','MeanPLCCLoss'} 
### 2. Perceptual Similarity on DREAM2024 (Table 1, Table 2)

```bash
python MixScentNet_Similarity.py --split random_cv --loss MAELoss
```

This script:
- Loads the DREAM2024 mixture-pair dataset (Snitz + Ravia + Bushdid subsets) from `datasets/DREAM2024/`.
- Encodes each mixture in a pair independently, then computes the L1 distance between the mixture embeddings as the predicted perceptual distance.
- Performs **5-fold cross-validation** under the same protocol as the label task.
- Reports PLCC (↑) and KRCC (↑) per fold and aggregated mean ± deviation.
- '--split' can be chosen as {'random_cv', 'random_cv_unseen'} for reproducing results in TABLE 1&2
- '--loss' can be chosen as {'MAELoss','MSELoss','PLCCLoss'}
 
### 3. Olfactory White Reproduction (Section 5 + Appendix E)

```bash
python "MixScentNet_OW_train&test.py" --train-sources Snitz+Ravia
```

This script:
- Implements the **leave-one-subset-out** protocol on DREAM2024.
- Trains MixScentNet on two of {Snitz, Ravia, Bushdid} and evaluates on the held-out subset.
- For every test mixture pair, records the geometric mean √(n₁·n₂) and the predicted perceptual distance.
- Computes Pearson r, Spearman ρ, and OLS regression statistics, and exports scatter data used to render the figures in the paper.

---

## 🔁 Reproducibility

| Aspect | Setting |
| --- | --- |
| Random seeds | Fixed per fold (`fold_idx = 0…4`); reported results are the mean±std across folds. |
| Cross-validation | 5-fold, 80% / 20% train–test, **no separate validation set**; the best-PLCC epoch per fold is selected. |
| Hardware | Single NVIDIA RTX 3090. |
| Comparability | All baselines (XGBoost+RDKit/POM/MOLT5, CheMeleon, MolSets, POM+CheMix, POMMix) are evaluated on **the exact same fold splits** used for MixScentNet. |

All hyperparameters, optimizer settings, and loss weighting schemes follow the Appendix and are encoded as defaults in the three entry-point scripts; no per-experiment manual tuning is required to reproduce the reported numbers.

---

## 📊 Mapping from Code to Main Results in the Paper

| Manuscript element | Reproducing artifact |
| --- | --- |
| Table 1 (standard 5-fold CV) | `MixScentNet_Label.py --split random_cv`, `MixScentNet_Similarity.py --split random_cv` + logs in `results-manuscript/EXP1/` |
| Table 2 (unseen-molecule CV) | Same scripts with the `--split random_cv_unseen` flag + logs in `results-manuscript/EXP2/` |
| Figure 4 + Appendix E (olfactory white) | `MixScentNet_OW_train&test.py` + logs in `results-manuscript/olfactory_white/` |

---

## 📜 License

Code and data in this anonymous repository are made available **for the sole purpose of NeurIPS 2026 peer review**. A formal open-source license (CC BY 4.0 for data, MIT for code) will be applied upon de-anonymization at the camera-ready stage.
