# DMPNN\_pretrain

Pre-training pipeline for the D-MPNN molecular fingerprint encoder used in MixScentNet.  
The model is trained to predict Mordred 2D descriptors from molecular graphs, which gives it a
chemistry-aware initialization before being fine-tuned on olfactory tasks.

---

🗺️ ## Directory structure

```
DMPNN_pretrain/
├── output/                             # Pretraining checkpoints folder
├── training_store/                     # Mordred descriptor folder (zarr)
├── filter_pubchem_olfactory.py         # filter rules for raw PubChem SMILE.
├── pubchem_olfactory_1million.smiles   # ~1.17 M curated SMILES
├── features_mordred.py                 # compute Mordred descriptors → zarr
├── pretrain_mordred.py                 # pre-train D-MPNN via masked descriptor prediction
└── torchford.py                        # Welford online mean/variance accumulator
```

---

🕹️ ## Pipeline overview

```
raw pubchem.smiles (7.1GB raw file, from https://zenodo.org/records/15733575)
        │
        ▼  filter_pubchem_olfactory.py
pubchem_olfactory_1million.smiles
        │
        ▼  features_mordred.py
training_store/          (zarr array, shape N × 1613)
        │
        ▼  pretrain_mordred.py
output/best.pt           (pre-trained D-MPNN weights)
```

---

⚙️ ## File descriptions

### `filter_pubchem_olfactory.py`

Filters a raw `pubchem.smiles` file down to a clean olfactory-relevant subset using six
sequential rules applied with 8 parallel workers:

| # | Rule | Level |
|---|------|-------|
| 1 | Remove SMILES longer than 150 characters | string |
| 2 | Remove multi-component SMILES (contain `.`) | string |
| 3 | Remove SMILES with inorganic atoms: He, Na, Mg, Al, K, Ca, Ti, V, Cr, Fe, Co, Cu, Zn, Bi | string (regex) |
| 4 | Remove SMILES that RDKit cannot parse | RDKit |
| 5 | Remove molecules for which `RemoveHs` raises an error | RDKit |
| 6 | Keep only molecules with molecular weight in **[30, 300] Da** | RDKit |

**Output:** `pubchem_olfactory_1million.smiles` (~1,168,920 molecules, one SMILES per line)

```bash
python filter_pubchem_olfactory.py
# reads  pubchem.smiles
# writes pubchem_olfactory_1million.smiles
```

---

### `features_mordred.py`

Computes all **Mordred 2D descriptors** (1,613 features) for every SMILES in the filtered file
and streams the results into a **zarr** array on disk. Memory usage is bounded because
descriptors are computed in batches and written incrementally.

**Output:** `training_store/` — zarr array of shape `(N, 1613)`, dtype `float32`, no compression.

```bash
python features_mordred.py [SMILES_FILE] [OUT_ZARR]

# Defaults:
#   SMILES_FILE = ./pubchem_olfactory_1million.smiles
#   OUT_ZARR    = ./training_store
```

---

### `pretrain_mordred.py`

Pre-trains a **Directed Message Passing Neural Network (D-MPNN)** to reconstruct Mordred
descriptors from molecular graphs, using a **masked-prediction objective** (15 % of descriptor
dimensions are randomly masked per step; loss is evaluated only on masked positions).

Key hyper-parameters:

| Parameter | Value |
|-----------|-------|
| Hidden size | 2,048 |
| Message-passing depth | 6 |
| Predictor hidden dim | 1,024 |
| Masking ratio | 0.15 |
| Winsorization factor | ±6 σ |
| Batch size | 128 |
| Max epochs | 500 |
| Early-stop patience | 50 |
| Train / val / test split | 70 / 20 / 10 % |

Feature statistics (per-descriptor mean and variance) are computed over the training split using
an online **Welford accumulator** (`torchford.py`) and cached to disk for reproducibility.
The best checkpoint (lowest validation loss) is saved via PyTorch Lightning's `ModelCheckpoint`.

**Output:**
- `output/checkpoints/` — top-2 Lightning checkpoints
- `output/best.pt` — full model object serialised with `torch.save`
- `feature_means_cached_training_store.pt` / `feature_vars_cached_training_store.pt` — cached statistics

```bash
python pretrain_mordred.py
# paths are currently hard-coded in __main__:
#   training_store  = ./training_store
#   output_dir      = ./output
#   smiles_file     = ./pubchem_olfactory_1million.smiles
```

---

### `torchford.py`

A PyTorch port of
[Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm)
for computing per-feature mean and population/sample variance in a single pass over the
training set without loading the full zarr array into memory.
NaN values (arising from missing or invariant Mordred dimensions) are handled by masking.

---

🖥️ ## Running the full pipeline

```bash
# 0. Obtain raw PubChem SMILES (e.g. from ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/)
#    Place as pubchem.smiles in this directory.

# 1. Filter
python filter_pubchem_olfactory.py

# 2. Compute descriptors  (CPU-intensive, ~hours on 8 cores)
python features_mordred.py

# 3. Pre-train  (GPU recommended)
python pretrain_mordred.py
```

Steps 1–2 are CPU-bound and benefit from multiple cores (8 workers by default).  
Step 3 uses PyTorch Lightning and automatically uses available GPUs.

---

## Requirements

See `requirements.txt`.  Install into the `torch2.4_py311` conda environment:

```bash
pip install -r requirements.txt
```
