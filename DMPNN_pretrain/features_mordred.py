"""
features_mordred.py

Reads a pre-filtered SMILES file (pubchem_olfactory_1million.smiles) and
computes Mordred 2D descriptors in streaming batches, writing results to a
zarr array for use as pretraining targets.

Difference from features_self.py:
  - Skips the first filtering pass (length, mixture, RDKit validation).
  - Input file is assumed to be clean single-component SMILES.

Usage (run from the models/ directory):
  python features_mordred.py [SMILES_FILE] [OUT_ZARR]
Defaults:
  SMILES_FILE = ./chemprop_mpnn/pubchem_olfactory_1million.smiles
  OUT_ZARR    = ./chemprop_mpnn/training_store
"""

import sys
import warnings
from multiprocessing import Pool

import numpy as np
import zarr
from mordred import Calculator, descriptors
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


def _to_mol(smi: str):
    """Convert a SMILES string to an RDKit Mol. Defined at module level for pickle compatibility."""
    return MolFromSmiles(smi)


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r") as f:
        for _ in f:
            n += 1
    return n


def iter_batches(smiles_file: str, batch_size: int):
    """Yield lists of SMILES of length batch_size, skipping blank lines."""
    batch = []
    with open(smiles_file, "r") as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            batch.append(smi)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


if __name__ == "__main__":
    # ── paths ─────────────────────────────────────────────────────────────────
    try:
        smiles_file = sys.argv[1] if len(sys.argv) > 1 else \
            "./pubchem_olfactory_1million.smiles"
        out_file    = sys.argv[2] if len(sys.argv) > 2 else \
            "./training_store"
    except Exception:
        print(__doc__)
        sys.exit(1)

    N_WORKERS = 8

    # ── Mordred calculator ────────────────────────────────────────────────────
    blocker    = rdBase.BlockLogs()
    calc       = Calculator(descriptors, ignore_3D=True)
    calc.config(timeout=1)
    n_features = len(calc)
    print(f"Mordred descriptors (2D): {n_features}")

    # ── count molecules (single-pass line count, no full load into memory) ────
    print(f"Counting lines in {smiles_file} ...")
    n_mols = count_lines(smiles_file)
    print(f"Total molecules: {n_mols:,}")

    # ── create zarr array ─────────────────────────────────────────────────────
    dtype          = np.float32
    bytes_per_val  = np.dtype(dtype).itemsize
    chunk_rows     = max((1 * 1024 * 1024) // (n_features * bytes_per_val), 1)
    print(f"Zarr chunk rows: {chunk_rows}  →  shape = ({n_mols}, {n_features})")

    z = zarr.create_array(
        store      = out_file,
        shape      = (n_mols, n_features),
        chunks     = (chunk_rows, n_features),
        dtype      = dtype,
        compressors= None,       # no compression — maximise read/write throughput
        fill_value = np.nan,
    )

    # ── compute descriptors in batches and write to zarr ──────────────────────
    written = 0
    p = Pool(N_WORKERS)

    with tqdm(total=n_mols, desc="Calculating Mordred descriptors", unit="mol") as pbar:
        for batch_smiles in iter_batches(smiles_file, batch_size=chunk_rows):
            # SMILES → Mol (parallel)
            mols = list(
                p.imap(_to_mol, batch_smiles,
                       chunksize=max(len(batch_smiles) // N_WORKERS, 1))
            )
            for mol in mols:
                if mol is not None:
                    mol.SetProp("_Name", "")   # prevent mordred from generating names (expensive)

            # compute Mordred descriptors (nproc workers)
            batch_feats = (
                calc.pandas(mols, quiet=True, nproc=N_WORKERS)
                    .fill_missing()
                    .to_numpy(dtype=np.float32)
            )

            end = written + batch_feats.shape[0]
            z[written:end, :] = batch_feats
            written = end
            pbar.update(batch_feats.shape[0])

    p.close()
    p.join()

    # ── done ──────────────────────────────────────────────────────────────────
    if written != n_mols:
        print(f"[warn] wrote {written} rows but expected {n_mols}; "
              f"tail rows remain filled with NaN.")
    else:
        print(f"\nDone. Wrote {written:,} rows × {n_features} features → {out_file}")
