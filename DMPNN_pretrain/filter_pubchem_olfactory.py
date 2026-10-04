"""
filter_pubchem_olfactory.py

Filters pubchem.smiles → pubchem_olfactory.smiles with 6 rules:
  1. Remove SMILES with > 150 characters (string-level)
  2. Remove SMILES containing '.' (mixtures, string-level)
  3. Remove SMILES containing inorganic atoms: He Na Mg Al K Ca Ti V Cr Fe Co Cu Zn Bi
  4. Remove if RDKit MolFromSmiles() returns None
  5. Remove if RDKit RemoveHs(mol, updateExplicitCount=True) raises any error
  6. Keep only molecules with molecular weight in [30, 300] Da

Target output: ~1,168,920 molecules.
"""

import re
from multiprocessing import Pool
from pathlib import Path

from rdkit import rdBase
from rdkit.Chem import Descriptors, MolFromSmiles, RemoveHs
from tqdm import tqdm

# ── configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = Path("pubchem.smiles")
OUTPUT_FILE = Path("pubchem_olfactory_1million.smiles")
N_WORKERS   = 8
CHUNK_SIZE  = 50_000   # lines accumulated before dispatching to workers
MW_MIN, MW_MAX = 30.0, 300.0

# Rule 3: inorganic atoms always appear as bracket atoms in SMILES
# Pattern matches the opening of any bracket containing one of the target symbols
INORGANIC_RE = re.compile(
    r"\[(?:He|Na|Mg|Al|Ca|Ti|Cr|Fe|Co|Cu|Zn|Bi"
    r"|K(?![a-z])"   # K but not Kr, etc.
    r"|V(?![a-z]))"  # V but not Vn, etc.
)


# ── per-SMILES filters ─────────────────────────────────────────────────────────

def _string_ok(smi: str) -> bool:
    """Rules 1-3: pure string operations, no RDKit."""
    if len(smi) > 150:          # rule 1
        return False
    if "." in smi:              # rule 2
        return False
    if INORGANIC_RE.search(smi):  # rule 3
        return False
    return True


def _rdkit_ok(smi: str) -> str | None:
    """Rules 4-6: RDKit validation + MW filter. Returns smi or None."""
    mol = MolFromSmiles(smi)        # rule 4
    if mol is None:
        return None
    try:
        mol = RemoveHs(mol, updateExplicitCount=True)   # rule 5
    except Exception:
        return None
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN <= mw <= MW_MAX):   # rule 6
        return None
    return smi


def _process_batch(smiles_batch: list[str]) -> list[str]:
    """Worker function: applies rules 4-6 to a list of pre-string-filtered SMILES."""
    _blocker = rdBase.BlockLogs()   # suppress RDKit stderr in worker
    return [s for s in (_rdkit_ok(smi) for smi in smiles_batch) if s is not None]


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    n_read = n_string = n_final = 0
    pending: list[str] = []

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout, Pool(N_WORKERS) as pool:
        pbar = tqdm(fin, desc="Filtering", unit=" lines", mininterval=5.0)

        def _flush(batch: list[str]):
            nonlocal n_string, n_final
            n_string += len(batch)
            # split evenly across workers
            sub = max(1, len(batch) // N_WORKERS)
            sub_batches = [batch[i : i + sub] for i in range(0, len(batch), sub)]
            for kept in pool.map(_process_batch, sub_batches):
                for smi in kept:
                    fout.write(smi + "\n")
                n_final += sum(len(k) for k in [kept])

        for line in pbar:
            smi = line.strip()
            if not smi:
                continue
            n_read += 1
            if _string_ok(smi):
                pending.append(smi)
                if len(pending) >= CHUNK_SIZE:
                    _flush(pending)
                    pending = []
                    pbar.set_postfix(
                        after_str=f"{n_string:,}",
                        kept=f"{n_final:,}",
                    )
                if n_final > 1168920:
                    break

        if pending:
            _flush(pending)

    print(f"\n{'─'*45}")
    print(f"  Total read           : {n_read:>12,}")
    print(f"  After string filters : {n_string:>12,}")
    print(f"  After RDKit + MW     : {n_final:>12,}")
    print(f"  Output file          : {OUTPUT_FILE}")
    print(f"{'─'*45}")


if __name__ == "__main__":
    main()
