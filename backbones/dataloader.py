import sys
sys.path.append("..")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

current_dir = Path(__file__).parent
DATASET_DIR = current_dir / "./datasets"

print(DATASET_DIR)

def get_mixture_cv_splits(labels_csv_path, smiles_csv_path, SEED=42):
    df_labels = pd.read_csv(labels_csv_path)
    df_smiles = pd.read_csv(smiles_csv_path, dtype=str)

    id_col_labels = df_labels.columns[0]
    id_col_smiles = df_smiles.columns[0]
    df_smiles.rename(columns={id_col_smiles: id_col_labels}, inplace=True)
    df_merged = pd.merge(df_labels, df_smiles, on=id_col_labels, how='inner')

    # 1. Collect the global set of all molecules
    smiles_cols = df_smiles.columns[1:11]
    all_unique_smiles = set()
    seed = SEED
    # Preprocessing: build the structured Features
    structured_features = []
    for _, row in df_merged[smiles_cols].iterrows():
        valid_smiles = [
            str(val).strip() for val in row
            if pd.notna(val) and str(val).strip() not in ['', 'nan', 'NaN', '0', '0.0']
        ]
        structured_features.append(np.array(valid_smiles))
        all_unique_smiles.update(valid_smiles)

    features_array = np.array(structured_features, dtype=object)
    labels_matrix = df_merged[df_labels.columns[1:52]].values.astype(np.float64)

    print(f"Found {len(all_unique_smiles)} unique molecules.")

    # 2. 5-fold split logic
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features_array)):
        X_train = features_array[train_idx]
        y_train = labels_matrix[train_idx]

        X_test = features_array[test_idx]
        y_test = labels_matrix[test_idx]

        # 3. Core check: detect molecules in the test set that never appear in the training set
        test_smiles = set()
        for mixture in X_test:
            test_smiles.update(mixture)

        train_smiles = set()
        for mixture in X_train:
            train_smiles.update(mixture)

        missing_smiles = test_smiles - train_smiles

        # If the test set contains molecules unseen in training, fix the split
        if missing_smiles:
            # Simple strategy: move the test samples that contain these "stray" molecules back to the training set
            # Find indices of test samples that contain any unseen molecule
            indices_to_move = [
                i for i, idx in enumerate(test_idx)
                if any(s in missing_smiles for s in X_test[i])
            ]

            # Re-split: pull these samples back into the training set
            move_idx = test_idx[indices_to_move]
            train_idx = np.concatenate([train_idx, move_idx])
            test_idx = np.delete(test_idx, indices_to_move)

            # Update X, y
            X_train, X_test = features_array[train_idx], features_array[test_idx]
            y_train, y_test = labels_matrix[train_idx], labels_matrix[test_idx]

        results.append((f"cv{fold_idx}", (X_train, y_train), (X_test, y_test)))
    print(f"Dataset split complete; all molecules are guaranteed to appear in the training set.")
    return results

def get_mixture_similarity_cv_splits(combined_csv_path, smiles_csv_path, SEED=42):
    # 1. Read data and build smiles_dict (same logic as the original)

    df_combined = pd.read_csv(combined_csv_path, dtype=str)
    df_smiles = pd.read_csv(smiles_csv_path, dtype=str)

    smiles_dict = {}
    dataset_col_smi, id_col_smi = df_smiles.columns[0], df_smiles.columns[1]
    smiles_cols = df_smiles.columns[5:48]

    for _, row in df_smiles.iterrows():
        key = (str(row[dataset_col_smi]).strip(), str(row[id_col_smi]).strip())
        valid_smiles = [str(val).strip() for val in row[smiles_cols]
                        if pd.notna(val) and str(val).strip() not in ['', 'nan', 'NaN', '0', '0.0', 'None']]
        smiles_dict[key] = valid_smiles

    # 2. Extract features and labels
    features_list, labels_list = [], []
    for _, row in df_combined.iterrows():
        key1 = (str(row.iloc[0]).strip(), str(row.iloc[1]).strip())
        key2 = (str(row.iloc[0]).strip(), str(row.iloc[2]).strip())
        try:
            score = float(row.iloc[3])
            if key1 in smiles_dict and key2 in smiles_dict:
                features_list.append([smiles_dict[key1], smiles_dict[key2]])
                labels_list.append(score)
        except ValueError:
            continue

    features_array = np.array(features_list, dtype=object)
    labels_matrix = np.array(labels_list, dtype=np.float64)

    # 3. K-Fold split with molecule-coverage check
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features_array)):
        X_train, X_test = features_array[train_idx], features_array[test_idx]
        y_train, y_test = labels_matrix[train_idx], labels_matrix[test_idx]

        # Helper: collect the set of all unique molecules from a sample set
        def get_all_smiles(X_set):
            all_s = set()
            for pair in X_set:
                all_s.update(pair[0])
                all_s.update(pair[1])
            return all_s

        # Check whether the test set contains molecules unseen in the training set
        train_smiles = get_all_smiles(X_train)
        test_smiles = get_all_smiles(X_test)
        missing_smiles = test_smiles - train_smiles

        if missing_smiles:
            # Find indices of test pairs that contain at least one unseen molecule
            move_indices = []
            for i, pair in enumerate(X_test):
                # If either molecule in the pair is unseen in training
                if any(s in missing_smiles for s in pair[0]) or any(s in missing_smiles for s in pair[1]):
                    move_indices.append(i)

            # Move them to the training set
            train_idx = np.concatenate([train_idx, test_idx[move_indices]])
            test_idx = np.delete(test_idx, move_indices)

            # Re-split
            X_train, X_test = features_array[train_idx], features_array[test_idx]
            y_train, y_test = labels_matrix[train_idx], labels_matrix[test_idx]

        results.append((f"cv{fold_idx}", (X_train, y_train), (X_test, y_test)))

    print(f"Dataset split complete; all molecules are guaranteed to appear in the training set.")
    return results


# ======================================================================
# Generalization test: split by molecule, test set contains molecules
# that never appeared in the training set
# ======================================================================
# Core helper: adaptively select unseen molecules so that the
# test-set sample ratio is approximately target_ratio
# ======================================================================

def _adaptive_unseen_split(
    sample_smiles_sets,      # list[set]   set of molecules contained in each sample
    unique_smiles_arr,       # np.array    globally unique molecules (sorted)
    seed,                    # int         random seed
    fold_idx,                # int         current fold index (used to offset the seed)
    n_samples,               # int         total number of samples
    target_ratio=0.20,       # float       desired test-set sample ratio
):
    """
    Iterate through the shuffled molecule list, marking molecules as unseen one by one
    while tracking the resulting test-set sample count. Stop once the test-set ratio
    first reaches target_ratio.

    Returns
    -------
    unseen_smiles : set          the chosen unseen molecules
    train_indices : np.ndarray   training sample indices
    test_indices  : np.ndarray   test sample indices
    """
    rng  = np.random.RandomState(seed + fold_idx)
    shuffled_mols = unique_smiles_arr.copy()
    rng.shuffle(shuffled_mols)

    target_test_count = int(np.ceil(n_samples * target_ratio))

    unseen_smiles = set()
    test_set = set()                      # sample indices currently assigned to the test set

    # Build an inverted index: molecule -> list of sample indices that contain it
    mol_to_samples = {}
    for idx, smi_set in enumerate(sample_smiles_sets):
        for s in smi_set:
            mol_to_samples.setdefault(s, []).append(idx)

    for mol in shuffled_mols:
        # Mark this molecule as unseen and add every sample containing it to the test set
        unseen_smiles.add(mol)
        affected = mol_to_samples.get(mol, [])
        test_set.update(affected)

        if len(test_set) >= target_test_count:
            break

    test_indices  = np.array(sorted(test_set))
    train_indices = np.array(sorted(set(range(n_samples)) - test_set))
    return unseen_smiles, train_indices, test_indices


def get_mixture_cv_splits_unseen_mol(labels_csv_path, smiles_csv_path, SEED=42):
    """
    "Unseen molecule" generalization split for the Label task.

    Split logic
    -----------
    1. Collect the globally unique molecules, shuffle them, and mark them as unseen one by one.
    2. Each time a molecule is marked, move every sample containing it into the test set.
    3. Stop when the test set reaches roughly 20% of all samples.
    4. Repeat for 5 folds, using a different shuffle order each fold.

    Final guarantee: ~80% train, ~20% test, with molecules in the test set that the
    training set has never seen.

    Returns
    -------
    list of (fold_id, (X_train, y_train), (X_test, y_test))
    """
    seed = SEED
    df_labels = pd.read_csv(labels_csv_path)
    df_smiles = pd.read_csv(smiles_csv_path, dtype=str)

    id_col_labels = df_labels.columns[0]
    id_col_smiles = df_smiles.columns[0]
    df_smiles.rename(columns={id_col_smiles: id_col_labels}, inplace=True)
    df_merged = pd.merge(df_labels, df_smiles, on=id_col_labels, how='inner')

    smiles_cols = df_smiles.columns[1:11]

    structured_features = []
    sample_smiles_sets  = []
    for _, row in df_merged[smiles_cols].iterrows():
        valid_smiles = [
            str(val).strip() for val in row
            if pd.notna(val) and str(val).strip() not in ['', 'nan', 'NaN', '0', '0.0']
        ]
        structured_features.append(np.array(valid_smiles))
        sample_smiles_sets.append(set(valid_smiles))

    features_array = np.array(structured_features, dtype=object)
    labels_matrix  = df_merged[df_labels.columns[1:52]].values.astype(np.float64)

    all_unique_smiles = sorted(set(s for sset in sample_smiles_sets for s in sset))
    unique_smiles_arr = np.array(all_unique_smiles)
    n_samples = len(features_array)
    print(f"Found {len(unique_smiles_arr)} unique molecules across {n_samples} samples.")

    results = []
    for fold_idx in range(5):
        unseen_smiles, train_indices, test_indices = _adaptive_unseen_split(
            sample_smiles_sets, unique_smiles_arr,
            seed=42, fold_idx=fold_idx, n_samples=n_samples,
        )

        X_train, y_train = features_array[train_indices], labels_matrix[train_indices]
        X_test,  y_test  = features_array[test_indices],  labels_matrix[test_indices]

        print(f"  Fold cv{fold_idx}: unseen mols={len(unseen_smiles)}, "
              f"train={len(train_indices)} ({len(train_indices)/n_samples:.1%}), "
              f"test={len(test_indices)} ({len(test_indices)/n_samples:.1%})")

        results.append((f"cv{fold_idx}", (X_train, y_train), (X_test, y_test)))

    print(f"Unseen-molecule split complete; every test sample contains molecules absent from the training set.")
    return results

def get_mixture_similarity_cv_splits_unseen_mol(combined_csv_path, smiles_csv_path, SEED=42):
    """
    "Unseen molecule" generalization split for the Similarity task.

    Split logic
    -----------
    1. Collect the globally unique molecules, shuffle them, and mark them as unseen one by one.
    2. Each time a molecule is marked, move every mixture-pair sample containing it into the test set.
    3. Stop when the test set reaches roughly 20% of all samples.
    4. Repeat for 5 folds, using a different shuffle order each fold.

    Returns
    -------
    list of (fold_id, (X_train, y_train), (X_test, y_test))
    """
    # 1. Read data and build smiles_dict
    df_combined = pd.read_csv(combined_csv_path, dtype=str)
    df_smiles   = pd.read_csv(smiles_csv_path, dtype=str)

    smiles_dict = {}
    dataset_col_smi, id_col_smi = df_smiles.columns[0], df_smiles.columns[1]
    smiles_cols = df_smiles.columns[5:48]

    for _, row in df_smiles.iterrows():
        key = (str(row[dataset_col_smi]).strip(), str(row[id_col_smi]).strip())
        valid_smiles = [str(val).strip() for val in row[smiles_cols]
                        if pd.notna(val) and str(val).strip() not in ['', 'nan', 'NaN', '0', '0.0', 'None']]
        smiles_dict[key] = valid_smiles

    # 2. Extract features and labels
    features_list, labels_list, sample_smiles_sets = [], [], []
    for _, row in df_combined.iterrows():
        key1 = (str(row.iloc[0]).strip(), str(row.iloc[1]).strip())
        key2 = (str(row.iloc[0]).strip(), str(row.iloc[2]).strip())
        try:
            score = float(row.iloc[3])
            if key1 in smiles_dict and key2 in smiles_dict:
                features_list.append([smiles_dict[key1], smiles_dict[key2]])
                labels_list.append(score)
                sample_smiles_sets.append(set(smiles_dict[key1]) | set(smiles_dict[key2]))
        except ValueError:
            continue

    features_array = np.array(features_list, dtype=object)
    labels_matrix  = np.array(labels_list, dtype=np.float64)

    # 3. Collect the globally unique molecules
    all_unique_smiles = sorted(set(s for sset in sample_smiles_sets for s in sset))
    unique_smiles_arr = np.array(all_unique_smiles)
    n_samples = len(features_array)
    print(f"Found {len(unique_smiles_arr)} unique molecules across {n_samples} samples.")

    # 4. Adaptive split
    results = []
    for fold_idx in range(5):
        unseen_smiles, train_indices, test_indices = _adaptive_unseen_split(
            sample_smiles_sets, unique_smiles_arr,
            seed=SEED, fold_idx=fold_idx, n_samples=n_samples,
        )

        X_train, y_train = features_array[train_indices], labels_matrix[train_indices]
        X_test,  y_test  = features_array[test_indices],  labels_matrix[test_indices]

        print(f"  Fold cv{fold_idx}: unseen mols={len(unseen_smiles)}, "
              f"train={len(train_indices)} ({len(train_indices)/n_samples:.1%}), "
              f"test={len(test_indices)} ({len(test_indices)/n_samples:.1%})")

        results.append((f"cv{fold_idx}", (X_train, y_train), (X_test, y_test)))

    print(f"Unseen-molecule split complete; every test sample contains molecules absent from the training set.")
    return results



