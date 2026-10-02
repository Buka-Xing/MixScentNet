import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 必须在 import torch 之前
import sys
from argparse import ArgumentParser
from pathlib import Path
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as F1
import tqdm

from backbones.DMPNN import DMPNN_Fingerprint
from backbones.model_gat import MixtureEncoder

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:])
sys.path.append(str(base_dir / "src/"))

MAX_LEN = 58
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# ============================================================
# Data loading
# ============================================================

def load_dream2024_train(dist_csv, smiles_csv):
    """
    Training_Dataset/TrainingData_mixturedist.csv
      columns: Dataset | Mixture 1 | Mixture 2 | Experimental Values
    Training_Dataset/Mixure_Definitions_Training_set_SMILES.csv
      columns: Dataset | Mixture Label | CID (SMILES) | CID.1 | … | CID.57
    Key = (Dataset, Mixture Label)
    """
    df_dist   = pd.read_csv(dist_csv,   dtype=str)
    df_smiles = pd.read_csv(smiles_csv, dtype=str)

    dataset_col = df_smiles.columns[0]
    label_col   = df_smiles.columns[1]
    smiles_cols = df_smiles.columns[2:]   # 58 SMILES columns

    smiles_dict = {}
    for _, row in df_smiles.iterrows():
        key = (str(row[dataset_col]).strip(), str(row[label_col]).strip())
        valid = [
            str(v).strip() for v in row[smiles_cols]
            if pd.notna(v) and str(v).strip() not in {'', 'nan', 'NaN', '0', '0.0', 'None'}
        ]
        if valid:
            smiles_dict[key] = valid

    features_list, labels_list = [], []
    for _, row in df_dist.iterrows():
        dataset = str(row.iloc[0]).strip()
        key1 = (dataset, str(row.iloc[1]).strip())
        key2 = (dataset, str(row.iloc[2]).strip())
        try:
            score = float(row.iloc[3])
        except (ValueError, TypeError):
            continue
        if key1 in smiles_dict and key2 in smiles_dict:
            features_list.append([smiles_dict[key1], smiles_dict[key2]])
            labels_list.append(score)

    print(f"[Train] {len(labels_list)} mixture pairs loaded")
    return np.array(features_list, dtype=object), np.array(labels_list, dtype=np.float64)


def load_dream2024_valid(dist_csv, smiles_csv):
    """
    Leaderboard_Dataset/Leaderboard_set_TrueValue.csv
      columns: Dataset | Mixture_1 | Mixture_2 | Predicted_Experimental_Values
    Leaderboard_Dataset/Mixure_Definitions_Leaderboard_set_SMILES.csv
      columns: Dataset | Mixture Label | smi_0 | smi_1 | … | smi_42
    Key = (Dataset, Mixture Label)
    """
    df_dist   = pd.read_csv(dist_csv,   dtype=str)
    df_smiles = pd.read_csv(smiles_csv, dtype=str)

    dataset_col = df_smiles.columns[0]
    label_col   = df_smiles.columns[1]
    smiles_cols = df_smiles.columns[2:]   # smi_0 … smi_42

    smiles_dict = {}
    for _, row in df_smiles.iterrows():
        key = (str(row[dataset_col]).strip(), str(row[label_col]).strip())
        valid = [
            str(v).strip() for v in row[smiles_cols]
            if pd.notna(v) and str(v).strip() not in {'', 'nan', 'NaN', '0', '0.0', 'None'}
        ]
        if valid:
            smiles_dict[key] = valid

    features_list, labels_list = [], []
    for _, row in df_dist.iterrows():
        dataset = str(row.iloc[0]).strip()
        key1 = (dataset, str(row.iloc[1]).strip())
        key2 = (dataset, str(row.iloc[2]).strip())
        try:
            score = float(row.iloc[3])
        except (ValueError, TypeError):
            continue
        if key1 in smiles_dict and key2 in smiles_dict:
            features_list.append([smiles_dict[key1], smiles_dict[key2]])
            labels_list.append(score)

    print(f"[Valid] {len(labels_list)} mixture pairs loaded")
    return np.array(features_list, dtype=object), np.array(labels_list, dtype=np.float64)


def load_dream2024_test(dist_csv, smiles_csv):
    """
    Test_Dataset/Test_set_mixturedist.csv
      columns: Dataset | Mixture 1 | Mixture 2 | Experimental values
    Test_Dataset/Test_set_Mixure_Definitions_SMILES.csv
      columns: Mixture Label | CID (numeric) | CID.1 (SMILES) | … | CID.9
      col 1 is a PubChem numeric CID — SMILES start at col 2.
    Key = Mixture Label only
    """
    df_dist   = pd.read_csv(dist_csv,   dtype=str)
    df_smiles = pd.read_csv(smiles_csv, dtype=str)

    label_col   = df_smiles.columns[0]
    smiles_cols = df_smiles.columns[2:]   # skip numeric CID at col 1

    smiles_dict = {}
    for _, row in df_smiles.iterrows():
        key = str(row[label_col]).strip()
        valid = [
            str(v).strip() for v in row[smiles_cols]
            if pd.notna(v) and str(v).strip() not in {'', 'nan', 'NaN', '0', '0.0', 'None'}
        ]
        if valid:
            smiles_dict[key] = valid

    col_names = list(df_dist.columns)
    features_list, labels_list, rows_meta = [], [], []
    for _, row in df_dist.iterrows():
        mix1 = str(row.iloc[1]).strip()
        mix2 = str(row.iloc[2]).strip()
        try:
            score = float(row.iloc[3])
        except (ValueError, TypeError):
            continue
        if mix1 in smiles_dict and mix2 in smiles_dict:
            rows_meta.append((str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2])))
            features_list.append([smiles_dict[mix1], smiles_dict[mix2]])
            labels_list.append(score)

    print(f"[Test]  {len(labels_list)} mixture pairs loaded")
    return (
        np.array(features_list, dtype=object),
        np.array(labels_list, dtype=np.float64),
        rows_meta,
        col_names,
    )


# ============================================================
# Feature helpers
# ============================================================

def create_bow_and_indices(features_list, max_len=MAX_LEN):
    unique_smiles = sorted(list(
        set([smi for mix in features_list[:, 0] for smi in mix]).union(
            set([smi for mix in features_list[:, 1] for smi in mix]))
    ))
    smi2idx = {smi: i for i, smi in enumerate(unique_smiles)}

    indices = []
    for mix in features_list:
        idx0 = [smi2idx[smi] for smi in mix[0]]
        idx1 = [smi2idx[smi] for smi in mix[1]]
        if len(idx0) < max_len:
            idx0 += [-1] * (max_len - len(idx0))
        else:
            idx0 = idx0[:max_len]
        if len(idx1) < max_len:
            idx1 += [-1] * (max_len - len(idx1))
        else:
            idx1 = idx1[:max_len]
        indices.append([idx0, idx1])

    return unique_smiles, torch.tensor(indices, dtype=torch.long).permute(0, 2, 1)


def get_dmpnn_mixture_features(bow_smiles, indices, dmpnn_model, device, unk_token=-999):
    unique_feats = dmpnn_model(bow_smiles)
    embed_dim    = unique_feats.shape[-1]

    pad_vec   = torch.full((1, embed_dim), unk_token, dtype=torch.float32, device=device)
    all_feats = torch.cat([unique_feats, pad_vec], dim=0)

    pad_idx = len(unique_feats)
    mapped  = indices.clone().to(device)
    mapped[mapped == -1] = pad_idx

    out = all_feats[mapped]
    return out.permute(0, 1, 3, 2)   # (batch, max_len, embed_dim, 2)


# ============================================================
# Loss
# ============================================================

class PLCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)
        pc = pred   - pred.mean()
        tc = target - target.mean()
        num = (pc * tc).sum()
        den = torch.sqrt((pc ** 2).sum() * (tc ** 2).sum()) + self.eps
        return 1.0 - num / den


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--loss", default="MAELoss",
                        choices=["MAELoss", "MSELoss", "PLCCLoss"], type=str)
    parser.add_argument("--gnn-lr",   default=5e-5, type=float)
    parser.add_argument("--dmpnn-lr", default=5e-5, type=float)
    FLAGS = parser.parse_args()

    SEED                = 2026723
    set_seed(SEED)
    EARLY_STOP_PATIENCE = 30
    num_epochs          = 45
    scheduler_step_size = 15

    MOL_DIM      = 512
    HIDDEN_DIM   = 512
    dmpnn_freeze = True

    train_dist_csv   = "./datasets/DREAM2024-official/Training_Dataset/TrainingData_mixturedist.csv"
    train_smiles_csv = "./datasets/DREAM2024-official/Training_Dataset/Mixure_Definitions_Training_set_SMILES.csv"
    valid_dist_csv   = "./datasets/DREAM2024-official/Leaderboard_Dataset/Leaderboard_set_TrueValue.csv"
    valid_smiles_csv = "./datasets/DREAM2024-official/Leaderboard_Dataset/Mixure_Definitions_Leaderboard_set_SMILES.csv"
    test_dist_csv    = "./datasets/DREAM2024-official/Test_Dataset/Test_set_mixturedist.csv"
    test_smiles_csv  = "./datasets/DREAM2024-official/Test_Dataset/Test_set_Mixure_Definitions_SMILES.csv"

    exp_name = f"DREAM2024official_validd_DMPNNfix_MixtureGATtrain_{FLAGS.loss}"
    fname    = Path(f"results/DREAM2024-official/{exp_name}")
    os.makedirs(fname, exist_ok=True)
    weights_dir = fname / "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # 1. Load data
    # ============================================================
    train_features, train_labels = load_dream2024_train(train_dist_csv, train_smiles_csv)
    valid_features, valid_labels = load_dream2024_valid(valid_dist_csv, valid_smiles_csv)
    test_features,  test_labels, test_meta, test_col_names = load_dream2024_test(test_dist_csv, test_smiles_csv)

    train_bow, train_indices = create_bow_and_indices(train_features, max_len=MAX_LEN)
    valid_bow, valid_indices = create_bow_and_indices(valid_features, max_len=MAX_LEN)
    test_bow,  test_indices  = create_bow_and_indices(test_features,  max_len=MAX_LEN)

    y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
    y_valid = torch.tensor(valid_labels, dtype=torch.float32).to(device)
    y_test  = torch.tensor(test_labels,  dtype=torch.float32).to(device)

    print(f"Training pairs   : {len(train_labels)}")
    print(f"Validation pairs : {len(valid_labels)}")
    print(f"Test pairs       : {len(test_labels)}")

    # ============================================================
    # 2. Initialize models
    # ============================================================
    print("Loading DMPNN Fingerprint Model...")
    DMPNN = DMPNN_Fingerprint(MOL_DIM, device=device)
    if dmpnn_freeze:
        for param in DMPNN.model.parameters():
            param.requires_grad = False

    # torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    mixture_encoder = MixtureEncoder(
        mol_dim=MOL_DIM, hidden_dim=HIDDEN_DIM, output_type="similarity"
    ).to(device)

    # ============================================================
    # 3. Loss, optimizer, scheduler
    # ============================================================
    if   FLAGS.loss == "MAELoss":  loss_fn = nn.L1Loss()
    elif FLAGS.loss == "MSELoss":  loss_fn = nn.MSELoss()
    elif FLAGS.loss == "PLCCLoss": loss_fn = PLCCLoss()
    else: raise ValueError(f"Unknown loss: {FLAGS.loss}")

    optimizer = torch.optim.Adam([
        {"params": mixture_encoder.parameters(), "lr": FLAGS.gnn_lr},
        {"params": DMPNN.model.parameters(),     "lr": FLAGS.dmpnn_lr},
        {"params": DMPNN.DimReduce.parameters(), "lr": FLAGS.dmpnn_lr},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=0.5
    )

    # ============================================================
    # 4. Training loop  (model selection on validation set)
    # ============================================================
    log = {k: [] for k in ["epoch", "train_loss", "valid_mse", "valid_pearson"]}

    best_valid_pearson = -float("inf")
    epochs_without_improvement = 0
    best_epoch = -1

    pbar = tqdm.tqdm(range(num_epochs))

    for epoch in pbar:
        # ---- Train ----
        mixture_encoder.train()
        optimizer.zero_grad()

        train_tensor = get_dmpnn_mixture_features(train_bow, train_indices, DMPNN, device)
        y_pred_train = mixture_encoder(train_tensor)
        loss         = loss_fn(y_pred_train, y_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.detach().cpu().item()

        # ---- Validate (Leaderboard) ----
        mixture_encoder.eval()
        with torch.no_grad():
            valid_tensor  = get_dmpnn_mixture_features(valid_bow, valid_indices, DMPNN, device)
            y_pred_valid  = mixture_encoder(valid_tensor)
            valid_mse    = F.mse_loss(y_pred_valid, y_valid).item()
            valid_pearson = F1.pearson_corrcoef(y_pred_valid, y_valid).item()

        log["epoch"].append(epoch)
        log["train_loss"].append(train_loss)
        log["valid_mse"].append(valid_mse)
        log["valid_pearson"].append(valid_pearson)

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(
            f"LR: {current_lr:.2e} | Train: {train_loss:.4f} | "
            f"Val MSE: {valid_mse:.4f} | Val Pearson: {valid_pearson:.4f}"
        )
        scheduler.step()

        # Model selection: save when validation Pearson improves
        if valid_pearson > best_valid_pearson:
            best_valid_pearson = valid_pearson
            best_epoch         = epoch
            epochs_without_improvement = 0

            torch.save({
                "epoch":            epoch,
                "best_valid_pearson": best_valid_pearson,
                "best_valid_mse":  valid_mse,
                "mixture_encoder":  mixture_encoder.state_dict(),
                "DMPNN_dim_reduce": DMPNN.DimReduce.state_dict(),
            }, weights_dir / "best_valid_pearson.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"No validation Pearson improvement for {EARLY_STOP_PATIENCE} epochs."
            )
            break

    # ============================================================
    # 5. Save training log
    # ============================================================
    pd.DataFrame(log).to_csv(fname / "training_log.csv", index=False)

    # ============================================================
    # 6. Load best model and evaluate on Test set
    # ============================================================
    print(f"\nLoading best model from epoch {best_epoch} (val Pearson = {best_valid_pearson:.4f})...")
    ckpt = torch.load(weights_dir / "best_valid_pearson.pt", map_location=device)
    mixture_encoder.load_state_dict(ckpt["mixture_encoder"])
    DMPNN.DimReduce.load_state_dict(ckpt["DMPNN_dim_reduce"])

    mixture_encoder.eval()
    with torch.no_grad():
        valid_tensor    = get_dmpnn_mixture_features(valid_bow, valid_indices, DMPNN, device)
        y_pred_valid    = mixture_encoder(valid_tensor).cpu().numpy().flatten()

        test_tensor     = get_dmpnn_mixture_features(test_bow, test_indices, DMPNN, device)
        y_pred_test_raw = mixture_encoder(test_tensor).cpu().numpy().flatten()


    valid_targets = y_valid.cpu().numpy().flatten()
    test_targets  = y_test.cpu().numpy().flatten()

    # ── Raw metrics ──────────────────────────────────────────────
    test_mse_raw     = float(np.mean((y_pred_test_raw - test_targets) ** 2))
    test_rmse_raw    = float(np.sqrt(test_mse_raw))
    test_pearson_raw = float(stats.pearsonr(y_pred_test_raw, test_targets)[0])
    test_krcc_raw    = float(stats.kendalltau(y_pred_test_raw, test_targets)[0])

    # ── Linear calibration fitted on the validation set ──────────
    # Finds a, b that minimize E[(a*pred_valid + b - target_valid)^2].
    # Closed-form: a = Cov(pred,target)/Var(pred),  b = mean(target) - a*mean(pred)
    # Equivalently: scipy.stats.linregress(pred, target).
    # Pearson is invariant to this transform; MSE drops to Var(y)*(1 - ρ²).
    calib_slope, calib_intercept, _, _, _ = stats.linregress(y_pred_valid, valid_targets)
    print(f"Calibration (fitted on Leaderboard): a={calib_slope:.4f}, b={calib_intercept:.4f}")

    y_pred_test_calib = calib_slope * y_pred_test_raw + calib_intercept

    # ── Test-set predictions CSV (calibrated, same format as Test_set_mixturedist.csv) ──
    test_pred_df = pd.DataFrame(test_meta, columns=test_col_names[:3])
    test_pred_df[test_col_names[3]] = y_pred_test_calib
    test_pred_df.to_csv(fname / "test_predictions.csv", index=False)
    test_mse_calib    = float(np.mean((y_pred_test_calib - test_targets) ** 2))
    test_rmse_calib   = float(np.sqrt(test_mse_calib))
    test_pearson_calib = float(stats.pearsonr(y_pred_test_calib, test_targets)[0])
    test_krcc_calib   = float(stats.kendalltau(y_pred_test_calib, test_targets)[0])

    # ============================================================
    # 7. Summary
    # ============================================================
    summary = (
        f"Experiment : {exp_name}\n"
        f"Stopped at epoch : {epoch}\n"
        f"Max mixture length (max_len) : {MAX_LEN}\n"
        f"\n"
        f"Best model selected by Validation (Leaderboard) Pearson:\n"
        f"  Epoch        : {best_epoch}\n"
        f"  Val Pearson  : {best_valid_pearson:.4f}\n"
        f"  Val MSE      : {ckpt['best_valid_mse']:.4f}\n"
        f"\n"
        f"Linear calibration (fitted on Leaderboard set):\n"
        f"  a (slope)    : {calib_slope:.4f}\n"
        f"  b (intercept): {calib_intercept:.4f}\n"
        f"\n"
        f"Final Test Set Performance — Raw:\n"
        f"  Pearson : {test_pearson_raw:.4f}\n"
        f"  MSE     : {test_mse_raw:.4f}\n"
        f"  RMSE    : {test_rmse_raw:.4f}\n"
        f"  KRCC    : {test_krcc_raw:.4f}\n"
        f"\n"
        f"Final Test Set Performance — After Linear Calibration:\n"
        f"  Pearson : {test_pearson_calib:.4f}\n"
        f"  MSE     : {test_mse_calib:.4f}\n"
        f"  RMSE    : {test_rmse_calib:.4f}\n"
        f"  KRCC    : {test_krcc_calib:.4f}\n"
    )

    print("\n" + "=" * 50)
    print(summary)

    with open(fname / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Results saved to: {fname}/")
