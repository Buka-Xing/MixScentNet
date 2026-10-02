import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
from argparse import ArgumentParser
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as F1
import tqdm
from scipy.stats import pearsonr as scipy_pearsonr
from scipy.spatial.distance import cosine as scipy_cosine_dist

from backbones.DMPNN import DMPNN_Fingerprint
from backbones.model_gat import MixtureEncoder

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:])
sys.path.append(str(base_dir / "src/"))

MAX_LEN = 10

# 51 label columns — order matches TASK2_Leaderboard_ActualValue.csv
LABEL_COLS = [
    'Green', 'Cucumber', 'Herbal', 'Mint', 'Woody', 'Pine', 'Floral', 'Powdery',
    'Fruity', 'Citrus', 'Tropical', 'Berry', 'Peach', 'Sweet', 'Caramellic', 'Vanilla',
    'BrownSpice', 'Smoky', 'Burnt', 'Roasted', 'Grainy', 'Meaty', 'Nutty', 'Fatty',
    'Coconut', 'Waxy', 'Dairy', 'Buttery', 'Cheesy', 'Sour', 'Fermented', 'Sulfurous',
    'Garlic.Onion', 'Earthy', 'Mushroom', 'Musty', 'Ammonia', 'Fishy', 'Fecal',
    'Rotten.Decay', 'Rubber', 'Phenolic', 'Animal', 'Medicinal', 'Cooling', 'Sharp',
    'Chlorine', 'Alcoholic', 'Plastic', 'Ozone', 'Metallic',
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================
# Data loading
# ============================================================

def load_smiles_dict(smiles_csv):
    df = pd.read_csv(smiles_csv, dtype=str)
    smi_cols = [c for c in df.columns if c.startswith('smi_')]
    smiles_dict = {}
    for _, row in df.iterrows():
        key = str(row['Mixture_Label']).strip()
        vals = [
            str(v).strip() for v in row[smi_cols]
            if pd.notna(v) and str(v).strip() not in {'', 'nan', 'NaN'}
        ]
        if vals:
            smiles_dict[key] = vals
    return smiles_dict


def load_data(labels_csv, smiles_dict, label_cols, tag='data'):
    df = pd.read_csv(labels_csv)
    features_list, labels_list, stimuli = [], [], []
    for _, row in df.iterrows():
        stimulus = str(row['stimulus']).strip()
        if stimulus not in smiles_dict:
            print(f"  [skip] {stimulus} not in SMILES dict")
            continue
        features_list.append(smiles_dict[stimulus])
        labels_list.append([float(row[c]) for c in label_cols])
        stimuli.append(stimulus)
    print(f"[{tag}] {len(stimuli)} samples loaded")
    return features_list, np.array(labels_list, dtype=np.float32), stimuli


# ============================================================
# Feature helpers
# ============================================================

def create_bow_and_indices(features_list, max_len=MAX_LEN):
    unique_smiles = sorted(set(smi for mix in features_list for smi in mix))
    smi2idx = {smi: i for i, smi in enumerate(unique_smiles)}
    indices = []
    for mix in features_list:
        idx = [smi2idx[s] for s in mix]
        if len(idx) < max_len:
            idx += [-1] * (max_len - len(idx))
        else:
            idx = idx[:max_len]
        indices.append(idx)
    return unique_smiles, torch.tensor(indices, dtype=torch.long)


def get_dmpnn_mixture_features(bow_smiles, indices, dmpnn_model, device, unk_token=-999):
    unique_feats = dmpnn_model(bow_smiles)
    embed_dim = unique_feats.shape[-1]
    pad_vec = torch.full((1, embed_dim), unk_token, dtype=torch.float32, device=device)
    all_feats = torch.cat([unique_feats, pad_vec], dim=0)
    pad_idx = len(unique_feats)
    mapped = indices.clone().to(device)
    mapped[mapped == -1] = pad_idx
    out = all_feats[mapped]        # (B, max_len, embed_dim)
    return out.unsqueeze(-1)       # (B, max_len, embed_dim, 1)


# ============================================================
# Loss functions
# ============================================================

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.4):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        loss = torch.where(
            err.abs() < self.delta,
            0.5 * err ** 2,
            self.delta * (err.abs() - 0.5 * self.delta),
        )
        return loss.mean()


class MeanPLCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c   = pred   - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        num = (pred_c * target_c).sum(dim=1)
        den = torch.sqrt((pred_c**2).sum(dim=1) * (target_c**2).sum(dim=1)) + self.eps
        return (1.0 - num / den).mean()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--loss", default="MSELoss",
                        choices=["HuberLoss", "MSELoss", "MAELoss", "MeanPLCCLoss"], type=str)
    parser.add_argument("--gnn-lr",   default=1e-4, type=float)
    parser.add_argument("--dmpnn-lr", default=1e-4, type=float)
    FLAGS = parser.parse_args()

    SEED                = 2026723
    set_seed(SEED)
    EARLY_STOP_PATIENCE = 1000
    num_epochs          = 5000
    scheduler_step_size = 1500

    MOL_DIM    = 512
    HIDDEN_DIM = 512
    dmpnn_freeze = True

    train_csv  = "./datasets/DREAM2025-official/TASK2_Train_mixture_Dataset_noLeaderboard.csv"
    lb_csv     = "./datasets/DREAM2025-official/TASK2_Leaderboard_ActualValue.csv"
    smiles_csv = "./datasets/DREAM2025-official/Mixture_SMILES_Converted.csv"

    exp_name    = f"DREAM2025official_Label_DMPNNfix_MixtureGATtrain_{FLAGS.loss}_SEED{SEED}_COSINE3"
    fname       = Path(f"results/DREAM2025-official/{exp_name}")
    os.makedirs(fname, exist_ok=True)
    weights_dir = fname / "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # 1. Load data
    # ============================================================
    smiles_dict = load_smiles_dict(smiles_csv)

    train_features, train_labels, train_stimuli = load_data(
        train_csv, smiles_dict, LABEL_COLS, tag="train"
    )
    lb_features, lb_labels, lb_stimuli = load_data(
        lb_csv, smiles_dict, LABEL_COLS, tag="leaderboard"
    )

    train_bow, train_indices = create_bow_and_indices(train_features, max_len=MAX_LEN)
    lb_bow,    lb_indices    = create_bow_and_indices(lb_features,    max_len=MAX_LEN)

    y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
    y_lb    = torch.tensor(lb_labels,    dtype=torch.float32).to(device)

    print(f"Training samples    : {len(train_stimuli)}")
    print(f"Leaderboard samples : {len(lb_stimuli)}")

    # ============================================================
    # 2. Initialize models
    # ============================================================
    print("Loading DMPNN Fingerprint Model...")
    DMPNN = DMPNN_Fingerprint(MOL_DIM, device=device)
    if dmpnn_freeze:
        for param in DMPNN.model.parameters():
            param.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    mixture_encoder = MixtureEncoder(
        mol_dim=MOL_DIM, hidden_dim=HIDDEN_DIM, output_type="label"
    ).to(device)

    # ============================================================
    # 3. Loss, optimizer, scheduler
    # ============================================================
    if   FLAGS.loss == "HuberLoss":    loss_fn = HuberLoss(delta=0.4)
    elif FLAGS.loss == "MAELoss":      loss_fn = nn.L1Loss()
    elif FLAGS.loss == "MSELoss":      loss_fn = nn.MSELoss()
    elif FLAGS.loss == "MeanPLCCLoss": loss_fn = MeanPLCCLoss()
    else: raise ValueError(f"Unknown loss: {FLAGS.loss}")

    optimizer = torch.optim.Adam([
        {"params": mixture_encoder.parameters(), "lr": FLAGS.gnn_lr},
        {"params": DMPNN.model.parameters(),     "lr": FLAGS.dmpnn_lr},
        {"params": DMPNN.DimReduce.parameters(), "lr": FLAGS.dmpnn_lr},
    ])
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=scheduler_step_size, gamma=0.5
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs , eta_min=1e-6)

    # ============================================================
    # 4. Training loop  (model selection on Leaderboard PLCC(sample))
    # ============================================================
    log = {k: [] for k in ["epoch", "train_loss", "lb_plcc_sample", "lb_cos_dist"]}

    best_plcc_sample       = -float("inf")
    best_cos_dist          = None
    best_epoch             = -1
    epochs_without_improvement = 0

    pbar = tqdm.tqdm(range(num_epochs), ncols=80)

    for epoch in pbar:
        # ---- Train ----
        mixture_encoder.train()
        optimizer.zero_grad()

        train_tensor = get_dmpnn_mixture_features(train_bow, train_indices, DMPNN, device)
        y_pred_train = mixture_encoder(train_tensor)
        loss = loss_fn(y_pred_train, y_train)
        loss.backward()
        optimizer.step()
        train_loss = loss.detach().cpu().item()

        # ---- Evaluate on Leaderboard ----
        mixture_encoder.eval()
        with torch.no_grad():
            lb_tensor   = get_dmpnn_mixture_features(lb_bow, lb_indices, DMPNN, device)
            y_pred_lb   = mixture_encoder(lb_tensor)           # (B, 51)

            # per-sample PLCC: for each sample, correlate its 51 predicted vs true labels
            # permute to (51, B) so pearson_corrcoef treats each column as one sample
            plcc_sample = F1.pearson_corrcoef(
                y_pred_lb.permute(1, 0), y_lb.permute(1, 0)
            ).mean().item()

            cos_dist = 1.0 - F.cosine_similarity(y_pred_lb, y_lb, dim=1).mean().item()

        log["epoch"].append(epoch)
        log["train_loss"].append(train_loss)
        log["lb_plcc_sample"].append(plcc_sample)
        log["lb_cos_dist"].append(cos_dist)

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(
            f"LR: {current_lr:.2e} | Train: {train_loss:.4f} | "
            f"PLCC(sample): {plcc_sample:.4f} | cos-dist: {cos_dist:.4f}"
        )
        scheduler.step()

        if plcc_sample > best_plcc_sample:
            best_plcc_sample = plcc_sample
            best_cos_dist    = cos_dist
            best_epoch       = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch":            epoch,
                "best_plcc_sample": best_plcc_sample,
                "best_cos_dist":    best_cos_dist,
                "mixture_encoder":  mixture_encoder.state_dict(),
                "DMPNN_dim_reduce": DMPNN.DimReduce.state_dict(),
            }, weights_dir / "best_lb_plcc.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"No leaderboard PLCC improvement for {EARLY_STOP_PATIENCE} epochs."
            )
            break

    # ============================================================
    # 5. Save training log
    # ============================================================
    pd.DataFrame(log).to_csv(fname / "training_log.csv", index=False)

    # ============================================================
    # 6. Load best model → predict on Leaderboard → save CSV
    # ============================================================
    print(f"\nLoading best model from epoch {best_epoch} (PLCC = {best_plcc_sample:.4f})...")
    ckpt = torch.load(weights_dir / "best_lb_plcc.pt", map_location=device)
    mixture_encoder.load_state_dict(ckpt["mixture_encoder"])
    DMPNN.DimReduce.load_state_dict(ckpt["DMPNN_dim_reduce"])

    mixture_encoder.eval()
    with torch.no_grad():
        lb_tensor  = get_dmpnn_mixture_features(lb_bow, lb_indices, DMPNN, device)
        y_pred_lb  = mixture_encoder(lb_tensor).cpu().numpy()  # (B, 51)

    pred_df = pd.DataFrame({"stimulus": lb_stimuli})
    for i, col in enumerate(LABEL_COLS):
        pred_df[col] = y_pred_lb[:, i]
    pred_df.to_csv(fname / "leaderboard_predictions.csv", index=False)

    # ============================================================
    # 7. Post-processing: clip values < 0.1 → 0.1
    #    (consistent with filter_and_PLCC-cosdist_test.py)
    # ============================================================
    y_pred_clipped = y_pred_lb.copy()
    mask_clip      = y_pred_clipped < 0.03
    y_pred_clipped[mask_clip] = 0

    pred_clipped_df = pd.DataFrame({"stimulus": lb_stimuli})
    for i, col in enumerate(LABEL_COLS):
        pred_clipped_df[col] = y_pred_clipped[:, i]
    pred_clipped_df.to_csv(fname / "leaderboard_predictions_clipped.csv", index=False)
    print(f"Clipped predictions saved")

    # Per-row PLCC and cosine distance on clipped predictions (scipy)
    y_true_np = lb_labels  # (B, 51) numpy
    plcc_vals, cosine_vals = [], []
    nan_plcc = nan_cosine = 0

    for i in range(len(lb_stimuli)):
        va = y_true_np[i]
        vb = y_pred_clipped[i]
        valid = ~(np.isnan(va) | np.isnan(vb))
        va_v, vb_v = va[valid], vb[valid]
        n_v = valid.sum()

        if n_v < 2 or np.std(va_v) == 0 or np.std(vb_v) == 0:
            p_val = float("nan")
        else:
            p_val = float(scipy_pearsonr(va_v, vb_v)[0])

        if n_v < 2 or np.linalg.norm(va_v) == 0 or np.linalg.norm(vb_v) == 0:
            c_val = float("nan")
        else:
            c_val = float(scipy_cosine_dist(va_v, vb_v))

        if np.isnan(p_val):
            nan_plcc += 1
        else:
            plcc_vals.append(p_val)
        if np.isnan(c_val):
            nan_cosine += 1
        else:
            cosine_vals.append(c_val)

    plcc_arr   = np.array(plcc_vals)
    cosine_arr = np.array(cosine_vals)
    mean_plcc_c   = float(np.mean(plcc_arr))   if len(plcc_arr)   > 0 else float("nan")
    mean_cosine_c = float(np.mean(cosine_arr)) if len(cosine_arr) > 0 else float("nan")
    std_plcc_c    = float(np.std(plcc_arr))    if len(plcc_arr)   > 0 else float("nan")
    std_cosine_c  = float(np.std(cosine_arr))  if len(cosine_arr) > 0 else float("nan")

    # ============================================================
    # 8. Summary
    # ============================================================
    y_pred_t = torch.tensor(y_pred_lb, dtype=torch.float32)
    y_lb_cpu = y_lb.cpu()
    final_plcc     = F1.pearson_corrcoef(
        y_pred_t.permute(1, 0), y_lb_cpu.permute(1, 0)
    ).mean().item()
    final_cos_dist = 1.0 - F.cosine_similarity(y_pred_t, y_lb_cpu, dim=1).mean().item()

    summary = (
        f"Experiment : {exp_name}\n"
        f"Stopped at epoch : {epoch}\n"
        f"Max mixture length (max_len) : {MAX_LEN}\n"
        f"Label columns : {len(LABEL_COLS)}\n"
        f"\n"
        f"Best model (selected by Leaderboard PLCC(sample)):\n"
        f"  Epoch          : {best_epoch}\n"
        f"\n"
        f"Final Leaderboard Performance (best checkpoint, raw):\n"
        f"  PLCC(sample)   : {final_plcc:.4f}\n"
        f"  cos-dist       : {final_cos_dist:.4f}\n"
        f"\n"
        f"After post-processing (clip < 0.1 → 0.1, scipy per-row):\n"
        f"  Mean PLCC(row)  : {mean_plcc_c:.6f}  (std={std_plcc_c:.6f}, NaN rows={nan_plcc})\n"
        f"  Mean cos-dist   : {mean_cosine_c:.6f}  (std={std_cosine_c:.6f}, NaN rows={nan_cosine})\n"
    )

    print("\n" + "=" * 50)
    print(summary)

    with open(fname / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Results saved to: {fname}/")
