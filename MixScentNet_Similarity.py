import os
from argparse import ArgumentParser
import scipy.stats as stats
import matplotlib

matplotlib.use('Agg')
import torch
import torch.nn as nn
import torchmetrics.functional as F1
import tqdm

from backbones.DMPNN import DMPNN_Fingerprint
from backbones.model_gat import MixtureEncoder
from backbones.dataloader import *

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:])
sys.path.append(str(base_dir / "src/"))

def create_bow_and_indices(features_list, max_len=43):
    unique_smiles = sorted(list(
        set([smi for mix in features_list[:, 0] for smi in mix]).union(
            set([smi for mix in features_list[:, 1] for smi in mix]))
    ))
    smi2idx = {smi: i for i, smi in enumerate(unique_smiles)}

    indices = []
    for mix in features_list:
        idx_list0 = [smi2idx[smi] for smi in mix[0]]
        idx_list1 = [smi2idx[smi] for smi in mix[1]]
        if len(idx_list0) < max_len:
            idx_list0.extend([-1] * (max_len - len(idx_list0)))
        else:
            idx_list0 = idx_list0[:max_len]
        if len(idx_list1) < max_len:
            idx_list1.extend([-1] * (max_len - len(idx_list1)))
        else:
            idx_list1 = idx_list1[:max_len]
        indices.append([idx_list0, idx_list1])

    # (batch, 2, 43) → (batch, 43, 2)
    return unique_smiles, torch.tensor(indices, dtype=torch.long).permute(0, 2, 1)

def get_dmpnn_mixture_features(bow_smiles, indices, dmpnn_model, device, unk_token=-999):
    """
    Output: (batch, 43, embed_dim, 2) — directly compatible with MixtureEncoder.forward(x)
    """
    unique_feats = dmpnn_model(bow_smiles)          # (n_unique, embed_dim)
    embed_dim = unique_feats.shape[-1]

    pad_vec = torch.full((1, embed_dim), unk_token, dtype=torch.float32, device=device)
    all_feats = torch.cat([unique_feats, pad_vec], dim=0)

    pad_idx = len(unique_feats)
    mapped_indices = indices.clone().to(device)
    mapped_indices[mapped_indices == -1] = pad_idx

    mixture_tensor = all_feats[mapped_indices]          # (batch, 43, 2, embed_dim)
    mixture_tensor = mixture_tensor.permute(0, 1, 3, 2) # (batch, 43, embed_dim, 2)
    return mixture_tensor

class PLCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (N,) or (N, 1) model predictions
            target: (N,) or (N, 1) ground-truth labels
        Returns:
            loss: 1 - PLCC, range [0, 2]; smaller is better
        """
        pred   = pred.view(-1)
        target = target.view(-1)

        pred_mean   = pred.mean()
        target_mean = target.mean()

        pred_centered   = pred   - pred_mean
        target_centered = target - target_mean

        numerator   = (pred_centered * target_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (target_centered ** 2).sum()
        ) + self.eps

        plcc = numerator / denominator          # range [-1, 1]
        return 1.0 - plcc                       # range [0, 2], optimal value is 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split", default="random_cv", choices=['random_cv', 'random_cv_unseen'], type=str)
    parser.add_argument("--loss", default='MAELoss', choices=['MAELoss','MSELoss','PLCCLoss'], type=str)
    parser.add_argument("--gnn-lr", default=1e-4, type=float,
                        help="Learning rate for MixtureEncoder")
    parser.add_argument("--dmpnn-lr", default=1e-4, type=float,
                        help="Learning rate for DMPNN backbone")
    FLAGS = parser.parse_args()

    SEED = 202644
    EARLY_STOP_PATIENCE = 1000
    num_epochs = 5000
    scheduler_step_size = 1500

    MOL_DIM = 512     # DMPNN output dimension
    HIDDEN_DIM = 512  # GNN hidden dimension

    dmpnn_freeze = True
    FLAGS.exp_name = 'DMPNNfix_MixtureGATtrain_SEED%s_%s' % (SEED, FLAGS.loss)
    labels_file = "./datasets/mixtures/mixtures_combined.csv"
    smiles_file  = "./datasets/mixtures/mixture_smi_definitions_clean.csv"
    fname = Path(f"results/{FLAGS.split}/similarity/{FLAGS.exp_name}")
    os.makedirs(f"{fname}/", exist_ok=True)
    weights_dir = fname / "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    if FLAGS.split == 'random_cv':
        cv_splits = get_mixture_similarity_cv_splits(labels_file, smiles_file, SEED)  # for normal split experiment
    elif FLAGS.split == 'random_cv_unseen':
        cv_splits = get_mixture_similarity_cv_splits_unseen_mol(labels_file, smiles_file, SEED=SEED)  # for unseen split experiment
    else:
        raise ValueError("Pick proper split in ['random_cv', 'random_cv_unseen']")

    summary_txt_path = fname / "all_splits_metrics_summary.txt"
    if summary_txt_path.exists():
        summary_txt_path.unlink()

    total_plcc = 0
    total_krcc = 0
    total_rmse = 0
    plcc_list = []
    krcc_list = []

    for id, train, test in cv_splits:
        # ================================================================
        # 1. Initialize the DMPNN feature extractor and MixtureEncoder
        # ================================================================
        print("Loading DMPNN Fingerprint Model...")
        DMPNN = DMPNN_Fingerprint(MOL_DIM, device=device)
        if dmpnn_freeze:
            for param in DMPNN.model.parameters():
                param.requires_grad = False

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        # MixtureEncoder: takes (batch, 43, 512, 2) and outputs (batch,) similarity scores
        mixture_encoder = MixtureEncoder(
            mol_dim=MOL_DIM, hidden_dim=HIDDEN_DIM, output_type='similarity'
        ).to(device)

        # ================================================================
        # 2. Data preparation (BOW indices; features are not pre-extracted, DMPNN is run each epoch)
        # ================================================================
        train_features, train_labels = train
        train_bow, train_indices = create_bow_and_indices(train_features, max_len=43)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)

        test_features, test_labels = test
        test_bow, test_indices = create_bow_and_indices(test_features, max_len=43)
        y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

        print(f"\nRunning split: {id}")
        print(f"Training set size: {len(train_labels)}")
        print(f"Testing set size:  {len(test_labels)}")

        # ================================================================
        # 3. Optimizer & scheduler
        # ================================================================
        if   FLAGS.loss == "MAELoss":
            loss_fn = nn.L1Loss()
        elif FLAGS.loss == "MSELoss":
            loss_fn   = nn.MSELoss()
        elif FLAGS.loss == "PLCCLoss":
            loss_fn   = PLCCLoss()
        else:
            raise ValueError("Pick proper training loss in ['MAELoss', 'MSELoss', 'PLCCLoss']")

        metric_fn = F1.pearson_corrcoef

        optimizer = torch.optim.Adam([
            {"params": mixture_encoder.parameters(), "lr": FLAGS.gnn_lr},
            {"params": DMPNN.model.parameters(), "lr": FLAGS.dmpnn_lr},
            {"params": DMPNN.DimReduce.parameters(), "lr": FLAGS.dmpnn_lr},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=0.5
        )

        # ================================================================
        # 4. Training loop
        # ================================================================
        log = {k: [] for k in ["epoch", "train_loss", "test_loss", "test_metric"]}
        pbar = tqdm.tqdm(range(num_epochs))

        best_plcc = 0
        epochs_without_improvement = 0

        for epoch in pbar:
            # --- Train ---
            mixture_encoder.train()
            optimizer.zero_grad()

            # (batch, 43, 512, 2) → MixtureEncoder → (batch,)
            train_mixture_tensor = get_dmpnn_mixture_features(train_bow, train_indices, DMPNN, device)
            y_pred = mixture_encoder(train_mixture_tensor)

            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.detach().cpu().item()

            # --- Eval ---
            mixture_encoder.eval()
            with torch.no_grad():
                test_mixture_tensor = get_dmpnn_mixture_features(test_bow, test_indices, DMPNN, device)
                y_pred_test = mixture_encoder(test_mixture_tensor)
                loss_test   = loss_fn(y_pred_test, y_test)
                metric      = metric_fn(y_pred_test, y_test)

                test_loss   = loss_test.detach().cpu().item()
                test_metric = metric.detach().cpu().item()

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["test_loss"].append(test_loss)
            log["test_metric"].append(test_metric)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"LR: {current_lr:.2e} | Train: {train_loss:.4f} | Test PLCC: {test_metric:.4f}"
            )
            scheduler.step()

            if test_metric > best_plcc:
                best_plcc  = test_metric
                preds_np   = y_pred_test.detach().cpu().numpy().flatten()
                targets_np = y_test.detach().cpu().numpy().flatten()
                best_krcc  = stats.kendalltau(preds_np, targets_np)[0]
                epochs_without_improvement = 0

                torch.save({
                    "epoch": epoch,
                    "best_plcc": best_plcc,
                    "mixture_encoder": mixture_encoder.state_dict(),
                    "DMPNN_dim_reduce": DMPNN.DimReduce.state_dict(),
                }, weights_dir / f"split_{id}_best.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"No improvement over {EARLY_STOP_PATIENCE} epochs."
                )
                break

        log = pd.DataFrame(log)
        log.to_csv(fname / f"{id}_training_log.txt", sep='\t', index=False)

        total_plcc += best_plcc
        total_krcc += best_krcc
        plcc_list.append(best_plcc)
        krcc_list.append(best_krcc)

        test_metrics = {'PLCC': best_plcc, 'KRCC': best_krcc}
        print(f"Split {id} best metrics: PLCC {best_plcc:.4f}, KRCC {best_krcc:.4f}")

        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write(f"========== Cross-Validation Split: {id} ==========\n")
            f.write(f"Stopped at epoch: {epoch}\n")
            f.write("Test Metrics on this split:\n")
            for m_name, m_val in test_metrics.items():
                f.write(f"  - {m_name}: {m_val:.4f}\n")
            f.write("\n")

        del DMPNN, mixture_encoder, optimizer, scheduler
        del id, train, test
        torch.cuda.empty_cache()

    # ================================================================
    # 5. Summary
    # ================================================================
    plcc_arr = np.array(plcc_list, dtype=np.float64)
    krcc_arr = np.array(krcc_list, dtype=np.float64)
    final_print = (
        f"PLCC: {plcc_arr.mean():.4f}±{plcc_arr.std():.4f}, "
        f"KRCC: {krcc_arr.mean():.4f}±{krcc_arr.std():.4f}"
    )
    print(f'{FLAGS.exp_name} training finished!')
    print(final_print)

    with open(summary_txt_path, "a", encoding="utf-8") as f:
        f.write(final_print + "\n")
