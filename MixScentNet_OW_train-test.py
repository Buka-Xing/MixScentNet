"""
Combined training + Olfactory White evaluation script.

Pipeline:
    1. Train DMPNN + MixtureGAT on a source-based split (e.g. Snitz+Ravia → Bushdid).
    2. After training, load the best checkpoint and run an Olfactory White
       reproduction test on the current split's test set.
    3. All artifacts (weights, training logs, OW CSV/figures, summaries) are
       saved under `olfactory_white_results/`.
"""

from argparse import ArgumentParser
import scipy.stats as stats
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchmetrics.functional as F1
import tqdm

from backbones.DMPNN import DMPNN_Fingerprint
from backbones.model_gat import MixtureEncoder
from backbones.dataloader import *

from MixScentNet_Similarity import (
    create_bow_and_indices,
    get_dmpnn_mixture_features,
)

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:])
sys.path.append(str(base_dir / "src/"))

plt.rcParams['font.size'] = 14

VALID_SOURCES = ["Bushdid", "Ravia", "Snitz"]


# ======================================================================
# Source / split helpers
# ======================================================================
def parse_source_combo(combo_str):
    """Parse a string like 'Snitz+Bushdid' into a list of source names. 'All' means every valid source."""
    if combo_str.lower() == "all":
        return list(VALID_SOURCES)
    names = [s.strip() for s in combo_str.split("+") if s.strip()]
    for n in names:
        if n not in VALID_SOURCES:
            raise ValueError(
                f"Unknown source '{n}'. Valid: {VALID_SOURCES} or combinations like 'Snitz+Bushdid'."
            )
    if len(names) == 0:
        raise ValueError("Empty source combination.")
    return names


def get_source_based_split(combined_csv_path, smiles_csv_path, train_sources):
    """
    Split the data by the source prefix in column 0 of mixtures_combined.csv.

    train_sources: list of source prefixes used for training (e.g. ["Snitz", "Ravia"]);
                   every other sample becomes the test set.

    Also returns the per-sample metadata (Dataset, Mixture1, Mixture2, len1, len2,
    Experimental) aligned with both train and test splits, so the Olfactory White
    analysis on the test set can be performed without reloading the data.
    """
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

    features_list, labels_list, source_list, meta_list = [], [], [], []
    for _, row in df_combined.iterrows():
        ds = str(row.iloc[0]).strip()
        key1 = (ds, str(row.iloc[1]).strip())
        key2 = (ds, str(row.iloc[2]).strip())
        try:
            score = float(row.iloc[3])
        except ValueError:
            continue
        if key1 in smiles_dict and key2 in smiles_dict:
            s1, s2 = smiles_dict[key1], smiles_dict[key2]
            if len(s1) == 0 or len(s2) == 0:
                continue
            features_list.append([s1, s2])
            labels_list.append(score)
            source_list.append(ds)
            meta_list.append({
                "Dataset": ds,
                "Mixture1": row.iloc[1],
                "Mixture2": row.iloc[2],
                "len1": len(s1),
                "len2": len(s2),
                "Experimental": score,
            })

    features_array = np.array(features_list, dtype=object)
    labels_matrix = np.array(labels_list, dtype=np.float64)
    sources_array = np.array(source_list, dtype=object)
    meta_df_full = pd.DataFrame(meta_list)

    def in_train(src):
        return any(src.startswith(t) for t in train_sources)

    train_mask = np.array([in_train(s) for s in sources_array])
    test_mask = ~train_mask

    X_train = features_array[train_mask]
    y_train = labels_matrix[train_mask]
    X_test = features_array[test_mask]
    y_test = labels_matrix[test_mask]
    meta_train = meta_df_full[train_mask].reset_index(drop=True)
    meta_test = meta_df_full[test_mask].reset_index(drop=True)

    split_id = "+".join(train_sources)
    print(f"[split] train sources={train_sources} | train={len(y_train)} | test={len(y_test)}")
    return [(split_id, (X_train, y_train), (X_test, y_test), meta_train, meta_test)]


# ======================================================================
# Olfactory White plotting helper (from OW_GeoMean_test.py)
# ======================================================================
def plot_distance_vs_geomean(df, value_col, y_label, title, out_png):
    sub_df = df.dropna(subset=[value_col, "GeoMeanLen"]).copy()
    if len(sub_df) == 0:
        print(f"[skip] no valid data for {value_col}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    datasets = sub_df["Dataset"].unique()
    cmap = plt.get_cmap("tab10")
    for i, ds in enumerate(datasets):
        s = sub_df[sub_df["Dataset"] == ds]
        ax.scatter(s["GeoMeanLen"], s[value_col],
                   s=18, alpha=0.55, color=cmap(i % 10), label=ds)

    # Bin by integer geometric mean and show mean ± std
    bins = np.arange(1, int(sub_df["GeoMeanLen"].max()) + 2)
    sub_df["_bin"] = pd.cut(sub_df["GeoMeanLen"], bins=bins, include_lowest=True)
    grouped = sub_df.groupby("_bin", observed=True)[value_col].agg(["mean", "std", "count"])
    centers = [iv.mid for iv in grouped.index]
    ax.errorbar(centers, grouped["mean"], yerr=grouped["std"],
                color="black", marker="o", linestyle="none", capsize=3,
                label="Mean ± Std (binned)")

    # Linear regression (red)
    x = sub_df["GeoMeanLen"].to_numpy(dtype=float)
    y = sub_df[value_col].to_numpy(dtype=float)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="red", linewidth=2,
            label="Linear fit")

    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)
    annotation = (f"Pearson r = {pearson_r:.4f} (p={pearson_p:.2e})\n"
                  f"Spearman ρ = {spearman_r:.4f} (p={spearman_p:.2e})\n"
                  f"R² = {r_value ** 2:.4f}")
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="red", alpha=0.85))

    ax.set_xlabel(r"Geometric mean of components $\sqrt{n_1\cdot n_2}$")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {out_png}")


# ======================================================================
# Olfactory White evaluation on the test split using best checkpoint
# ======================================================================
def run_olfactory_white_eval(
    ckpt_path, X_test, meta_test, out_dir, sources_tag, split_id,
    mol_dim, hidden_dim, max_len, batch_size, device,
):
    """
    Load the best checkpoint and predict perceptual distance for every
    test-split mixture pair. Save predictions to CSV and produce predicted
    vs. experimental Olfactory-White figures.
    """
    print(f"\n[OW] Running Olfactory White evaluation with ckpt: {ckpt_path}")

    # Rebuild model
    DMPNN = DMPNN_Fingerprint(mol_dim, device=device)
    for p in DMPNN.model.parameters():
        p.requires_grad = False

    mixture_encoder = MixtureEncoder(
        mol_dim=mol_dim, hidden_dim=hidden_dim, output_type='similarity'
    ).to(device)

    # LazyLinear needs one forward pass to materialize parameter shapes
    dummy = torch.zeros(1, max_len, mol_dim, 2, device=device)
    with torch.no_grad():
        mixture_encoder(dummy)

    ckpt = torch.load(ckpt_path, map_location=device)
    mixture_encoder.load_state_dict(ckpt["mixture_encoder"])
    DMPNN.DimReduce.load_state_dict(ckpt["DMPNN_dim_reduce"])
    print(f"[OW] Loaded ckpt @ epoch={ckpt.get('epoch')}, best_plcc={ckpt.get('best_plcc')}")

    mixture_encoder.eval()
    DMPNN.model.eval()
    DMPNN.DimReduce.eval()

    # Batched prediction over the test split
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            batch = X_test[start:start + batch_size]
            bow, indices = create_bow_and_indices(batch, max_len=max_len)
            mix_tensor = get_dmpnn_mixture_features(bow, indices, DMPNN, device)
            y_pred = mixture_encoder(mix_tensor).detach().cpu().numpy().flatten()
            preds.extend(y_pred.tolist())
            print(f"  [OW] Processed {min(start + batch_size, len(X_test))}/{len(X_test)}")

    meta_df = meta_test.copy()
    meta_df["PredictedDistance"] = preds
    meta_df["GeoMeanLen"] = np.sqrt(
        meta_df["len1"].astype(float) * meta_df["len2"].astype(float)
    )

    ckpt_tag = Path(ckpt_path).stem
    out_csv = out_dir / f"olfactory_white_predictions_{ckpt_tag}_{sources_tag}_TEST-{split_id}.csv"
    meta_df.to_csv(out_csv, index=False)
    print(f"[OW] Saved predictions to {out_csv}")

    # Figures
    plot_distance_vs_geomean(
        meta_df, value_col="PredictedDistance",
        y_label="Predicted perceptual distance",
        title=f"Olfactory White — MixScentNet (Predicted) [train={sources_tag}]",
        out_png=out_dir / f"olfactory_white_predicted_{ckpt_tag}_{sources_tag}.png",
    )

    plot_distance_vs_geomean(
        meta_df, value_col="Experimental",
        y_label="Experimental perceptual distance",
        title=f"Olfactory White — Human Experiment (Ground truth) [train={sources_tag}]",
        out_png=out_dir / f"olfactory_white_experimental_{ckpt_tag}_{sources_tag}.png",
    )

    # Cleanup
    del DMPNN, mixture_encoder
    torch.cuda.empty_cache()


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split", default="Ofactory-white_split", type=str)
    parser.add_argument("--train-sources", default="Snitz+Ravia", type=str,
                        help="Training source combination, e.g. 'Snitz', 'Snitz+Bushdid', "
                             "'Bushdid+Ravia+Snitz', or 'All'")
    parser.add_argument("--gnn-lr", default=1e-4, type=float,
                        help="Learning rate for MixtureEncoder")
    parser.add_argument("--dmpnn-lr", default=1e-4, type=float,
                        help="Learning rate for DMPNN backbone")
    FLAGS = parser.parse_args()

    SEED = 202644
    EARLY_STOP_PATIENCE = 1000
    num_epochs = 5000
    scheduler_step_size = 1500

    MOL_DIM = 512
    HIDDEN_DIM = 512
    MAX_LEN = 43
    OW_BATCH_SIZE = 64

    dmpnn_freeze = True
    train_sources = parse_source_combo(FLAGS.train_sources)
    sources_tag = "+".join(train_sources)

    FLAGS.exp_name = (
        f'DMPNNfix_MixtureGATtrain_MAE_TRAIN-{sources_tag}_SEED{SEED}_new'
    )
    labels_file = "./datasets/DREAM2024/mixtures_combined.csv"
    smiles_file = "./datasets/DREAM2024/mixture_smi_definitions_clean.csv"

    # ------------------------------------------------------------------
    # All training/eval artifacts go under olfactory_white_results/
    # ------------------------------------------------------------------
    OUT_ROOT = Path("olfactory_white_results")
    fname = OUT_ROOT / FLAGS.split / FLAGS.train_sources
    fname.mkdir(parents=True, exist_ok=True)
    weights_dir = fname / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ow_dir = fname / "Figure-CSV"
    ow_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    cv_splits = get_source_based_split(labels_file, smiles_file, train_sources)

    summary_txt_path = fname / "all_splits_metrics_summary.txt"
    if summary_txt_path.exists():
        summary_txt_path.unlink()

    total_plcc = 0
    total_krcc = 0

    for split_id, train, test, meta_train, meta_test in cv_splits:
        # ================================================================
        # 1. Initialize DMPNN feature extractor and MixtureEncoder
        # ================================================================
        print("Loading DMPNN Fingerprint Model...")
        DMPNN = DMPNN_Fingerprint(MOL_DIM, device=device)
        if dmpnn_freeze:
            for param in DMPNN.model.parameters():
                param.requires_grad = False

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        mixture_encoder = MixtureEncoder(
            mol_dim=MOL_DIM, hidden_dim=HIDDEN_DIM, output_type='similarity'
        ).to(device)

        # ================================================================
        # 2. Data preparation
        # ================================================================
        train_features, train_labels = train
        train_bow, train_indices = create_bow_and_indices(train_features, max_len=MAX_LEN)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)

        test_features, test_labels = test
        test_bow, test_indices = create_bow_and_indices(test_features, max_len=MAX_LEN)
        y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

        print(f"\nRunning split: {split_id}")
        print(f"Training set size: {len(train_labels)}")
        print(f"Testing set size:  {len(test_labels)}")

        # ================================================================
        # 3. Optimizer & scheduler
        # ================================================================
        loss_fn = nn.L1Loss()
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
        best_krcc = 0
        epochs_without_improvement = 0
        best_ckpt_path = weights_dir / f"split_{split_id}_best.pt"

        for epoch in pbar:
            mixture_encoder.train()
            optimizer.zero_grad()

            train_mixture_tensor = get_dmpnn_mixture_features(
                train_bow, train_indices, DMPNN, device
            )
            y_pred = mixture_encoder(train_mixture_tensor)

            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.detach().cpu().item()

            mixture_encoder.eval()
            with torch.no_grad():
                test_mixture_tensor = get_dmpnn_mixture_features(
                    test_bow, test_indices, DMPNN, device
                )
                y_pred_test = mixture_encoder(test_mixture_tensor)
                loss_test = loss_fn(y_pred_test, y_test)
                metric = metric_fn(y_pred_test, y_test)

                test_loss = loss_test.detach().cpu().item()
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
                best_plcc = test_metric
                preds_np = y_pred_test.detach().cpu().numpy().flatten()
                targets_np = y_test.detach().cpu().numpy().flatten()
                best_krcc = stats.kendalltau(preds_np, targets_np)[0]
                epochs_without_improvement = 0

                torch.save({
                    "epoch": epoch,
                    "best_plcc": best_plcc,
                    "mixture_encoder": mixture_encoder.state_dict(),
                    "DMPNN_dim_reduce": DMPNN.DimReduce.state_dict(),
                }, best_ckpt_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"No improvement over {EARLY_STOP_PATIENCE} epochs."
                )
                break

        log = pd.DataFrame(log)
        log.to_csv(fname / f"{split_id}_training_log.txt", sep='\t', index=False)

        total_plcc += best_plcc
        total_krcc += best_krcc

        test_metrics = {'PLCC': best_plcc, 'KRCC': best_krcc}
        print(f"Split {split_id} best metrics: PLCC {best_plcc:.4f}, KRCC {best_krcc:.4f}")

        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write(f"========== Source Split: {split_id} ==========\n")
            f.write(f"Train sources: {train_sources}\n")
            f.write(f"Stopped at epoch: {epoch}\n")
            f.write("Test Metrics on this split:\n")
            for m_name, m_val in test_metrics.items():
                f.write(f"  - {m_name}: {m_val:.4f}\n")
            f.write("\n")

        # Free training models before reloading the best checkpoint for OW eval
        del DMPNN, mixture_encoder, optimizer, scheduler
        torch.cuda.empty_cache()

        # ================================================================
        # 5. Olfactory White evaluation on the current test split using
        #    the best-checkpoint weights.
        # ================================================================
        if best_ckpt_path.exists():
            run_olfactory_white_eval(
                ckpt_path=best_ckpt_path,
                X_test=test_features,
                meta_test=meta_test,
                out_dir=ow_dir,
                sources_tag=sources_tag,
                split_id=split_id,
                mol_dim=MOL_DIM,
                hidden_dim=HIDDEN_DIM,
                max_len=MAX_LEN,
                batch_size=OW_BATCH_SIZE,
                device=device,
            )
        else:
            print(f"[OW] No best checkpoint found at {best_ckpt_path}; skipping OW eval.")

        del train, test, meta_train, meta_test, split_id

    # ================================================================
    # 6. Summary
    # ================================================================
    n_splits = len(cv_splits)
    final_print = (
        f"Avg PLCC: {total_plcc / n_splits:.4f}, "
        f"Avg KRCC: {total_krcc / n_splits:.4f}"
    )
    print(f'{FLAGS.exp_name} training finished!')
    print(final_print)

    with open(summary_txt_path, "a", encoding="utf-8") as f:
        f.write(final_print + "\n")
