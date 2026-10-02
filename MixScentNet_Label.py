import os
from argparse import ArgumentParser

import matplotlib

matplotlib.use('Agg')

import torch
import torch.nn as nn
import torchmetrics.functional as F1
import torch.nn.functional as F2
import tqdm

# 导入相关模块
from backbones.DMPNN import DMPNN_Fingerprint
from backbones.dataloader import *
from backbones.model_gat import MixtureEncoder

script_dir = Path(__file__).parent
base_dir = Path(*script_dir.parts[:])
sys.path.append(str(base_dir / "src/"))

def create_bow_and_indices(features_list, max_len=43):
    """
    输入:
        features_list: 包含每个混合物组分 SMILES 的列表。例如 [['C', 'O'],['CC', 'CCC']]
    输出:
        unique_smiles: 该集合中所有独立的 SMILES 组成的 Bag-of-Words
        indices: 形状为 (Batch, max_len) 的 Tensor，-1 表示没有该组分
    """
    unique_smiles = list(set([smi for mix in features_list for smi in mix]))
    smi2idx = {smi: i for i, smi in enumerate(unique_smiles)}

    indices = []
    for mix in features_list:
        idx_list = [smi2idx[smi] for smi in mix]
        # 统一长度至 max_len，不足补 -1
        if len(idx_list) < max_len:
            idx_list.extend([-1] * (max_len - len(idx_list)))
        else:
            idx_list = idx_list[:max_len]
        indices.append(idx_list)

    return unique_smiles, torch.tensor(indices, dtype=torch.long)

def get_dmpnn_mixture_features(bow_smiles, indices, dmpnn_model, device, unk_token=-999):
    """
    输入:
        bow_smiles: 独立的分子池 (list of str)
        indices: 映射索引 Tensor (Batch, 43)
        dmpnn_model: 预初始化的 DMPNN_Fingerprint 特征提取器
    输出:
        mixture_tensor: 形状为 (Batch, 43, embed_dim, 1) 的混合物特征
    """
    # 1. 提取独立分子的 DMPNN 特征 (一次性处理所有独特分子)
    unique_feats = dmpnn_model(bow_smiles)
    unique_feats = torch.tensor(unique_feats, dtype=torch.float32, device=device)

    embed_dim = unique_feats.shape[-1]

    # 2. 生成对应 -1 (padding) 的特征占位符，值为 unk_token(-999)
    pad_vec = torch.full((1, embed_dim), unk_token, dtype=torch.float32, device=device)
    all_feats = torch.cat([unique_feats, pad_vec], dim=0)

    # 3. 将原先为 -1 的位置，映射到刚才拼接到最后的 pad_vec 索引上
    pad_idx = len(unique_feats)
    mapped_indices = indices.clone().to(device)
    mapped_indices[mapped_indices == -1] = pad_idx

    # 4. 根据 index 进行 Gather，快速重构为 (Batch, 43, embed_dim)
    mixture_tensor = all_feats[mapped_indices]

    # 5. 增加最后一个维度，变成 (Batch, 43, embed_dim, 1)
    #    以兼容 MixtureEncoder.forward 中 torch.unbind(x, dim=-1) 的逻辑
    mixture_tensor = mixture_tensor.unsqueeze(-1)

    return mixture_tensor

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.4):
        """
        delta=0.4 对低std标签友好：
        绝大多数误差 < 0.4（因为标签本身范围就窄），走二次区间
        极端离群值进入线性区间，不会主导梯度
        """
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, N) 模型预测值
            target: (B, N) 真实标签
        Returns:
            loss: 标量
        """
        err = pred - target
        loss = torch.where(
            err.abs() < self.delta,
            0.5 * err ** 2,
            self.delta * (err.abs() - 0.5 * self.delta)
        )
        return loss.mean()

class MeanPLCCLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, N) 模型预测值
            target: (B, N) 真实标签
        Returns:
            loss: 在 B 上平均的 (1 - PLCC)，范围 [0, 2]
        """
        # 沿 N 维度去均值，保持维度用于广播: (B, 1)
        pred_mean   = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)

        pred_centered   = pred   - pred_mean    # (B, N)
        target_centered = target - target_mean  # (B, N)

        numerator = (pred_centered * target_centered).sum(dim=1)  # (B,)

        denominator = torch.sqrt(
            (pred_centered ** 2).sum(dim=1) *
            (target_centered ** 2).sum(dim=1)
        ) + self.eps  # (B,)

        plcc_per_sample = numerator / denominator   # (B,)，每个样本一个 PLCC 值
        loss_per_sample = 1.0 - plcc_per_sample     # (B,)

        return loss_per_sample.mean()               # 标量

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split", default="random_cv", choices=['random_cv', 'random_cv_unseen'], type=str)
    parser.add_argument("--loss", default="HuberLoss", choices=['HuberLoss', 'MSELoss','MAELoss','MeanPLCCLoss'], type=str)
    parser.add_argument("--gnn-lr", default=1e-4, type=float)
    parser.add_argument("--dmpnn-lr", default=1e-4, type=float)
    FLAGS = parser.parse_args()

    SEED = 202644  # 202644
    EARLY_STOP_PATIENCE = 2000
    EARLY_STOP_DELTA = 0.001
    num_epochs = 5000
    scheduler_step_size = 1500

    MOL_DIM = 512     # D-MPNN 输出维度
    HIDDEN_DIM = 512  # GATv2 隐层维度

    DMPNN_freeze = False
    FLAGS.exp_name = 'DMPNNTrain_MixtureGATtrain_SEED%s_%s' % (SEED, FLAGS.loss)
    labels_file = "./datasets/DREAM2025/TASK2_final_mixture_dataset.csv"
    smiles_file = "./datasets/DREAM2025/Mixture_SMILES_Converted.csv"
    fname = Path(f"results/{FLAGS.split}/label/{FLAGS.exp_name}")
    os.makedirs(f"{fname}/", exist_ok=True)
    weights_dir = fname / "weights"
    os.makedirs(weights_dir, exist_ok=True)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ===============================
    # 数据划分与训练主循环
    # ===============================
    if FLAGS.split == 'random_cv':
        cv_splits = get_mixture_cv_splits(labels_file, smiles_file, SEED)  # for normal split experiment
    elif FLAGS.split == 'random_cv_unseen':
        cv_splits = get_mixture_cv_splits_unseen_mol(labels_file, smiles_file, SEED=SEED)  # for unseen split experiment
    else:
        raise ValueError("Pick proper split in ['random_cv', 'random_cv_unseen']")

    summary_txt_path = fname / "all_splits_metrics_summary.txt"
    if summary_txt_path.exists():
        summary_txt_path.unlink()

    total_plcc_label  = 0
    total_plcc_sample = 0
    total_cosine_dist = 0
    plcc_sample_list = []
    cosine_dist_list = []
    for id, train, test in cv_splits:
        print("Loading DMPNN Fingerprint Model...")
        DMPNN = DMPNN_Fingerprint(MOL_DIM, device=device)
        if DMPNN_freeze:
            for name, param in DMPNN.model.named_parameters():
                param.requires_grad = False

        # 每个 fold 重置随机种子，保证初始化一致
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        train_features, train_labels = train
        train_bow, train_indices = create_bow_and_indices(train_features, max_len=43)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)

        test_features, test_labels = test
        test_bow, test_indices = create_bow_and_indices(test_features, max_len=43)
        y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

        print(f"\nRunning split: {id}")
        print(f"Training set size: {len(train_labels)}")
        print(f"Testing set size: {len(test_labels)}")

        # 用 MixtureEncoder 替代 Chemix，输入 (Batch, 43, 512, 1)，输出 (Batch, 51)
        mixture_encoder = MixtureEncoder(
            mol_dim=MOL_DIM,
            hidden_dim=HIDDEN_DIM,
            output_type='label',
        ).to(device)

        # different loss to train
        if FLAGS.loss == 'HuberLoss':
            loss_fn = HuberLoss(delta=0.4)
        elif FLAGS.loss == 'MAELoss':
            loss_fn = nn.L1Loss()
        elif FLAGS.loss == 'MSELoss':
            loss_fn = nn.MSELoss()
        elif FLAGS.loss == 'MeanPLCCLoss':
            loss_fn = MeanPLCCLoss()
        else:
            raise ValueError("Pick proper training loss in ['HuberLoss', 'MAELoss', 'MSELoss', 'MeanPLCCLoss']")

        metric_fn = F1.pearson_corrcoef

        optimizer = torch.optim.Adam([
            {"params": mixture_encoder.parameters(), "lr": FLAGS.gnn_lr},
            {"params": DMPNN.model.parameters(),      "lr": FLAGS.dmpnn_lr},
            {"params": DMPNN.DimReduce.parameters(),  "lr": FLAGS.dmpnn_lr},
        ])

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=0.5,
        )

        log = {k: [] for k in ["epoch", "train_loss", "test_loss", "test_metric_sample"]}
        pbar = tqdm.tqdm(range(num_epochs))

        best_plcc_label = 0
        best_plcc_sample = 0
        best_plcc_label_vec = None
        epochs_without_improvement = 0
        for epoch in pbar:
            # === Training Phase ===
            mixture_encoder.train()
            optimizer.zero_grad()

            train_mixture_tensor = get_dmpnn_mixture_features(
                train_bow, train_indices, DMPNN, device
            )
            y_pred = mixture_encoder(train_mixture_tensor)   # (Batch, 51)

            loss = loss_fn(y_pred, y_train)  # Huber loss / MSE loss / MAE loss / MeanPLCCloss
            loss.backward()
            optimizer.step()
            train_loss = loss.detach().cpu().item()

            # === Evaluation Phase ===
            mixture_encoder.eval()
            with torch.no_grad():
                test_mixture_tensor = get_dmpnn_mixture_features(
                    test_bow, test_indices, DMPNN, device
                )
                y_pred_test = mixture_encoder(test_mixture_tensor)   # (Batch, 51)
                loss_test = loss_fn(y_pred_test, y_test)  # Huber loss / MSE loss / MAE loss / MeanPLCCloss

                # per-sample PLCC: each sample's correlation across labels
                metric_sample = metric_fn(
                    y_pred_test.permute(1, 0), y_test.permute(1, 0)
                ).mean()

                test_loss = loss_test.detach().cpu().item()
                test_metric_sample = metric_sample.detach().cpu().item()

            log["epoch"].append(epoch)
            log["train_loss"].append(train_loss)
            log["test_loss"].append(test_loss)
            log["test_metric_sample"].append(test_metric_sample)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_description(
                f"LR: {current_lr:.2e} | Train: {train_loss:.4f} | "
                f"PLCC(sample): {test_metric_sample:.4f} | test loss: {test_loss:.4f}"
            )
            scheduler.step()

            if test_metric_sample > best_plcc_sample:
                best_plcc_sample = test_metric_sample
                best_cos = 1 - F2.cosine_similarity(
                    y_pred_test, y_test, dim=1
                ).mean().detach().cpu().item()
                epochs_without_improvement = 0

                ## Save best epoch weights
                torch.save({
                    "epoch": epoch,
                    "best_plcc": best_plcc_sample,
                    "mixture_encoder": mixture_encoder.state_dict(),
                    "DMPNN_dim_reduce": DMPNN.DimReduce.state_dict(),
                }, weights_dir / f"split_{id}_best.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"\nEarly stopping triggered at Epoch {epoch}. Best PLCC hasn't improved by "
                    f"{EARLY_STOP_DELTA} over the last {EARLY_STOP_PATIENCE} epochs."
                )
                break

        log = pd.DataFrame(log)
        log.to_csv(fname / f"{id}_training_log.txt", sep='\t', index=False)

        total_plcc_sample += best_plcc_sample
        total_cosine_dist += best_cos
        plcc_sample_list.append(best_plcc_sample)
        cosine_dist_list.append(best_cos)

        test_metrics = {'PLCC(sample)': best_plcc_sample, 'cos-dist': best_cos}
        print(f"Split {id} best Metrics: PLCC(sample) {best_plcc_sample:.4f}, cos-dist {best_cos:.4f}")

        with open(summary_txt_path, "a", encoding="utf-8") as f:
            f.write(f"========== Cross-Validation Split: {id} ==========\n")
            f.write(f"Best Test Metric (from Early Stopping): PLCC(sample)={best_plcc_sample:.4f}\n")
            f.write("Test Metrics on this split:\n")
            for m_name, m_val in test_metrics.items():
                f.write(f"  - {m_name}: {m_val:.4f}\n")
            f.write("\n")

        del DMPNN, mixture_encoder, optimizer, scheduler
        del id, train, test
        torch.cuda.empty_cache()

    plcc_sample_arr = np.array(plcc_sample_list, dtype=np.float64)
    cosine_dist_arr = np.array(cosine_dist_list, dtype=np.float64)
    plcc_mean, plcc_std = plcc_sample_arr.mean(), plcc_sample_arr.std()
    cos_mean,  cos_std  = cosine_dist_arr.mean(), cosine_dist_arr.std()

    final_print = (
        f'Avg PLCC(sample): {plcc_mean:.4f}±{plcc_std:.4f}, '
        f'Avg cosine_dist: {cos_mean:.4f}±{cos_std:.4f}'
    )
    print('%s training finished!' % FLAGS.exp_name)
    print(final_print)

    with open(summary_txt_path, "a", encoding="utf-8") as f:
        f.write(final_print + "\n")
