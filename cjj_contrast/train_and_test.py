from typing import List

from tqdm import tqdm
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import wandb

class WeightedL1Loss(nn.Module):
    """
    PENN 损失函数 - 加权 L1 损失

    论文公式 (12):
    L(Θ̂, Θ) = Σ_{k=1}^M λ_k |θ̂_k - θ_k|

    其中 λ_k 是每个参数的权重，用于平衡不同范围的参数
    """

    def __init__(self, weight: List[float]):
        super().__init__()
        self.eps = torch.tensor(1e-8)
        self.loss = nn.L1Loss()
        self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        """
        计算加权 L1 损失。

        Returns:
            [total_loss, loss_1, loss_2, ..., loss_M]
        """
        M = x.shape[1]
        loss = []
        l_sum = 0

        for i in range(M):
            l = torch.mean(self.loss(x[:, i], y[:, i]))
            l_sum = l_sum + l * self.weight[i]
            loss.append(l)

        loss.insert(0, l_sum)
        return loss

class SupConLoss(nn.Module):
    """
    有监督对比学习损失 (Supervised Contrastive Loss)。
    使用参数标签 y 来定义正样本对（相同参数配置的轨迹）和负样本对。
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim] 特征向量
            labels: [batch_size, n_params] 参数标签，用于判断正负样本对
        Returns:
            loss: 标量损失
        """
        device = features.device
        batch_size = features.shape[0]
        
        # L2 归一化特征
        features = F.normalize(features, dim=1)
        
        # 统一 label 形状为 [batch_size, n_params]
        if labels.dim() > 2:
            labels = labels.view(batch_size, -1)
        elif labels.dim() == 2 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        
        # 计算余弦相似度矩阵 [B, B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 正样本 mask：参数完全相同的样本对
        labels_i = labels.unsqueeze(1)  # [B, 1, n_params]
        labels_j = labels.unsqueeze(0)  # [1, B, n_params]
        mask = torch.all(torch.abs(labels_i - labels_j) < 1e-6, dim=2).float().to(device)
        
        # 排除自身
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask - mask_self
        
        # 计算 exp(similarity)，排除自身
        exp_sim = torch.exp(similarity_matrix) * (1 - mask_self)
        
        # 分子：正样本的相似度之和
        pos_sum = (exp_sim * mask_pos).sum(dim=1)
        
        # 分母：所有样本的相似度之和
        denom = exp_sim.sum(dim=1)
        
        # 只对存在正样本的 anchor 计算损失
        valid = (mask_pos.sum(dim=1) > 0).float()
        loss = -torch.log(pos_sum / (denom + 1e-8) + 1e-8)
        loss = (loss * valid).sum() / (valid.sum() + 1e-8)
        
        return loss


def train(models, train_loader, optimizer=None, epochs: int = 10, 
          temperature: float = 0.1, log_interval: int = 10):
    """
    使用对比学习范式训练 FeatureExtractor。
    
    Args:
        models: (FeatureExtractor, ParamPredictor) tuple
        train_loader: 训练数据加载器
        optimizer: 优化器，若为 None 则默认使用 Adam(lr=1e-3)
        epochs: 训练轮数
        temperature: 对比学习温度系数
        log_interval: 打印间隔（batch 数）
    
    Returns:
        model: 训练后的模型
    """
    extractor, predictor = models
    
    # Pre-training: Contrastive Learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    if optimizer is None:
        optimizer_e = torch.optim.AdamW(extractor.parameters(), lr=1e-3)
    criterion_e = SupConLoss(temperature=temperature)
    for epoch in range(epochs):
        extractor.train()
        predictor.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Contrastive Training Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer_e.zero_grad()
            features = extractor(x)
            loss = criterion_e(features, y)
            loss.backward()
            clip_grad_norm_(extractor.parameters(), max_norm=100)
            optimizer_e.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            #! Uncomment the following line when testing predictor
            # break
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        wandb.log({"train/contrastive_loss": avg_loss, "epoch": epoch})
        
        #! Uncomment the following line when testing predictor
        # break
    
    torch.cuda.empty_cache()
    
    # Post-training: Regressing parameters
    predictor = predictor.to(device)
    if optimizer is None:
        optimizer_p = torch.optim.AdamW(predictor.parameters(), lr=1e-4)
    criterion_p = WeightedL1Loss([2, 2, 1, 1, 8, 8, 20, 20])
    for epoch in range(epochs):
        predictor.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            optimizer_p.zero_grad()
            optimizer_e.zero_grad()
            
            features = extractor(x)
            preds = predictor(features)
            
            loss, *_ = criterion_p(preds, y)
            loss.backward()
            
            clip_grad_norm_(predictor.parameters(), max_norm=100)
            clip_grad_norm_(extractor.parameters(), max_norm=1)
            
            optimizer_p.step()
            optimizer_e.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        wandb.log({"train/regression_loss": avg_loss, "epoch": epochs + epoch})
    
    return extractor, predictor


def plot_parameter_predictions(pred_params: np.ndarray, true_params: np.ndarray,
                               save_path: str = None):
    """绘制参数预测对比图（仿照 josephson_junction_param_learning_v2.py）"""
    param_names = [
        r'$\beta_{1}$', r'$\beta_{2}$',
        r'$i_1$', r'$i_2$',
        r'$\kappa_1$', r'$\kappa_2$',
        r'$\sigma_1$', r'$\sigma_2$'
    ]
    n_params = len(param_names)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        pred_vals = pred_params[:, i]
        true_vals = true_params[:, i]

        ax.scatter(true_vals, pred_vals, color='r', s=0.5, alpha=0.5)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin], 'b', linewidth=1)

        ax.axis('equal')
        ax.grid('minor')
        ax.set_xlabel('True values', fontsize=12)
        ax.set_ylabel('Estimated values', fontsize=12)
        ax.set_title(f'({alphabet[i]}) {name}', fontsize=14)

        # 计算 R²
        ss_res = np.sum((pred_vals - true_vals) ** 2)
        ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'$R^2$={r2:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Parameter prediction plot saved to: {save_path}")
    plt.close()
    return fig


@torch.no_grad()
def test(models, test_loader, temperature: float = 0.1):
    """
    在测试集上同时评估 FeatureExtractor（对比学习指标）和 ParamPredictor（回归指标）。
    
    Args:
        models: (FeatureExtractor, ParamPredictor) tuple
        test_loader: 测试数据加载器
        temperature: 对比学习温度系数
    
    Returns:
        dict: 包含对比学习损失、正/负样本相似度、回归损失、每个参数的 MAPE 和 R2
    """
    from sklearn.metrics import r2_score
    
    extractor, predictor = models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    predictor = predictor.to(device)
    extractor.eval()
    predictor.eval()
    
    criterion_contrast = SupConLoss(temperature=temperature)
    criterion_reg = WeightedL1Loss([2, 2, 1, 1, 8, 8, 20, 20])
    
    total_contrast_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    all_features = []
    all_labels = []
    all_preds = []
    all_trues = []
    
    for x, y in tqdm(test_loader, desc="Testing"):
        x, y = x.to(device), y.to(device)
        
        # FeatureExtractor 校验
        features = extractor(x)
        loss_contrast = criterion_contrast(features, y)
        total_contrast_loss += loss_contrast.item()
        
        all_features.append(features.cpu())
        all_labels.append(y.cpu())
        
        # ParamPredictor 校验
        preds = predictor(features)
        loss_reg, *_ = criterion_reg(preds, y)
        total_reg_loss += loss_reg.item()
        
        all_preds.append(preds.cpu())
        all_trues.append(y.cpu())
        
        num_batches += 1
    
    # === FeatureExtractor 对比学习指标 ===
    avg_contrast_loss = total_contrast_loss / max(num_batches, 1)
    print(f"Test Contrastive Loss: {avg_contrast_loss:.4f}")
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if all_labels.dim() > 2:
        all_labels = all_labels.view(all_labels.shape[0], -1)
    elif all_labels.dim() == 2 and all_labels.shape[-1] == 1:
        all_labels = all_labels.squeeze(-1)
    
    all_features = F.normalize(all_features, dim=1)
    sim_matrix = torch.matmul(all_features, all_features.T)
    
    labels_i = all_labels.unsqueeze(1)
    labels_j = all_labels.unsqueeze(0)
    pos_mask = torch.all(torch.abs(labels_i - labels_j) < 1e-6, dim=2).float()
    eye_mask = torch.eye(pos_mask.shape[0])
    pos_mask = pos_mask - eye_mask
    
    pos_sims = (sim_matrix * pos_mask).sum() / (pos_mask.sum() + 1e-8)
    neg_mask = 1 - pos_mask - eye_mask
    neg_sims = (sim_matrix * neg_mask).sum() / (neg_mask.sum() + 1e-8)
    
    print(f"Average Positive Similarity: {pos_sims.item():.4f}")
    print(f"Average Negative Similarity: {neg_sims.item():.4f}")
    
    # === ParamPredictor 回归指标 ===
    avg_reg_loss = total_reg_loss / max(num_batches, 1)
    print(f"Test Regression Loss: {avg_reg_loss:.4f}")
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_trues = torch.cat(all_trues, dim=0).numpy()
    
    eps = 1e-8
    mape_per_param = np.mean(np.abs(all_preds - all_trues) / (np.abs(all_trues) + eps), axis=0) * 100.0
    r2_per_param = r2_score(all_trues, all_preds, multioutput='raw_values')
    
    param_names = ['beta1', 'beta2', 'i1', 'i2', 'kappa1', 'kappa2', 'sigma1', 'sigma2']
    print("-" * 50)
    for i, name in enumerate(param_names):
        print(f"Param {name}: MAPE={mape_per_param[i]:.2f}%, R2={r2_per_param[i]:.4f}")
    print("-" * 50)
    
    log_dict = {
        "test/contrastive_loss": avg_contrast_loss,
        "test/pos_similarity": pos_sims.item(),
        "test/neg_similarity": neg_sims.item(),
        "test/regression_loss": avg_reg_loss,
    }
    for i, name in enumerate(param_names):
        log_dict[f"test/{name}_mape"] = mape_per_param[i]
        log_dict[f"test/{name}_r2"] = r2_per_param[i]
    wandb.log(log_dict)
    
    # 绘制并记录参数预测对比图
    fig = plot_parameter_predictions(all_preds, all_trues, save_path="parameter_predictions.png")
    wandb.log({"parameter_predictions": wandb.Image(fig)})
    
    return {
        "contrastive_loss": avg_contrast_loss,
        "pos_similarity": pos_sims.item(),
        "neg_similarity": neg_sims.item(),
        "regression_loss": avg_reg_loss,
        "mape_per_param": mape_per_param.tolist(),
        "r2_per_param": r2_per_param.tolist(),
    }


if __name__ == "__main__":
    from dataloader import create_dataloaders
    from models import ParamPredictor, FeatureExtractor
    from utils import seed_everything
    
    seed_everything(42)
    
    wandb.init(project="cjj-contrast", name="contrast+regression")
    
    train_loader, test_loader = create_dataloaders(batchsize=32)
    
    # 初始化 FeatureExtractor
    extractor = FeatureExtractor(in_dim=16, hidden_dim=64, n_layer=3, out_dim=128)
    predictor = ParamPredictor(in_dim=128, hid_dim=128, n_params=8)
    
    # 对比学习训练
    extractor, predictor = train((extractor, predictor), train_loader, epochs=5, temperature=0.1)
    
    # 测试评估
    results = test((extractor, predictor), test_loader, temperature=0.1)
    
    wandb.finish()
