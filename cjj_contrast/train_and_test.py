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
    物理感知对比损失：参数越近，越倾向于拉近；
    但动力学行为不同时，即使参数相同也推开（处理多稳态）
    """
    def __init__(self, temperature: float = 0.1, 
                 param_scales: torch.Tensor = None,
                 psd_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.psd_weight = psd_weight  # 功率谱特征的权重
        
        # 各参数的物理尺度（用于归一化距离）
        # 例如：i∈[0,2], β∈[0,1], κ∈[0,0.2], σ∈[0,0.05]
        self.register_buffer('param_scales', param_scales or torch.ones(1))
        
    def compute_kinematic_similarity(self, traj_i, traj_j):
        """
        计算两条轨迹的动力学相似度（0-1之间，1表示完全相同）
        输入: [T, 4] 的轨迹 (phi1, v1, phi2, v2)
        """
        with torch.no_grad():  # 不梯度回传，作为软标签
            # 1. 平均电压（Running/Trapped 态的关键区分）
            v_mean_i = traj_i[:, [1, 3]].mean(dim=0)  # [v1_mean, v2_mean]
            v_mean_j = traj_j[:, [1, 3]].mean(dim=0)
            v_sim = torch.exp(-torch.norm(v_mean_i - v_mean_j) / 0.5)
            
            # 2. 功率谱密度相似度（Bursting/周期/混沌的区分）
            # 使用周期图法快速估计PSD
            psd_i = self._compute_psd(traj_i[:, [1, 3]])  # 基于电压
            psd_j = self._compute_psd(traj_j[:, [1, 3]])
            psd_sim = torch.exp(-torch.norm(psd_i - psd_j) / (torch.norm(psd_i) + 1e-8))
            
            # 3. 相位缠绕数（区分旋转与静止）
            winding_i = self._compute_winding_number(traj_i[:, [0, 2]])
            winding_j = self._compute_winding_number(traj_j[:, [0, 2]])
            winding_sim = (winding_i == winding_j).float()
            
            # 综合相似度（可调整权重）
            total_sim = 0.4 * v_sim + 0.4 * psd_sim + 0.2 * winding_sim
            return total_sim
    
    def _compute_psd(self, voltage_signal):
        """快速功率谱密度估计（使用FFT）"""
        # voltage_signal: [T, 2]
        T = voltage_signal.shape[0]
        # Hann窗减少频谱泄漏
        window = torch.hann_window(T, device=voltage_signal.device).unsqueeze(1)
        signal = voltage_signal * window
        
        # FFT并计算功率谱
        fft = torch.fft.rfft(signal, dim=0)
        psd = torch.abs(fft)**2
        # 只保留低频部分（约瑟夫森结动力学通常在特定频段）
        return psd[:T//10].flatten()  # 取前10%频段
    
    def _compute_winding_number(self, phase_signal):
        """计算相位缠绕数（检测 2π 跳变）"""
        # phase_signal: [T, 2]
        delta = torch.diff(phase_signal, dim=0)
        # 检测 2π 跳变（考虑周期性）
        jumps = torch.abs(delta) > torch.pi
        winding = (delta / (2 * torch.pi)).sum(dim=0)
        return torch.round(winding).abs()
    
    def forward(self, features: torch.Tensor, 
                labels: torch.Tensor,
                trajectories: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: [B, D] 嵌入特征
            labels: [B, n_params] 参数标签
            trajectories: [B, T, 4] 原始轨迹（用于计算动力学相似度）
        """
        device = features.device
        batch_size = features.shape[0]
        
        features = F.normalize(features, dim=1)
        
        # 参数距离矩阵 [B, B]
        labels_i = labels.unsqueeze(1)  # [B, 1, n_params]
        labels_j = labels.unsqueeze(0)  # [1, B, n_params]
        
        # 归一化参数距离（考虑各参数物理尺度）
        param_diff = torch.abs(labels_i - labels_j) / self.param_scales.to(device)
        param_dist = torch.norm(param_diff, dim=2)  # [B, B]
        
        # 软正样本权重：参数越近权重越高（高斯核）
        # 但同时需要考虑动力学行为
        if trajectories is not None:
            # 计算动力学相似度矩阵（较慢，只在训练后期或每隔几步计算）
            kin_sim = torch.zeros(batch_size, batch_size, device=device)
            for i in range(batch_size):
                for j in range(i+1, batch_size):
                    sim = self.compute_kinematic_similarity(trajectories[i], trajectories[j])
                    kin_sim[i, j] = sim
                    kin_sim[j, i] = sim
            
            # 最终相似度权重：参数相近且动力学相似 -> 高权重正样本
            # 参数相近但动力学不同（多稳态）-> 低权重（视为负样本）
            mask_pos_weight = torch.exp(-param_dist / 0.1) * kin_sim
        else:
            # 退化为纯参数距离（训练初期使用，更快）
            mask_pos_weight = torch.exp(-param_dist / 0.1)
        
        # 计算对比损失（Soft InfoNCE）
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        exp_sim = torch.exp(similarity_matrix)
        
        # 排除自身
        mask_self = torch.eye(batch_size, device=device).bool()
        exp_sim = exp_sim.masked_fill(mask_self, 0)
        
        # 分子：加权正样本
        pos_weighted = (exp_sim * mask_pos_weight).sum(dim=1)
        
        # 分母：所有样本
        denom = exp_sim.sum(dim=1)
        
        loss = -torch.log((pos_weighted + 1e-8) / (denom + 1e-8))
        return loss.mean()


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
    
    # 训练结束后绘制训练集上的参数预测对比图
    extractor.eval()
    predictor.eval()
    train_preds = []
    train_trues = []
    with torch.no_grad():
        for x, y in tqdm(train_loader, desc="Collecting train predictions"):
            x = x.to(device)
            features = extractor(x)
            preds = predictor(features)
            train_preds.append(preds.cpu())
            train_trues.append(y.cpu())
    
    train_preds = torch.cat(train_preds, dim=0).numpy()
    train_trues = torch.cat(train_trues, dim=0).numpy()
    train_fig = plot_parameter_predictions(train_preds, train_trues, save_path="train_parameter_predictions.png")
    wandb.log({"train_parameter_predictions": wandb.Image(train_fig)})
    
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
