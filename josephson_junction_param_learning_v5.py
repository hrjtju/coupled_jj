"""
Coupled Josephson Junction SDE - v5 (Target: Loss < 1.0)
=========================================================

v4 问题分析及 v5 改进：

1. 损失权重不平衡: sigma 权重 20 太高 → 改为自适应归一化
2. LSTM 可能欠拟合 → 增加到 6 层 + 64 隐藏单元
3. 缺乏直接监督信号 → 添加残差辅助输出
4. 学习率可能不合适 → 使用 OneCycleLR
5. 梯度消失风险 → 添加更多跳跃连接
"""

import os
import sys
import random
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import torchsde

# =============================================================================
# 工具函数
# =============================================================================

def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 物理参数配置
# =============================================================================

class PhysicalParamsConfig:
    PARAM_RANGES = {
        'beta1': (0.05, 0.5), 'beta2': (0.05, 0.5),
        'i1': (0.3, 1.5), 'i2': (0.3, 1.5),
        'kappa1': (0.0, 0.15), 'kappa2': (0.0, 0.15),
        'sigma1': (0.01, 0.05), 'sigma2': (0.01, 0.05),
    }
    
    # 归一化后的权重（使所有参数对损失的贡献大致相等）
    PARAM_LOSS_WEIGHTS = {
        'beta1': 4.0, 'beta2': 4.0,      # 范围 0.45
        'i1': 1.0, 'i2': 1.0,            # 范围 1.2
        'kappa1': 8.0, 'kappa2': 8.0,    # 范围 0.15
        'sigma1': 25.0, 'sigma2': 25.0,  # 范围 0.04
    }
    
    N_RANGE = (100, 400)
    T_RANGE = (5.0, 15.0)
    PARAM_NAMES = ['beta1', 'beta2', 'i1', 'i2', 'kappa1', 'kappa2', 'sigma1', 'sigma2']
    PARAM_NAMES_LATEX = [r'$\beta_{1}$', r'$\beta_{2}$', r'$i_{1}$', r'$i_{2}$',
                         r'$\kappa_{1}$', r'$\kappa_{2}$', r'$\sigma_{1}$', r'$\sigma_{2}$']
    
    @classmethod
    def get_weights(cls):
        return [cls.PARAM_LOSS_WEIGHTS[n] for n in cls.PARAM_NAMES]
    
    @classmethod
    def sample_params(cls, n: int = 1, device: str = 'cpu'):
        params = []
        for _ in range(n):
            p = [random.uniform(*cls.PARAM_RANGES[name]) for name in cls.PARAM_NAMES]
            params.append(p)
        return torch.tensor(params, dtype=torch.float64, device=device)
    
    @classmethod
    def sample_n_and_t(cls, n_samples: int):
        N = np.random.randint(cls.N_RANGE[0], cls.N_RANGE[1] + 1, size=n_samples)
        T = np.random.uniform(cls.T_RANGE[0], cls.T_RANGE[1], size=n_samples)
        return N, T

# =============================================================================
# SDE 模拟器
# =============================================================================

class JosephsonJunctionSDE(nn.Module):
    def __init__(self, params: torch.Tensor):
        super().__init__()
        self.register_buffer('params', params.clone())
        self.sde_type = "ito"
        self.noise_type = "diagonal"
    
    def f(self, t, y):
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        beta1, beta2 = self.params[0], self.params[1]
        i1, i2 = self.params[2], self.params[3]
        kappa1, kappa2 = self.params[4], self.params[5]
        
        d_phi1 = v1
        dv1 = i1 - beta1 * v1 - torch.sin(phi1) + kappa1 * (phi2 - phi1)
        d_phi2 = v2
        dv2 = i2 - beta2 * v2 - torch.sin(phi2) + kappa2 * (phi1 - phi2)
        return torch.stack([d_phi1, dv1, d_phi2, dv2], dim=-1)
    
    def g(self, t, y):
        batch = y.shape[0]
        device = y.device
        noise = torch.zeros(batch, 4, device=device)
        noise[..., 1], noise[..., 3] = self.params[6], self.params[7]
        return noise
    
    def simulate(self, T: float, N: int, y0: torch.Tensor, n_samples: int = 1):
        device = y0.device if torch.is_tensor(y0) else torch.device('cpu')
        times = torch.linspace(0, T, N, device=device)
        dt = T / N
        
        if y0.dim() == 1:
            y0 = y0.unsqueeze(0).expand(n_samples, -1)
        
        with torch.no_grad():
            traj = torchsde.sdeint(self, y0, times, dt=dt, method='euler')
        return times, traj.permute(1, 0, 2)

# =============================================================================
# 数据生成
# =============================================================================

def generate_data(n_samples: int, seed: int = 42, discard_T: float = 10.0, device: str = 'cpu'):
    if seed is not None:
        seed_everything(seed)
    
    print(f"Generating {n_samples} samples...")
    params = PhysicalParamsConfig.sample_params(n_samples, device='cpu')
    N_arr, T_arr = PhysicalParamsConfig.sample_n_and_t(n_samples)
    dt_arr = T_arr / N_arr
    discard_N = np.maximum(1, (discard_T / dt_arr).astype(int))
    
    xs, ys, hs = [], [], []
    
    for i in tqdm(range(n_samples)):
        p = params[i].to(device)
        N, T, dN = int(N_arr[i]), float(T_arr[i]), int(discard_N[i])
        total_N = N + dN
        
        sde = JosephsonJunctionSDE(p).to(device)
        y0 = torch.randn(4, device=device) * 0.5
        _, traj = sde.simulate(total_N * (T / N), total_N, y0, 1)
        
        xs.append(traj[0, dN:, :].cpu().numpy())
        ys.append(p.cpu().tolist())
        hs.append(T)
    
    return {'X': xs, 'Y': ys, 'H': hs}

def save_pickle(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump([data, {}], f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)[0]

# =============================================================================
# 数据集
# =============================================================================

class JSDataset(Dataset):
    def __init__(self, path: str, max_samples: int = -1):
        super().__init__()
        data = load_pickle(path)
        
        X = [torch.from_numpy(x) for x in data['X']]
        Y = torch.tensor(data['Y'])
        H = torch.tensor(data['H']).reshape(-1, 1)
        
        if max_samples > 0:
            X, Y, H = X[:max_samples], Y[:max_samples], H[:max_samples]
        
        self.lengths = [x.shape[0] for x in X]
        self.max_len = max(self.lengths)
        
        # Padding
        self.X = torch.stack([nn.ZeroPad2d((0, 0, 0, self.max_len - x.shape[0]))(x) for x in X])
        self.Y = Y
        self.H = H
        self.Z = torch.stack([nn.ZeroPad2d((0, 0, 0, self.max_len - l))(torch.ones((l, 1))) for l in self.lengths])
        self.L = torch.tensor(self.lengths, dtype=torch.float32).reshape(-1, 1)
        
        print(f"Loaded: X{self.X.shape}, Y{self.Y.shape}")
    
    def get_dim(self):
        return self.X.shape[2], self.Y.shape[1]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.L[idx], self.H[idx], self.Z[idx]
    
    def __len__(self):
        return len(self.X)

# =============================================================================
# v5 模型 - 更深更宽 + 辅助任务
# =============================================================================

class PENNv5(nn.Module):
    """
    v5 改进：
    - 更深的 LSTM (6层)
    - 更宽的隐藏层 (64)
    - 辅助输出头：直接从 LSTM 输出预测（多任务学习）
    - LayerNorm + Dropout
    """
    
    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 6, 
                 out_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        
        self.hidden = hidden
        self.n_layers = n_layers
        
        # LSTM Encoder - 更深
        self.lstm = nn.LSTM(in_dim, hidden, n_layers, batch_first=True, 
                           dropout=dropout, bidirectional=False)
        
        # 辅助头：直接从 LSTM 输出预测（帮助梯度流动）
        self.aux_head = nn.Linear(hidden, out_dim)
        
        # 主网络 - 更宽
        self.fc1 = nn.Linear(hidden + 2, 64)
        self.ln1 = nn.LayerNorm(64)
        
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)
        
        self.fc_out = nn.Linear(32, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.ones = torch.ones((1, hidden))
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                init.orthogonal_(p)
            elif 'bias' in name:
                p.data.zero_()
        
        for m in [self.fc1, self.fc2, self.fc3, self.fc_out, self.aux_head]:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, l, h, z):
        # LSTM
        out, _ = self.lstm(x.double())
        
        # Mask and mean pool
        out = out * z
        out_sum = torch.sum(out, dim=1)
        out = out_sum / torch.mm(l, self.ones.to(x.device))
        
        # 辅助预测（用于多任务损失）
        aux_pred = self.aux_head(out)
        
        # 主网络
        feat = torch.cat([out, h, l], dim=1)
        
        x = F.elu(self.ln1(self.fc1(feat)))
        x = self.dropout(x)
        
        x = F.elu(self.ln2(self.fc2(x))) + x  # 残差
        x = self.dropout(x)
        
        x = F.elu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        main_pred = self.fc_out(x)
        
        return main_pred, aux_pred

# =============================================================================
# v5 损失函数 - 多任务 + 自适应权重
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    多任务损失：
    1. 主预测 L1 损失
    2. 辅助预测 L1 损失（帮助训练稳定性）
    3. 一致性损失（主预测和辅助预测应该接近）
    """
    
    def __init__(self, weights: List[float], aux_weight: float = 0.3, consistency_weight: float = 0.1):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float64))
        self.aux_weight = aux_weight
        self.consistency_weight = consistency_weight
    
    def forward(self, main_pred, aux_pred, target):
        # 主损失（加权 L1）
        diff = torch.abs(main_pred - target).to(self.weights.device)
        main_loss = (diff * self.weights).mean()
        
        # 辅助损失
        aux_loss = (torch.abs(aux_pred - target).to(self.weights.device) * self.weights).mean()
        
        # 一致性损失（主预测和辅助预测应该一致）
        consistency_loss = F.mse_loss(main_pred, aux_pred)
        
        total = main_loss + self.aux_weight * aux_loss + self.consistency_weight * consistency_loss
        
        return {
            'total': total,
            'main': main_loss,
            'aux': aux_loss,
            'consistency': consistency_loss,
        }

# =============================================================================
# 可视化
# =============================================================================

def plot_history(history, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history) + 1)
    
    axes[0, 0].semilogy(epochs, [h['total'] for h in history], 'b-')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(epochs, [h['main'] for h in history], 'g-', label='Main')
    axes[0, 1].semilogy(epochs, [h['aux'] for h in history], 'r--', label='Aux', alpha=0.7)
    axes[0, 1].set_title('Main vs Aux Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, [h['consistency'] for h in history], 'purple')
    axes[1, 0].set_title('Consistency Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率
    axes[1, 1].text(0.5, 0.5, f'Epochs: {len(history)}\nBest: {min([h["total"] for h in history]):.4f}',
                    ha='center', va='center', fontsize=14)
    axes[1, 1].set_title('Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

def plot_preds(pred, true, save_path=None):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, PhysicalParamsConfig.PARAM_NAMES_LATEX)):
        p, t = pred[:, i].numpy(), true[:, i].numpy()
        ax.scatter(t, p, s=1, alpha=0.5)
        
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, 'r--', lw=1)
        ax.set_title(name)
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        
        r2 = 1 - np.sum((p-t)**2) / np.sum((t-t.mean())**2)
        ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

# =============================================================================
# 主程序
# =============================================================================

def main():
    config = {
        'seed': 42,
        'n_all': 2000,
        'eval_ratio': 0.15,
        'test_ratio': 0.20,
        'discard_T': 10.0,
        
        'data_dir': './data/josephson',
        'train_file': './data/josephson/train_v5.pkl',
        'eval_file': './data/josephson/eval_v5.pkl',
        'test_file': './data/josephson/test_v5.pkl',
        
        # 模型参数 - 更深更宽
        'hidden': 64,
        'n_layers': 6,
        'dropout': 0.2,
        
        # 训练参数
        'batch_size': 64,
        'epochs': 2000,
        'lr': 0.002,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # 损失权重
        'loss_weights': PhysicalParamsConfig.get_weights(),
        'aux_weight': 0.3,
        'consistency_weight': 0.1,
    }
    
    device = get_device()
    print(f"Device: {device}")
    seed_everything(config['seed'])
    
    save_dir = os.path.join(config['data_dir'], 'penn_v5')
    os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)
    
    print("=" * 60)
    print("Josephson Junction - v5 (Target: Loss < 1.0)")
    print("=" * 60)
    
    # 数据生成
    if not all(os.path.exists(f) for f in [config['train_file'], config['eval_file'], config['test_file']]):
        print("\nGenerating data...")
        data = generate_data(config['n_all'], config['seed'], config['discard_T'], device)
        
        indices = np.random.permutation(config['n_all'])
        n_train = int(config['n_all'] * (1 - config['test_ratio'] - config['eval_ratio']))
        n_eval = int(config['n_all'] * (1 - config['test_ratio']))
        
        def subset(d, idx):
            return {'X': [d['X'][i] for i in idx], 'Y': [d['Y'][i] for i in idx], 'H': [d['H'][i] for i in idx]}
        
        save_pickle(subset(data, indices[:n_train]), config['train_file'])
        save_pickle(subset(data, indices[n_train:n_eval]), config['eval_file'])
        save_pickle(subset(data, indices[n_eval:]), config['test_file'])
    
    # 加载数据
    print("\nLoading data...")
    train_ds = JSDataset(config['train_file'])
    eval_ds = JSDataset(config['eval_file'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=len(eval_ds))
    
    in_dim, out_dim = train_ds.get_dim()
    print(f"Input: {in_dim}, Output: {out_dim}")
    
    # 模型
    model = PENNv5(in_dim, config['hidden'], config['n_layers'], out_dim, config['dropout']).to(device)
    model = model.double()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器 - OneCycleLR 更好的收敛
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr'], epochs=config['epochs'],
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    criterion = MultiTaskLoss(config['loss_weights'], config['aux_weight'], config['consistency_weight'])
    
    # 加载 eval 数据
    for x_eval, y_eval, l_eval, h_eval, z_eval in eval_loader:
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        l_eval = l_eval.to(device)
        h_eval = h_eval.to(device)
        z_eval = z_eval.to(device)
    
    # 训练
    print("\nTraining...")
    history = []
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        start = time.time()
        model.train()
        
        epoch_losses = []
        for x, y, l, h, z in train_loader:
            x, y, l, h, z = x.to(device), y.to(device), l.to(device), h.to(device), z.to(device)
            
            main_pred, aux_pred = model(x, l, h, z)
            losses = criterion(main_pred, aux_pred, y)
            
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append({k: v.item() for k, v in losses.items()})
        
        # Eval
        model.eval()
        with torch.no_grad():
            main_pred, aux_pred = model(x_eval, l_eval, h_eval, z_eval)
            eval_losses = criterion(main_pred, aux_pred, y_eval)
            eval_loss = {k: v.item() for k, v in eval_losses.items()}
        
        train_loss = {k: sum([l[k] for l in epoch_losses]) / len(epoch_losses) for k in epoch_losses[0]}
        history.append(train_loss)
        
        if (epoch + 1) % 10 == 0 or epoch < 3:
            print(f"Epoch {epoch+1}/{config['epochs']} | "
                  f"Train: {train_loss['total']:.4f} | "
                  f"Eval: {eval_loss['total']:.4f} | "
                  f"Time: {time.time()-start:.1f}s")
        
        # Save best
        if eval_loss['total'] < best_loss:
            best_loss = eval_loss['total']
            patience_counter = 0
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 
                       'optimizer': optimizer.state_dict(), 'loss': best_loss, 'config': config},
                      os.path.join(save_dir, 'model', 'best.ckpt'))
        else:
            patience_counter += 1
        
        if best_loss < 1.0:
            print(f"\n✓ Target achieved! Loss = {best_loss:.4f} at epoch {epoch+1}")
            break
        
        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break
    
    # Final eval
    print("\n" + "=" * 60)
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 60)
    
    checkpoint = torch.load(os.path.join(save_dir, 'model', 'best.ckpt'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    with torch.no_grad():
        main_pred, _ = model(x_eval, l_eval, h_eval, z_eval)
    
    print("\nPer-parameter MAE:")
    for i, name in enumerate(PhysicalParamsConfig.PARAM_NAMES):
        mae = torch.mean(torch.abs(main_pred[:, i] - y_eval[:, i])).item()
        print(f"  {name}: {mae:.6f}")
    
    plot_history(history, os.path.join(save_dir, 'history.png'))
    plot_preds(main_pred.cpu(), y_eval.cpu(), os.path.join(save_dir, 'preds.png'))
    
    print(f"\nResults saved to: {save_dir}")
    return model, history

def test(model_file):
    device = get_device()
    ckpt = torch.load(model_file, map_location=device)
    config = ckpt['config']
    
    test_ds = JSDataset(config['test_file'])
    test_loader = DataLoader(test_ds, batch_size=256)
    
    in_dim, out_dim = test_ds.get_dim()
    model = PENNv5(in_dim, config['hidden'], config['n_layers'], out_dim).to(device)
    model = model.double()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    preds, targets = [], []
    with torch.no_grad():
        for x, y, l, h, z in test_loader:
            x, y, l, h, z = x.to(device), y.to(device), l.to(device), h.to(device), z.to(device)
            p, _ = model(x, l, h, z)
            preds.append(p.cpu())
            targets.append(y.cpu())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    print("\nTest Results:")
    for i, name in enumerate(PhysicalParamsConfig.PARAM_NAMES):
        mae = torch.mean(torch.abs(preds[:, i] - targets[:, i])).item()
        r2 = 1 - torch.sum((preds[:, i] - targets[:, i])**2) / torch.sum((targets[:, i] - targets[:, i].mean())**2)
        print(f"  {name}: MAE={mae:.6f}, R²={r2:.4f}")
    
    total_mae = torch.mean(torch.abs(preds - targets)).item()
    print(f"\nTotal MAE: {total_mae:.4f}")
    
    plot_preds(preds, targets, os.path.join(os.path.dirname(model_file), 'test_preds.png'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    
    if args.mode == 'train':
        main()
    else:
        test(args.model)
