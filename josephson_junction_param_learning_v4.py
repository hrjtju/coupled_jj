"""
Coupled Josephson Junction SDE 参数学习代码 - v4 (稳定改进版)
=================================================================

基于 v3 的问题修复：

v3 的问题分析：
1. 周期性编码可能破坏了原始数据的时序结构
2. 二次变差计算数值不稳定
3. sigma_weight=10 太强，导致优化失衡
4. 物理正则化约束可能不适合

v4 的改进策略：
1. 回滚周期性编码，保持原始 4 维输入
2. 简化二次变差计算，降低权重 (sigma_weight=1.0)
3. 移除物理正则化，专注标准 L1 + 轻量 sigma 约束
4. 添加学习率调度 (ReduceLROnPlateau)
5. 增加梯度裁剪，防止梯度爆炸
6. 改进 LSTM 初始化
7. 添加残差连接帮助梯度流动

作者: Assistant
日期: 2026-04-09
版本: v4 (稳定改进版)
"""

import os
import sys
import random
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Optional, List, Union
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import torchsde

# =============================================================================
# 全局配置与工具函数
# =============================================================================

def seed_everything(seed: int):
    """设置所有随机种子以确保结果可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """获取可用的计算设备（CUDA 或 CPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 物理参数配置
# =============================================================================

class PhysicalParamsConfig:
    """物理参数的采样范围配置"""

    PARAM_RANGES = {
        'beta1': (0.05, 0.5),
        'beta2': (0.05, 0.5),
        'i1': (0.3, 1.5),
        'i2': (0.3, 1.5),
        'kappa1': (0.0, 0.15),
        'kappa2': (0.0, 0.15),
        'sigma1': (0.01, 0.05),
        'sigma2': (0.01, 0.05),
    }

    PARAM_LOSS_WEIGHTS = {
        'beta1': 2.0,
        'beta2': 2.0,
        'i1': 1.0,
        'i2': 1.0,
        'kappa1': 6.0,
        'kappa2': 6.0,
        'sigma1': 20.0,
        'sigma2': 20.0,
    }

    N_RANGE = (100, 400)
    T_RANGE = (5.0, 15.0)
    PARAM_NAMES = ['beta1', 'beta2', 'i1', 'i2', 'kappa1', 'kappa2', 'sigma1', 'sigma2']
    PARAM_NAMES_LATEX = [
        r'$\beta_{J_1}$', r'$\beta_{J_2}$',
        r'$i_1$', r'$i_2$',
        r'$\kappa_1$', r'$\kappa_2$',
        r'$\sigma_1$', r'$\sigma_2$'
    ]

    @classmethod
    def get_param_weights(cls):
        """延迟计算权重，避免类定义时的 NameError"""
        return [cls.PARAM_LOSS_WEIGHTS[name] for name in cls.PARAM_NAMES]

    @classmethod
    def sample_params(cls, n: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, List[Dict]]:
        """随机采样 n 组物理参数"""
        params_dicts = []
        params_list = []

        for _ in range(n):
            params_dict = {}
            for name, (min_val, max_val) in cls.PARAM_RANGES.items():
                params_dict[name] = random.uniform(min_val, max_val)
            params_dicts.append(params_dict)
            params_list.append([
                params_dict['beta1'], params_dict['beta2'],
                params_dict['i1'], params_dict['i2'],
                params_dict['kappa1'], params_dict['kappa2'],
                params_dict['sigma1'], params_dict['sigma2'],
            ])

        params_tensor = torch.tensor(params_list, dtype=torch.float64, device=device)
        return params_tensor, params_dicts

    @classmethod
    def sample_n_and_t(cls, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """随机采样轨迹长度 N 和时间跨度 T"""
        N = np.random.randint(cls.N_RANGE[0], cls.N_RANGE[1] + 1, size=n_samples)
        T = np.random.uniform(cls.T_RANGE[0], cls.T_RANGE[1], size=n_samples)
        return N, T


# =============================================================================
# Josephson Junction SDE 模拟器
# =============================================================================

class JosephsonJunctionSDE(nn.Module):
    """耦合约瑟夫森结 SDE 模拟器"""

    def __init__(self, params: torch.Tensor):
        super().__init__()
        self.register_buffer('params', params.clone())
        self.sde_type = "ito"
        self.noise_type = "diagonal"

    @property
    def beta1(self): return self.params[0]
    @property
    def beta2(self): return self.params[1]
    @property
    def i1(self): return self.params[2]
    @property
    def i2(self): return self.params[3]
    @property
    def kappa1(self): return self.params[4]
    @property
    def kappa2(self): return self.params[5]
    @property
    def sigma1(self): return self.params[6]
    @property
    def sigma2(self): return self.params[7]

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """漂移项"""
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        d_phi1 = v1
        dv1 = (self.i1 - self.beta1 * v1 - torch.sin(phi1) + self.kappa1 * (phi2 - phi1))
        d_phi2 = v2
        dv2 = (self.i2 - self.beta2 * v2 - torch.sin(phi2) + self.kappa2 * (phi1 - phi2))
        return torch.stack([d_phi1, dv1, d_phi2, dv2], dim=-1)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """扩散项"""
        batch_size = y.shape[0]
        device = y.device
        noise = torch.zeros(batch_size, 4, device=device)
        noise[..., 1] = self.sigma1
        noise[..., 3] = self.sigma2
        return noise

    def simulate(self, T: float, N: int, initial_state: torch.Tensor,
                 n_samples: int = 1, method: str = 'euler') -> Tuple[torch.Tensor, torch.Tensor]:
        """模拟 SDE"""
        device = initial_state.device if torch.is_tensor(initial_state) else torch.device('cpu')
        times = torch.linspace(0, T, N, device=device)
        dt = T / N

        if initial_state.dim() == 1:
            y0 = initial_state.unsqueeze(0).expand(n_samples, -1)
        else:
            y0 = initial_state

        with torch.no_grad():
            trajectories = torchsde.sdeint(
                sde=self, y0=y0, ts=times, dt=dt, method=method
            )
        trajectories = trajectories.permute(1, 0, 2)
        return times, trajectories


# =============================================================================
# 数据生成与处理
# =============================================================================

def generate_trajectory_data(n_samples: int, seed: Optional[int] = None,
                             discard_T: float = 10.0, device: str = 'cpu') -> Dict:
    """生成轨迹数据"""
    if seed is not None:
        seed_everything(seed)

    print(f"Generating {n_samples} trajectory samples...")

    params_tensor, params_dicts = PhysicalParamsConfig.sample_params(n_samples, device='cpu')
    N_array, T_array = PhysicalParamsConfig.sample_n_and_t(n_samples)
    dt_array = T_array / N_array
    discard_N_array = np.maximum(1, (discard_T / dt_array).astype(int))

    xs, ys, T_list = [], [], []
    time_start = time.time()
    NUM_FOR_SHOW = max(1, n_samples // 10)

    for i in tqdm(range(n_samples)):
        if (i + 1) % NUM_FOR_SHOW == 0:
            time_end = time.time()
            time_used = time_end - time_start
            time_left = time_used / (i + 1) * (n_samples - i)
            print(f'{i+1}/{n_samples}, {time_used:.1f}s used, {time_left:.1f}s left')

        params = params_tensor[i].to(device)
        N = int(N_array[i])
        T = float(T_array[i])
        discard_N = int(discard_N_array[i])
        total_N = N + discard_N

        sde = JosephsonJunctionSDE(params).to(device)
        initial_state = torch.randn(4, device=device) * 0.5

        times, trajectories = sde.simulate(
            T=total_N * (T / N), N=total_N,
            initial_state=initial_state, n_samples=1, method='euler'
        )

        trajectory = trajectories[0, discard_N:, :].cpu().numpy()
        xs.append(trajectory)
        ys.append([params[j].item() for j in range(8)])
        T_list.append(T)

    return {'X': xs, 'Y': ys, 'H': T_list}


def save_data_pickle(data_dict: Dict, save_path: str):
    """保存数据"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump([data_dict, {}], f)
    print(f"Data saved to: {save_path}")


def load_data_pickle(file_path: str) -> Tuple[Dict, Dict]:
    """加载数据"""
    with open(file_path, 'rb') as f:
        data_dict, param_dic = pickle.load(f)
    return data_dict, param_dic


class MyDataset(Dataset):
    """数据集类 - 保持原始 4 维输入"""

    def __init__(self, file_name: str, max_sample: int = -1):
        super().__init__()
        print('Loading data:', file_name)

        with open(file_name, "rb") as fp:
            data_dic, self.param_dic = pickle.load(fp)

        X = [torch.from_numpy(x) for x in data_dic['X']]
        Y = data_dic['Y']
        H = data_dic['H']

        if max_sample > 0:
            X, Y, H = X[:max_sample], Y[:max_sample], H[:max_sample]

        self.lengths = [_.shape[0] for _ in X]
        self.max_lengths = max(self.lengths)

        # 原始 4 维输入，不使用周期性编码
        self.X = torch.stack([
            nn.ZeroPad2d((0, 0, 0, self.max_lengths - _.shape[0]))(_)
            for _ in X
        ])

        self.Y = torch.from_numpy(np.array(Y))
        self.H = torch.from_numpy(np.array(H).reshape((-1, 1)))
        self.Z = torch.stack([
            nn.ZeroPad2d((0, 0, 0, self.max_lengths - i))(torch.ones((i, 1)))
            for i in self.lengths
        ])
        self.lengths = torch.from_numpy(np.reshape(np.array(self.lengths), (-1, 1))).float()

        print(f'  X shape: {self.X.shape}, Y shape: {self.Y.shape}')

    def get_dim(self) -> Tuple[int, int]:
        return self.X.shape[2], self.Y.shape[1]

    def __getitem__(self, index: int):
        return (self.X[index], self.Y[index], self.lengths[index],
                self.H[index], self.Z[index])

    def __len__(self) -> int:
        return len(self.X)


# =============================================================================
# v4 改进版 PENN 模型
# =============================================================================

class PENNv4(nn.Module):
    """
    v4 改进版 PENN

    改进点：
    1. 使用 LayerNorm 替代 BatchNorm（对变长序列更稳定）
    2. 添加残差连接
    3. 改进初始化
    4. 使用更宽的隐藏层
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layer: int,
                 n_class: int, activation: str = 'elu', dropout: float = 0.1):
        super().__init__()

        self.type = 'LSTM'
        self.activation = activation
        self.init_param = [in_dim, hidden_dim, n_layer, n_class]
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim

        # LSTM - 使用 dropout
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True,
                           dropout=dropout if n_layer > 1 else 0)

        # FCNN - 使用 LayerNorm 和残差连接
        self.nn_size = 32  # 增加宽度

        # 第一层
        self.fc1 = nn.Linear(hidden_dim + 2, self.nn_size)
        self.ln1 = nn.LayerNorm(self.nn_size)

        # 隐藏层（带残差）
        self.fc2 = nn.Linear(self.nn_size, self.nn_size)
        self.ln2 = nn.LayerNorm(self.nn_size)
        self.fc3 = nn.Linear(self.nn_size, self.nn_size)
        self.ln3 = nn.LayerNorm(self.nn_size)

        # 输出层
        self.fc_out = nn.Linear(self.nn_size, n_class)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 辅助张量
        self.ones = torch.ones((1, self.hidden_dim))

        self.init_weights()

    def init_weights(self):
        """改进的权重初始化"""
        # LSTM 使用 Xavier 初始化
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                param.data.zero_()

        # FC 使用 He 初始化
        for m in [self.fc1, self.fc2, self.fc3, self.fc_out]:
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """激活函数"""
        if self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leakyrelu':
            return F.leaky_relu(x, 0.1)
        else:
            return torch.tanh(x)

    def forward(self, x: torch.Tensor, l: torch.Tensor, h: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # LSTM
        _out, _ = self.lstm(x.to(torch.float64))
        out = _out * z
        out = torch.sum(out, dim=1)
        out = torch.div(out, torch.mm(l, self.ones.to(x.device)))

        # 拼接时间信息
        out = torch.cat([out, h, l], dim=1)

        # FCNN with residual
        out = self.fc1(out)
        out = self.ln1(out)
        out = self._activation(out)
        out = self.dropout(out)

        # Residual block 1
        residual = out
        out = self.fc2(out)
        out = self.ln2(out)
        out = self._activation(out)
        out = self.dropout(out)
        out = out + residual  # 残差连接

        # Residual block 2
        residual = out
        out = self.fc3(out)
        out = self.ln3(out)
        out = self._activation(out)
        out = self.dropout(out)
        out = out + residual  # 残差连接

        # 输出
        out = self.fc_out(out)
        return out


# =============================================================================
# 改进的损失函数
# =============================================================================

class StableLoss(nn.Module):
    """
    稳定的复合损失函数

    简化设计：
    1. 标准加权 L1 损失（所有参数）
    2. 轻量 sigma 一致性损失（权重 1.0，不强推）
    """

    def __init__(self, weight: List[float], sigma_weight: float = 1.0):
        super().__init__()
        self.weight = torch.tensor(weight, dtype=torch.float64)
        self.sigma_weight = sigma_weight

    def compute_sigma_target(self, x: torch.Tensor, l: torch.Tensor,
                            pred_params: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        简化的 sigma 目标计算
        使用速度差分的标准差作为 sigma 的估计
        """
        batch_size = x.shape[0]
        dt = (h / l).mean().item()  # 使用平均 dt

        # 提取 v1, v2
        v1, v2 = x[..., 1], x[..., 3]

        # 计算速度差分（使用更稳定的计算方式）
        dv1 = torch.diff(v1, dim=1) if v1.shape[1] > 1 else torch.zeros_like(v1[:, :1])
        dv2 = torch.diff(v2, dim=1) if v2.shape[1] > 1 else torch.zeros_like(v2[:, :1])

        # 使用标准差估计 sigma（更稳定）
        # sigma ≈ std(dv) / sqrt(dt)
        sigma1_est = torch.std(dv1, dim=1, unbiased=False) / np.sqrt(dt)
        sigma2_est = torch.std(dv2, dim=1, unbiased=False) / np.sqrt(dt)

        # 限制范围，避免极端值
        sigma1_est = torch.clamp(sigma1_est, 0.001, 0.5)
        sigma2_est = torch.clamp(sigma2_est, 0.001, 0.5)

        return torch.stack([sigma1_est, sigma2_est], dim=1).to(x.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                x: torch.Tensor, l: torch.Tensor, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失"""
        self.weight = self.weight.to(pred.device)

        # 1. 标准 L1 损失（加权）
        diff = torch.abs(pred - target)
        weighted_diff = diff * self.weight.unsqueeze(0)
        standard_loss = weighted_diff.mean()

        # 各参数单独损失（用于监控）
        individual = [diff[:, i].mean() for i in range(8)]

        # 2. Sigma 一致性损失（轻量）
        with torch.no_grad():
            sigma_target = self.compute_sigma_target(x, l, pred, h)

        sigma_pred = pred[:, 6:8]
        sigma_target = sigma_target.to(pred.dtype)
        sigma_consistency = F.mse_loss(sigma_pred, sigma_target)

        # 总损失
        total = standard_loss + self.sigma_weight * sigma_consistency

        return {
            'total': total,
            'standard': standard_loss,
            'sigma_consistency': sigma_consistency,
            'individual': individual
        }


# =============================================================================
# 可视化函数
# =============================================================================

def plot_training_history(history: List[Dict], save_path: str = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history) + 1)

    # 总损失
    ax = axes[0, 0]
    ax.plot(epochs, [h['total'] for h in history], 'b-', label='Total')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 标准损失
    ax = axes[0, 1]
    ax.plot(epochs, [h['standard'] for h in history], 'g-', label='Standard')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Standard L1 Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Sigma 一致性
    ax = axes[1, 0]
    ax.plot(epochs, [h['sigma_consistency'] for h in history], 'r-', label='Sigma')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Sigma Consistency Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 各参数损失
    ax = axes[1, 1]
    for i, name in enumerate(PhysicalParamsConfig.PARAM_NAMES):
        ax.plot(epochs, [h['individual'][i] for h in history],
                label=name, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Per-Parameter Loss')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_predictions(pred: torch.Tensor, true: torch.Tensor, save_path: str = None):
    """绘制预测对比"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, PhysicalParamsConfig.PARAM_NAMES_LATEX)):
        p = pred[:, i].numpy()
        t = true[:, i].numpy()

        ax.scatter(t, p, c='r', s=1, alpha=0.5)

        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, 'b--', lw=1)

        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)

        # R²
        r2 = 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
        mae = np.mean(np.abs(p - t))
        ax.text(0.05, 0.95, f'R²={r2:.3f}\nMAE={mae:.4f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主训练函数"""

    config = {
        'seed': 42,
        'n_all': 2000,
        'eval_ratio': 0.15,
        'test_ratio': 0.20,
        'discard_T': 10.0,

        'data_dir': './data/josephson',
        'train_file': './data/josephson/train_v4.pkl',
        'eval_file': './data/josephson/eval_v4.pkl',
        'test_file': './data/josephson/test_v4.pkl',

        # 网络参数
        'lstm_layers': 4,
        'lstm_hidden': 32,  # 增加隐藏维度
        'fc_hidden': 32,
        'activation': 'elu',
        'dropout': 0.1,

        # 训练参数
        'batch_size': 64,
        'num_epochs': 2000,
        'learning_rate': 0.001,
        'lr_patience': 10,  # 学习率调度耐心值
        'lr_factor': 0.5,   # 学习率衰减因子
        'weight_decay': 1e-5,
        'grad_clip': 1.0,   # 梯度裁剪阈值

        # 损失权重
        'loss_weight': PhysicalParamsConfig.get_param_weights(),
        'sigma_weight': 1.0,  # 降低 sigma 权重

        # 保存
        'save_every': 100,
    }

    device = get_device()
    print(f"Device: {device}")
    seed_everything(config['seed'])

    save_dir = os.path.join(config['data_dir'], 'penn_v4')
    os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)

    print("=" * 60)
    print("Josephson Junction Parameter Learning - v4")
    print("=" * 60)

    # 生成数据
    if not all(os.path.exists(f) for f in [config['train_file'], config['eval_file'], config['test_file']]):
        print("\nGenerating data...")
        data = generate_trajectory_data(config['n_all'], config['seed'], config['discard_T'], device)

        indices = np.random.permutation(config['n_all'])
        n_train = int(config['n_all'] * (1 - config['test_ratio'] - config['eval_ratio']))
        n_eval = int(config['n_all'] * (1 - config['test_ratio']))

        def subset(d, idx):
            return {'X': [d['X'][i] for i in idx], 'Y': [d['Y'][i] for i in idx], 'H': [d['H'][i] for i in idx]}

        save_data_pickle(subset(data, indices[:n_train]), config['train_file'])
        save_data_pickle(subset(data, indices[n_train:n_eval]), config['eval_file'])
        save_data_pickle(subset(data, indices[n_eval:]), config['test_file'])

    # 加载数据
    print("\nLoading datasets...")
    train_data = MyDataset(config['train_file'])
    eval_data = MyDataset(config['eval_file'])

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                              shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_data, batch_size=len(eval_data))

    in_dim, n_class = train_data.get_dim()
    print(f"Input dim: {in_dim}, Output dim: {n_class}")

    # 创建模型
    model = PENNv4(in_dim, config['lstm_hidden'], config['lstm_layers'],
                   n_class, config['activation'], config['dropout']).to(device)
    model = model.double()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器和学习率调度
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_factor'],
        patience=config['lr_patience'], verbose=True
    )

    # 损失函数
    criterion = StableLoss(config['loss_weight'], config['sigma_weight'])

    # 加载评估数据
    for x_eval, y_eval, l_eval, h_eval, z_eval in eval_loader:
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        l_eval = l_eval.to(device)
        h_eval = h_eval.to(device)
        z_eval = z_eval.to(device)

    # 训练循环
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    history = []
    best_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        start_time = time.time()
        model.train()

        epoch_losses = []
        for x, y, l, h, z in train_loader:
            x, y, l, h, z = x.to(device), y.to(device), l.to(device), h.to(device), z.to(device)

            pred = model(x, l, h, z)
            losses = criterion(pred, y, x, l, h)

            optimizer.zero_grad()
            losses['total'].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()
            epoch_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v
                                for k, v in losses.items()})

        # 评估
        model.eval()
        with torch.no_grad():
            eval_pred = model(x_eval, l_eval, h_eval, z_eval)
            eval_losses = criterion(eval_pred, y_eval, x_eval, l_eval, h_eval)
            eval_loss = {k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in eval_losses.items()}

        # 平均训练损失
        train_loss = {
            'total': np.mean([l['total'] for l in epoch_losses]),
            'standard': np.mean([l['standard'] for l in epoch_losses]),
            'sigma_consistency': np.mean([l['sigma_consistency'] for l in epoch_losses]),
            'individual': [np.mean([l['individual'][i] for l in epoch_losses]) for i in range(8)]
        }

        history.append(train_loss)

        # 学习率调度
        scheduler.step(eval_loss['total'])

        # 打印
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                  f"Train: {train_loss['total']:.4f} | "
                  f"Eval: {eval_loss['total']:.4f} | "
                  f"Time: {time.time()-start_time:.1f}s | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if eval_loss['total'] < best_loss:
            best_loss = eval_loss['total']
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, os.path.join(save_dir, 'model', 'best.ckpt'))
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # 检查收敛目标
        if best_loss < 1.0:
            print(f"\n✓ Target reached! Loss < 1.0 at epoch {epoch + 1}")
            break

        # 定期保存
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config
            }, os.path.join(save_dir, 'model', f'epoch_{epoch+1}.ckpt'))

    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_dir, 'model', 'best.ckpt'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    with torch.no_grad():
        final_pred = model(x_eval, l_eval, h_eval, z_eval).cpu()
        final_true = y_eval.cpu()

    print("\nPer-parameter MAE:")
    for i, name in enumerate(PhysicalParamsConfig.PARAM_NAMES):
        mae = torch.mean(torch.abs(final_pred[:, i] - final_true[:, i])).item()
        print(f"  {name}: {mae:.6f}")

    print(f"\nBest eval loss: {best_loss:.4f}")

    # 可视化
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    plot_predictions(final_pred, final_true, os.path.join(save_dir, 'predictions.png'))

    # 保存结果
    with open(os.path.join(save_dir, 'results.pkl'), 'wb') as f:
        pickle.dump({
            'pred': final_pred.numpy(),
            'true': final_true.numpy(),
            'history': history,
            'best_loss': best_loss
        }, f)

    print(f"\nResults saved to: {save_dir}")
    return model, history


def test(model_file: str, test_file: str = None):
    """测试模型"""
    device = get_device()

    checkpoint = torch.load(model_file, map_location=device)
    config = checkpoint['config']

    if test_file is None:
        test_file = config['test_file']

    test_data = MyDataset(test_file)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    in_dim, n_class = test_data.get_dim()
    model = PENNv4(in_dim, config['lstm_hidden'], config['lstm_layers'],
                   n_class, config['activation']).to(device)
    model = model.double()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    all_pred, all_true = [], []

    with torch.no_grad():
        for x, y, l, h, z in test_loader:
            x, y, l, h, z = x.to(device), y.to(device), l.to(device), h.to(device), z.to(device)
            pred = model(x, l, h, z)
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())

    pred = torch.cat(all_pred, dim=0)
    true = torch.cat(all_true, dim=0)

    print("\nTest Results:")
    for i, name in enumerate(PhysicalParamsConfig.PARAM_NAMES):
        mae = torch.mean(torch.abs(pred[:, i] - true[:, i])).item()
        r2 = 1 - torch.sum((pred[:, i] - true[:, i])**2) / torch.sum((true[:, i] - true[:, i].mean())**2)
        print(f"  {name}: MAE={mae:.6f}, R²={r2:.4f}")

    total_mae = torch.mean(torch.abs(pred - true)).item()
    print(f"\nTotal MAE: {total_mae:.4f}")

    plot_predictions(pred, true, os.path.join(os.path.dirname(model_file), 'test_predictions.png'))

    return pred, true


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--testfile', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        main()
    else:
        test(args.model, args.testfile)
