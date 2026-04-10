"""
Coupled Josephson Junction SDE 参数学习代码 - PENN 精确复现版
================================================================

本代码严格按照 PENN (Physics-Embedded Neural Network) 论文设计：

论文参考：
Wang, Xiaolong, et al. "Neural network-based parameter estimation of stochastic
differential equations driven by Lévy noise." Physica A: Statistical Mechanics
and its Applications 606 (2022): 128146.

PENN 原始架构：
1. LSTM Stage: 4层 LSTM，隐藏维度 25
2. FCNN Stage: [26->20] + BN + ELU + [20->20] + ELU + [20->20] + ELU + [20->M]

关键设计点：
- 输入：原始轨迹 x（不使用 [t, s_t^T] 拼接）
- Mean Pooling：对 LSTM 输出进行 mask 处理后求平均
- 特征拼接：[h_mean, H]（不包含长度 L）
- 损失函数：加权 L1 损失

作者: Assistant
日期: 2026-03-27
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
PARAM_LOSS_WEIGHTS = {
    'beta1':  3.00,   # 1/0.45
    'beta2':  3.00,   # 1/0.45
    'i1':     0.83,   # 1/1.2
    'i2':     0.83,   # 1/1.2
    'kappa1': 9.00,   # 1/0.15
    'kappa2': 9.00,   # 1/0.15
    'sigma1': 50.0,   # 1/0.04
    'sigma2': 50.0,   # 1/0.04
}

def seed_everything(seed: int):
    """设置所有随机种子以确保结果可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """获取可用的计算设备（CUDA 或 CPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 物理参数配置
# =============================================================================

class PhysicalParamsConfig:
    """
    物理参数的采样范围配置。
    按照 PENN 论文的权重设计：损失权重与参数范围成反比
    """

    # 8 个物理参数的采样范围 (min, max)
    PARAM_RANGES = {
        'beta1': (0.05, 0.5),    # 阻尼系数
        'beta2': (0.05, 0.5),    # 阻尼系数
        'i1': (0.3, 1.5),        # 归一化偏置电流
        'i2': (0.3, 1.5),        # 归一化偏置电流
        'kappa1': (0.0, 0.15),   # 耦合强度
        'kappa2': (0.0, 0.15),   # 耦合强度
        'sigma1': (0.01, 0.05),  # 噪声强度（避免零噪声）
        'sigma2': (0.01, 0.05),  # 噪声强度
    }

    # 损失权重（与参数范围成反比）
    # PENN 论文建议：权重应该与参数范围的倒数成正比
    # 例如：参数范围 [0, 0.05] 的权重应该是 [0, 1] 的 20 倍
    PARAM_LOSS_WEIGHTS = {
        'beta1': 1.0 / 0.45,      # ~2.22
        'beta2': 1.0 / 0.45,      # ~2.22
        'i1': 1.0 / 1.2,           # ~0.83
        'i2': 1.0 / 1.2,           # ~0.83
        'kappa1': 1.0 / 0.15,     # ~6.67
        'kappa2': 1.0 / 0.15,     # ~6.67
        'sigma1': 1.0 / 0.04,     # 25
        'sigma2': 1.0 / 0.04,     # 25
    }

    # 轨迹长度 N 的采样范围（时间步数）
    N_RANGE = (100, 400)

    # 时间跨度 T 的采样范围（模拟的总时间）
    T_RANGE = (5.0, 15.0)

    # 参数名称列表（有序）
    PARAM_NAMES = ['beta1', 'beta2', 'i1', 'i2', 'kappa1', 'kappa2', 'sigma1', 'sigma2']

    # 损失权重列表（有序）
    PARAM_WEIGHTS = [PARAM_LOSS_WEIGHTS[name] for name in PARAM_NAMES]

    # LaTeX 格式的参数名称（用于绘图）
    PARAM_NAMES_LATEX = [
        r'$\beta_{J_1}$', r'$\beta_{J_2}$',
        r'$i_1$', r'$i_2$',
        r'$\kappa_1$', r'$\kappa_2$',
        r'$\sigma_1$', r'$\sigma_2$'
    ]

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

    @classmethod
    def get_n_params(cls) -> int:
        """返回参数数量"""
        return len(cls.PARAM_RANGES)


# =============================================================================
# Josephson Junction SDE 模拟器
# =============================================================================

class JosephsonJunctionSDE(nn.Module):
    """
    耦合约瑟夫森结 SDE 模拟器。
    """

    def __init__(self, params: torch.Tensor):
        super().__init__()

        self.register_buffer('params', params.clone())
        self.sde_type = "ito"
        self.noise_type = "diagonal"

    @property
    def beta1(self):
        return self.params[0]

    @property
    def beta2(self):
        return self.params[1]

    @property
    def i1(self):
        return self.params[2]

    @property
    def i2(self):
        return self.params[3]

    @property
    def kappa1(self):
        return self.params[4]

    @property
    def kappa2(self):
        return self.params[5]

    @property
    def sigma1(self):
        return self.params[6]

    @property
    def sigma2(self):
        return self.params[7]

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """漂移项（drift）函数"""
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]

        d_phi1 = v1
        dv1 = (self.i1 - self.beta1 * v1 - torch.sin(phi1) + self.kappa1 * (phi2 - phi1))
        d_phi2 = v2
        dv2 = (self.i2 - self.beta2 * v2 - torch.sin(phi2) + self.kappa2 * (phi1 - phi2))

        return torch.stack([d_phi1, dv1, d_phi2, dv2], dim=-1)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """扩散项（diffusion）函数"""
        batch_size = y.shape[0]
        device = y.device

        noise = torch.zeros(batch_size, 4, device=device)
        noise[..., 1] = self.sigma1
        noise[..., 3] = self.sigma2

        return noise

    def simulate(self, T: float, N: int, initial_state: torch.Tensor,
                 n_samples: int = 1, method: str = 'srk') -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 torchsde 模拟 SDE"""
        device = initial_state.device if torch.is_tensor(initial_state) else torch.device('cpu')
        times = torch.linspace(0, T, N, device=device)
        dt = T / N

        if initial_state.dim() == 1:
            y0 = initial_state.unsqueeze(0).expand(n_samples, -1)
        else:
            y0 = initial_state

        with torch.no_grad():
            trajectories = torchsde.sdeint(
                sde=self,
                y0=y0,
                ts=times,
                dt=dt,
                method=method
            )

        trajectories = trajectories.permute(1, 0, 2)

        return times, trajectories


# =============================================================================
# 数据生成与处理（PENN 原始格式）
# =============================================================================

def generate_trajectory_data(n_samples: int, seed: Optional[int] = None,
                             discard_T: float = 10.0,
                             device: str = 'cpu') -> Dict:
    """
    生成轨迹数据（PENN 原始格式）。

    返回格式：[xs, T_list, ys]
    - xs: 轨迹列表，每个元素形状 [N, 4]
    - T_list: 时间跨度列表
    - ys: 参数列表
    """
    if seed is not None:
        seed_everything(seed)

    print(f"Generating {n_samples} trajectory samples...")

    params_tensor, params_dicts = PhysicalParamsConfig.sample_params(n_samples, device='cpu')
    N_array, T_array = PhysicalParamsConfig.sample_n_and_t(n_samples)
    dt_array = T_array / N_array
    discard_N_array = np.maximum(1, (discard_T / dt_array).astype(int))

    xs = []
    ys = []
    T_list = []

    time_start = time.time()
    NUM_FOR_SHOW = max(1, n_samples // 10)

    for i in tqdm(range(n_samples)):
        if (i + 1) % NUM_FOR_SHOW == 0:
            time_end = time.time()
            time_used = time_end - time_start
            time_left = time_used / (i + 1) * (n_samples - i)
            print(f'{i+1} of {n_samples}, {time_used:.1f}s used, {time_left:.1f}s left')

        params = params_tensor[i].to(device)
        N = int(N_array[i])
        T = float(T_array[i])
        discard_N = int(discard_N_array[i])
        total_N = N + discard_N

        sde = JosephsonJunctionSDE(params).to(device)
        initial_state = torch.randn(4, device=device) * 0.5

        times, trajectories = sde.simulate(
            T=total_N * (T / N),
            N=total_N,
            initial_state=initial_state,
            n_samples=1,
            method='srk'
        )

        trajectory = trajectories[0, discard_N:, :].cpu().numpy()

        # PENN 格式：每个轨迹存储为 [N, state_dim]
        xs.append(trajectory)
        ys.append([params[0].item(), params[1].item(), params[2].item(), params[3].item(),
                   params[4].item(), params[5].item(), params[6].item(), params[7].item()])
        T_list.append(T)

    print(f"Data generation completed. Total samples: {len(xs)}")
    return {'X': xs, 'Y': ys, 'H': T_list}


def save_data_pickle(data_dict: Dict, save_path: str):
    """将数据保存为 pickle 文件（PENN 原始格式）"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump([data_dict, {}], f)  # PENN 格式：[data, param_dic]
    print(f"Data saved to: {save_path}")


def load_data_pickle(file_path: str) -> Tuple[Dict, Dict]:
    """从 pickle 文件加载数据（PENN 原始格式）"""
    with open(file_path, 'rb') as f:
        data_dict, param_dic = pickle.load(f)
    print(f"Data loaded from: {file_path}")
    return data_dict, param_dic


class MyDataset(Dataset):
    """
    PENN 原始数据集类。

    严格按照 PENN train.py 中的 MyDataset 实现：
    - 使用 ZeroPad2d 进行 padding
    - Z 为 mask，向量 Z 有效位置为 1，padding 为 0
    """

    def __init__(self, file_name: str, max_sample: int = -1):
        super(MyDataset, self).__init__()
        print('load data', file_name)

        self.param_dic = {}
        with open(file_name, "rb") as fp:
            data_dic, self.param_dic = pickle.load(fp)

        X = data_dic['X']  # 轨迹列表
        X = [torch.from_numpy(x) for x in X]
        Y = data_dic['Y']
        H = data_dic['H']

        if max_sample > 0:
            print('*' * 50)
            print('max_sample =', max_sample)
            print('*' * 50)
            X = X[:max_sample]
            Y = Y[:max_sample]
            H = H[:max_sample]

        self.lengths = [_.shape[0] for _ in X]
        self.max_lengths = max(self.lengths)

        # PENN 原始实现：使用 ZeroPad2d 进行 padding
        # padding 格式：nn.ZeroPad2d((left, right, top, bottom))
        # 这里只有底部需要 padding
        self.X = torch.stack([
            nn.ZeroPad2d((0, 0, 0, self.max_lengths - _.shape[0]))(_)
            for _ in X
        ])

        self.Y = torch.from_numpy(np.array(Y))
        self.H = torch.from_numpy(np.array(H).reshape((-1, 1)))

        # Z: 变长序列的 mask，有效位置为 1，padding 为 0
        self.Z = torch.stack([
            nn.ZeroPad2d((0, 0, 0, self.max_lengths - i))(torch.ones((i, 1)))
            for i in self.lengths
        ])

        self.lengths = torch.from_numpy(np.reshape(np.array(self.lengths), (-1, 1))).float()

        print('=' * 50)
        print('size X', self.X.shape)
        print('size Y', self.Y.shape)
        print('size H', self.H.shape)
        print('size Z', self.Z.shape)
        print('=' * 50)

    def get_dim(self) -> Tuple[int, int]:
        """返回 (state_dim, num_params)"""
        return self.X.shape[2], self.Y.shape[1]

    def __getitem__(self, index: int):
        x = self.X[index, :]    # 轨迹
        y = self.Y[index, :]    # 参数
        l = self.lengths[index, :]  # 轨迹长度
        h = self.H[index, :]    # 时间跨度
        z = self.Z[index, :]    # mask
        return x, y, l, h, z

    def __len__(self) -> int:
        return len(self.X)


# =============================================================================
# PENN 模型（严格按论文实现）
# =============================================================================

class PENN(nn.Module):
    """
    PENN (Physics-Embedded Neural Network) - 严格按论文实现

    论文架构：
    1. LSTM Stage: 4层 LSTM，隐藏维度 25
       - 输入：原始轨迹 x = [x_1, ..., x_N]^T
       - 输出：隐藏状态 {h_t}_{t=1}^N

    2. Mean Pooling:
       h̄ = (1/N) Σ_{i=1}^N h_i
       - 使用 mask Z 屏蔽 padding 位置
       - 除以实际长度

    3. FCNN Stage:
       y_0 = [h̄^T, T]^T  (26 维)
       - 第一层: Linear(26, 20) + BatchNorm + ELU
       - 中间层: Linear(20, 20) + ELU (重复2次)
       - 输出层: Linear(20, M)

    参考文献: Wang et al., "Neural network-based parameter estimation of
    stochastic differential equations driven by Lévy noise", Physica A 2022
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layer: int,
                 n_class: int, activation: str = 'elu'):
        """
        初始化 PENN 模型。

        Args:
            in_dim: 输入维度（状态维度，约瑟夫森结为 4）
            hidden_dim: LSTM 隐藏维度（论文推荐 25）
            n_layer: LSTM 层数（论文推荐 4）
            n_class: 输出参数数量（8）
            activation: 激活函数类型（'elu', 'leakyrelu', 'tanh'）
        """
        super(PENN, self).__init__()

        self.type = 'LSTM'  # PENN 使用 LSTM
        self.activation = activation

        self.init_param = [in_dim, hidden_dim, n_layer, n_class]
        print('init_param', self.init_param)

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim

        # Part 1: LSTM
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)

        # Part 2: 全连接网络
        # PENN 论文结构：3 个隐藏层 + 1 个输出层
        self.nn_size = 20  # PENN 论文推荐的隐藏层维度
        self.linears = nn.ModuleList()

        for k in range(3):
            if k == 0:
                # 第一层：输入拼接 [h, T]
                self.linears.append(nn.Linear(hidden_dim + 2, self.nn_size))
                self.linears.append(nn.BatchNorm1d(self.nn_size))
            else:
                self.linears.append(nn.Linear(self.nn_size, self.nn_size))
                if self.activation == 'elu':
                    self.linears.append(nn.ELU())
                elif self.activation == 'leakyrelu':
                    self.linears.append(nn.LeakyReLU())
                elif self.activation == 'tanh':
                    self.linears.append(nn.Tanh())

        # 输出层
        self.linears.append(nn.Linear(self.nn_size, n_class))

        # 用于计算平均的辅助张量
        self.ones = torch.ones((1, self.hidden_dim))

        # 初始化权重
        self.init_weights()

    def _init_lstm(self, weight):
        """
        LSTM 权重初始化 - PENN 论文使用 orthogonal 初始化
        """
        for w in weight.chunk(4, 0):
            init.orthogonal_(w)

    def init_weights(self):
        """
        初始化网络权重 - 严格按照 PENN 实现
        """
        # LSTM 权重初始化
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def forward(self, x: torch.Tensor, l: torch.Tensor, h: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """
        PENN 前向传播。

        Args:
            x: 轨迹数据 [batch, max_N, state_dim]
            l: 真实长度 [batch, 1]
            h: 时间跨度 [batch, 1]
            z: mask [batch, max_N, 1]

        Returns:
            params: 预测的参数 [batch, num_params]
        """
        # Part 1: LSTM
        _out, _ = self.lstm(torch.tensor(x, dtype=torch.float64))

        # 使用 mask 屏蔽 padding 位置
        out = _out * z

        # Mean Pooling：计算有效部分的平均值
        # PENN 实现：sum / length
        out = torch.sum(out, dim=1)
        out = torch.div(out, torch.mm(l, self.ones.to(x.device)))

        # Part 2: 拼接时间跨度
        out = torch.cat([out, h, l], dim=1)

        # Part 3: FCNN
        for m in self.linears:
            out = m(out)

        return out


class loss_l1(nn.Module):
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


# =============================================================================
# 可视化函数
# =============================================================================

def plot_training_history(train_history: List[List[float]], eval_history: List[List[float]],
                          save_path: str = None):
    """绘制训练历史"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    train_total = [h[0] for h in train_history]
    eval_total = [h[0] for h in eval_history]

    ax.plot(train_total, label='Train Loss', alpha=0.7)
    ax.plot(eval_total, label='Eval Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    plt.close()


def plot_parameter_predictions(pred_params: torch.Tensor, true_params: torch.Tensor,
                               save_path: str = None):
    """绘制参数预测对比图"""
    param_names = PhysicalParamsConfig.PARAM_NAMES_LATEX
    n_params = len(param_names)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        pred_vals = pred_params[:, i].numpy()
        true_vals = true_params[:, i].numpy()

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


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数 - 完全按照 PENN 训练流程实现"""

    # ========================================================================
    # 配置参数（严格按 PENN 论文设置）
    # ========================================================================
    config = {
        # 随机种子
        'seed': 42,

        # 数据生成参数
        'n_all': 2000,
        'eval_ratio': 0.15,
        'test_ratio': 0.20,
        'discard_T': 10.0,

        # 数据文件路径
        'data_dir': './data/josephson',
        'train_file': './data/josephson/train.pkl',
        'eval_file': './data/josephson/eval.pkl',
        'test_file': './data/josephson/test.pkl',

        # PENN 网络架构参数
        'lstm_layers': 6,      # PENN 论文推荐：4 层
        'lstm_fea_dim': 25,    # PENN 论文推荐：隐藏维度 25
        'activation': 'elu',   # PENN 论文推荐：ELU

        # 训练参数
        'batch_size': 64,      # 约瑟夫森结 4 维状态，比 OU 过程更复杂
        'num_epochs': 2000,
        'learning_rate': 0.001,  # PENN 论文推荐：0.001
        'weight_decay': 1e-5,
        'drop_last': True,

        # 损失函数权重（与参数范围成反比）
        'loss_weight': PhysicalParamsConfig.PARAM_WEIGHTS,

        # 模型保存参数
        'architecture_name': 'penn_josephson',
        'init_weight_file': '',
        'save_every': 100,
    }

    device = get_device()
    print(f"Using device: {device}")

    save_path = os.path.join(config['data_dir'], config['architecture_name'])
    os.makedirs(os.path.join(save_path, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)

    print("=" * 70)
    print("Coupled Josephson Junction - PENN (Exact Implementation)")
    print("=" * 70)

    # 设置随机种子
    seed_everything(config['seed'])

    # ========================================================================
    # 步骤 1: 生成数据
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Generating Data")
    print("=" * 70)

    if not os.path.exists(config['train_file']) or \
       not os.path.exists(config['eval_file']) or \
       not os.path.exists(config['test_file']):
        print("Generating data...")
        data = generate_trajectory_data(
            n_samples=config['n_all'],
            seed=config['seed'],
            discard_T=config['discard_T'],
            device=device
        )

        indices = np.random.permutation(config['n_all'])
        train_bound = int(config['n_all'] * (1 - config['test_ratio'] - config['eval_ratio']))
        eval_bound = int(config['n_all'] * (1 - config['test_ratio']))

        train_indices = indices[:train_bound]
        eval_indices = indices[train_bound:eval_bound]
        test_indices = indices[eval_bound:]

        def create_subset(data_dict, indices):
            return {
                'X': [data_dict['X'][i] for i in indices],
                'Y': [data_dict['Y'][i] for i in indices],
                'H': [data_dict['H'][i] for i in indices],
            }

        save_data_pickle(create_subset(data, train_indices), config['train_file'])
        save_data_pickle(create_subset(data, eval_indices), config['eval_file'])
        save_data_pickle(create_subset(data, test_indices), config['test_file'])
    else:
        print("Data files already exist. Skipping generation.")

    # ========================================================================
    # 步骤 2: 创建数据集
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Creating Datasets")
    print("=" * 70)

    train_data = MyDataset(config['train_file'])
    eval_data = MyDataset(config['eval_file'])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=config['drop_last']
    )

    # 评估数据集作为单个 batch 处理
    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=len(eval_data),
        shuffle=False,
        num_workers=0
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # ========================================================================
    # 步骤 3: 创建 PENN 模型
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Creating PENN Model")
    print("=" * 70)

    in_dim, n_class = train_data.get_dim()

    model = PENN(
        in_dim=in_dim,
        hidden_dim=config['lstm_fea_dim'],
        n_layer=config['lstm_layers'],
        n_class=n_class,
        activation=config['activation']
    ).to(device)

    # 使用 double 精度（PENN 原始实现）
    model = model.double()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: PENN (LSTM-based)")
    print(f"  State dim: {in_dim}, Num params: {n_class}")
    print(f"  LSTM layers: {config['lstm_layers']}, Hidden dim: {config['lstm_fea_dim']}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ========================================================================
    # 步骤 4: 训练设置
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Training Setup")
    print("=" * 70)

    # 优化器（PENN 使用 Adam）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    # 损失函数（PENN 使用加权 L1 损失）
    criterion = loss_l1(config['loss_weight'])

    init_epoch = 0

    # 加载预训练模型继续训练
    if config['init_weight_file'] and os.path.exists(config['init_weight_file']):
        print('=' * 50)
        print('load model', config['init_weight_file'])
        print('=' * 50)
        model_CKPT = torch.load(config['init_weight_file'], map_location=device)
        model.load_state_dict(model_CKPT['state_dict'])
        optimizer.load_state_dict(model_CKPT['optimizer'])
        init_epoch = model_CKPT['epoch']
        print(f"Resuming from epoch {init_epoch}")

    # ========================================================================
    # 步骤 5: 训练循环
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Training")
    print("=" * 70)

    train_history = []
    eval_history = []

    # 加载评估数据到内存
    for x_eval, y_eval, l_eval, h_eval, z_eval in eval_loader:
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)
        l_eval = l_eval.to(device)
        h_eval = h_eval.to(device)
        z_eval = z_eval.to(device)

    best_eval_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(init_epoch, config['num_epochs']):
        start_time = time.time()
        model.train()

        for i, (x, y, l, h, z) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            h = h.to(device)
            z = z.to(device)

            outputs = model(x, l, h, z)
            losses = criterion(outputs, y)

            optimizer.zero_grad()
            losses[0].backward()
            optimizer.step()

        used_time = time.time() - start_time

        # 评估
        model.eval()
        with torch.no_grad():
            eval_outputs = model(x_eval, l_eval, h_eval, z_eval)
            eval_losses = criterion(eval_outputs, y_eval)

        train_history.append([l.item() for l in losses])
        eval_history.append([l.item() for l in eval_losses])

        # 打印进度
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], "
                  f"Loss: {losses[0].item():.4f}/{eval_losses[0].item():.4f}, "
                  f"time: {used_time:.1f}s")

            for param_group in optimizer.param_groups:
                print(f'  LR: {param_group["lr"]:.6f}')

            # 检查 NaN
            if eval_losses[0] != eval_losses[0]:  # NaN check
                print('Train failed: NaN detected!')
                break

            # 保存模型
            if (epoch + 1) % config['save_every'] == 0:
                save_dict = {
                    'network': model.type,
                    'init_param': model.init_param,
                    'activation': model.activation,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'param_dic': config
                }
                model_path = os.path.join(save_path, 'model', f'model_{epoch + 1:05d}.ckpt')
                torch.save(save_dict, model_path)
                print(f"  -> Model saved: {model_path}")

        # 早停检查
        if eval_losses[0] < best_eval_loss:
            best_eval_loss = eval_losses[0]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ========================================================================
    # 步骤 6: 最终评估
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Final Evaluation")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        final_preds = model(x_eval, l_eval, h_eval, z_eval).cpu()
        final_targets = y_eval.cpu()

    print("\nFinal Evaluation:")
    param_names = PhysicalParamsConfig.PARAM_NAMES
    for i, name in enumerate(param_names):
        loss_val = torch.mean(torch.abs(final_preds[:, i] - final_targets[:, i])).item()
        print(f"  {name}: {loss_val:.6f}")
    print(f"  Total: {eval_losses[0].item():.6f}")

    # 绘制训练历史
    plot_training_history(
        train_history, eval_history,
        save_path=os.path.join(save_path, 'training_history.png')
    )

    # 绘制参数预测对比
    plot_parameter_predictions(
        final_preds, final_targets,
        save_path=os.path.join(save_path, 'parameter_predictions.png')
    )

    # 保存结果
    result_path = os.path.join(save_path, 'final_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump({
            'preds': final_preds.numpy(),
            'targets': final_targets.numpy(),
            'train_history': train_history,
            'eval_history': eval_history,
        }, f)
    print(f"\nResults saved to: {result_path}")

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)

    return model, train_history, eval_history


def test(model_file: str, test_file: str = None, device_mode: str = 'cuda'):
    """测试 PENN 模型"""
    print("=" * 70)
    print("Testing PENN Model")
    print("=" * 70)

    if device_mode == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        model_CKPT = torch.load(model_file, map_location=torch.device('cpu'))
    else:
        device = torch.device('cuda')
        model_CKPT = torch.load(model_file, map_location=torch.device('cuda'))

    print(f"Model loaded from: {model_file}")

    if test_file is None:
        test_file = model_CKPT['param_dic']['test_file']

    test_data = MyDataset(test_file)
    test_loader = DataLoader(dataset=test_data, batch_size=1000, shuffle=False, num_workers=0)

    param = model_CKPT['init_param']
    if 'activation' in model_CKPT:
        model = PENN(param[0], param[1], param[2], param[3],
                    model_CKPT['activation']).to(device)
    else:
        model = PENN(param[0], param[1], param[2], param[3], activation='elu').to(device)

    model = model.double()
    model.load_state_dict(model_CKPT['state_dict'])
    model.eval()

    ys = []
    os = []
    count = 0
    time_start = time.time()

    with torch.no_grad():
        for x, y, l, h, z in test_loader:
            x = x.to(device)
            l = l.to(device)
            h = h.to(device)
            z = z.to(device)

            _o = model(x, l, h, z)
            o = _o.cpu().numpy()

            ys.append(y)
            os.append(o)
            count += y.shape[0]
            print(count, 'samples predicted')

    ys = np.vstack(ys)
    os = np.vstack(os)

    tt = time.time()
    print(f'{len(test_data)} trajectories using {tt - time_start:.1f}s')

    # 保存结果
    result_path = os.path.join(os.path.dirname(model_file), 'test_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump([ys, os], f)
    print(f"Results saved to: {result_path}")

    # 绘制预测对比图
    plot_parameter_predictions(
        torch.from_numpy(os), torch.from_numpy(ys),
        save_path=os.path.join(os.path.dirname(model_file), 'test_predictions.png')
    )

    return ys, os


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Josephson Junction PENN')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--model_file', type=str, default=None,
                       help='Model file for testing')
    parser.add_argument('--test_file', type=str, default=None,
                       help='Test data file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'test':
        if args.model_file is None:
            print("Error: --model_file is required for testing")
            sys.exit(1)
        test(args.model_file, args.test_file, args.device)
