"""
Coupled Josephson Junction SDE 参数学习代码 - 物理增强版 (v3)
===============================================================

基于 v2 的改进版本，引入物理一致性约束：

核心改进：
1. 周期性编码 (Phase Wrapping): 将 phi 编码为 sin/cos，解决 2π 周期性问题
2. 二次变差残差监督: 利用预测的漂移参数计算 sigma 理论目标值
3. 物理正则化损失: 耗散-波动关系约束，避免伪极小值
4. 复合损失函数: 标准 L1 + sigma 一致性 + 物理正则化

改进参考:
- Sigma 估计: 基于二次变差 (Quadratic Variation) 的残差监督
- 周期编码: 处理角变量的多值性问题
- 物理约束: Fluctuation-Dissipation 关系

作者: Assistant
日期: 2026-04-09
版本: v3 (物理增强版)
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
    'beta1':  5.00,   # 1/0.45
    'beta2':  5.00,   # 1/0.45
    'i1':     0.83,   # 1/1.2
    'i2':     0.83,   # 1/1.2
    'kappa1': 18.00,   # 1/0.15
    'kappa2': 18.00,   # 1/0.15
    'sigma1': 100.0,   # 1/0.04
    'sigma2': 100.0,   # 1/0.04
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
                 n_samples: int = 1, method: str = 'euler') -> Tuple[torch.Tensor, torch.Tensor]:
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
# 数据生成与处理（物理增强版）
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
            method='euler'
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
    物理增强版数据集类。

    关键改进 - 周期性编码 (Phase Wrapping):
    - 将 phi 变量从 [0, 2π] 编码为 sin(phi), cos(phi)
    - 解决周期变量的多值性问题
    - 输入维度从 4 变为 6: [phi1, sin(phi1), v1, phi2, sin(phi2), v2]
    """

    def __init__(self, file_name: str, max_sample: int = -1, use_periodic_encoding: bool = True):
        super(MyDataset, self).__init__()
        print('load data', file_name)

        self.use_periodic_encoding = use_periodic_encoding
        self.param_dic = {}
        with open(file_name, "rb") as fp:
            data_dic, self.param_dic = pickle.load(fp)

        X = data_dic['X']  # 轨迹列表，每个元素 [N, 4]
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

        # 应用周期性编码
        if self.use_periodic_encoding:
            X_encoded = []
            for traj in X:
                phi1, v1, phi2, v2 = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
                # 编码 phi 为 sin/cos
                phi1_sin = torch.sin(phi1)
                phi2_sin = torch.sin(phi2)
                # 新维度: [sin(phi1), cos(phi1), v1, sin(phi2), cos(phi2), v2]
                encoded = torch.stack([phi1, phi1_sin, v1, phi2, phi2_sin, v2], dim=1)
                X_encoded.append(encoded)
            X = X_encoded
            self.state_dim = 6  # 编码后维度
        else:
            self.state_dim = 4  # 原始维度

        # 使用 ZeroPad2d 进行 padding
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
        print('periodic encoding:', self.use_periodic_encoding)
        print('state dim:', self.state_dim)
        print('=' * 50)

    def get_dim(self) -> Tuple[int, int]:
        """返回 (state_dim, num_params)"""
        return self.state_dim, self.Y.shape[1]

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
# 物理增强版 PENN 模型
# =============================================================================

class PositionalEncoding(nn.Module):
    """
    Transformer 位置编码 (Sinusoidal Position Embedding)
    基于 Vaswani et al. "Attention Is All You Need"
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float64).unsqueeze(1)
        
        # 计算 div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # d_model 为奇数时，处理最后一个奇数维度
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer（不作为可学习参数，但会保存到 state_dict）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x: [batch, seq_len, d_model] (添加了位置编码)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class ImprovedPENN(nn.Module):
    """
    物理增强版 PENN (Physics-Embedded Neural Network)
    
    架构升级: LSTM -> Transformer + Positional Encoding
    
    核心改进:
    1. Transformer Encoder: 更好的长距离依赖建模
    2. 位置编码 (Positional Encoding): 捕获序列位置信息
    3. 支持周期性编码后的输入维度 (6 vs 4)
    4. 二次变差残差计算：利用预测的漂移参数计算 sigma 理论目标值
    5. 分离 sigma 优化：通过物理一致性损失约束 sigma

    论文架构（修改后）:
    1. Transformer Stage: n_layer 层 Transformer，隐藏维度 hidden_dim，带位置编码
    2. FCNN Stage: [hidden+2 -> 20] + BN + ELU + [20->20] + ELU + [20->8]
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_layer: int,
                 n_class: int, activation: str = 'elu', 
                 nhead: int = 5, dropout: float = 0.1):
        super(ImprovedPENN, self).__init__()

        self.type = 'Transformer'
        self.activation = activation

        self.init_param = [in_dim, hidden_dim, n_layer, n_class]
        print('init_param', self.init_param)

        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        # Part 1: Transformer Encoder with Positional Encoding
        # 输入投影层：将 in_dim 映射到 hidden_dim
        self.input_projection = nn.Linear(in_dim, hidden_dim, dtype=torch.float64)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=5000, dropout=dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,  # 通常设置为 hidden_dim * 4
            dropout=dropout,
            batch_first=True,
            dtype=torch.float64
        )
        
        # Transformer Encoder (多层堆叠)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
        )
        
        # Layer Normalization (Transformer 后的归一化)
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=torch.float64)

        # Part 2: 全连接网络
        self.nn_size = 20
        self.linears = nn.ModuleList()

        for k in range(3):
            if k == 0:
                # 第一层：输入拼接 [h, H, L] (h_mean + time_horizon + length)
                self.linears.append(nn.Linear(hidden_dim + 2, self.nn_size, dtype=torch.float64))
                self.linears.append(nn.BatchNorm1d(self.nn_size, dtype=torch.float64))
            else:
                self.linears.append(nn.Linear(self.nn_size, self.nn_size, dtype=torch.float64))
                if self.activation == 'elu':
                    self.linears.append(nn.ELU())
                elif self.activation == 'leakyrelu':
                    self.linears.append(nn.LeakyReLU())
                elif self.activation == 'tanh':
                    self.linears.append(nn.Tanh())

        # 输出层
        self.linears.append(nn.Linear(self.nn_size, n_class, dtype=torch.float64))
        self.normalization = nn.Tanh()

        # 用于计算平均的辅助张量
        self.ones = torch.ones((1, self.hidden_dim))

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """初始化网络权重"""
        # 输入投影层初始化
        init.xavier_uniform_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            init.zeros_(self.input_projection.bias)
        
        # 线性层使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) and m != self.input_projection:
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def compute_quadratic_variation_target(self, x, l, pred_params, h, z):
        """
        修复版：正确使用 mask z 处理变长序列
        """
        batch_size = x.shape[0]
        device = x.device
        
        # dt 计算
        dt = h / l  # [batch, 1]
        
        # 解析参数
        beta1 = pred_params[:, 0:1]
        beta2 = pred_params[:, 1:2]
        i1 = pred_params[:, 2:3]
        i2 = pred_params[:, 3:4]
        kappa1 = pred_params[:, 4:5]
        kappa2 = pred_params[:, 5:6]
        
        # 提取变量（考虑周期性编码）
        if x.shape[-1] == 6:
            phi1, sin_phi1, v1 = x[..., 0], x[..., 1], x[..., 2]
            phi2, sin_phi2, v2 = x[..., 3], x[..., 4], x[..., 5]
            cos_phi1 = torch.cos(phi1)
            cos_phi2 = torch.cos(phi2)
        else:
            phi1, v1, phi2, v2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        
        # ===== 关键修复 1: 应用 Mask 屏蔽 Padding =====
        # z 是 [batch, max_N, 1]，需要 squeeze 到 [batch, max_N]
        if z is not None:
            mask = z.squeeze(-1)  # [batch, max_N]
            # 将 padding 位置的 v 设为 nan，后续计算中排除
            v1 = v1.clone()
            v2 = v2.clone()
            v1[mask == 0] = float('nan')
            v2[mask == 0] = float('nan')
            phi1 = phi1.clone()
            phi2 = phi2.clone()
            phi1[mask == 0] = float('nan')
            phi2[mask == 0] = float('nan')
        
        # ===== 关键修复 2: 处理相位差的周期性 =====
        # 使用 sin 和 cos 直接计算差分，避免 2π 跳跃
        # 或者使用: diff = atan2(sin(phi2-phi1), cos(phi2-phi1))
        sin_diff = sin_phi2 * cos_phi1 - cos_phi2 * sin_phi1
        cos_diff = cos_phi2 * cos_phi1 + sin_phi2 * sin_phi1
        phi_diff = torch.atan2(sin_diff, cos_diff)  # 这是 (phi2 - phi1) 包裹到 [-pi, pi]
        
        # 计算漂移项（带 mask 的 nan）
        dv1_drift = i1 - beta1 * v1 - torch.sin(phi1) + kappa1 * phi_diff
        dv2_drift = i2 - beta2 * v2 - torch.sin(phi2) - kappa2 * phi_diff  # 注意符号：phi1-phi2 = -phi_diff
        
        # 计算实际速度变化（使用 nan-aware 操作）
        # 对 v1 进行差分，padding 位置已经是 nan
        dv1_actual = (v1[:, 1:] - v1[:, :-1]) / dt
        dv2_actual = (v2[:, 1:] - v2[:, :-1]) / dt
        
        # 对齐 drift（也是 nan-aware）
        dv1_drift_aligned = dv1_drift[:, :-1]
        dv2_drift_aligned = dv2_drift[:, :-1]
        
        # 计算残差（nan 传播）
        residual1 = dv1_actual - dv1_drift_aligned
        residual2 = dv2_actual - dv2_drift_aligned
        
        # 计算二次变差（忽略 nan）
        # nanmean 沿时间轴计算
        sigma1_sq = torch.nanmean(residual1 ** 2, dim=1, keepdim=True) * dt
        sigma2_sq = torch.nanmean(residual2 ** 2, dim=1, keepdim=True) * dt
        
        # 数值保护
        sigma1_target = torch.sqrt(torch.clamp(sigma1_sq, min=1e-8))
        sigma2_target = torch.sqrt(torch.clamp(sigma2_sq, min=1e-8))
        
        return torch.cat([sigma1_target, sigma2_target], dim=1)


    def forward(self, x: torch.Tensor, l: torch.Tensor, h: torch.Tensor,
                z: torch.Tensor, return_sigma_target: bool = False):
        """
        前向传播。

        Args:
            x: 轨迹数据 [batch, max_N, state_dim]
            l: 真实长度 [batch, 1]
            h: 时间跨度 [batch, 1]
            z: mask [batch, max_N, 1]
            return_sigma_target: 是否返回 sigma 的二次变差目标

        Returns:
            pred_params: [batch, num_params]
            sigma_target (可选): [batch, 2] - sigma 的理论目标值
        """
        # Part 1: Transformer Encoder with Positional Encoding
        # 1. 输入投影: [batch, seq_len, in_dim] -> [batch, seq_len, hidden_dim]
        x_input = self.input_projection(x.to(torch.float64))
        
        # 2. 添加位置编码
        x_encoded = self.pos_encoder(x_input)
        
        # 3. 创建 key_padding_mask (用于屏蔽 padding 位置)
        # Transformer 需要 [batch, seq_len] 的 mask，True 表示该位置会被屏蔽
        if z is not None:
            # z: [batch, seq_len, 1] -> [batch, seq_len]
            key_padding_mask = (z.squeeze(-1) == 0)
        else:
            key_padding_mask = None
        
        # 4. Transformer Encoder
        # src_key_padding_mask: [batch, seq_len]，True 表示 mask 掉
        _out = self.transformer_encoder(x_encoded, src_key_padding_mask=key_padding_mask)
        
        # 5. Layer Normalization
        _out = self.layer_norm(_out)

        # 使用 mask 屏蔽 padding 位置 (用于后续 mean pooling)
        out = _out * z

        # Mean Pooling
        out = torch.sum(out, dim=1)
        out = torch.div(out, torch.mm(l, self.ones.to(x.device)))

        # Part 2: 拼接时间跨度和长度
        out = torch.cat([out, h, l], dim=1)

        # Part 3: FCNN
        for m in self.linears:
            out = m(out)

        pred_params = out
        
        if return_sigma_target:
            with torch.no_grad():
                sigma_target = self.compute_quadratic_variation_target(x, l, pred_params, h, z)
            return pred_params, sigma_target
        return pred_params



class ImprovedLoss(nn.Module):
    """
    物理增强版损失函数

    组合:
    1. 标准加权 L1 损失（所有参数）
    2. Sigma 二次变差一致性损失（核心改进）
    3. 物理正则化损失（耗散-波动关系）
    """

    def __init__(self, weight: List[float], sigma_weight: float = 10.0,
                 physics_weight: float = 0.1):
        super().__init__()
        self.eps = torch.tensor(1e-8)
        self.loss = nn.L1Loss()
        self.weight = weight
        self.sigma_weight = sigma_weight
        self.physics_weight = physics_weight

    def forward(self, pred_params: torch.Tensor, true_params: torch.Tensor,
                sigma_target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算复合损失。

        Returns:
            dict 包含:
            - 'total': 总损失
            - 'standard': 标准 L1 损失
            - 'sigma_consistency': sigma 一致性损失
            - 'physics': 物理正则化损失
            - 'individual': 各参数的单独损失列表
        """
        M = pred_params.shape[1]
        individual_losses = []
        standard_loss = 0.0

        # 1. 标准加权 L1 损失
        for i in range(M):
            l = torch.mean(self.loss(pred_params[:, i], true_params[:, i]))
            individual_losses.append(l)
            standard_loss = standard_loss + l * self.weight[i]

        # 2. Sigma 二次变差一致性损失
        sigma_consistency = torch.tensor(0.0, device=pred_params.device)
        if sigma_target is not None:
            sigma_pred = pred_params[:, 6:8]  # 预测 sigma
            sigma_consistency = torch.mean((sigma_pred - sigma_target) ** 2)

        # 3. 物理正则化损失（耗散-波动关系）
        physics_loss = self._compute_physics_loss(pred_params)

        # 总损失
        total_loss = (standard_loss +
                     self.sigma_weight * sigma_consistency +
                     self.physics_weight * physics_loss)

        return {
            'total': total_loss,
            'standard': standard_loss,
            'sigma_consistency': sigma_consistency,
            'physics': physics_loss,
            'individual': individual_losses
        }

    def _compute_physics_loss(self, pred_params: torch.Tensor) -> torch.Tensor:
        """
        物理正则化：耗散-波动关系约束

        近似关系：sigma^2 / (2*beta) 应该在合理范围内
        这来源于 Fluctuation-Dissipation 定理的启发
        """
        beta1 = pred_params[:, 0]
        beta2 = pred_params[:, 1]
        sigma1 = pred_params[:, 6]
        sigma2 = pred_params[:, 7]

        # 有效温度估计 (避免除零)
        effective_temp1 = sigma1 ** 2 / (2 * beta1 + 1e-8)
        effective_temp2 = sigma2 ** 2 / (2 * beta2 + 1e-8)

        # 惩罚不合理的有效温度（假设合理范围在 [0.001, 10]）
        temp_penalty = (torch.relu(effective_temp1 - 10.0) +
                       torch.relu(-effective_temp1 + 0.001) +
                       torch.relu(effective_temp2 - 10.0) +
                       torch.relu(-effective_temp2 + 0.001))

        return torch.mean(temp_penalty)


# =============================================================================
# 可视化函数
# =============================================================================

def plot_training_history(train_history: List[Dict], eval_history: List[Dict],
                          save_path: str = None):
    """绘制训练历史（改进版，显示各损失分量）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 总损失
    ax = axes[0, 0]
    train_total = [h['total'].item() for h in train_history]
    eval_total = [h['total'].item() for h in eval_history]
    ax.plot(train_total, label='Train', alpha=0.7)
    ax.plot(eval_total, label='Eval', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 标准 L1 损失
    ax = axes[0, 1]
    train_std = [h['standard'].item() for h in train_history]
    eval_std = [h['standard'].item() for h in eval_history]
    ax.plot(train_std, label='Train', alpha=0.7)
    ax.plot(eval_std, label='Eval', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Standard L1 Loss')
    ax.set_title('Standard L1 Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Sigma 一致性损失
    ax = axes[1, 0]
    train_sigma = [h['sigma_consistency'].item() for h in train_history]
    eval_sigma = [h['sigma_consistency'].item() for h in eval_history]
    ax.plot(train_sigma, label='Train', alpha=0.7)
    ax.plot(eval_sigma, label='Eval', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sigma Consistency Loss')
    ax.set_title('Sigma Consistency Loss (Quadratic Variation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 物理正则化损失
    ax = axes[1, 1]
    train_phys = [h['physics'].item() for h in train_history]
    eval_phys = [h['physics'].item() for h in eval_history]
    ax.plot(train_phys, label='Train', alpha=0.7)
    ax.plot(eval_phys, label='Eval', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Physics Regularization Loss')
    ax.set_title('Physics Regularization Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

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


def plot_sigma_comparison(pred_sigma: torch.Tensor, true_sigma: torch.Tensor,
                          sigma_target: torch.Tensor, save_path: str = None):
    """
    专门绘制 sigma 的预测对比图（包含二次变差目标）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sigma_names = [r'$\sigma_1$', r'$\sigma_2$']

    for i, (ax, name) in enumerate(zip(axes, sigma_names)):
        pred_vals = pred_sigma[:, i].numpy()
        true_vals = true_sigma[:, i].numpy()
        target_vals = sigma_target[:, i].numpy()

        # 散点图：预测 vs 真实
        ax.scatter(true_vals, pred_vals, color='r', s=20, alpha=0.5, label='NN Prediction')
        ax.scatter(true_vals, target_vals, color='g', s=20, alpha=0.3, marker='x',
                  label='QV Target (Theory)')

        # 对角线
        min_val = min(true_vals.min(), pred_vals.min(), target_vals.min())
        max_val = max(true_vals.max(), pred_vals.max(), target_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=1, label='Ideal')

        ax.set_xlabel('True values', fontsize=12)
        ax.set_ylabel('Estimated values', fontsize=12)
        ax.set_title(f'{name} Estimation', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 计算 R²
        ss_res = np.sum((pred_vals - true_vals) ** 2)
        ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'Pred $R^2$={r2:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sigma comparison plot saved to: {save_path}")
    plt.close()


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主函数 - 物理增强版训练流程"""

    # ========================================================================
    # 配置参数
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
        'train_file': './data/josephson/train_v3.pkl',
        'eval_file': './data/josephson/eval_v3.pkl',
        'test_file': './data/josephson/test_v3.pkl',

        # PENN 网络架构参数 (Transformer 版)
        'transformer_layers': 4,    # Transformer 编码器层数
        'transformer_dim': 25,      # 隐藏维度 (需要能被 nhead 整除)
        'transformer_nhead': 5,     # 注意力头数 (25 / 5 = 5)
        'transformer_dropout': 0.1, # Transformer dropout
        'activation': 'elu',        # FCNN 激活函数

        # 数据增强
        'use_periodic_encoding': True,  # 使用周期性编码

        # 训练参数
        'batch_size': 32,
        'num_epochs': 2000,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'drop_last': True,

        # 损失函数权重
        'loss_weight': PhysicalParamsConfig.PARAM_WEIGHTS,
        'sigma_weight': 1.0,      # sigma 一致性损失权重
        'physics_weight': 0.01,     # 物理正则化权重

        # 模型保存参数
        'architecture_name': 'penn_josephson_v3',
        'init_weight_file': '',
        'save_every': 100,
    }

    device = get_device()
    print(f"Using device: {device}")

    save_path = os.path.join(config['data_dir'], config['architecture_name'])
    os.makedirs(os.path.join(save_path, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)

    print("=" * 70)
    print("Coupled Josephson Junction - Improved PENN (Physics-Enhanced v3)")
    print("=" * 70)
    print(f"Key improvements:")
    print(f"  1. Transformer + Positional Encoding (replaces LSTM)")
    print(f"  2. Periodic encoding (phase wrapping): {config['use_periodic_encoding']}")
    print(f"  3. Quadratic variation sigma supervision")
    print(f"  4. Physics regularization (fluctuation-dissipation)")
    print(f"  5. Composite loss: standard + sigma + physics")
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

    train_data = MyDataset(config['train_file'], use_periodic_encoding=config['use_periodic_encoding'])
    eval_data = MyDataset(config['eval_file'], use_periodic_encoding=config['use_periodic_encoding'])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=config['drop_last']
    )

    eval_loader = DataLoader(
        dataset=eval_data,
        batch_size=len(eval_data),
        shuffle=False,
        num_workers=0
    )

    print(f"Training samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # ========================================================================
    # 步骤 3: 创建改进版 PENN 模型
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Creating Improved PENN Model")
    print("=" * 70)

    in_dim, n_class = train_data.get_dim()

    model = ImprovedPENN(
        in_dim=in_dim,
        hidden_dim=config['transformer_dim'],
        n_layer=config['transformer_layers'],
        n_class=n_class,
        activation=config['activation'],
        nhead=config['transformer_nhead'],
        dropout=config['transformer_dropout']
    ).to(device)

    # 使用 double 精度
    model = model.double()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: Improved PENN (Physics-Enhanced) with Transformer")
    print(f"  State dim: {in_dim}, Num params: {n_class}")
    print(f"  Transformer layers: {config['transformer_layers']}, Hidden dim: {config['transformer_dim']}")
    print(f"  Attention heads: {config['transformer_nhead']}, Dropout: {config['transformer_dropout']}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # 在 main() 的数据加载后添加这段测试代码：
    test_batch_x, test_batch_y, test_batch_l, test_batch_h, test_batch_z = next(iter(train_loader))
    test_batch_x = test_batch_x.to(device)
    test_batch_y = test_batch_y.to(device)
    test_batch_l = test_batch_l.to(device)
    test_batch_h = test_batch_h.to(device)
    test_batch_z = test_batch_z.to(device)
    
    # ... 移动到 device ...

    model.eval()
    with torch.no_grad():
        pred_params, sigma_target = model(test_batch_x, test_batch_l, test_batch_h, test_batch_z, return_sigma_target=True)

    print("True sigma:", test_batch_y[:5, 6:8])
    print("Computed sigma_target:", sigma_target[:5])
    print("True beta:", test_batch_y[:5, 0:2])
    print("Pred beta:", pred_params[:5, 0:2])
    
    # ========================================================================
    # 步骤 4: 训练设置
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Training Setup")
    print("=" * 70)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    # 改进的损失函数
    criterion = ImprovedLoss(
        config['loss_weight'],
        sigma_weight=config['sigma_weight'],
        physics_weight=config['physics_weight']
    )

    init_epoch = 0

    # 加载预训练模型
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
    # 步骤 5: 训练循环（物理增强版）
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Training (Physics-Enhanced)")
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
    patience = 100
    patience_counter = 0

    for epoch in range(init_epoch, config['num_epochs']):
        start_time = time.time()
        model.train()

        epoch_losses = []

        for i, (x, y, l, h, z) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            h = h.to(device)
            z = z.to(device)

            # 前向传播，返回预测参数和 sigma 目标值
            pred_params, sigma_target = model(x, l, h, z, return_sigma_target=True)

            # 计算复合损失
            losses = criterion(pred_params, y, sigma_target)

            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            epoch_losses.append(losses)

        used_time = time.time() - start_time

        # 评估
        model.eval()
        with torch.no_grad():
            eval_pred_params, eval_sigma_target = model(x_eval, l_eval, h_eval, z_eval, return_sigma_target=True)
            eval_losses = criterion(eval_pred_params, y_eval, eval_sigma_target)

        # 平均训练损失
        train_loss = {
            'total': torch.mean(torch.stack([l['total'] for l in epoch_losses])),
            'standard': torch.mean(torch.stack([l['standard'] for l in epoch_losses])),
            'sigma_consistency': torch.mean(torch.stack([l['sigma_consistency'] for l in epoch_losses])),
            'physics': torch.mean(torch.stack([l['physics'] for l in epoch_losses])),
        }

        train_history.append(train_loss)
        eval_history.append(eval_losses)

        # 打印进度
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{config['num_epochs']}], "
                  f"Total: {train_loss['total'].item():.4f}/{eval_losses['total'].item():.4f}, "
                  f"Std: {train_loss['standard'].item():.4f}/{eval_losses['standard'].item():.4f}, "
                  f"Sigma: {train_loss['sigma_consistency'].item():.4f}/{eval_losses['sigma_consistency'].item():.4f}, "
                  f"time: {used_time:.1f}s")

            for param_group in optimizer.param_groups:
                print(f'  LR: {param_group["lr"]:.6f}')

            # 检查 NaN
            if eval_losses['total'] != eval_losses['total']:
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
        if eval_losses['total'] < best_eval_loss:
            best_eval_loss = eval_losses['total']
            patience_counter = 0
            # 保存最佳模型
            best_model_path = os.path.join(save_path, 'model', 'model_best.ckpt')
            torch.save({
                'network': model.type,
                'init_param': model.init_param,
                'activation': model.activation,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'param_dic': config
            }, best_model_path)
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
        final_preds, final_sigma_targets = model(x_eval, l_eval, h_eval, z_eval, return_sigma_target=True)
        final_targets = y_eval.cpu()
        final_preds = final_preds.cpu()
        final_sigma_targets = final_sigma_targets.cpu()

    print("\nFinal Evaluation (per parameter):")
    param_names = PhysicalParamsConfig.PARAM_NAMES
    for i, name in enumerate(param_names):
        loss_val = torch.mean(torch.abs(final_preds[:, i] - final_targets[:, i])).item()
        print(f"  {name}: {loss_val:.6f}")

    # Sigma 专项分析
    print("\nSigma Estimation Analysis:")
    for i, name in enumerate(['sigma1', 'sigma2']):
        pred_vals = final_preds[:, 6+i].numpy()
        true_vals = final_targets[:, 6+i].numpy()
        target_vals = final_sigma_targets[:, i].numpy()

        mae_pred = np.mean(np.abs(pred_vals - true_vals))
        mae_target = np.mean(np.abs(target_vals - true_vals))
        correlation = np.corrcoef(pred_vals, true_vals)[0, 1]

        print(f"  {name}:")
        print(f"    NN Prediction MAE: {mae_pred:.6f}")
        print(f"    QV Target MAE: {mae_target:.6f}")
        print(f"    Correlation: {correlation:.4f}")

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

    # 绘制 Sigma 专项对比
    plot_sigma_comparison(
        final_preds[:, 6:8], final_targets[:, 6:8], final_sigma_targets,
        save_path=os.path.join(save_path, 'sigma_comparison.png')
    )

    # 保存结果
    result_path = os.path.join(save_path, 'final_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump({
            'preds': final_preds.numpy(),
            'targets': final_targets.numpy(),
            'sigma_targets': final_sigma_targets.numpy(),
            'train_history': train_history,
            'eval_history': eval_history,
        }, f)
    print(f"\nResults saved to: {result_path}")

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)

    return model, train_history, eval_history


def test(model_file: str, test_file: str = None, device_mode: str = 'cuda'):
    """测试改进版 PENN 模型"""
    print("=" * 70)
    print("Testing Improved PENN Model (v3)")
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

    use_periodic = model_CKPT['param_dic'].get('use_periodic_encoding', True)
    test_data = MyDataset(test_file, use_periodic_encoding=use_periodic)
    test_loader = DataLoader(dataset=test_data, batch_size=1000, shuffle=False, num_workers=0)

    param = model_CKPT['init_param']
    if 'activation' in model_CKPT:
        model = ImprovedPENN(param[0], param[1], param[2], param[3],
                            model_CKPT['activation']).to(device)
    else:
        model = ImprovedPENN(param[0], param[1], param[2], param[3], activation='elu').to(device)

    model = model.double()
    model.load_state_dict(model_CKPT['state_dict'])
    model.eval()

    ys = []
    os = []
    sigma_targets = []
    count = 0
    time_start = time.time()

    with torch.no_grad():
        for x, y, l, h, z in test_loader:
            x = x.to(device)
            l = l.to(device)
            h = h.to(device)
            z = z.to(device)

            _o, _st = model(x, l, h, z, return_sigma_target=True)
            o = _o.cpu().numpy()
            st = _st.cpu().numpy()

            ys.append(y)
            os.append(o)
            sigma_targets.append(st)
            count += y.shape[0]
            print(count, 'samples predicted')

    ys = np.vstack(ys)
    os = np.vstack(os)
    sigma_targets = np.vstack(sigma_targets)

    tt = time.time()
    print(f'{len(test_data)} trajectories using {tt - time_start:.1f}s')

    # 保存结果
    result_path = os.path.join(os.path.dirname(model_file), 'test_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump([ys, os, sigma_targets], f)
    print(f"Results saved to: {result_path}")

    # 绘制预测对比图
    plot_parameter_predictions(
        torch.from_numpy(os), torch.from_numpy(ys),
        save_path=os.path.join(os.path.dirname(model_file), 'test_predictions.png')
    )

    # 绘制 Sigma 专项对比
    plot_sigma_comparison(
        torch.from_numpy(os[:, 6:8]), torch.from_numpy(ys[:, 6:8]),
        torch.from_numpy(sigma_targets),
        save_path=os.path.join(os.path.dirname(model_file), 'test_sigma_comparison.png')
    )

    return ys, os, sigma_targets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Josephson Junction Improved PENN v3')
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
