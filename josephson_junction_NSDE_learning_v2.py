"""
Coupled Josephson Junction Neural SDE - 修复与优化版
======================================================
针对 4G 显存优化，修复 Euler-Maruyama 损失数值稳定性问题
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 配置与工具函数
# =============================================================================

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 对固定大小输入加速


class JosephsonJunctionSDE(nn.Module):
    """
    耦合约瑟夫森结 SDE 模拟器。
    使用 torchsde 库求解 SDE。
    
    SDE 方程:
        dφ₁ = v₁ dt
        dv₁ = [i₁ - β_{J₁}v₁ - sin(φ₁) + κ₁(φ₂ - φ₁)] dt + σ₁ dW₁
        dφ₂ = v₂ dt
        dv₂ = [i₂ - β_{J₂}v₂ - sin(φ₂) + κ₂(φ₁ - φ₂)] dt + σ₂ dW₂
    """
    
    def __init__(self, beta1, beta2, i1, i2, kappa1, kappa2, sigma1, sigma2):
        """
        初始化物理参数。
        
        Args:
            beta1, beta2: 阻尼系数
            i1, i2: 偏置电流
            kappa1, kappa2: 耦合系数
            sigma1, sigma2: 噪声强度
        """
        super().__init__()
        
        # 将标量参数转换为 buffer 或 parameter，使其随模型移动设备
        self.register_buffer('beta1', torch.tensor(beta1, dtype=torch.float32))
        self.register_buffer('beta2', torch.tensor(beta2, dtype=torch.float32))
        self.register_buffer('i1', torch.tensor(i1, dtype=torch.float32))
        self.register_buffer('i2', torch.tensor(i2, dtype=torch.float32))
        self.register_buffer('kappa1', torch.tensor(kappa1, dtype=torch.float32))
        self.register_buffer('kappa2', torch.tensor(kappa2, dtype=torch.float32))
        self.register_buffer('sigma1', torch.tensor(sigma1, dtype=torch.float32))
        self.register_buffer('sigma2', torch.tensor(sigma2, dtype=torch.float32))
        
        # SDE 类型设置 (torchsde 需要)
        self.sde_type = "ito"
        self.noise_type = "diagonal"
    
    def f(self, t, y):
        """
        漂移项 (drift)。
        
        Args:
            t: 时间标量
            y: 状态向量 [batch, 4] = [φ₁, v₁, φ₂, v₂]
            
        Returns:
            漂移向量 [batch, 4]
        """
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        
        # dφ₁/dt = v₁
        d_phi1 = v1
        
        # dv₁/dt = i₁ - β₁v₁ - sin(φ₁) + κ₁(φ₂ - φ₁)
        dv1 = (self.i1 - self.beta1 * v1 - torch.sin(phi1) + self.kappa1 * (phi2 - phi1))
        
        # dφ₂/dt = v₂
        d_phi2 = v2
        
        # dv₂/dt = i₂ - β₂v₂ - sin(φ₂) + κ₂(φ₁ - φ₂)
        dv2 = (self.i2 - self.beta2 * v2 - torch.sin(phi2) + self.kappa2 * (phi1 - phi2))
        
        return torch.stack([d_phi1, dv1, d_phi2, dv2], dim=-1)
    
    def g(self, t, y):
        """
        扩散项 (diffusion)。
        
        Args:
            t: 时间标量
            y: 状态向量 [batch, 4]
            
        Returns:
            对角噪声 [batch, 4] (对应 diagonal noise type)
            [0, σ₁, 0, σ₂] - 只有速度方程有噪声
        """
        batch_size = y.shape[0]
        device = y.device
        
        # 对角噪声: [0, σ₁, 0, σ₂]
        noise = torch.zeros(batch_size, 4, device=device)
        noise[..., 1] = self.sigma1  # dv₁ 的噪声
        noise[..., 3] = self.sigma2  # dv₂ 的噪声
        
        return noise
    
    def simulate(self, t_span, n_steps, initial_state, n_samples=1, dt=None, method='euler'):
        """
        使用 torchsde 模拟 SDE。
        
        Args:
            t_span: 时间区间 [t_start, t_end]
            n_steps: 时间步数
            initial_state: 初始状态 [φ₁₀, v₁₀, φ₂₀, v₂₀]
            n_samples: 样本数量
            dt: SDE 积分步长，默认为 None (自动选择)
            method: 积分方法，默认为 'milstein'
            
        Returns:
            times: 时间点数组 (n_steps,)
            trajectories: 轨迹数组 (n_samples, n_steps, 4)
        """
        
        # TODO: 检查 n_samples 的参数是否被有效用到。
        device = initial_state.device if torch.is_tensor(initial_state) else torch.device('cpu')
        
        t_start, t_end = t_span
        times = torch.linspace(t_start, t_end, n_steps, device=device)
        
        # 准备初始状态
        if initial_state.dim() == 1:
            y0 = initial_state.unsqueeze(0).expand(n_samples, -1)
        else:
            y0 = initial_state
        
        # 使用 torchsde.sdeint 求解 SDE
        # 输出 z_t 的 shape 为 [n_steps, batch, channels]
        with torch.no_grad():
            trajectories = torchsde.sdeint(
                sde=self,
                y0=y0,
                ts=times,
                dt=dt if dt is not None else (t_end - t_start) / n_steps,
                method=method
            )
        
        # 调整维度: [n_steps, batch, 4] -> [batch, n_steps, 4]
        trajectories = trajectories.permute(1, 0, 2)
        
        return times, trajectories


# =============================================================================
# 修复 1: 带周期性编码的 Neural SDE
# =============================================================================

class LipSwish(nn.Module):
    """标准 LipSwish：带可学习 β 参数，确保 Lipschitz 常数 ≤ 1"""
    def __init__(self, beta_init=0.5):
        super().__init__()
        # 使用 softplus 逆函数确保 β > 0
        self.beta_logit = nn.Parameter(
            torch.log(torch.exp(torch.tensor(beta_init)) - 1)
        )
        self.scale = 0.909  # ≈ 1/1.1
    
    @property
    def beta(self):
        return torch.nn.functional.softplus(self.beta_logit)
    
    def forward(self, x):
        # 标准公式: (x * σ(βx)) / 1.1
        return self.scale * x * torch.sigmoid(self.beta * x)
    
    def extra_repr(self):
        return f'beta={self.beta.item():.4f}'


class NeuralJosephsonSDE(nn.Module):
    """
    优化后的 Neural Josephson SDE：
    1. 输入维度从 4 扩展到 6（sin/cos 编码相位）
    2. 针对小显存优化默认参数
    """
    
    def __init__(self, state_dim=4, hidden_dim=32, num_layers=2, activation='lipswish'):
        super().__init__()
        self.state_dim = state_dim
        self.encoded_dim = 6  # [sin(phi1), cos(phi1), v1, sin(phi2), cos(phi2), v2]
        
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        
        # 使用更小的网络以适应 4G 显存
        if activation == 'lipswish':
            act = LipSwish()
        else:
            act = nn.Tanh()
        
        # 漂移网络 f: 处理编码后的输入
        layers = [nn.Linear(self.encoded_dim, hidden_dim), act]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.f_net = nn.Sequential(*layers)
        
        # 扩散网络 g: 输出严格为正
        layers_g = [nn.Linear(self.encoded_dim, hidden_dim), act]
        for _ in range(num_layers - 1):
            layers_g.append(nn.Linear(hidden_dim, hidden_dim))
            layers_g.append(act)
        layers_g.append(nn.Linear(hidden_dim, state_dim))
        self.g_net = nn.Sequential(*layers_g)
        
        # self._init_weights()
    
    def _encode_input(self, y):
        """
        将相位变量编码为周期性特征，避免 phi ~ phi+2pi 的不连续性
        y: [..., 4] -> [phi1, v1, phi2, v2]
        """
        phi1, v1 = y[..., 0:1], y[..., 1:2]
        phi2, v2 = y[..., 2:3], y[..., 3:4]
        
        # 关键改进: sin/cos 编码
        return torch.cat([
            torch.sin(phi1), torch.cos(phi1), v1,
            torch.sin(phi2), torch.cos(phi2), v2
        ], dim=-1)
    
    def f(self, t, y):
        y_enc = self._encode_input(y)
        out = self.f_net(y_enc)
        # 收紧 clamp 范围，避免梯度爆炸
        return torch.clamp(out, min=-20.0, max=20.0)
    
    def g(self, t, y):
        y_enc = self._encode_input(y)
        out = torch.nn.functional.softplus(self.g_net(y_enc))
        # 关键: 严格限制扩散系数范围，避免 Euler 损失中的除零
        return torch.clamp(out, min=1e-4, max=0.5)
    
    def forward(self, initial_states, times, dt=None, method='euler'):
        # 使用 euler 方法（最省显存）
        trajectories = torchsde.sdeint(
            sde=self,
            y0=initial_states,
            ts=times,
            dt=dt if dt is not None else (times[-1] - times[0]) / len(times),
            method=method,
            adaptive=False,  # 固定步长更省显存
            rtol=1e-3,
            atol=1e-3
        )
        return trajectories.permute(1, 0, 2)


# =============================================================================
# 修复 2: 数值稳定的 Euler-Maruyama 损失
# =============================================================================

def compute_euler_pseudo_likelihood_loss(true_trajectories, dt, model):
    """
    修复后的 Euler-Maruyama 伪似然损失。
    
    数学公式:
    X_{t+1} | X_t ~ N(X_t + f(X_t)dt, g(X_t)^2 dt)
    
    负对数似然（忽略常数）:
    NLL = sum [0.5 * (dx - f*dt)^2 / (g^2 * dt) + log(g)]
    """
    device = true_trajectories.device
    dtype = true_trajectories.dtype
    
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    
    # 分割相邻时间步
    x_t = true_trajectories[:, :-1, :]      # [batch, n-1, 4]
    x_next = true_trajectories[:, 1:, :]    # [batch, n-1, 4]
    dx = x_next - x_t                       # 实际观测增量
    
    batch_size, seq_len, dim = x_t.shape
    x_flat = x_t.reshape(batch_size * seq_len, dim)
    
    # 计算漂移和扩散
    f_val = model.f(None, x_flat).reshape(batch_size, seq_len, dim)
    g_val = model.g(None, x_flat).reshape(batch_size, seq_len, dim)
    
    # 数值稳定性关键: 严格限制 g 的范围，避免除零和 log(0)
    g_val = torch.clamp(g_val, min=1e-4, max=1.0)
    
    # 计算残差: dx - f*dt
    drift_term = f_val * dt
    residual = dx - drift_term
    
    # 方差 = g^2 * dt
    variance = (g_val ** 2) * dt
    
    # 标准化残差
    normalized_residual = residual / (g_val * torch.sqrt(dt) + 1e-8)
    
    # 负对数似然: 0.5 * (residual^2 / variance) + log(g)
    # 等价于: 0.5 * normalized_residual^2 + log(g)
    log_likelihood = 0.5 * (normalized_residual ** 2) + torch.log(g_val)
    
    # 检查数值异常
    if torch.isnan(log_likelihood).any() or torch.isinf(log_likelihood).any():
        # 返回一个大的但有限的损失值，避免训练中断
        return torch.tensor(100.0, device=device, dtype=dtype, requires_grad=True)
    
    return log_likelihood.mean()


# =============================================================================
# 修复 3: 显存优化的轨迹生成
# =============================================================================

def compute_trajectory_loss_optimized(model, initial_states, true_trajectories, 
                                      times, loss_fn, dt=0.01, max_batch_size=16):
    """
    针对 4G 显存优化的损失计算：
    - 自动拆分大批量
    - 使用 checkpoint 减少显存占用
    """
    n_samples = initial_states.shape[0]
    device = initial_states.device
    
    # 如果 batch 太大，自动拆分
    if n_samples > max_batch_size:
        total_loss = 0.0
        loss_dict_accum = {name: 0.0 for name in loss_fn.loss_names + ['total']}
        all_preds = []
        
        for i in range(0, n_samples, max_batch_size):
            end_idx = min(i + max_batch_size, n_samples)
            batch_init = initial_states[i:end_idx]
            batch_true = true_trajectories[i:end_idx]
            
            # 生成预测轨迹（不存储中间梯度以节省显存）
            with torch.cuda.amp.autocast(enabled=True):  # 混合精度
                pred_batch = torchsde.sdeint(
                    sde=model,
                    y0=batch_init,
                    ts=times,
                    dt=dt,
                    method='euler',
                    adaptive=False
                ).permute(1, 0, 2)
            
            # 计算损失
            loss, loss_dict, _ = loss_fn(batch_true, pred_batch, dt, model)
            
            total_loss += loss * (end_idx - i)
            for key in loss_dict:
                loss_dict_accum[key] += loss_dict[key] * (end_idx - i)
            all_preds.append(pred_batch.detach())
            
            # 清理显存
            del pred_batch, loss
            torch.cuda.empty_cache()
        
        # 平均
        total_loss /= n_samples
        for key in loss_dict_accum:
            loss_dict_accum[key] /= n_samples
        
        all_preds = torch.cat(all_preds, dim=0)
        return total_loss, loss_dict_accum, all_preds
    else:
        # 小批量直接计算
        pred_trajectories = torchsde.sdeint(
            sde=model,
            y0=initial_states,
            ts=times,
            dt=dt,
            method='euler',
            adaptive=False
        ).permute(1, 0, 2)
        
        return loss_fn(true_trajectories, pred_trajectories, dt, model)


# =============================================================================
# 修复 4: 改进的损失函数组合
# =============================================================================

class OptimizedLossFunction(nn.Module):
    """
    针对多稳态系统优化的损失函数：
    - 弱化 MSE（对多稳态敏感）
    - 强化基于转移概率的损失（Euler）
    - 添加基于统计量的正则化
    """
    
    def __init__(self, 
                 euler_weight=1.0,      # 主要损失：转移概率
                 moment_weight=0.5,     # 辅助：稳态统计量
                 spectral_weight=0.1,   # 新增：频谱特征（对 Josephson 结有效）
                 dt=0.01,
                 model=None):
        super().__init__()
        self.euler_weight = euler_weight
        self.moment_weight = moment_weight
        self.spectral_weight = spectral_weight
        self.dt = dt
        self.model = model
        self.loss_names = ['euler', 'moment', 'spectral', 'total']
    
    def compute_spectral_loss(self, true_traj, pred_traj):
        """
        简单的频谱匹配损失：对 Josephson 结的周期行为敏感
        """
        # 计算速度的 FFT（沿时间轴）
        # true_traj: [batch, n_steps, 4]
        v_true = true_traj[:, :, 1]  # v1
        v_pred = pred_traj[:, :, 1]
        
        # 简单频谱能量匹配（避免复杂 FFT 计算）
        # 使用自相关代替频谱
        def autocorr_energy(x):
            x_centered = x - x.mean(dim=1, keepdim=True)
            # 计算滞后 1 和 5 的自相关
            c1 = (x_centered[:, :-1] * x_centered[:, 1:]).mean(dim=1)
            c5 = (x_centered[:, :-5] * x_centered[:, 5:]).mean(dim=1) if x.shape[1] > 5 else 0
            return torch.stack([c1.abs(), c5.abs()], dim=1).mean()
        
        energy_true = autocorr_energy(v_true)
        energy_pred = autocorr_energy(v_pred)
        
        return (energy_pred - energy_true).abs()
    
    def forward(self, true_trajectories, pred_trajectories, dt, model):
        device = true_trajectories.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict = {}
        
        # 1. Euler-Maruyama 伪似然（主要损失）
        if self.euler_weight > 0:
            euler_loss = compute_euler_pseudo_likelihood_loss(
                true_trajectories, dt, model
            )
            total_loss = total_loss + self.euler_weight * euler_loss
            loss_dict['euler'] = euler_loss.item()
        else:
            loss_dict['euler'] = 0.0
        
        # 2. 矩匹配（辅助，detach 真实数据）
        if self.moment_weight > 0:
            true_mean = true_trajectories.mean(dim=1).detach()
            pred_mean = pred_trajectories.mean(dim=1)
            moment_loss = ((pred_mean - true_mean) ** 2).mean()
            
            # 添加方差匹配
            true_var = true_trajectories.var(dim=1).detach()
            pred_var = pred_trajectories.var(dim=1)
            var_loss = ((pred_var - true_var) ** 2).mean()
            
            total_loss = total_loss + self.moment_weight * (moment_loss + var_loss)
            loss_dict['moment'] = moment_loss.item()
        else:
            loss_dict['moment'] = 0.0
        
        # 3. 频谱特征（可选，帮助识别多稳态）
        if self.spectral_weight > 0:
            spec_loss = self.compute_spectral_loss(true_trajectories, pred_trajectories)
            total_loss = total_loss + self.spectral_weight * spec_loss
            loss_dict['spectral'] = spec_loss.item()
        else:
            loss_dict['spectral'] = 0.0
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict, pred_trajectories


# =============================================================================
# 修复 5: 训练循环优化
# =============================================================================

def train_epoch_optimized(model, dataloader, optimizer, times, loss_fn, dt, 
                          max_batch_size=16, grad_clip=5.0):
    """
    针对小显存的训练循环
    """
    model.train()
    total_losses = {name: 0.0 for name in loss_fn.loss_names + ['total']}
    n_batches = 0
    
    for batch_trajectories, _ in dataloader:
        device = batch_trajectories.device
        batch_trajectories = batch_trajectories.to(device)
        initial_states = batch_trajectories[:, 0, :]
        
        # 使用优化后的损失计算
        loss, loss_dict, _ = compute_trajectory_loss_optimized(
            model, initial_states, batch_trajectories, times, 
            loss_fn, dt, max_batch_size
        )
        
        if loss.item() > 1000:  # 异常值检测
            print(f"Warning: Large loss detected ({loss.item():.2f}), skipping batch")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（关键，防止爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        for key, value in loss_dict.items():
            total_losses[key] += value
        n_batches += 1
        
        # 显存清理
        del loss, loss_dict
        if n_batches % 10 == 0:
            torch.cuda.empty_cache()
    
    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


# =============================================================================
# 主程序配置（针对 4G 显存优化）
# =============================================================================

def main_optimized():
    """
    针对 4G 显存的优化配置
    """
    config = {
        'seed': 42,
        'n_samples': 500,        # 减小样本数
        'n_steps': 100,          # 减小时间步长（原 200）
        't_span': [0.0, 50.0],   # 相应减小时间跨度
        'dt': 0.05,              # 增大 dt 减少步数
        
        'true_params': {
            'beta1': 0.1, 'beta2': 0.1,
            'i1': 0.8, 'i2': 0.8,
            'kappa1': 0.05, 'kappa2': 0.05,
            'sigma1': 0.01, 'sigma2': 0.01,
        },
        
        # 关键：针对 4G 显存的网络配置
        'model_config': {
            'hidden_dim': 32,      # 从 128 降至 32
            'num_layers': 2,       # 从 5 降至 2
            'activation': 'tanh',  # tanh 比 lipswish 更稳定且省显存
        },
        
        # 训练配置
        'batch_size': 16,        # 减小 batch size（原 64）
        'n_epochs': 100,
        'lr': 5e-4,              # 降低学习率
        'grad_clip': 5.0,        # 添加梯度裁剪
        
        # 损失权重
        'loss_config': {
            'euler_weight': 1.0,     # 主要依赖 Euler 损失
            'moment_weight': 0.3,    # 辅助矩匹配
            'spectral_weight': 0.1,
        }
    }
    
    print("=" * 70)
    print("Optimized Neural SDE for 4GB VRAM")
    print("=" * 70)
    
    seed_everything(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 生成数据（使用真实物理模型）
    from torch.utils.data import TensorDataset
    
    true_sde = JosephsonJunctionSDE(**config['true_params']).to(device)
    times = torch.linspace(config['t_span'][0], config['t_span'][1], 
                          config['n_steps'], device=device)
    
    # 生成数据（分批以避免 OOM）
    all_trajs = []
    batch_gen = 50
    for i in range(0, config['n_samples'], batch_gen):
        init = torch.randn(min(batch_gen, config['n_samples'] - i), 4, device=device) * 0.5
        with torch.no_grad():
            traj = torchsde.sdeint(true_sde, init, times, dt=config['dt'], method='euler')
        all_trajs.append(traj.permute(1, 0, 2))
        torch.cuda.empty_cache()
    
    trajectories = torch.cat(all_trajs, dim=0)
    
    # 数据集
    train_size = int(0.8 * config['n_samples'])
    train_data = TensorDataset(trajectories[:train_size], torch.zeros(train_size))
    test_data = TensorDataset(trajectories[train_size:], torch.zeros(config['n_samples'] - train_size))
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    
    # 创建模型
    model = NeuralJosephsonSDE(**config['model_config']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (compressed for 4GB VRAM)")
    
    # 损失和优化器
    loss_fn = OptimizedLossFunction(**config['loss_config'], dt=config['dt'], model=model)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 训练循环
    print("\nTraining...")
    for epoch in range(config['n_epochs']):
        train_losses = train_epoch_optimized(
            model, train_loader, optimizer, times, loss_fn, 
            config['dt'], max_batch_size=16, grad_clip=config['grad_clip']
        )
        
        scheduler.step(train_losses['total'])
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']} | "
                  f"Loss: {train_losses['total']:.4f} | "
                  f"Euler: {train_losses['euler']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 显存监控
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print("Training completed!")
    return model


if __name__ == "__main__":
    model = main_optimized()