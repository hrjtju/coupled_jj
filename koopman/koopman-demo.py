from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

# ==================== 0. 全局设备配置 ====================

def get_device():
    """获取最优可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 启用 cudnn 加速
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device

DEVICE = get_device()

# 检测是否支持 torch.compile (PyTorch 2.0+)
HAS_COMPILE = hasattr(torch, 'compile')
# Windows 上通常缺少 Triton，禁用 compile
import platform
if platform.system() == 'Windows':
    HAS_COMPILE = False
    print("Windows 系统，禁用 torch.compile (需要 Triton)")
elif HAS_COMPILE:
    print("检测到 PyTorch 2.0+, 可使用 torch.compile 加速")


# ==================== 1. Loss Scheduler ====================

class LossScheduler:
    """动态调整各个 Loss 权重的调度器 - 平滑版本"""
    def __init__(self, total_epochs, initial_weights=None, schedule_type='adaptive',
                 max_delta_per_epoch=0.01):  # 每轮最大变化量
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.schedule_type = schedule_type
        self.max_delta = max_delta_per_epoch  # 限制每轮权重变化幅度
        
        # 默认权重配置 - 优先保证线性一致性和重构质量
        # 简化权重配置，专注核心损失
        default_weights = {
            'linear': 1.0,      # 线性一致性（核心）
            'recon': 1.0,       # 重构损失（核心）
            'orth': 1.0,       # 正交性（辅助）
            'stable': 0.02,     # 稳定性（轻微）
            'smooth': 0.0,      # 禁用平滑性
            'sparse': 0.0       # 禁用稀疏性
        }
        
        self.weights = initial_weights if initial_weights else default_weights
        self.initial_weights = self.weights.copy()
        self.target_weights = self.weights.copy()  # 目标权重（平滑过渡用）
        self.weight_history = {k: [] for k in self.weights.keys()}
    
    def _smooth_update(self, key, target_value):
        """平滑更新权重，限制每轮变化幅度"""
        current = float(self.weights[key])
        target = float(target_value)
        delta = target - current
        # 限制变化幅度
        if delta > self.max_delta:
            delta = self.max_delta
        elif delta < -self.max_delta:
            delta = -self.max_delta
        self.weights[key] = current + delta
    
    def step(self, epoch_losses=None):
        """更新权重 - 平滑版本，避免突变"""
        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        
        if self.schedule_type == 'constant':
            pass  # 常数权重，不做任何调整
        
        elif self.schedule_type == 'linear_decay':
            # 线性插值到目标值，而非直接赋值
            target_orth = self.initial_weights['orth'] * (1.0 - 0.5 * progress)
            target_stable = self.initial_weights['stable'] * (1.0 - 0.5 * progress)
            target_recon = self.initial_weights['recon'] * (1.0 + 0.3 * progress)
            
            self._smooth_update('orth', target_orth)
            self._smooth_update('stable', target_stable)
            self._smooth_update('recon', target_recon)
        
        elif self.schedule_type == 'cosine':
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            target_orth = self.initial_weights['orth'] * (0.1 + 0.9 * cosine_factor)
            self._smooth_update('orth', target_orth)
        
        elif self.schedule_type == 'adaptive' and epoch_losses:
            recon_loss = epoch_losses.get('recon', 0)
            linear_loss = epoch_losses.get('linear', 1.0)
            
            # 计算目标权重（极小幅调整，每轮最多0.01）
            if recon_loss > linear_loss * 1.5 and recon_loss > 0.05:
                self.target_weights['recon'] = min(1.1, self.target_weights['recon'] + 0.01)
                self.target_weights['sparse'] = max(0.01, self.target_weights['sparse'] - 0.005)
            elif recon_loss < 0.01:  # 重构很好时
                self.target_weights['sparse'] = min(0.05, self.target_weights['sparse'] + 0.001)
            
            # 平滑过渡到目标权重
            for key in ['recon', 'sparse']:
                self._smooth_update(key, self.target_weights[key])
        
        # 阶段性调整已移除，所有调整都通过 _smooth_update 平滑进行
        
        for k in self.weights:
            self.weight_history[k].append(self.weights[k])
    
    def get_weights(self):
        return self.weights.copy()
    
    def get_weight_summary(self):
        return " | ".join([f"{k}={v:.3f}" for k, v in self.weights.items()])


# ==================== 2. OU过程数据生成器（优化版） ====================

class OUProcess:
    """Ornstein-Uhlenbeck过程 - 向量化生成"""
    def __init__(self, theta=1.0, mu=0.0, sigma=0.5, device=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.device = device if device is not None else DEVICE
    
    def generate_trajectories(self, n_traj=1000, n_steps=1000, dt=0.05, x0_range=(-2, 2)):
        """向量化生成轨迹数据 [n_traj, n_steps+1, 1] (生成在CPU，训练时再移到GPU)"""
        # 在CPU上生成数据，以便使用 DataLoader 的 pin_memory
        x0 = torch.FloatTensor(n_traj, 1).uniform_(*x0_range)
        
        # 预分配内存
        trajectories = torch.empty(n_traj, n_steps+1, 1)
        trajectories[:, 0] = x0
        
        # 向量化 Euler-Maruyama
        sqrt_dt = np.sqrt(dt)
        dW = torch.randn(n_traj, n_steps, 1) * sqrt_dt * self.sigma
        
        X = x0
        for t in range(n_steps):
            dX = self.theta * (self.mu - X) * dt + dW[:, t]
            X = X + dX
            trajectories[:, t+1] = X
            
        return trajectories
    
    def theoretical_eigenvalues(self, max_n=5):
        """理论Koopman特征值"""
        return [-n * self.theta for n in range(max_n)]


# ==================== 3. CNN Koopman 网络（加速版） ====================

class CNNKoopmanNet(nn.Module):
    """
    基于 1D CNN 的 Koopman 网络 - 优化训练速度版本
    """
    def __init__(self, state_dim=1, koopman_dim=16, hidden_channels=32, device=None, 
                 sparsity_target=0.4, sparsity_type='hoyer'):
        super().__init__()
        self.state_dim = state_dim
        self.koopman_dim = koopman_dim
        self.hidden_channels = hidden_channels
        self.device = device if device is not None else DEVICE
        self.sparsity_target = sparsity_target
        self.sparsity_type = sparsity_type
        
        # ===== 编码器 =====
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(state_dim, hidden_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=1, padding=0),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Tanh(),
            nn.Conv1d(hidden_channels * 2, koopman_dim, kernel_size=1, padding=0),
            nn.BatchNorm1d(koopman_dim),
            nn.Tanh()
        )
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)
        
        # 生成元矩阵
        init_diag = torch.linspace(0, 0, koopman_dim)
        self.K = nn.Parameter(torch.diag(init_diag))
        
        # ===== 解码器 =====
        self.decoder_expand = nn.Linear(koopman_dim, koopman_dim * 2)
        
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.ConvTranspose1d(koopman_dim * 2, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.Tanh(),
            nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.Tanh(),
            nn.ConvTranspose1d(hidden_channels, state_dim, kernel_size=3, padding=1)
        )
        self.decoder_pool = nn.AdaptiveAvgPool1d(1)
        
        self.to(self.device)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """编码: x -> z"""
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D).unsqueeze(-1)
            squeeze_output = True
        
        z = self.encoder_conv(x)
        z = self.encoder_pool(z).squeeze(-1)
        return z
    
    def decode(self, z):
        """解码: z -> x"""
        if z.dim() == 3:
            B, T, D = z.shape
            z = z.reshape(B * T, D)
            squeeze_output = True
        else:
            squeeze_output = False
        
        h = self.decoder_expand(z).unsqueeze(-1)
        x = self.decoder_conv(h)
        x = self.decoder_pool(x).squeeze(-1)
        
        if squeeze_output:
            x = x.reshape(B, T, -1)
        return x
    
    def compute_sparsity_loss(self, z):
        """计算稀疏性损失 - 温和版本，避免过度稀疏"""
        if self.sparsity_target == 0.0 or self.sparsity_type == 'none':
            return torch.tensor(0.0, device=self.device)
        
        if self.sparsity_type == 'l1':
            # L1 正则化，但进行归一化避免数值过大
            return torch.mean(torch.abs(z)) / np.sqrt(self.koopman_dim)
        
        elif self.sparsity_type == 'hoyer':
            # Hoyer 稀疏度，但使用更温和的目标
            n = self.koopman_dim
            l1 = torch.sum(torch.abs(z), dim=-1)
            l2 = torch.sqrt(torch.sum(z**2, dim=-1) + 1e-8)
            hoyer = (np.sqrt(n) - l1 / (l2 + 1e-8)) / (np.sqrt(n) - 1)
            # 使用 soft 约束，避免过度惩罚
            target_hoyer = min(self.sparsity_target, 0.5)  # 限制最大稀疏度
            deviation = torch.abs(hoyer - target_hoyer)
            # 只对过度稠密进行惩罚，对过度稀疏放宽
            loss = torch.where(hoyer > target_hoyer, 
                              deviation ** 2,  # 稠密时严格惩罚
                              deviation * 0.3)  # 稀疏时宽松
            return torch.mean(loss)
        
        elif self.sparsity_type == 'entropy':
            z_sq = z ** 2 + 1e-8
            probs = z_sq / torch.sum(z_sq, dim=-1, keepdim=True)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            # 归一化熵，0=最稀疏，1=最均匀
            normalized_entropy = entropy / np.log(self.koopman_dim)
            # 目标：适中的熵值（不太稀疏也不太均匀）
            target_entropy = 0.6  # 中等活跃度
            return torch.mean((normalized_entropy - target_entropy) ** 2)
        
        return torch.tensor(0.0, device=self.device)
    
    def evolve(self, z0, t):
        """
        Koopman空间线性演化 - 优化版
        使用批量矩阵指数避免循环
        """
        batch_size = z0.shape[0]
        
        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), t, device=z0.device, dtype=z0.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        
        # 如果所有时间相同，可以批量计算
        if torch.allclose(t, t[0]):
            K_exp = torch.matrix_exp(self.K * t[0])
            return torch.matmul(z0, K_exp.T)
        
        # 否则逐个计算（很少发生）
        results = []
        for i in range(batch_size):
            K_exp = torch.matrix_exp(self.K * t[i])
            zt = K_exp @ z0[i]
            results.append(zt)
        return torch.stack(results)
    
    def forward(self, x0, t):
        """完整前向"""
        z0 = self.encode(x0)
        zt = self.evolve(z0, t)
        return self.decode(zt), zt
    
    def compute_loss(self, trajectory_batch, dt, weights=None):
        """计算Koopman学习损失 - 优化版"""
        if weights is None:
            weights = {
                'linear': 1.0, 'recon': 1.0, 'orth': 1.0,
                'stable': 0.02, 'smooth': 0.0, 'sparse': 0.0
            }
        
        B, T, D = trajectory_batch.shape
        device = trajectory_batch.device
        
        # 编码
        z_flat = self.encode(trajectory_batch.reshape(B * T, D))
        z = z_flat.reshape(B, T, self.koopman_dim)
        
        # 1. 线性一致性损失 - 使用向量化计算
        if T > 1:
            z_t = z[:, :-1, :]  # [B, T-1, K]
            z_next = z[:, 1:, :]  # [B, T-1, K]
            
            # 批量演化预测
            B_flat, T_flat, K = z_t.shape
            z_t_flat = z_t.reshape(B_flat * T_flat, K)
            
            # 假设所有 dt 相同（常见情况），批量计算
            z_pred = self.evolve(z_t_flat, dt)
            z_pred = z_pred.reshape(B_flat, T_flat, K)
            
            loss_lin = torch.mean((z_pred - z_next)**2)
        else:
            loss_lin = torch.tensor(0.0, device=device)
        
        # 2. 重构损失
        x_rec_flat = self.decode(z_flat)
        x_rec = x_rec_flat.reshape(B, T, D)
        loss_rec = torch.mean((trajectory_batch - x_rec)**2)
        
        # 3. 正交归一化损失
        cov = torch.matmul(z_flat.T, z_flat) / z_flat.shape[0]
        diff = (cov - torch.eye(self.koopman_dim, device=device))**2
        loss_orth = torch.mean(diff) + torch.max(diff)
        
        # 4. 稳定性约束
        eigs = torch.linalg.eigvals(self.K).real
        loss_stable = torch.mean(torch.relu(eigs))
        
        # 5. 平滑性约束
        loss_smooth = torch.mean(torch.diff(z, dim=1)**2) if T > 1 else torch.tensor(0.0, device=device)
        
        # 6. 稀疏性损失
        loss_sparse = self.compute_sparsity_loss(z_flat)
        
        # 加权总损失
        # 重构保护：仅当重构误差非常大时才调整，且保持辅助损失的微小贡献
        if loss_rec.item() > 2.0:
            # 重构很差时，提升重构权重，但保持其他损失的最小贡献
            effective_weights = {
                'linear': weights.get('linear', 1.0) * 0.5,
                'recon': weights.get('recon', 1.0) * 2.0,
                'orth': 0.01,
                'stable': 0.01,
                'smooth': 0.0,
                'sparse': 0.0
            }
        else:
            effective_weights = weights
        
        total_loss = (
            effective_weights.get('linear', 1.0) * loss_lin +
            effective_weights.get('recon', 0.5) * loss_rec +
            effective_weights.get('orth', 0.1) * loss_orth +
            effective_weights.get('stable', 0.1) * loss_stable +
            effective_weights.get('smooth', 0.05) * loss_smooth +
            effective_weights.get('sparse', 0.15) * loss_sparse
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'linear': loss_lin.item(),
            'recon': loss_rec.item(),
            'orth': loss_orth.item(),
            'stable': loss_stable.item(),
            'smooth': loss_smooth.item(),
            'sparse': loss_sparse.item()
        }


# ==================== 4. 训练函数（加速版） ====================

def train_koopman(model, ou_process, n_epochs=200, batch_size=512, dt=0.05, 
                  lr=1e-3, device=None, schedule_type='adaptive', use_amp=True,
                  compile_model=False):
    """训练Koopman网络 - 加速版本"""
    if device is None:
        device = DEVICE
    model = model.to(device)
    
    # 使用 torch.compile 加速 (PyTorch 2.0+)
    # if compile_model and HAS_COMPILE:
    #     print("编译模型 (torch.compile)...")
    #     try:
    #         # model = torch.compile(model, mode='default')
    #         print("模型编译完成")
    #     except Exception as e:
    #         print(f"模型编译失败: {e}, 使用普通模式")
    
    # 混合精度训练
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    amp_enabled = scaler is not None
    if amp_enabled:
        print("启用混合精度训练 (AMP)")
    
    loss_scheduler = LossScheduler(
        total_epochs=n_epochs,
        schedule_type=schedule_type
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    print("生成训练数据...")
    trajectories = ou_process.generate_trajectories(
        n_traj=3000, n_steps=25, dt=dt, x0_range=(-3, 3)
    )
    
    # DataLoader 优化
    dataset = TensorDataset(trajectories)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows 下建议用 0
        pin_memory=device.type == 'cuda',
        persistent_workers=False
    )
    
    print(f"开始训练... 设备: {device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e3:.1f}K)")
    print(f"Batch Size: {batch_size}")
    print(f"AMP: {amp_enabled}")
    print(f"Compile: {compile_model and HAS_COMPILE}")
    print("-" * 110)
    
    print(f"{'Epoch':>5} | {'Time':>6} | {'Total':>7} | {'Linear':>7} | {'Recon':>7} | {'Orth':>7} | "
          f"{'Stable':>7} | {'Sparse':>7} | {'Eigenvalues':<20} | {'Weights':<15}")
    print("-" * 110)
    
    start_time = time.time()
    best_recon_loss = float('inf')  # 记录最佳重构误差
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_losses = {k: [] for k in ['total', 'linear', 'recon', 'orth', 'stable', 'smooth', 'sparse']}
        
        weights = loss_scheduler.get_weights()
        
        for batch in loader:
            traj = batch[0].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向
            if amp_enabled:
                with autocast(device_type="cuda"):
                    loss, loss_dict = model.compute_loss(traj, dt, weights)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = model.compute_loss(traj, dt, weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)
        
        scheduler.step()
        
        avg_loss = {k: np.mean(v) for k, v in epoch_losses.items()}
        loss_scheduler.step(avg_loss)
        
        epoch_time = time.time() - epoch_start
        
        # 监控重构误差（仅记录，不直接干预权重，由 scheduler 统一处理）
        if avg_loss['recon'] < best_recon_loss:
            best_recon_loss = avg_loss['recon']
        # 移除直接修改 weights 的代码，避免与 scheduler 冲突
        
        # 每 10 轮打印一次（降低打印频率）
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                eigs = torch.linalg.eigvals(model.K).real.cpu().numpy()
                eigs_sorted = np.sort(eigs)[::-1]
                # 显示前4个特征值，更容易看出偏离
                eigs_str = ', '.join([f'{e:.2f}' for e in eigs_sorted[:4]])
            
            # 显示当前权重（简洁格式）
            w = loss_scheduler.get_weights()
            weight_str = f"R={w['recon']:.2f},S={w['sparse']:.2f}"
            
            print(f"{epoch:5d} | {epoch_time:5.2f}s | {avg_loss['total']:7.4f} | "
                  f"{avg_loss['linear']:7.4f} | {avg_loss['recon']:7.4f} | "
                  f"{avg_loss['orth']:7.4f} | {avg_loss['stable']:7.4f} | "
                  f"{avg_loss['sparse']:7.4f} | [{eigs_str}] | {weight_str}")
    
    total_time = time.time() - start_time
    print("-" * 110)
    print(f"训练完成！总耗时: {total_time:.1f}s, 平均每轮: {total_time/n_epochs:.2f}s")
    
    return model, loss_scheduler


# ==================== 5. 验证与可视化 ====================

def validate_and_plot(model, ou_process, dt=0.05, device=None, loss_scheduler=None):
    """验证训练结果并绘图"""
    if device is None:
        device = DEVICE
    device = next(model.parameters()).device
    
    # 设置为评估模式（避免 BatchNorm 在 batch_size=1 时出错）
    model.eval()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 测试1: 长期预测能力对比
    print("\n=== 验证长期预测 ===")
    with torch.no_grad():
        x0 = torch.tensor([[2.0]], device=device)
        n_steps = 100
        
        true_traj = [x0.item()]
        x = x0.clone()
        for _ in range(n_steps):
            dW = torch.randn_like(x) * np.sqrt(dt)
            x = x + ou_process.theta * (ou_process.mu - x) * dt + ou_process.sigma * dW
            true_traj.append(x.item())
        
        pred_traj = [x0.item()]
        x_current = x0
        for _ in range(n_steps):
            x_pred, _ = model(x_current, dt)
            pred_traj.append(x_pred.item())
            x_current = x_pred
    
    # 图1: 轨迹对比
    t = np.arange(len(true_traj)) * dt
    ax1.plot(t, true_traj, 'b-', label='True OU', alpha=0.7, linewidth=2)
    ax1.plot(t, pred_traj, 'r--', label='Koopman CNN', alpha=0.7, linewidth=2)
    ax1.axhline(y=ou_process.mu, color='k', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X(t)')
    ax1.set_title('Long-term Prediction (CNN)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 学到的特征函数可视化
    with torch.no_grad():
        x_range = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)
        z = model.encode(x_range).cpu().numpy()
        x_np = x_range.cpu().numpy()
        
        n_show = min(6, model.koopman_dim)
        for i in range(n_show):
            ax2.plot(x_np, z[:, i], label=f'φ_{i}', linewidth=2, alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Feature value')
    ax2.set_title(f'Eigenfunctions ({n_show}/{model.koopman_dim})')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 图3: 特征值对比
    with torch.no_grad():
        learned_eigs = torch.linalg.eigvals(model.K).real.cpu().numpy()
        theoretical_eigs = np.array(ou_process.theoretical_eigenvalues(model.koopman_dim))
        
        learned_eigs = np.sort(learned_eigs)[::-1]
        theoretical_eigs = np.sort(theoretical_eigs)[::-1]
        
        n_eigs = min(8, model.koopman_dim)
        x_pos = np.arange(n_eigs)
        width = 0.35
        
        ax3.bar(x_pos - width/2, learned_eigs[:n_eigs], width, 
                label='Learned', alpha=0.8, color='steelblue')
        ax3.bar(x_pos + width/2, theoretical_eigs[:n_eigs], width, 
                label='Theory', alpha=0.8, color='coral')
        ax3.set_xlabel('Mode')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title(f'Eigenvalues (top {n_eigs})')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{i}' for i in range(n_eigs)])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 图4: Koopman空间中的轨迹
    with torch.no_grad():
        x0s = torch.linspace(-2, 2, 5, device=device).unsqueeze(1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(x0s)))
        
        for i, x0 in enumerate(x0s):
            z0 = model.encode(x0.unsqueeze(0))
            z_traj = [z0[0, :2].cpu().numpy()]
            
            z = z0
            for _ in range(20):
                z = model.evolve(z, dt)
                z_traj.append(z[0, :2].cpu().numpy())
            
            z_traj = np.array(z_traj)
            ax4.plot(z_traj[:, 0], z_traj[:, 1], 'o-', color=colors[i], 
                    markersize=4, label=f'x0={x0.item():.1f}')
            ax4.scatter(z_traj[0, 0], z_traj[0, 1], s=100, c=[colors[i]], marker='*', zorder=5)
    
    ax4.set_xlabel('z_0')
    ax4.set_ylabel('z_1')
    ax4.set_title('Trajectories in Koopman Space')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 图5: 特征活跃度
    with torch.no_grad():
        # 确保 z 是标准 numpy 数组
        if isinstance(z, torch.Tensor):
            z_np = z.cpu().numpy()
        else:
            z_np = np.array(z)
        z_abs_mean = np.mean(np.abs(z_np), axis=0)
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(z_abs_mean)))
        
        n_bar = min(12, model.koopman_dim)
        ax5.bar(range(n_bar), z_abs_mean[:n_bar], color=colors_bar[:n_bar])
        ax5.axhline(y=np.mean(z_abs_mean), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(z_abs_mean):.3f}')
        ax5.set_xlabel('Feature dim')
        ax5.set_ylabel('Mean |value|')
        ax5.set_title('Feature Activity')
        ax5.set_xticks(range(n_bar))
        ax5.set_xticklabels([f'{i}' for i in range(n_bar)], fontsize=8)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 图6: Loss 权重变化历史
    if loss_scheduler is not None and len(loss_scheduler.weight_history['linear']) > 0:
        epochs_hist = range(len(loss_scheduler.weight_history['linear']))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for key, color in zip(['linear', 'recon', 'orth', 'stable', 'smooth', 'sparse'], colors):
            if key in loss_scheduler.weight_history:
                ax6.plot(epochs_hist, loss_scheduler.weight_history[key], 
                        label=f'{key}', linewidth=2, color=color)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Weight')
        ax6.set_title('Dynamic Loss Weights')
        ax6.legend(loc='upper right', ncol=2, fontsize=8)
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No Scheduler Data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Dynamic Loss Weights')
    
    plt.tight_layout()
    plt.savefig('ou_koopman_cnn_fast.png', dpi=150, bbox_inches='tight')
    print("结果已保存至 ou_koopman_cnn_fast.png")
    plt.show()
    
    # 打印最终统计
    print(f"\n学习到的特征值 (top 8): {np.sort(learned_eigs)[::-1][:8].round(2)}")
    print(f"理论特征值 (top 8):     {theoretical_eigs[:8].round(2)}")
    
    with torch.no_grad():
        z_all = model.encode(x_range)
        z_np = z_all.cpu().numpy()
        l1_norm = np.mean(np.abs(z_np))
        active_ratio = np.mean(np.abs(z_np) > 0.1 * np.max(np.abs(z_np)))
        print(f"\n稀疏度统计:")
        print(f"  平均 L1 范数: {l1_norm:.4f}")
        print(f"  激活比例: {active_ratio:.2%}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    theta, mu, sigma, dt = 1.0, 0.0, 0.5, 0.05
    
    print("=" * 70)
    print(f"OU参数: θ={theta}, μ={mu}, σ={sigma}")
    print(f"理论特征值: {OUProcess(theta, mu, sigma).theoretical_eigenvalues(16)[:8]}")
    print("=" * 70)
    
    ou = OUProcess(theta, mu, sigma, device=DEVICE)
    model = CNNKoopmanNet(
        state_dim=1, 
        koopman_dim=4,
        hidden_channels=32,        # 减少通道数
        device=DEVICE,
        sparsity_target=0.0,       # 禁用稀疏性
        sparsity_type='none'
    )
    
    # 训练 - 保守设置以确保稳定性
    # schedule_type 可选: 'constant' | 'linear_decay' | 'cosine' | 'adaptive'
    trained_model, loss_scheduler = train_koopman(
        model, ou, 
        n_epochs=600,
        batch_size=256,
        dt=dt,
        device=DEVICE,
        schedule_type='adaptive',  # 使用常数权重，避免动态调整导致不稳定
        use_amp=False,             # 禁用混合精度，避免数值问题
        compile_model=False
    )
    
    validate_and_plot(trained_model, ou, dt=dt, device=DEVICE, loss_scheduler=loss_scheduler)
    
    print("\n" + "=" * 70)
    print("加速版 CNN 模型配置：")
    print(f"  - Koopman 维度: {model.koopman_dim}")
    print(f"  - 隐藏通道数: {model.hidden_channels}")
    print(f"  - 混合精度训练 (AMP): 启用")
    print(f"  - torch.compile: {'启用' if HAS_COMPILE else '未启用 (PyTorch < 2.0)'}")
    print(f"  - cuDNN 加速: 启用")
    print("=" * 70)
