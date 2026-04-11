import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ==================== 0. 全局设备配置 ====================

def get_device():
    """获取最优可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用 MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device

# 全局 device 变量
DEVICE = get_device()


# ==================== 1. Loss Scheduler ====================

class LossScheduler:
    """
    动态调整各个 Loss 权重的调度器
    支持多种调度策略：常数、线性衰减、余弦退火、阶段性调整
    """
    def __init__(self, total_epochs, initial_weights=None, schedule_type='adaptive'):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.schedule_type = schedule_type
        
        # 默认权重
        default_weights = {
            'linear': 1.0,
            'recon': 0.5,
            'orth': 0.1,
            'stable': 0.1,
            'smooth': 0.05,
            'sparse': 0.2  # 新增稀疏损失
        }
        
        self.weights = initial_weights if initial_weights else default_weights
        self.initial_weights = self.weights.copy()
        self.weight_history = {k: [] for k in self.weights.keys()}
    
    def step(self, epoch_losses=None):
        """更新权重，可选择基于当前 loss 自适应调整"""
        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        
        if self.schedule_type == 'constant':
            # 保持初始权重不变
            pass
        
        elif self.schedule_type == 'linear_decay':
            # 线性衰减：某些 loss 随着训练进行逐渐减小
            decay_factor = 1.0 - 0.5 * progress
            self.weights['orth'] = self.initial_weights['orth'] * decay_factor
            self.weights['stable'] = self.initial_weights['stable'] * decay_factor
            self.weights['smooth'] = self.initial_weights['smooth'] * decay_factor
            # 重构损失逐渐增加重要性
            self.weights['recon'] = self.initial_weights['recon'] * (1.0 + 0.5 * progress)
        
        elif self.schedule_type == 'cosine':
            # 余弦调度
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            self.weights['orth'] = self.initial_weights['orth'] * (0.1 + 0.9 * cosine_factor)
            self.weights['sparse'] = self.initial_weights['sparse'] * (0.2 + 0.8 * cosine_factor)
        
        elif self.schedule_type == 'adaptive' and epoch_losses:
            # 自适应调整：根据当前 loss 大小动态平衡
            # 如果重构 loss 很大，增加其权重
            if epoch_losses.get('recon', 0) > epoch_losses.get('linear', 1.0):
                self.weights['recon'] = min(2.0, self.weights['recon'] * 1.02)
            else:
                self.weights['recon'] = max(0.1, self.weights['recon'] * 0.995)
            
            # 训练后期增加稀疏性约束
            if progress > 0.5:
                self.weights['sparse'] = self.initial_weights['sparse'] * (1.0 + progress)
        
        # 阶段性调整：特定 epoch 大幅改变权重
        if self.current_epoch == self.total_epochs // 3:
            # 第一阶段结束，增加稀疏性
            self.weights['sparse'] *= 1.5
        elif self.current_epoch == 2 * self.total_epochs // 3:
            # 第二阶段结束，强调重构精度
            self.weights['recon'] *= 1.3
            self.weights['linear'] *= 1.2
        
        # 记录历史
        for k in self.weights:
            self.weight_history[k].append(self.weights[k])
    
    def get_weights(self):
        """获取当前权重"""
        return self.weights.copy()
    
    def get_weight_summary(self):
        """获取权重摘要字符串"""
        return " | ".join([f"{k}={v:.4f}" for k, v in self.weights.items()])


# ==================== 2. OU过程数据生成器 ====================

class OUProcess:
    """
    Ornstein-Uhlenbeck过程: dX = theta*(mu - X)*dt + sigma*dW
    理论Koopman特征值: lambda_n = -n*theta (n=0,1,2,...)
    理论特征函数: Hermite多项式
    """
    def __init__(self, theta=1.0, mu=0.0, sigma=0.5, device=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.device = device if device is not None else DEVICE
    
    def generate_trajectories(self, n_traj=1000, n_steps=50, dt=0.05, x0_range=(-2, 2)):
        """生成轨迹数据 [n_traj, n_steps+1, 1]"""
        x0 = torch.FloatTensor(n_traj, 1).uniform_(*x0_range).to(self.device)
        trajectories = torch.zeros(n_traj, n_steps+1, 1).to(self.device)
        trajectories[:, 0] = x0
        
        # Euler-Maruyama离散化
        for t in range(n_steps):
            dW = torch.randn_like(x0) * np.sqrt(dt)
            X = trajectories[:, t]
            dX = self.theta * (self.mu - X) * dt + self.sigma * dW
            trajectories[:, t+1] = X + dX
            
        return trajectories
    
    def theoretical_eigenvalues(self, max_n=5):
        """理论Koopman特征值: 0, -theta, -2*theta, ..."""
        return [-n * self.theta for n in range(max_n)]


# ==================== 3. Koopman神经网络（增强版） ====================

class SimpleKoopmanNet(nn.Module):
    """
    学习SDE的Koopman算子近似（高维特征版）：
    - 编码器: 物理空间 -> Koopman特征空间 (更深的非线性变换)
    - 生成元K: 线性演化规则 (可学习矩阵)
    - 解码器: Koopman空间 -> 物理空间
    - 稀疏性: 鼓励特征表示稀疏
    """
    def __init__(self, state_dim=1, koopman_dim=32, hidden_dim=256, device=None, 
                 sparsity_target=0.3, sparsity_type='l1'):
        super().__init__()
        self.koopman_dim = koopman_dim
        self.device = device if device is not None else DEVICE
        self.sparsity_target = sparsity_target  # 目标稀疏度 (0-1)
        self.sparsity_type = sparsity_type  # 'l1' 或 'hoyer' 或 'entropy'
        
        # 编码器：更深的网络结构，学习特征函数 phi_1, phi_2, ..., phi_K
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, koopman_dim)
        )
        
        # 生成元矩阵 (Koopman算子的有限维近似)
        # 初始化为接近理论值的对角矩阵: 0, -1, -2, -3...
        init_diag = torch.linspace(0, -koopman_dim+1, koopman_dim)
        self.K = nn.Parameter(torch.diag(init_diag))
        
        # 添加非对角元素，允许模态间弱耦合
        self.K_offdiag = nn.Parameter(torch.randn(koopman_dim, koopman_dim) * 0.01)
        
        # 解码器：更深的网络结构，从特征空间重构原始状态
        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 将模型移到指定设备
        self.to(self.device)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """He初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 非对角元素初始化为小值
        nn.init.normal_(self.K_offdiag, mean=0, std=0.01)
    
    def get_K_matrix(self):
        """获取完整的生成元矩阵（对角 + 非对角）"""
        # 使用掩码确保对角线来自 K，非对角来自 K_offdiag
        mask = torch.eye(self.koopman_dim, device=self.device)
        return self.K * mask + self.K_offdiag * (1 - mask)
    
    def encode(self, x):
        """Koopman特征函数: x -> z"""
        return self.encoder(x)
    
    def decode(self, z):
        """重构: z -> x"""
        return self.decoder(z)
    
    def compute_sparsity_loss(self, z):
        """
        计算稀疏性损失
        
        Args:
            z: [batch, koopman_dim] 特征向量
        """
        if self.sparsity_type == 'l1':
            # L1 稀疏性
            return torch.mean(torch.abs(z))
        
        elif self.sparsity_type == 'hoyer':
            # Hoyer 稀疏度 (0 = 稠密, 1 = 稀疏)
            # H(z) = (sqrt(n) - L1/L2) / (sqrt(n) - 1)
            n = self.koopman_dim
            l1 = torch.sum(torch.abs(z), dim=-1)
            l2 = torch.sqrt(torch.sum(z**2, dim=-1) + 1e-8)
            hoyer = (np.sqrt(n) - l1 / (l2 + 1e-8)) / (np.sqrt(n) - 1)
            # 我们希望 hoyer 接近目标稀疏度
            target_hoyer = self.sparsity_target
            return torch.mean((hoyer - target_hoyer)**2)
        
        elif self.sparsity_type == 'entropy':
            # 熵稀疏性 (鼓励概率分布集中在少数维度)
            z_sq = z ** 2
            probs = z_sq / (torch.sum(z_sq, dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # 最大熵是 log(n)，我们希望熵小
            max_entropy = np.log(self.koopman_dim)
            return torch.mean(entropy / max_entropy)
        
        elif self.sparsity_type == 'k_sparse':
            # Top-k 稀疏性
            k = max(1, int(self.koopman_dim * (1 - self.sparsity_target)))
            z_sorted = torch.sort(torch.abs(z), dim=-1, descending=True)[0]
            # 鼓励非 top-k 元素接近 0
            tail = z_sorted[:, k:]
            return torch.mean(tail ** 2)
        
        else:
            return torch.tensor(0.0, device=self.device)
    
    def evolve(self, z0, t):
        """
        Koopman空间线性演化: z(t) = exp(K*t) @ z0
        解析可积，无累积误差！
        """
        batch_size = z0.shape[0]
        K_full = self.get_K_matrix()
        
        # 统一时间维度
        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), t, device=z0.device, dtype=z0.dtype)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        
        # 对每个batch元素计算矩阵指数演化
        results = []
        for i in range(batch_size):
            # 矩阵指数: exp(K * t[i])
            K_exp = torch.matrix_exp(K_full * t[i])
            zt = K_exp @ z0[i]
            results.append(zt)
        
        return torch.stack(results)
    
    def forward(self, x0, t):
        """完整前向: 编码 -> 线性演化 -> 解码"""
        z0 = self.encode(x0)
        zt = self.evolve(z0, t)
        return self.decode(zt), zt
    
    def compute_loss(self, trajectory_batch, dt, weights=None):
        """
        计算Koopman学习损失
        
        Args:
            trajectory_batch: [batch, time_steps, state_dim] 轨迹数据
            dt: 时间步长
            weights: 各 loss 的权重字典
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典
        """
        if weights is None:
            weights = {
                'linear': 1.0,
                'recon': 0.5,
                'orth': 0.1,
                'stable': 0.1,
                'smooth': 0.05,
                'sparse': 0.2
            }
        
        B, T, D = trajectory_batch.shape
        device = trajectory_batch.device
        
        # 编码整条轨迹到Koopman空间
        z = self.encode(trajectory_batch.reshape(-1, D)).reshape(B, T, self.koopman_dim)
        
        # 1. 线性一致性损失 (核心): z_{t+1} ≈ exp(K*dt) @ z_t
        loss_lin = 0
        for t in range(T-1):
            z_pred = self.evolve(z[:, t], dt)
            loss_lin += torch.mean((z_pred - z[:, t+1])**2)
        loss_lin = loss_lin / (T-1)
        
        # 2. 重构损失: 确保信息不丢失
        x_rec = self.decode(z.reshape(-1, self.koopman_dim)).reshape(B, T, D)
        loss_rec = torch.mean((trajectory_batch - x_rec)**2)
        
        # 3. 正交归一化损失: 鼓励特征函数形成正交基
        z_flat = z.reshape(-1, self.koopman_dim)
        cov = (z_flat.T @ z_flat) / z_flat.shape[0]
        loss_orth = torch.mean((cov - torch.eye(self.koopman_dim, device=device))**2)
        
        # 4. 稳定性约束: 特征值实部应为负（衰减模态）
        K_full = self.get_K_matrix()
        eigs = torch.linalg.eigvals(K_full).real
        loss_stable = torch.mean(torch.relu(eigs))
        
        # 5. 平滑性约束: 特征函数应该平滑
        z_diff = torch.diff(z, dim=1)
        loss_smooth = torch.mean(z_diff**2)
        
        # 6. 稀疏性损失
        loss_sparse = self.compute_sparsity_loss(z_flat)
        
        # 加权总损失
        total_loss = (
            weights.get('linear', 1.0) * loss_lin +
            weights.get('recon', 0.5) * loss_rec +
            weights.get('orth', 0.1) * loss_orth +
            weights.get('stable', 0.1) * loss_stable +
            weights.get('smooth', 0.05) * loss_smooth +
            weights.get('sparse', 0.2) * loss_sparse
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


# ==================== 4. 训练函数 ====================

def train_koopman(model, ou_process, n_epochs=400, batch_size=512, dt=0.05, 
                  lr=1e-3, device=None, schedule_type='adaptive'):
    """训练Koopman网络"""
    if device is None:
        device = DEVICE
    model = model.to(device)
    
    # 初始化 Loss Scheduler
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
        n_traj=15000, n_steps=40, dt=dt, x0_range=(-4, 4)
    )
    
    dataset = TensorDataset(trajectories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"开始训练... 设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Koopman 维度: {model.koopman_dim}")
    print(f"Loss 调度策略: {schedule_type}")
    print("-" * 100)
    
    # 打印表头
    print(f"{'Epoch':>6} | {'Total':>8} | {'Linear':>8} | {'Recon':>8} | {'Orth':>8} | "
          f"{'Stable':>8} | {'Smooth':>8} | {'Sparse':>8} | {'Eigenvalues (top 5)':<30}")
    print("-" * 100)
    
    for epoch in range(n_epochs):
        epoch_losses = {k: [] for k in ['total', 'linear', 'recon', 'orth', 'stable', 'smooth', 'sparse']}
        
        # 获取当前权重
        weights = loss_scheduler.get_weights()
        
        for batch in loader:
            traj = batch[0].to(device)
            
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(traj, dt, weights)
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k].append(v)
        
        scheduler.step()
        
        # 计算平均 loss
        avg_loss = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Loss Scheduler 步进
        loss_scheduler.step(avg_loss)
        
        # 每5轮打印进度
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                K_full = model.get_K_matrix()
                eigs = torch.linalg.eigvals(K_full).real.cpu().numpy()
                eigs_sorted = np.sort(eigs)[::-1]
                eigs_str = ', '.join([f'{e:.2f}' for e in eigs_sorted[:5]])
            
            print(f"{epoch:6d} | {avg_loss['total']:8.4f} | {avg_loss['linear']:8.4f} | "
                  f"{avg_loss['recon']:8.4f} | {avg_loss['orth']:8.4f} | "
                  f"{avg_loss['stable']:8.4f} | {avg_loss['smooth']:8.4f} | "
                  f"{avg_loss['sparse']:8.4f} | [{eigs_str}]")
        
        # 每20轮打印权重信息
        if epoch % 20 == 0 and epoch > 0:
            print(f"       | 当前权重: {loss_scheduler.get_weight_summary()}")
    
    print("-" * 100)
    return model, loss_scheduler


# ==================== 5. 验证与可视化 ====================

def validate_and_plot(model, ou_process, dt=0.05, device=None, loss_scheduler=None):
    """验证训练结果并绘图"""
    if device is None:
        device = DEVICE
    device = next(model.parameters()).device
    
    # 创建大图，增加一行显示权重变化和特征热力图
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])  # 轨迹对比
    ax2 = fig.add_subplot(gs[0, 1])  # 特征函数（前6个）
    ax3 = fig.add_subplot(gs[0, 2])  # 特征值对比
    ax4 = fig.add_subplot(gs[1, 0])  # Koopman空间轨迹
    ax5 = fig.add_subplot(gs[1, 1])  # 特征热力图
    ax6 = fig.add_subplot(gs[1, 2])  # 特征活跃度
    ax7 = fig.add_subplot(gs[2, :])  # Loss权重变化
    
    # 测试1: 长期预测能力对比
    print("\n=== 验证长期预测 ===")
    with torch.no_grad():
        x0 = torch.tensor([[2.0]], device=device)
        n_steps = 100
        
        # 真实轨迹（欧拉离散）
        true_traj = [x0.item()]
        x = x0.clone()
        for _ in range(n_steps):
            dW = torch.randn_like(x) * np.sqrt(dt)
            x = x + ou_process.theta * (ou_process.mu - x) * dt + ou_process.sigma * dW
            true_traj.append(x.item())
        
        # Koopman预测（单步矩阵指数演化，无累积误差）
        pred_traj = [x0.item()]
        x_current = x0
        for _ in range(n_steps):
            x_pred, _ = model(x_current, dt)
            pred_traj.append(x_pred.item())
            x_current = x_pred
    
    # 图1: 轨迹对比
    t = np.arange(len(true_traj)) * dt
    ax1.plot(t, true_traj, 'b-', label='True OU (Euler)', alpha=0.7, linewidth=2)
    ax1.plot(t, pred_traj, 'r--', label='Koopman Pred', alpha=0.7, linewidth=2)
    ax1.axhline(y=ou_process.mu, color='k', linestyle=':', alpha=0.5, label='Equilibrium')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('X(t)')
    ax1.set_title('Long-term Prediction: True vs Koopman')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 学到的特征函数可视化（展示前 min(6, koopman_dim) 个）
    with torch.no_grad():
        x_range = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)
        z = model.encode(x_range).cpu().numpy()
        x_np = x_range.cpu().numpy()
        
        n_show = min(6, model.koopman_dim)
        for i in range(n_show):
            ax2.plot(x_np, z[:, i], label=f'φ_{i}(x)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Feature value')
    ax2.set_title(f'Learned Koopman Eigenfunctions (showing {n_show}/{model.koopman_dim})')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 图3: 特征值对比（理论 vs 学习）
    with torch.no_grad():
        K_full = model.get_K_matrix()
        learned_eigs = torch.linalg.eigvals(K_full).real.cpu().numpy()
        theoretical_eigs = np.array(ou_process.theoretical_eigenvalues(model.koopman_dim))
        
        learned_eigs = np.sort(learned_eigs)[::-1]
        theoretical_eigs = np.sort(theoretical_eigs)[::-1]
        
        # 只显示前10个特征值
        n_eigs = min(10, model.koopman_dim)
        x_pos = np.arange(n_eigs)
        width = 0.35
        
        ax3.bar(x_pos - width/2, learned_eigs[:n_eigs], width, 
                label='Learned', alpha=0.8, color='steelblue')
        ax3.bar(x_pos + width/2, theoretical_eigs[:n_eigs], width, 
                label='Theory (-nθ)', alpha=0.8, color='coral')
        ax3.set_xlabel('Mode index')
        ax3.set_ylabel('Eigenvalue')
        ax3.set_title(f'Koopman Eigenvalues (top {n_eigs})')
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
    
    ax4.set_xlabel('z_0 (1st mode)')
    ax4.set_ylabel('z_1 (2nd mode)')
    ax4.set_title('Trajectories in Koopman Space (Linear Evolution)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 图5: 特征热力图（不同输入位置的特征值）
    with torch.no_grad():
        x_range = torch.linspace(-3, 3, 50, device=device).unsqueeze(1)
        z = model.encode(x_range).cpu().numpy()
        
        # 显示前 16 个特征维度
        n_heatmap = min(16, model.koopman_dim)
        im = ax5.imshow(z[:, :n_heatmap].T, aspect='auto', cmap='RdBu_r', 
                        extent=[-3, 3, n_heatmap-0.5, -0.5], vmin=-3, vmax=3)
        ax5.set_xlabel('Input x')
        ax5.set_ylabel('Feature dimension')
        ax5.set_title(f'Feature Heatmap (first {n_heatmap} dims)')
        plt.colorbar(im, ax=ax5, label='Feature value')
    
    # 图6: 特征活跃度（各维度特征值的平均绝对值）
    with torch.no_grad():
        z_abs_mean = np.mean(np.abs(z), axis=0)
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(z_abs_mean)))
        
        # 只显示前20个维度
        n_bar = min(20, model.koopman_dim)
        bars = ax6.bar(range(n_bar), z_abs_mean[:n_bar], color=colors_bar[:n_bar])
        ax6.axhline(y=np.mean(z_abs_mean), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(z_abs_mean):.3f}')
        ax6.set_xlabel('Feature dimension')
        ax6.set_ylabel('Mean |feature value|')
        ax6.set_title('Feature Activity Distribution')
        ax6.set_xticks(range(n_bar))
        ax6.set_xticklabels([f'{i}' for i in range(n_bar)], fontsize=8)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 图7: Loss 权重变化历史
    if loss_scheduler is not None and len(loss_scheduler.weight_history['linear']) > 0:
        epochs_hist = range(len(loss_scheduler.weight_history['linear']))
        for key, color in zip(['linear', 'recon', 'orth', 'stable', 'smooth', 'sparse'],
                              ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']):
            if key in loss_scheduler.weight_history:
                ax7.plot(epochs_hist, loss_scheduler.weight_history[key], 
                        label=f'{key}', linewidth=2, color=color)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Loss Weight')
        ax7.set_title('Dynamic Loss Weight Scheduling')
        ax7.legend(loc='upper right', ncol=3)
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale('log')
    else:
        ax7.text(0.5, 0.5, 'Loss Scheduler History Not Available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=14)
        ax7.set_title('Dynamic Loss Weight Scheduling')
    
    plt.tight_layout()
    plt.savefig('ou_koopman_sparse_highdim.png', dpi=150, bbox_inches='tight')
    print("结果已保存至 ou_koopman_sparse_highdim.png")
    plt.show()
    
    # 打印最终统计
    print(f"\n最终学习到的生成元矩阵 K (对角):")
    print(np.diag(model.K.detach().cpu().numpy()).round(3))
    print(f"\n学习到的特征值 (top 10): {np.sort(learned_eigs)[::-1][:10].round(3)}")
    print(f"理论特征值 (top 10):     {theoretical_eigs[:10].round(3)}")
    
    # 稀疏度统计
    with torch.no_grad():
        z_all = model.encode(x_range)
        z_np = z_all.cpu().numpy()
        l1_norm = np.mean(np.abs(z_np))
        l0_approx = np.mean(np.abs(z_np) > 0.1 * np.max(np.abs(z_np)))
        print(f"\n稀疏度统计:")
        print(f"  平均 L1 范数: {l1_norm:.4f}")
        print(f"  近似激活比例 (>10% max): {l0_approx:.4f}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 如果使用GPU，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # OU过程参数
    theta = 1.0   # 回归速率
    mu = 0.0      # 长期均值  
    sigma = 0.5   # 波动率
    dt = 0.05     # 时间步长
    
    print("=" * 80)
    print(f"OU过程参数: θ={theta}, μ={mu}, σ={sigma}")
    print(f"理论Koopman特征值: {OUProcess(theta, mu, sigma).theoretical_eigenvalues(32)[:10]}...")
    print("=" * 80)
    
    # 初始化 - 使用更高的特征维度
    ou = OUProcess(theta, mu, sigma, device=DEVICE)
    model = SimpleKoopmanNet(
        state_dim=1, 
        koopman_dim=8,       # 大幅增加到32维
        hidden_dim=256,       # 更大的隐藏层
        device=DEVICE,
        sparsity_target=0.5,  # 目标50%稀疏度
        sparsity_type='hoyer' # 使用Hoyer稀疏度
    )
    
    # 训练
    trained_model, loss_scheduler = train_koopman(
        model, ou, 
        n_epochs=400,         # 更多训练轮数
        batch_size=128,
        dt=dt,
        device=DEVICE,
        schedule_type='adaptive'  # 自适应权重调度
    )
    
    # 验证与可视化
    validate_and_plot(trained_model, ou, dt=dt, device=DEVICE, loss_scheduler=loss_scheduler)
    
    print("\n" + "=" * 80)
    print("训练完成！高维稀疏版网络特点：")
    print(f"1. 高维特征空间: {model.koopman_dim} 维")
    print(f"2. 稀疏性约束: {model.sparsity_type} 类型，目标稀疏度 {model.sparsity_target}")
    print(f"3. 动态Loss权重: 使用 {loss_scheduler.schedule_type} 调度策略")
    print(f"4. 完整日志: 输出所有 loss 项的值")
    print("=" * 80)
