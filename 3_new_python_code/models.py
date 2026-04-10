"""
模型定义模块
============
包含所有 SDE 模型的定义，包括物理 SDE 和 Neural SDE。
"""

import torch
import torch.nn as nn
import torchsde

from activations import get_activation


class JosephsonJunctionSDE(nn.Module):
    """
    耦合约瑟夫森结 SDE 模拟器（物理模型）。
    
    SDE 方程:
        dφ₁ = v₁ dt
        dv₁ = [i₁ - β₁v₁ - sin(φ₁) + κ₁(φ₂ - φ₁)] dt + σ₁ dW₁
        dφ₂ = v₂ dt
        dv₂ = [i₂ - β₂v₂ - sin(φ₂) + κ₂(φ₁ - φ₂)] dt + σ₂ dW₂
    """
    
    def __init__(self, beta1, beta2, i1, i2, kappa1, kappa2, sigma1, sigma2):
        super().__init__()
        
        # 将标量参数转换为 buffer
        self.register_buffer('beta1', torch.tensor(beta1, dtype=torch.float32))
        self.register_buffer('beta2', torch.tensor(beta2, dtype=torch.float32))
        self.register_buffer('i1', torch.tensor(i1, dtype=torch.float32))
        self.register_buffer('i2', torch.tensor(i2, dtype=torch.float32))
        self.register_buffer('kappa1', torch.tensor(kappa1, dtype=torch.float32))
        self.register_buffer('kappa2', torch.tensor(kappa2, dtype=torch.float32))
        self.register_buffer('sigma1', torch.tensor(sigma1, dtype=torch.float32))
        self.register_buffer('sigma2', torch.tensor(sigma2, dtype=torch.float32))
        
        # SDE 类型设置
        self.sde_type = "ito"
        self.noise_type = "diagonal"
    
    def f(self, t, y):
        """漂移项 (drift)"""
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        
        d_phi1 = v1
        dv1 = (self.i1 - self.beta1 * v1 - torch.sin(phi1) + 
               self.kappa1 * (phi2 - phi1))
        d_phi2 = v2
        dv2 = (self.i2 - self.beta2 * v2 - torch.sin(phi2) + 
               self.kappa2 * (phi1 - phi2))
        
        return torch.stack([d_phi1, dv1, d_phi2, dv2], dim=-1)
    
    def g(self, t, y):
        """扩散项 (diffusion)"""
        batch_size = y.shape[0]
        device = y.device
        
        noise = torch.zeros(batch_size, 4, device=device)
        noise[..., 1] = self.sigma1
        noise[..., 3] = self.sigma2
        
        return noise
    
    def simulate(self, t_span, n_steps, initial_state, n_samples=1, 
                 dt=None, method='euler'):
        """
        使用 torchsde 模拟 SDE。
        
        Args:
            t_span: 时间区间 [t_start, t_end]
            n_steps: 时间步数
            initial_state: 初始状态
            n_samples: 样本数量
            dt: SDE 积分步长
            method: 积分方法
            
        Returns:
            times: 时间点数组
            trajectories: 轨迹数组 [n_samples, n_steps, 4]
        """
        device = initial_state.device if torch.is_tensor(initial_state) else torch.device('cpu')
        
        t_start, t_end = t_span
        times = torch.linspace(t_start, t_end, n_steps, device=device)
        
        # 准备初始状态
        if initial_state.dim() == 1:
            y0 = initial_state.unsqueeze(0).expand(n_samples, -1)
        else:
            y0 = initial_state
        
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


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, in_size, out_size, hidden_dim, num_layers, 
                 activation='lipswish'):
        super().__init__()
        
        activation_fn = get_activation(activation)
        
        layers = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, out_size))
        
        self._model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._model(x)


class NeuralJosephsonSDE(nn.Module):
    """
    神经约瑟夫森结 SDE 模型。
    使用神经网络学习 SDE 的漂移项和扩散项。
    """
    
    def __init__(self, state_dim=4, hidden_dim=64, num_layers=3, 
                 activation='lipswish'):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # SDE 类型设置
        self.sde_type = "ito"
        self.noise_type = "diagonal"
        
        # 漂移项网络 f(y)
        self.f_net = MLP(
            in_size=state_dim,
            out_size=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation
        )
        
        # 扩散项网络 g(y)
        self.g_net = MLP(
            in_size=state_dim,
            out_size=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def f(self, t, y):
        """漂移项 (drift)"""
        f_out = self.f_net(y)
        return torch.clamp(f_out, min=-100.0, max=100.0)
    
    def g(self, t, y):
        """扩散项 (diffusion)"""
        g_out = torch.nn.functional.softplus(self.g_net(y))
        return torch.clamp(g_out, min=1e-4)
    
    def forward(self, initial_states, times, dt=None, method='euler'):
        """
        前向传播: 从初始状态积分 SDE。
        
        Args:
            initial_states: 初始状态 [n_samples, state_dim]
            times: 时间点 [n_steps]
            dt: 积分步长
            method: 积分方法
            
        Returns:
            trajectories: 轨迹 [n_samples, n_steps, state_dim]
        """
        trajectories = torchsde.sdeint(
            sde=self,
            y0=initial_states,
            ts=times,
            dt=dt if dt is not None else (times[-1] - times[0]) / len(times),
            method=method
        )
        
        # 调整维度: [n_steps, batch, state_dim] -> [batch, n_steps, state_dim]
        return trajectories.permute(1, 0, 2)
