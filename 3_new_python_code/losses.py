"""
损失函数模块
============
包含各种损失函数的实现。
"""

import torch
import torch.nn as nn


def compute_euler_pseudo_likelihood_loss(true_trajectories, pred_trajectories, 
                                         dt, model):
    """
    Euler-Maruyama 伪似然损失。
    
    Args:
        true_trajectories: 真实轨迹 [n_samples, n_steps, n_dims]
        pred_trajectories: 预测轨迹 [n_samples, n_steps, n_dims]
        dt: 时间步长
        model: SDE 模型，需要提供 f 和 g 方法
        
    Returns:
        euler_loss: Euler-Maruyama 伪似然损失
    """
    device = true_trajectories.device
    dtype = true_trajectories.dtype
    
    # 确保 dt 是 tensor
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, device=device, dtype=dtype)
    else:
        dt = dt.to(device=device, dtype=dtype)
    
    # 计算增量
    x_ = true_trajectories[:, :-1, :]
    dx_ = true_trajectories[:, 1:, :] - x_
    
    # 计算模型输出的方差和漂移
    var_x_ = model.g(None, x_)**2
    bias_x_ = model.f(None, x_)
    
    # 计算损失
    euler_loss = 0.5 * torch.sum(
        ((bias_x_ - dx_/dt) / torch.sqrt(var_x_/dt))**2
    ) + 0.5 * torch.sum(torch.log(var_x_**2))
    
    # 数值稳定性检查
    if torch.isnan(euler_loss) or torch.isinf(euler_loss):
        return torch.tensor(0.0, device=device, dtype=dtype)
    
    return euler_loss


def compute_moment_loss(true_trajectories, pred_trajectories):
    """
    矩匹配损失。匹配均值和方差。
    
    Args:
        true_trajectories: 真实轨迹 [n_samples, n_steps, n_dims]
        pred_trajectories: 预测轨迹 [n_samples, n_steps, n_dims]
        
    Returns:
        moment_loss: 矩匹配损失
    """
    # 检查输入
    if torch.isnan(true_trajectories).any() or torch.isnan(pred_trajectories).any():
        return torch.tensor(0.0, device=true_trajectories.device, 
                           dtype=true_trajectories.dtype)
    
    # 计算均值
    true_mean = torch.mean(true_trajectories, dim=0)
    pred_mean = torch.mean(pred_trajectories, dim=0)
    
    # 计算方差
    true_var = torch.var(true_trajectories, dim=0, unbiased=True)
    pred_var = torch.var(pred_trajectories, dim=0, unbiased=True)
    
    # 均值损失
    mean_loss = torch.mean((pred_mean - true_mean) ** 2)
    
    # 方差损失
    var_loss = torch.mean((pred_var - true_var) ** 2)
    
    # 数值稳定性检查
    if torch.isnan(mean_loss):
        mean_loss = torch.tensor(0.0, device=true_trajectories.device, 
                                dtype=true_trajectories.dtype)
    if torch.isnan(var_loss):
        var_loss = torch.tensor(0.0, device=true_trajectories.device, 
                               dtype=true_trajectories.dtype)
    
    return mean_loss + var_loss


def compute_path_mse_loss(true_trajectories, pred_trajectories):
    """
    路径级 MSE 损失。
    
    Args:
        true_trajectories: 真实轨迹 [n_samples, n_steps, n_dims]
        pred_trajectories: 预测轨迹 [n_samples, n_steps, n_dims]
        
    Returns:
        mse_loss: MSE 损失
    """
    if torch.isnan(true_trajectories).any() or torch.isnan(pred_trajectories).any():
        return torch.tensor(0.0, device=true_trajectories.device, 
                           dtype=true_trajectories.dtype)
    
    mse_loss = torch.mean((pred_trajectories - true_trajectories) ** 2)
    
    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
        return torch.tensor(0.0, device=true_trajectories.device, 
                           dtype=true_trajectories.dtype)
    
    return mse_loss


def compute_soft_histogram_kl_loss(true_trajectories, pred_trajectories,
                                    num_points=5, n_bins=10, sigma=0.1):
    """
    软直方图 KL 散度损失（随机时间片）。
    
    Args:
        true_trajectories: 真实轨迹 [n_samples, n_steps, n_dims]
        pred_trajectories: 预测轨迹 [n_samples, n_steps, n_dims]
        num_points: 随机选择的时间点数量
        n_bins: 直方图的 bin 数量
        sigma: 软分箱的高斯带宽
        
    Returns:
        kl_loss: KL 散度损失
    """
    if torch.isnan(true_trajectories).any() or torch.isnan(pred_trajectories).any():
        return torch.tensor(0.0, device=true_trajectories.device, 
                           dtype=true_trajectories.dtype)
    
    n_samples, n_steps, n_dims = true_trajectories.shape
    device = true_trajectories.device
    dtype = true_trajectories.dtype
    
    # 随机选择时间点
    if num_points >= n_steps - 1:
        time_indices = torch.arange(1, n_steps, device=device)
    else:
        time_indices = torch.randperm(n_steps - 1, device=device)[:num_points] + 1
    
    total_kl = 0.0
    num_comparisons = 0
    
    for t_idx in time_indices:
        true_vals = true_trajectories[:, t_idx, :]
        pred_vals = pred_trajectories[:, t_idx, :]
        
        for dim_idx in range(n_dims):
            true_dim = true_vals[:, dim_idx]
            pred_dim = pred_vals[:, dim_idx]
            
            if (true_dim.std() < 1e-6) or (pred_dim.std() < 1e-6):
                continue
            
            # 动态确定直方图范围
            vmin = min(true_dim.min().item(), pred_dim.min().item()) - 0.1
            vmax = max(true_dim.max().item(), pred_dim.max().item()) + 0.1
            
            bins = torch.linspace(vmin, vmax, n_bins, device=device, dtype=dtype)
            
            def soft_histogram(values, bins, sigma):
                diff = torch.abs(values.unsqueeze(1) - bins.unsqueeze(0))
                weights = torch.exp(-diff.pow(2) / (2 * sigma ** 2))
                return weights.sum(dim=0)
            
            true_hist = soft_histogram(true_dim, bins, sigma)
            pred_hist = soft_histogram(pred_dim, bins, sigma)
            
            # 归一化为概率分布
            true_prob = (true_hist + 1e-6) / (true_hist.sum() + 1e-6 * n_bins)
            pred_prob = (pred_hist + 1e-6) / (pred_hist.sum() + 1e-6 * n_bins)
            
            pred_prob_clipped = torch.clamp(pred_prob, min=1e-8)
            true_prob_clipped = torch.clamp(true_prob, min=1e-8)
            
            kl = torch.sum(true_prob * torch.log(true_prob_clipped / pred_prob_clipped))
            
            if not torch.isnan(kl) and not torch.isinf(kl):
                weight = max(1.0, (t_idx.item() / n_steps) ** 0.5)
                total_kl += kl * weight
                num_comparisons += 1
    
    if num_comparisons > 0:
        avg_kl = total_kl / num_comparisons
        if torch.isnan(avg_kl) or torch.isinf(avg_kl):
            return torch.tensor(0.0, device=device, dtype=dtype)
        return avg_kl
    else:
        return torch.tensor(0.0, device=device, dtype=dtype)


class LossFunction(nn.Module):
    """
    综合损失函数，支持多种损失项的灵活组合。
    """
    
    def __init__(self,
                 mse_weight=1.0,
                 euler_weight=0.0,
                 moment_weight=0.0,
                 kl_weight=0.0,
                 dt=0.01,
                 kl_points=5,
                 kl_bins=10,
                 kl_sigma=0.1,
                 model=None):
        super().__init__()
        self.mse_weight = mse_weight
        self.euler_weight = euler_weight
        self.moment_weight = moment_weight
        self.kl_weight = kl_weight
        self.dt = dt
        self.kl_points = kl_points
        self.kl_bins = kl_bins
        self.kl_sigma = kl_sigma
        self.model = model
        
        self.loss_names = ['mse', 'euler', 'moment', 'kl']
    
    def forward(self, true_trajectories, pred_trajectories):
        """
        计算综合损失。
        
        Args:
            true_trajectories: 真实轨迹 [n_samples, n_steps, n_dims]
            pred_trajectories: 预测轨迹 [n_samples, n_steps, n_dims]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各损失项的字典
        """
        device = true_trajectories.device
        dtype = true_trajectories.dtype
        
        if torch.isnan(true_trajectories).any() or torch.isnan(pred_trajectories).any():
            loss_dict = {name: 0.0 for name in self.loss_names}
            loss_dict['total'] = 0.0
            return torch.tensor(0.0, device=device, dtype=dtype), loss_dict
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        
        # MSE 损失
        if self.mse_weight > 0:
            mse_loss = compute_path_mse_loss(true_trajectories, pred_trajectories)
            if not torch.isnan(mse_loss) and not torch.isinf(mse_loss):
                loss_dict['mse'] = mse_loss.item()
                total_loss += self.mse_weight * mse_loss
            else:
                loss_dict['mse'] = 0.0
        else:
            loss_dict['mse'] = 0.0
        
        # Euler-Maruyama 伪似然
        if self.euler_weight > 0:
            euler_loss = compute_euler_pseudo_likelihood_loss(
                true_trajectories, pred_trajectories, self.dt, model=self.model
            )
            if not torch.isnan(euler_loss) and not torch.isinf(euler_loss):
                loss_dict['euler'] = euler_loss.item()
                total_loss += self.euler_weight * euler_loss
            else:
                loss_dict['euler'] = 0.0
        else:
            loss_dict['euler'] = 0.0
        
        # 矩匹配损失
        if self.moment_weight > 0:
            moment_loss = compute_moment_loss(true_trajectories, pred_trajectories)
            if not torch.isnan(moment_loss) and not torch.isinf(moment_loss):
                loss_dict['moment'] = moment_loss.item()
                total_loss += self.moment_weight * moment_loss
            else:
                loss_dict['moment'] = 0.0
        else:
            loss_dict['moment'] = 0.0
        
        # KL 散度损失
        if self.kl_weight > 0:
            kl_loss = compute_soft_histogram_kl_loss(
                true_trajectories,
                pred_trajectories,
                num_points=self.kl_points,
                n_bins=self.kl_bins,
                sigma=self.kl_sigma
            )
            if not torch.isnan(kl_loss) and not torch.isinf(kl_loss):
                loss_dict['kl'] = kl_loss.item()
                total_loss += self.kl_weight * kl_loss
            else:
                loss_dict['kl'] = 0.0
        else:
            loss_dict['kl'] = 0.0
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
            for key in loss_dict:
                loss_dict[key] = 0.0
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
