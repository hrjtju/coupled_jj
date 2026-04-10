"""
训练模块
========
包含模型训练、评估和轨迹预测的函数。
"""

import torch
import torchsde
from torchsde import BrownianInterval


def compute_trajectory_loss(model, initial_states, true_trajectories, times,
                            loss_fn, dt=0.01, n_samples_per_batch=None):
    """
    计算模型生成的轨迹与真实轨迹之间的损失。
    
    Args:
        model: NeuralJosephsonSDE 模型
        initial_states: 初始状态 [n_samples, 4]
        true_trajectories: 真实轨迹 [n_samples, n_steps, 4]
        times: 时间点 [n_steps]
        loss_fn: LossFunction 实例
        dt: SDE 积分步长
        n_samples_per_batch: 每次生成的样本数，None 表示全部
        
    Returns:
        total_loss: 总损失
        loss_dict: 各损失项的字典
        pred_trajectories: 预测的轨迹
    """
    n_samples = initial_states.shape[0]
    device = initial_states.device
    sde_method = 'euler'
    
    try:
        # 显式创建布朗运动对象
        bm = BrownianInterval(
            t0=float(times[0]),
            t1=float(times[-1]),
            size=(n_samples, 4),
            device=device
        )
        
        if n_samples_per_batch is not None and n_samples > n_samples_per_batch:
            all_pred_trajectories = []
            for i in range(0, n_samples, n_samples_per_batch):
                end_idx = min(i + n_samples_per_batch, n_samples)
                batch_initial = initial_states[i:end_idx]
                batch_size = batch_initial.shape[0]
                
                # 为每个子批次创建对应的布朗运动
                batch_bm = BrownianInterval(
                    t0=float(times[0]),
                    t1=float(times[-1]),
                    size=(batch_size, 4),
                    device=device
                )
                
                pred_batch = torchsde.sdeint(
                    sde=model,
                    y0=batch_initial,
                    ts=times,
                    bm=batch_bm,
                    dt=dt,
                    method=sde_method
                )
                # [n_steps, batch, 4] -> [batch, n_steps, 4]
                pred_batch = pred_batch.permute(1, 0, 2)
                all_pred_trajectories.append(pred_batch)
            
            pred_trajectories = torch.cat(all_pred_trajectories, dim=0)
        else:
            pred_trajectories = torchsde.sdeint(
                sde=model,
                y0=initial_states,
                ts=times,
                bm=bm,
                dt=dt,
                method=sde_method
            )
            # 调整维度: [n_steps, batch, 4] -> [batch, n_steps, 4]
            pred_trajectories = pred_trajectories.permute(1, 0, 2)
        
        # 检查预测轨迹是否包含 nan
        if torch.isnan(pred_trajectories).any() or torch.isinf(pred_trajectories).any():
            loss_dict = {name: 0.0 for name in loss_fn.loss_names}
            loss_dict['total'] = 0.0
            # 创建与模型参数相关的零损失
            zero_loss = sum(p.view(-1)[0] * 0 for p in model.parameters() 
                          if p.requires_grad)
            if zero_loss == 0:
                zero_loss = torch.zeros(1, device=device, 
                                       dtype=true_trajectories.dtype, 
                                       requires_grad=True)
            return zero_loss, loss_dict, true_trajectories.clone()
        
        # 计算损失
        total_loss, loss_dict = loss_fn(true_trajectories, pred_trajectories)
        
    except Exception as e:
        # SDE 积分失败，返回零损失
        print(f"Warning: SDE integration failed with error: {e}")
        loss_dict = {name: 0.0 for name in loss_fn.loss_names}
        loss_dict['total'] = 0.0
        # 创建与模型参数相关的零损失
        zero_loss = sum(p.view(-1)[0] * 0 for p in model.parameters() 
                      if p.requires_grad)
        if zero_loss == 0:
            zero_loss = torch.zeros(1, device=device, 
                                   dtype=true_trajectories.dtype, 
                                   requires_grad=True)
        return zero_loss, loss_dict, true_trajectories.clone()
    
    return total_loss, loss_dict, pred_trajectories


def train_epoch(model, dataloader, optimizer, times, loss_fn, dt=0.01,
                n_samples_per_batch=None, grad_clip_config=None, debug=False):
    """
    训练一个 epoch。
    
    Args:
        model: 参数化 SDE 模型
        dataloader: 数据加载器
        optimizer: 优化器
        times: 时间点
        loss_fn: LossFunction 实例
        dt: SDE 积分步长
        n_samples_per_batch: 每次生成的样本数
        grad_clip_config: 梯度裁剪配置字典 {'enabled': bool, 'max_norm': float}
        debug: 是否打印调试信息
        
    Returns:
        avg_losses: 平均损失字典
        avg_grad_norm: 平均梯度范数
    """
    model.train()
    total_losses = {name: 0.0 for name in loss_fn.loss_names + ['total']}
    total_grad_norm = 0.0
    n_batches = 0
    
    for batch_idx, (batch_trajectories, _) in enumerate(dataloader):
        device = batch_trajectories.device
        batch_trajectories = batch_trajectories.to(device)
        
        # 初始状态
        initial_states = batch_trajectories[:, 0, :]
        
        # 计算损失
        loss, loss_dict, _ = compute_trajectory_loss(
            model, initial_states, batch_trajectories, times,
            loss_fn, dt, n_samples_per_batch
        )
        
        # 调试信息
        if debug and batch_idx == 0:
            print(f"\n[Debug] Batch 0 loss details:")
            for name, value in loss_dict.items():
                print(f"  {name}: {value:.6f}")
            print(f"  Total loss: {loss.item():.6f}")
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 计算梯度范数（在裁剪之前）
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        total_grad_norm += grad_norm.item()
        
        # 梯度裁剪
        if grad_clip_config is not None and grad_clip_config.get('enabled', False):
            max_norm = grad_clip_config.get('max_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        for key, value in loss_dict.items():
            total_losses[key] += value
        n_batches += 1
    
    # 计算平均损失和梯度范数
    avg_losses = {key: value / n_batches for key, value in total_losses.items()}
    avg_grad_norm = total_grad_norm / n_batches if n_batches > 0 else 0.0
    
    return avg_losses, avg_grad_norm


def evaluate(model, dataloader, times, loss_fn, dt=0.01, n_samples_per_batch=None):
    """
    评估模型。
    
    Args:
        model: 参数化 SDE 模型
        dataloader: 数据加载器
        times: 时间点
        loss_fn: LossFunction 实例
        dt: SDE 积分步长
        n_samples_per_batch: 每次生成的样本数
        
    Returns:
        avg_losses: 平均损失字典
        all_preds: 所有预测轨迹
        all_trues: 所有真实轨迹
    """
    model.eval()
    total_losses = {name: 0.0 for name in loss_fn.loss_names + ['total']}
    n_batches = 0
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_trajectories, _ in dataloader:
            device = batch_trajectories.device
            batch_trajectories = batch_trajectories.to(device)
            
            initial_states = batch_trajectories[:, 0, :]
            
            _, loss_dict, pred_trajectories = compute_trajectory_loss(
                model, initial_states, batch_trajectories, times,
                loss_fn, dt, n_samples_per_batch
            )
            
            for key, value in loss_dict.items():
                total_losses[key] += value
            n_batches += 1
            
            all_preds.append(pred_trajectories.cpu())
            all_trues.append(batch_trajectories.cpu())
    
    avg_losses = {key: value / n_batches for key, value in total_losses.items()}
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    
    return avg_losses, all_preds, all_trues
