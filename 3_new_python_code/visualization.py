"""
可视化模块
==========
包含轨迹对比、训练历史、相空间等可视化函数。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch


def plot_trajectories_comparison(true_trajectories, pred_trajectories, times,
                                 n_samples=3, save_path=None, compare=False):
    """
    可视化真实轨迹和预测轨迹的对比。
    
    Args:
        true_trajectories: 真实轨迹 [n_samples, n_steps, 4]
        pred_trajectories: 预测轨迹 [n_samples, n_steps, 4]
        times: 时间点
        n_samples: 要绘制的样本数
        save_path: 保存路径，如果为 None 则显示
        compare: 是否绘制预测轨迹进行对比
    """
    # 转换为 numpy
    times_np = times.cpu().numpy() if torch.is_tensor(times) else times
    true_np = true_trajectories[:n_samples].cpu().numpy() if torch.is_tensor(
        true_trajectories) else true_trajectories[:n_samples]
    pred_np = pred_trajectories[:n_samples].cpu().numpy() if torch.is_tensor(
        pred_trajectories) else pred_trajectories[:n_samples]
    
    # 提取单个样本的数据
    true_sample = true_np[0]
    pred_sample = pred_np[0]
    
    phi1_true, v1_true = true_sample[:, 0], true_sample[:, 1]
    phi2_true, v2_true = true_sample[:, 2], true_sample[:, 3]
    phi1_pred, v1_pred = pred_sample[:, 0], pred_sample[:, 1]
    phi2_pred, v2_pred = pred_sample[:, 2], pred_sample[:, 3]
    
    # 创建 2x3 子图
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    axs = axs.flatten()
    
    # 相位演化对比
    axs[0].plot(times_np, phi1_true, 'b-', alpha=0.7, label=r'$\phi_1$ (True)')
    axs[0].plot(times_np, phi2_true, 'g-', alpha=0.7, label=r'$\phi_2$ (True)')
    if compare:
        axs[0].plot(times_np, phi1_pred, 'b--', alpha=0.7, label=r'$\phi_1$ (Pred)')
        axs[0].plot(times_np, phi2_pred, 'g--', alpha=0.7, label=r'$\phi_2$ (Pred)')
        axs[0].legend(loc='best', fontsize=8)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"Phase $\phi$")
    axs[0].set_title("Phase Evolution")
    
    # 速度演化对比
    axs[1].plot(times_np, v1_true, 'b-', alpha=0.7, label=r'$v_1$ (True)')
    axs[1].plot(times_np, v2_true, 'g-', alpha=0.7, label=r'$v_2$ (True)')
    if compare:
        axs[1].plot(times_np, v1_pred, 'b--', alpha=0.7, label=r'$v_1$ (Pred)')
        axs[1].plot(times_np, v2_pred, 'g--', alpha=0.7, label=r'$v_2$ (Pred)')
        axs[1].legend(loc='best', fontsize=8)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity $v$")
    axs[1].set_title("Velocity Evolution")
    
    # 相空间轨迹对比
    axs[2].plot(phi1_true, v1_true, 'b-', alpha=0.5, label='Junction 1 (True)')
    axs[2].plot(phi2_true, v2_true, 'g-', alpha=0.5, label='Junction 2 (True)')
    if compare:
        axs[2].plot(phi1_pred, v1_pred, 'b--', alpha=0.5, label='Junction 1 (Pred)')
        axs[2].plot(phi2_pred, v2_pred, 'g--', alpha=0.5, label='Junction 2 (Pred)')
        axs[2].legend(loc='best', fontsize=8)
    axs[2].grid(True, alpha=0.3)
    axs[2].set_xlabel(r"$\phi$")
    axs[2].set_ylabel(r"$v$")
    axs[2].set_title("Phase Space Trajectories")
    
    # 相位相关性
    axs[3].plot(phi1_true, phi2_true, 'b-', alpha=0.5, label='True')
    if compare:
        axs[3].plot(phi1_pred, phi2_pred, 'r--', alpha=0.5, label='Pred')
        axs[3].legend(loc='best', fontsize=8)
    axs[3].grid(True, alpha=0.3)
    axs[3].set_xlabel(r"$\phi_1$")
    axs[3].set_ylabel(r"$\phi_2$")
    axs[3].set_title("Phase Correlation")
    
    # 相位差
    axs[4].plot(times_np, phi1_true - phi2_true, 'b-', alpha=0.7, label='True')
    if compare:
        axs[4].plot(times_np, phi1_pred - phi2_pred, 'r--', alpha=0.7, label='Pred')
        axs[4].legend(loc='best', fontsize=8)
    axs[4].grid(True, alpha=0.3)
    axs[4].set_xlabel("Time")
    axs[4].set_ylabel(r"$\phi_1 - \phi_2$")
    axs[4].set_title("Phase Difference")
    
    # 速度差
    axs[5].plot(times_np, v1_true - v2_true, 'b-', alpha=0.7, label='True')
    if compare:
        axs[5].plot(times_np, v1_pred - v2_pred, 'r--', alpha=0.7, label='Pred')
        axs[5].legend(loc='best', fontsize=8)
    axs[5].grid(True, alpha=0.3)
    axs[5].set_xlabel("Time")
    axs[5].set_ylabel(r"$v_1 - v_2$")
    axs[5].set_title("Velocity Difference")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', transparent=False)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase_space(trajectories, n_samples=100, save_path=None):
    """
    绘制相空间图。
    
    Args:
        trajectories: 轨迹 [n_samples, n_steps, 4]
        n_samples: 要绘制的样本数
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    traj_np = trajectories[:n_samples].numpy() if torch.is_tensor(
        trajectories) else trajectories[:n_samples]
    
    # φ₁ vs v₁
    ax = axes[0]
    for i in range(min(n_samples, len(traj_np))):
        ax.plot(traj_np[i, :, 0], traj_np[i, :, 1], alpha=0.3)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$v_1$')
    ax.set_title('Phase Space: Junction 1')
    ax.grid(True, alpha=0.3)
    
    # φ₂ vs v₂
    ax = axes[1]
    for i in range(min(n_samples, len(traj_np))):
        ax.plot(traj_np[i, :, 2], traj_np[i, :, 3], alpha=0.3)
    ax.set_xlabel(r'$\phi_2$')
    ax.set_ylabel(r'$v_2$')
    ax.set_title('Phase Space: Junction 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(train_history, test_history, loss_names, 
                          save_path_prefix="training_history"):
    """
    绘制训练历史。
    
    Args:
        train_history: 训练损失历史列表
        test_history: 测试损失历史列表
        loss_names: 损失名称列表（不包括 'total'）
        save_path_prefix: 保存路径前缀
    """
    # 绘制各损失项
    if len(loss_names) > 0:
        fig, axes = plt.subplots(1, len(loss_names), 
                                 figsize=(5 * len(loss_names), 4))
        if len(loss_names) == 1:
            axes = [axes]
        
        for idx, loss_name in enumerate(loss_names):
            ax = axes[idx]
            train_values = [h[loss_name] for h in train_history]
            test_values = [h[loss_name] for h in test_history]
            
            ax.plot(train_values, label=f'Train {loss_name}', alpha=0.7)
            ax.plot(test_values, label=f'Test {loss_name}', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{loss_name} Loss')
            ax.semilogy()
            ax.set_title(f'{loss_name} Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}_components.png", dpi=150, bbox_inches='tight')
        print(f"Training history components saved to >>> {save_path_prefix}_components.png")
        plt.close()
    
    # 绘制总损失
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    train_total = [h['total'] for h in train_history]
    test_total = [h['total'] for h in test_history]
    ax.plot(train_total, label='Train Total', color='blue', alpha=0.7)
    ax.plot(test_total, label='Test Total', color='red', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_total.png", dpi=150, bbox_inches='tight')
    print(f"Total loss history saved to >>> {save_path_prefix}_total.png")
    plt.close()
