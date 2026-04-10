"""
主程序
======
耦合约瑟夫森结 Neural SDE 参数学习主程序。

本代码按功能模块拆分，包含：
- config.py: 配置参数
- utils.py: 工具函数
- activations.py: 激活函数
- models.py: 模型定义
- dataset.py: 数据集处理
- losses.py: 损失函数
- trainer.py: 训练和评估
- visualization.py: 可视化
- main.py: 主程序入口
"""

import torch
import torch.optim as optim

from config import get_default_config
from utils import seed_everything, get_device, count_parameters
from models import JosephsonJunctionSDE, NeuralJosephsonSDE
from dataset import create_dataloaders
from losses import LossFunction
from trainer import train_epoch, evaluate
from visualization import (
    plot_trajectories_comparison, 
    plot_phase_space, 
    plot_training_history
)


def main():
    """
    主函数: 完整的参数学习流程演示。
    """
    # 加载配置
    config = get_default_config()
    
    print("=" * 70)
    print("Learning SDE of Coupled Josephson Junctions")
    print("=" * 70)
    
    # 设置随机种子和设备
    seed_everything(config['seed'])
    device = get_device()
    print(f"Using device: {device}")
    
    # ========================================================================
    # 步骤 1: 生成训练数据
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Generating Training Data")
    print("=" * 70)
    
    # 使用真实参数创建模拟器
    true_sde = JosephsonJunctionSDE(**config['true_params']).to(device)
    
    print("Ground Truth Params:")
    for name, value in config['true_params'].items():
        print(f"  {name}: {value}")
    
    # 生成数据
    initial_state = config['initial_state'].to(device)
    times, trajectories = true_sde.simulate(
        t_span=config['t_span'],
        n_steps=config['n_steps'],
        initial_state=initial_state,
        n_samples=config['n_samples']
    )
    
    print(f"\nTrajectories Generated:")
    print(f"  #Samples    : {config['n_samples']}")
    print(f"  Time Steps  : {config['n_steps']}")
    print(f"  Time Span   : {config['t_span']}")
    print(f"  State Space : 4 [φ₁, v₁, φ₂, v₂]")
    
    # 绘制真实数据的可视化
    print("\nPlotting Generated Ground Truth Trajectories...")
    sample_indices = torch.tensor([0, 1, 2])
    sample_trajs = trajectories[sample_indices]
    plot_trajectories_comparison(
        sample_trajs, sample_trajs, times,
        n_samples=1,
        save_path="true_trajectories_visualization.png"
    )
    print("Figure Saved to >>> true_trajectories_visualization.png")
    
    # ========================================================================
    # 步骤 2: 划分训练集和测试集
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Splitting Training and Testing Sets")
    print("=" * 70)
    
    train_loader, test_loader, n_train = create_dataloaders(
        trajectories, times,
        train_ratio=config['train_ratio'],
        batch_size=config['batch_size'],
        shuffle_train=True
    )
    
    print(f"Training Samples: {n_train}")
    print(f"Testing Samples : {config['n_samples'] - n_train}")
    
    # ========================================================================
    # 步骤 3: 创建 Neural SDE 模型和损失函数
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Constructing Neural SDE Model and Loss Functions")
    print("=" * 70)
    
    # 创建模型
    model = NeuralJosephsonSDE(
        state_dim=4,
        hidden_dim=config['model_config']['hidden_dim'],
        num_layers=config['model_config']['num_layers'],
        activation=config['model_config']['activation']
    ).to(device)
    
    print(f"Neural SDE Model Constructed:")
    print(f"  Hidden Dim : {config['model_config']['hidden_dim']}")
    print(f"  #Layers    : {config['model_config']['num_layers']}")
    print(f"  Activation : {config['model_config']['activation']}")
    print(f"  #Params    : {count_parameters(model):,}")
    
    # 创建损失函数
    loss_fn = LossFunction(
        mse_weight=config['loss_config']['mse_weight'],
        euler_weight=config['loss_config']['euler_weight'],
        moment_weight=config['loss_config']['moment_weight'],
        kl_weight=config['loss_config']['kl_weight'],
        dt=config['dt'],
        kl_points=config['loss_config']['kl_points'],
        kl_bins=config['loss_config']['kl_bins'],
        kl_sigma=config['loss_config']['kl_sigma'],
        model=model
    ).to(device)
    
    print("\nLoss Function:")
    for name in loss_fn.loss_names:
        weight_key = f"{name}_weight"
        if weight_key in config['loss_config'] and config['loss_config'][weight_key] > 0:
            print(f"  {name}: {config['loss_config'][weight_key]}")
    
    print("\nGradient Clipping:")
    for name, value in config['grad_clip'].items():
        print(f"  {name}: {value}")
    
    # ========================================================================
    # 步骤 4: 训练模型
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Training")
    print("=" * 70)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8)
    
    # 记录训练历史
    train_history = []
    test_history = []
    grad_norm_history = []
    
    for epoch in range(config['n_epochs']):
        # 训练
        train_losses, avg_grad_norm = train_epoch(
            model, train_loader, optimizer, times, loss_fn, config['dt'],
            grad_clip_config=config['grad_clip'],
            debug=(epoch == 0)
        )
        train_history.append(train_losses)
        grad_norm_history.append(avg_grad_norm)
        
        # 评估
        test_losses, _, _ = evaluate(
            model, test_loader, times, loss_fn, config['dt']
        )
        test_history.append(test_losses)
        
        # 学习率调整
        scheduler.step(test_losses['total'])
        
        # 打印进度
        print(f"Epoch {epoch + 1}/{config['n_epochs']} | "
              f"Train: {train_losses['total']:.4f} | "
              f"Test: {test_losses['total']:.4f} | "
              f"Grad: {avg_grad_norm:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # ========================================================================
    # 步骤 5: 最终评估和可视化
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Evaluation")
    print("=" * 70)
    
    final_test_losses, final_preds, final_trues = evaluate(
        model, test_loader, times, loss_fn, config['dt']
    )
    print(f"Testing Loss:")
    for name, value in final_test_losses.items():
        print(f"  {name}: {value:.6f}")
    
    print("\nModel Info:")
    print(f"  Type     : {config['model_config']['model_type']}")
    print(f"  #Params  : {count_parameters(model):,}")
    
    # 绘制训练历史
    print("\nGenerating Training History Figures...")
    loss_names = [name for name in train_history[0].keys() if name != 'total']
    plot_training_history(train_history, test_history, loss_names)
    
    # 绘制轨迹对比
    print("\nPlotting Trajectory Comparison...")
    plot_trajectories_comparison(
        final_trues, final_preds, times,
        n_samples=1,
        save_path="trajectory_comparison.png",
        compare=True
    )
    
    # 绘制预测轨迹的相空间
    print("Plotting Predicted Phase Space...")
    plot_phase_space(final_preds, n_samples=50, save_path="predicted_phase_space.png")
    
    # 保存模型
    model_path = "neural_josephson_sde.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("Neural SDE Training Completed!")
    print("=" * 70)
    
    return model, train_history, test_history


if __name__ == "__main__":
    model, train_history, test_history = main()
