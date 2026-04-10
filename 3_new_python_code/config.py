"""
配置文件
========
包含所有可配置的参数和超参数。
"""

import torch


def get_default_config():
    """
    获取默认配置。
    
    Returns:
        dict: 配置字典
    """
    config = {
        # 随机种子
        'seed': 42,
        
        # 数据生成参数
        'n_samples': 1000,        # 样本数量
        'n_steps': 200,           # 时间步数
        't_span': [0.0, 100],      # 时间区间
        'dt': 0.01,               # SDE 积分步长
        
        # 真实参数 (用于生成数据)
        'true_params': {
            'beta1': 0.1,
            'beta2': 0.1,
            'i1': 0.8,
            'i2': 0.8,
            'kappa1': 0.05,
            'kappa2': 0.05,
            'sigma1': 0.01,
            'sigma2': 0.01,
        },
        
        # 初始状态
        'initial_state': torch.tensor([0.0, 0.0, 0.0, 0.0]),
        
        # 训练参数
        'train_ratio': 0.8,
        'batch_size': 16,
        'n_epochs': 50,
        'lr': 1e-3,
        
        # 损失函数配置
        'loss_config': {
            'mse_weight': 0.0,
            'euler_weight': 1.0,
            'moment_weight': 1.0,
            'autocorr_weight': 0.0,
            'wasserstein_weight': 0.0,
            'kl_weight': 1.0,
            'max_lags': 10,
            'wasserstein_points': 20,
            'kl_points': 5,
            'kl_bins': 20,
            'kl_sigma': 0.1,
        },
        
        # 梯度裁剪配置
        'grad_clip': {
            'enabled': True,
            'max_norm': 10.0,
        },
        
        # Neural SDE 模型配置
        'model_config': {
            'model_type': 'neural',
            'hidden_dim': 128,
            'num_layers': 5,
            'activation': 'lipswish',
        }
    }
    return config
