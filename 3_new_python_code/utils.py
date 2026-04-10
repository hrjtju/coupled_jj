"""
工具函数模块
===========
包含随机种子设置、路径处理等通用工具函数。
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed):
    """
    设置所有随机种子以确保结果可复现。
    
    Args:
        seed: 随机种子值
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device():
    """
    获取可用的计算设备。
    
    Returns:
        torch.device: 计算设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    计算模型参数数量。
    
    Args:
        model: PyTorch 模型
        
    Returns:
        int: 可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
