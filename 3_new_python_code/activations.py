"""
激活函数模块
============
包含自定义的激活函数实现。
"""

import torch
import torch.nn as nn


class LipSwish(nn.Module):
    """
    LipSwish 激活函数：带可学习 β 参数，确保 Lipschitz 常数 ≤ 1
    """
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


def get_activation(activation='lipswish'):
    """
    根据名称获取激活函数。
    
    Args:
        activation: 激活函数名称 ('lipswish', 'tanh', 'relu')
        
    Returns:
        nn.Module: 激活函数模块
    """
    if activation == 'lipswish':
        return LipSwish()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")
