"""
数据集模块
==========
包含数据集的加载和处理。
"""

import torch
from torch.utils.data import Dataset, DataLoader


class JosephsonDataset(Dataset):
    """
    约瑟夫森结轨迹数据集。
    """
    
    def __init__(self, trajectories, times):
        """
        Args:
            trajectories: 轨迹数据 [n_samples, n_steps, 4]
            times: 时间点 [n_steps]
        """
        self.trajectories = trajectories
        self.times = times
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return self.trajectories[idx], self.times


def create_dataloaders(trajectories, times, train_ratio=0.8, batch_size=64, 
                       shuffle_train=True):
    """
    创建训练集和测试集的数据加载器。
    
    Args:
        trajectories: 轨迹数据 [n_samples, n_steps, 4]
        times: 时间点 [n_steps]
        train_ratio: 训练集比例
        batch_size: 批次大小
        shuffle_train: 是否打乱训练集
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        n_train: 训练样本数
    """
    n_samples = len(trajectories)
    n_train = int(n_samples * train_ratio)
    
    # 随机划分
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_trajectories = trajectories[train_indices]
    test_trajectories = trajectories[test_indices]
    
    # 创建数据集
    train_dataset = JosephsonDataset(train_trajectories, times)
    test_dataset = JosephsonDataset(test_trajectories, times)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False)
    
    return train_loader, test_loader, n_train
