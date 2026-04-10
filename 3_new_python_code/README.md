# Coupled Josephson Junction Neural SDE - 模块化代码

本目录包含按功能拆分后的耦合约瑟夫森结 Neural SDE 参数学习代码。

## 文件结构

```
3_new_python_code/
├── README.md               # 本说明文件
├── config.py              # 配置参数
├── utils.py               # 工具函数
├── activations.py         # 激活函数
├── models.py              # 模型定义
├── dataset.py             # 数据集处理
├── losses.py              # 损失函数
├── trainer.py             # 训练和评估
├── visualization.py       # 可视化
└── main.py                # 主程序入口
```

## 模块说明

### 1. config.py
包含所有可配置的参数，包括：
- 数据生成参数（样本数、时间步数、时间区间等）
- 物理参数（阻尼系数、偏置电流、耦合强度、噪声强度）
- 训练参数（批次大小、学习率、训练轮数等）
- 损失函数权重配置
- 梯度裁剪配置
- 模型架构配置

### 2. utils.py
通用工具函数：
- `seed_everything()`: 设置随机种子
- `get_device()`: 获取计算设备
- `count_parameters()`: 计算模型参数数量

### 3. activations.py
激活函数实现：
- `LipSwish`: 带可学习参数的 LipSwish 激活函数
- `get_activation()`: 根据名称获取激活函数

### 4. models.py
模型定义：
- `JosephsonJunctionSDE`: 物理 SDE 模型（用于生成真实数据）
- `MLP`: 多层感知机
- `NeuralJosephsonSDE`: Neural SDE 模型（用于学习）

### 5. dataset.py
数据集处理：
- `JosephsonDataset`: 数据集类
- `create_dataloaders()`: 创建训练和测试数据加载器

### 6. losses.py
损失函数实现：
- `compute_euler_pseudo_likelihood_loss()`: Euler-Maruyama 伪似然损失
- `compute_moment_loss()`: 矩匹配损失
- `compute_path_mse_loss()`: 路径级 MSE 损失
- `compute_soft_histogram_kl_loss()`: 软直方图 KL 散度损失
- `LossFunction`: 综合损失函数类

### 7. trainer.py
训练和评估函数：
- `compute_trajectory_loss()`: 计算轨迹损失
- `train_epoch()`: 训练一个 epoch
- `evaluate()`: 评估模型

### 8. visualization.py
可视化函数：
- `plot_trajectories_comparison()`: 绘制轨迹对比图
- `plot_phase_space()`: 绘制相空间图
- `plot_training_history()`: 绘制训练历史

### 9. main.py
主程序入口，整合所有模块完成完整的训练流程。

## 使用方法

```bash
cd 3_new_python_code
python main.py
```

## 依赖项

```
torch
torchsde
numpy
matplotlib
```

## 关键修复

相比原始代码，本版本修复了以下问题：

1. **梯度问题**: 移除了不必要的 `@torch.no_grad()` 和 `.detach()`，确保损失可以正确传播梯度

2. **SDE 积分失败**: 显式创建 `BrownianInterval` 对象传递给 `torchsde.sdeint`

3. **零损失无梯度**: 异常处理中返回与模型参数相关的零损失，而非常数

4. **拼写错误**: 修复了 `torch.nn.functional.softplus` 的拼写错误
