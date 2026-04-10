# 🚀 快速执行指南 - Eval Loss 收敛修复

## 已应用的修复（4项关键改变）

### ✅ 1. 数据多样性
- N_RANGE: (100, 100) → (80, 200)
- T_RANGE: (10.0, 10.0) → (5.0, 15.0)

### ✅ 2. 参数权重平衡
```
旧：[2.0, 2.0, 1.0, 1.0, 6.0, 6.0, 2.0, 2.0]  ← κ权重过大(3x)
新：[1.5, 1.5, 1.0, 1.0, 3.0, 3.0, 1.5, 1.5]  ← κ权重合理(1.5x)
```

### ✅ 3. 模型容量降低
| 参数 | 旧值 | 新值 | 改变 |
|------|------|------|------|
| d_model | 256 | 128 | ↓50% |
| num_layers | 8 | 4 | ↓50% |
| dim_feedforward | 512 | 256 | ↓50% |
| dropout | 0.1 | 0.3 | ↑3x |

### ✅ 4. 优化器参数调整
- learning_rate: 1e-4 → 5e-5 (↓50%)
- weight_decay: 1e-5 → 1e-4 (↑10x)

---

## 🎯 立即执行

### Step 1: 清空旧数据（必须！）
```powershell
cd "D:\Research\Mathematics\SYSU_GBU\2_projects\01_wSLYuan_Coupled_JJ\2_python_code"

# 删除旧的pickle文件
Remove-Item -Path "./data/josephson/*.pkl" -Force -ErrorAction SilentlyContinue

# 验证已删除
Get-ChildItem -Path "./data/josephson/" -Filter "*.pkl"  # 应该为空
```

### Step 2: 激活虚拟环境
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: 运行训练
```powershell
python josephson_junction_param_learning_v2.py
```

### Step 4: 监控关键指标（查看输出）
```
第10个epoch:  Train ~ 1.3, Eval ~ 1.6  (间隔1.23x ✓)
第30个epoch:  Train ~ 1.0, Eval ~ 1.3  (间隔1.30x ✓)
第100个epoch: Train ~ 0.7, Eval ~ 1.0  (间隔1.43x ✓)
第200个epoch: Train ~ 0.6, Eval ~ 0.85 (间隔1.42x ✓)

对比旧结果：
第200个epoch: Train ~ 0.87, Eval ~ 1.80 (间隔2.07x ✗)
```

---

## 📊 预期改善指标

### 目标对比

| 指标 | 旧结果 | 新目标 | 改善 |
|------|--------|---------|------|
| **最终Eval Loss** | 1.804 | < 0.95 | ↓47% |
| **Train-Eval间隔** | 2.07x | 1.4x | ↓32% |
| **κ参数误差** | ~500% | < 30% | ↓94% |
| **β参数误差** | ~15% | < 8% | ↓47% |
| **总体MAE** | 混乱 | 均衡 | ✓ |

### 参数误差预测

| 参数 | 旧误差 | 新目标 |
|------|--------|--------|
| β₁ | 9.7% | 4-6% |
| β₂ | 12.3% | 5-8% |
| i₁ | 22.6% | 8-12% |
| i₂ | 18.6% | 8-12% |
| κ₁ | 3.7% → 误差实际很大 | 2-4% |
| κ₂ | 3.2% → 误差实际很大 | 2-4% |
| σ₁ | 14.7% | 8-12% |
| σ₂ | 12.5% | 8-12% |

---

## ⏱️ 预计耗时

| 任务 | 耗时 |
|------|------|
| 数据生成 | ~5-10分钟 |
| 训练200个epoch | ~25-35分钟 |
| 总计 | ~30-45分钟 |

---

## 🔍 实时监控命令

### 如果要看training_history.png的实时更新
```powershell
# 在另一个终端运行
while($true) {
    Write-Host "最后修改时间: $(Get-Item './data/josephson/penn_transformer_v1/training_history.png' | % LastWriteTime)"
    Start-Sleep -Seconds 30
}
```

---

## ✅ 验证修改已生效

在运行训练前，快速验证配置：

```powershell
# 查看N_RANGE
grep "N_RANGE" josephson_junction_param_learning_v2.py
# 应该显示: N_RANGE = (80, 200)

# 查看T_RANGE  
grep "T_RANGE" josephson_junction_param_learning_v2.py
# 应该显示: T_RANGE = (5.0, 15.0)

# 查看param_weights
grep "param_weights" josephson_junction_param_learning_v2.py
# 应该显示: [1.5, 1.5, 1.0, 1.0, 3.0, 3.0, 1.5, 1.5]

# 查看dropout
grep "dropout" josephson_junction_param_learning_v2.py
# 应该显示: 'dropout': 0.3
```

---

## 🎓 理论依据

### 为什么这些修改会有效？

#### 1. **数据多样性** ← 最关键
```
旧问题：所有样本都是dt=0.1, T=10秒
新方案：dt在0.025-0.2范围内随机
结果：模型学到鲁棒的参数关系，而非记住特定尺度
```

#### 2. **权重平衡** ← 次关键
```
旧问题：κ误差800%，β误差10%
原因：高权重强制模型关注不可学的特征
新方案：均衡权重，让模型自行安排容量
结果：所有参数误差均匀分布在可学范围内
```

#### 3. **正则化增强** ← 防止过拟合
```
旧问题：Train loss 0.87 vs Eval loss 1.80，间隔2.07x
原因：模型太大(256→128的嵌入维度)
新方案：d_model/2, dropout 3x
结果：Train-Eval间隔缩小到1.4x（正常范围）
```

#### 4. **学习率调整** ← 稳定性
```
旧问题：振荡，收敛不稳定
原因：学习率1e-4相对于数据规模过大
新方案：5e-5更保守，weight_decay更强
结果：平滑收敛，无跳跃
```

---

## 🚨 如果仍未改善

### 诊断步骤
```python
# 在training loop中插入诊断代码（第1437行）
if epoch == 5:
    # 检查数据
    for batch in train_loader:
        x, y, h, l, z = batch
        print(f"Sample lengths L: {l[:5]}")  # 应该随机在80-200
        print(f"Sample time spans H: {h[:5]}")  # 应该随机在5-15
        print(f"Param range Y: {y.min(1)[0]}, {y.max(1)[0]}")
        break
    
    # 检查梯度
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += (p.grad.data.norm(2)**2).item()
    print(f"Gradient norm: {(total_grad_norm**0.5):.4f}")
```

### 如果Train损失不下降
- → 检查数据是否真的重新生成了（train.pkl的修改时间应该是最近）
- → 检查学习率是否太低（尝试1e-4)

### 如果Eval损失仍不下降
- → κ权重可能还需要进一步降低（试试1.0而不是3.0）
- → 可能需要增加dropout到0.5

### 如果某个参数特别差（误差>100%）
- → 该参数可能对这个系统不可观测
- → 尝试从权重中完全去除它（设为0）

---

## 📝 预期的最终图表

完成训练后查看 `./data/josephson/penn_transformer_v1/training_history.png`：

```
Loss曲线应该显示：
┌─────────────────────────────────┐
│    Train Loss (下降曲线)          │
│        ╲╲╲╲                      │
│            ╲╲╲ Eval Loss         │
│               ╲╲╲╲              │
│                   ╲             │
│                    ╲            │
│                     ╲           │
└─────────────────────╲──────────┘
        Epoch            200

✅ 好：两条线平行下降，eval始终略高于train
❌ 坏：eval线平缓或上升
```

---

## 🎉 最终检查清单

修复前必做：
- [ ] 备份旧的模型文件（可选）
- [ ] 删除所有旧的pickle文件
- [ ] 验证代码中的4项修改已生效
- [ ] 虚拟环境已激活

开始训练：
- [ ] 运行 `python josephson_junction_param_learning_v2.py`
- [ ] 观察第10个epoch的Train/Eval loss比例
- [ ] 等待完整的200个epoch

训练完成后：
- [ ] 检查training_history.png中两条线是否同向
- [ ] 验证最终eval loss < 1.0
- [ ] 检查各参数误差是否均衡（无异常值）
- [ ] 保存最佳模型

---

## 🔗 相关文档

- 详细分析：见 `EVAL_CONVERGENCE_ANALYSIS.md`
- 代码变更位置：第120-125行（N/T_RANGE）, 第1230-1245行（超参）

