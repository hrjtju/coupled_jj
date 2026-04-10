"""快速测试 v4 代码"""
import sys
import os

# 设置目录
os.chdir(r'D:\Research\Mathematics\SYSU_GBU\2_projects\01_wSLYuan_Coupled_JJ\2_python_code')

from josephson_junction_param_learning_v4 import *

# 快速测试配置
config = {
    'seed': 42,
    'n_all': 200,  # 小数据集
    'eval_ratio': 0.15,
    'test_ratio': 0.20,
    'discard_T': 10.0,
    'data_dir': './data/josephson',
    'train_file': './data/josephson/train_v4_test.pkl',
    'eval_file': './data/josephson/eval_v4_test.pkl',
    'test_file': './data/josephson/test_v4_test.pkl',
    'lstm_layers': 2,
    'lstm_hidden': 16,
    'activation': 'elu',
    'dropout': 0.1,
    'batch_size': 16,
    'num_epochs': 20,  # 只跑20个epoch
    'learning_rate': 0.001,
    'lr_patience': 10,
    'lr_factor': 0.5,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'loss_weight': PhysicalParamsConfig.get_param_weights(),
    'sigma_weight': 1.0,
    'save_every': 100,
}

device = get_device()
print(f"Device: {device}")
seed_everything(config['seed'])

# 生成数据
if not all(os.path.exists(f) for f in [config['train_file'], config['eval_file'], config['test_file']]):
    print("\nGenerating test data (200 samples)...")
    data = generate_trajectory_data(config['n_all'], config['seed'], config['discard_T'], device)

    indices = np.random.permutation(config['n_all'])
    n_train = int(config['n_all'] * (1 - config['test_ratio'] - config['eval_ratio']))
    n_eval = int(config['n_all'] * (1 - config['test_ratio']))

    def subset(d, idx):
        return {'X': [d['X'][i] for i in idx], 'Y': [d['Y'][i] for i in idx], 'H': [d['H'][i] for i in idx]}

    save_data_pickle(subset(data, indices[:n_train]), config['train_file'])
    save_data_pickle(subset(data, indices[n_train:n_eval]), config['eval_file'])
    save_data_pickle(subset(data, indices[n_eval:]), config['test_file'])
    print("Data saved!")

# 加载数据
print("\nLoading datasets...")
train_data = MyDataset(config['train_file'])
eval_data = MyDataset(config['eval_file'])

print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, drop_last=True)
eval_loader = DataLoader(eval_data, batch_size=len(eval_data))

in_dim, n_class = train_data.get_dim()
print(f"Input dim: {in_dim}, Output dim: {n_class}")

# 创建模型
model = PENNv4(in_dim, config['lstm_hidden'], config['lstm_layers'],
               n_class, config['activation'], config['dropout']).to(device)
model = model.double()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                             weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=config['lr_factor'],
    patience=config['lr_patience'], verbose=True
)

# 损失函数
criterion = StableLoss(config['loss_weight'], config['sigma_weight'])

# 加载评估数据
for x_eval, y_eval, l_eval, h_eval, z_eval in eval_loader:
    x_eval = x_eval.to(device)
    y_eval = y_eval.to(device)
    l_eval = l_eval.to(device)
    h_eval = h_eval.to(device)
    z_eval = z_eval.to(device)

# 训练循环
print("\n" + "=" * 60)
print("Training (Quick Test - 20 epochs)")
print("=" * 60)

history = []
best_loss = float('inf')

for epoch in range(config['num_epochs']):
    start_time = time.time()
    model.train()

    epoch_losses = []
    for x, y, l, h, z in train_loader:
        x, y, l, h, z = x.to(device), y.to(device), l.to(device), h.to(device), z.to(device)

        pred = model(x, l, h, z)
        losses = criterion(pred, y, x, l, h)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        epoch_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in losses.items()})

    # 评估
    model.eval()
    with torch.no_grad():
        eval_pred = model(x_eval, l_eval, h_eval, z_eval)
        eval_losses = criterion(eval_pred, y_eval, x_eval, l_eval, h_eval)
        eval_loss = {k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in eval_losses.items()}

    # 平均训练损失
    train_loss = {
        'total': np.mean([l['total'] for l in epoch_losses]),
        'standard': np.mean([l['standard'] for l in epoch_losses]),
        'sigma_consistency': np.mean([l['sigma_consistency'] for l in epoch_losses]),
    }

    history.append(train_loss)
    scheduler.step(eval_loss['total'])

    print(f"Epoch {epoch+1}/{config['num_epochs']} | "
          f"Train: {train_loss['total']:.4f} | "
          f"Eval: {eval_loss['total']:.4f} | "
          f"Time: {time.time()-start_time:.1f}s")

    if eval_loss['total'] < best_loss:
        best_loss = eval_loss['total']

print(f"\nBest eval loss: {best_loss:.4f}")
print("\nTest completed successfully!")
