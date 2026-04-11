from ast import List
from typing import Tuple

from matplotlib.axes import Axes
import torch
import torchsde
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

# ==================== 参数设置 ====================
params = {
    'i1': 0.8,
    'i2': 0.8,
    'beta_J1': 0.1,
    'beta_J2': 0.1,
    'k1': 0.05,
    'k2': 0.05,
    'sigma1': 0.05,
    'sigma2': 0.05
}

i1 = params['i1']
i2 = params['i2']
beta1 = params['beta_J1']
beta2 = params['beta_J2']
kappa1 = params['k1']
kappa2 = params['k2']
sigma1 = params['sigma1']
sigma2 = params['sigma2']

T = 50.0
N = 5000
dt = T / N
ts = torch.linspace(0, T, N)

# 初始条件：phi1=0, v1=0, phi2=0, v2=0
y0 = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

torch.manual_seed(42)
np.random.seed(42)

# ==================== SDE模型定义 ====================
class CoupledPhaseOscillators(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        
        # 模 2π 处理，映射到 [-π, π]
        phi1_w = (phi1 + np.pi) % (2 * np.pi) - np.pi
        phi2_w = (phi2 + np.pi) % (2 * np.pi) - np.pi
        
        # 最短路径相位差
        diff = phi2_w - phi1_w
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        # 漂移函数
        d_phi1 = v1
        d_v1 = i1 - beta1 * v1 - torch.sin(phi1_w) + kappa1 * diff
        d_phi2 = v2
        d_v2 = i2 - beta2 * v2 - torch.sin(phi2_w) - kappa2 * diff
        
        return torch.stack([d_phi1, d_v1, d_phi2, d_v2], dim=-1)
    
    def g(self, t, y):
        noise = torch.zeros_like(y)
        noise[..., 1] = sigma1
        noise[..., 3] = sigma2
        return noise

sde = CoupledPhaseOscillators()

# ==================== 运行模拟 ====================
print("Running simulation...")
ys = torchsde.sdeint(sde, y0, ts, method='euler', dt=dt)

phi1 = ys[:, 0, 0].detach().numpy()
v1 = ys[:, 0, 1].detach().numpy()
phi2 = ys[:, 0, 2].detach().numpy()
v2 = ys[:, 0, 3].detach().numpy()
ts_np = ts.numpy()

# 数据后处理
phi1_w = (phi1 + np.pi) % (2 * np.pi) - np.pi
phi2_w = (phi2 + np.pi) % (2 * np.pi) - np.pi
diff_continuous = phi1 - phi2
diff_w = (diff_continuous + np.pi) % (2 * np.pi) - np.pi

print(f"Simulation completed. Final phase diff: {diff_w[-1]:.4f} rad")

# ==================== 标准五图 ====================
fig = plt.figure(figsize=(15, 15))

fig1, axes = plt.subplots(3, 3, figsize=(15, 15))
axes: Tuple[Axes] = axes.flatten()


# 1. 相位轨道
axes[0].plot(ts_np, phi1_w, label='φ₁', color='#1f77b4', alpha=0.8, linewidth=1.2)
axes[0].plot(ts_np, phi2_w, label='φ₂', color='#ff7f0e', alpha=0.8, linewidth=1.2)
axes[0].set_title('1. Phase Trajectories [-π, π]', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time t')
axes[0].set_ylabel('Phase (rad)')
axes[0].legend()
axes[0].set_ylim([-np.pi, np.pi])
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# 2. 角速度轨道
axes[1].plot(ts_np, v1, label='v₁', color='#2ca02c', alpha=0.8, linewidth=1.2)
axes[1].plot(ts_np, v2, label='v₂', color='#d62728', alpha=0.8, linewidth=1.2)
axes[1].set_title('2. Angular Velocity Trajectories', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time t')
axes[1].set_ylabel('Velocity v')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. 相空间
axes[2].scatter(phi1_w, v1, c=ts_np, cmap='Blues', s=8, alpha=0.6, label='Oscillator 1')
axes[2].scatter(phi2_w, v2, c=ts_np, cmap='Oranges', s=8, alpha=0.6, label='Oscillator 2')
axes[2].set_title('3. Phase-Space (φ vs v)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Phase φ [-π, π]')
axes[2].set_ylabel('Velocity v')
axes[2].set_xlim([-np.pi, np.pi])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. 相位-相位关系（平面）
axes[3].plot(phi1_w, phi2_w, color='#9467bd', alpha=0.6, linewidth=1)
axes[3].scatter(phi1_w[0], phi2_w[0], color='green', s=100, marker='o', 
               label='Start', zorder=5, edgecolors='black')
axes[3].scatter(phi1_w[-1], phi2_w[-1], color='red', s=100, marker='s', 
               label='End', zorder=5, edgecolors='black')
axes[3].plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', alpha=0.3, linewidth=1, label='Sync')
axes[3].set_title('4. Phase-Phase Relationship', fontsize=12, fontweight='bold')
axes[3].set_xlabel('φ₁ [-π, π]')
axes[3].set_ylabel('φ₂ [-π, π]')
axes[3].set_xlim([-np.pi, np.pi])
axes[3].set_ylim([-np.pi, np.pi])
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].set_aspect('equal')

# 5. 相位差
ax_twin = axes[4].twinx()
line1 = axes[4].plot(ts_np, diff_w, color='#8c564b', linewidth=1.5, label='Wrapped (mod 2π)')
line2 = ax_twin.plot(ts_np, diff_continuous, color='#e377c2', alpha=0.5, 
                    linestyle='--', linewidth=1, label='Continuous')
axes[4].set_title('5. Phase Difference', fontsize=12, fontweight='bold')
axes[4].set_xlabel('Time t')
axes[4].set_ylabel('Phase Diff (mod 2π) [rad]', color='#8c564b')
ax_twin.set_ylabel('Cumulative Diff [rad]', color='#e377c2')
axes[4].tick_params(axis='y', labelcolor='#8c564b')
ax_twin.tick_params(axis='y', labelcolor='#e377c2')
axes[4].set_xlim([0, T])
axes[4].grid(True, alpha=0.3)
axes[4].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
lines = line1 + line2
labs = [l.get_label() for l in lines]
axes[4].legend(lines, labs, loc='upper right')

axes[5].axis('off')

# ==================== 3D环面可视化 ====================
# 矮胖环面参数
R = 2.5
r = 0.35
flatten_factor = 0.5

phi1_torus = phi1_w + np.pi  # [0, 2π]
phi2_torus = phi2_w + np.pi  # [0, 2π]

# 重新计算环面坐标
x_torus_flat = (R + r * np.cos(phi2_torus)) * np.cos(phi1_torus)
y_torus_flat = (R + r * np.cos(phi2_torus)) * np.sin(phi1_torus)
z_torus_flat = r * flatten_factor * np.sin(phi2_torus)

# 同步线
phi_sync = np.linspace(0, 2*np.pi, 100)

x_sync_flat = (R + r * np.cos(phi_sync)) * np.cos(phi_sync)
y_sync_flat = (R + r * np.cos(phi_sync)) * np.sin(phi_sync)
z_sync_flat = r * flatten_factor * np.sin(phi_sync)

fig = plt.figure(figsize=(16, 7))

# 左图：扁平环面（无网格）
ax1 = fig.add_subplot(131, projection='3d')

# 绘制环面表面（无网格线）
phi1_grid = np.linspace(0, 2*np.pi, 50)
phi2_grid = np.linspace(0, 2*np.pi, 30)
phi1_grid, phi2_grid = np.meshgrid(phi1_grid, phi2_grid)
x_grid = (R + r * np.cos(phi2_grid)) * np.cos(phi1_grid)
y_grid = (R + r * np.cos(phi2_grid)) * np.sin(phi1_grid)
z_grid = r * flatten_factor * np.sin(phi2_grid)
ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.08, color='lightgray', 
                linewidth=0, antialiased=False, rstride=1, cstride=1)

# 绘制轨迹
points = np.array([x_torus_flat, y_torus_flat, z_torus_flat]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = Line3DCollection(segments, cmap='RdBu', alpha=0.95, linewidth=2.5)
lc.set_array(ts_np)
ax1.add_collection3d(lc)

# 标记点
ax1.scatter(x_torus_flat[0], y_torus_flat[0], z_torus_flat[0], color='green', s=150, marker='o', 
           label='Start', edgecolors='black', linewidth=2, zorder=10)
ax1.scatter(x_torus_flat[-1], y_torus_flat[-1], z_torus_flat[-1], color='red', s=150, marker='s', 
           label='End', edgecolors='black', linewidth=2, zorder=10)
ax1.plot(x_sync_flat, y_sync_flat, z_sync_flat, 'r--', alpha=0.9, linewidth=2, label='Sync line')

ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([-0.5, 0.5])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(r'Flattened Torus $\mathbb{T}^2$', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(False)  # 去掉坐标网格
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')
ax1.set_axis_off()


# 中图：俯视图（无网格）
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x_grid, y_grid, z_grid, alpha=0.08, color='lightgray', linewidth=0)

segments2 = np.concatenate([points[:-1], points[1:]], axis=1)
lc2 = Line3DCollection(segments2, cmap='RdBu', alpha=0.95, linewidth=2.5)
lc2.set_array(ts_np)
ax2.add_collection3d(lc2)
ax2.scatter(x_torus_flat[0], y_torus_flat[0], z_torus_flat[0], color='green', s=150, marker='o', 
           edgecolors='black', linewidth=2, zorder=10)
ax2.scatter(x_torus_flat[-1], y_torus_flat[-1], z_torus_flat[-1], color='red', s=150, marker='s', 
           edgecolors='black', linewidth=2, zorder=10)
ax2.plot(x_sync_flat, y_sync_flat, z_sync_flat, 'r--', alpha=0.9, linewidth=2)
ax2.view_init(elev=90, azim=-90)
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([-0.5, 0.5])
ax2.set_title('Top View', fontsize=12, fontweight='bold')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.grid(False)
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('w')
ax2.yaxis.pane.set_edgecolor('w')
ax2.zaxis.pane.set_edgecolor('w')
ax2.set_axis_off()


# 右图：侧视图（无网格）
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x_grid, y_grid, z_grid, alpha=0.08, color='lightgray', linewidth=0)

segments3 = np.concatenate([points[:-1], points[1:]], axis=1)
lc3 = Line3DCollection(segments3, cmap='RdBu', alpha=0.95, linewidth=2.5)
lc3.set_array(ts_np)
ax3.add_collection3d(lc3)
ax3.scatter(x_torus_flat[0], y_torus_flat[0], z_torus_flat[0], color='green', s=150, marker='o', 
           edgecolors='black', linewidth=2, zorder=10)
ax3.scatter(x_torus_flat[-1], y_torus_flat[-1], z_torus_flat[-1], color='red', s=150, marker='s', 
           edgecolors='black', linewidth=2, zorder=10)
ax3.plot(x_sync_flat, y_sync_flat, z_sync_flat, 'r--', alpha=0.9, linewidth=2)
ax3.view_init(elev=0, azim=0)
ax3.set_xlim([-3, 3])
ax3.set_ylim([-3, 3])
ax3.set_zlim([-0.5, 0.5])
ax3.set_title('Side View', fontsize=12, fontweight='bold')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.grid(False)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor('w')
ax3.yaxis.pane.set_edgecolor('w')
ax3.zaxis.pane.set_edgecolor('w')
ax3.set_axis_off()



plt.tight_layout()
plt.savefig('coupled_oscillators.png', dpi=300, bbox_inches='tight')
plt.close()


print("All figures saved.")