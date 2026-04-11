import torch
import torchsde
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec

# ==================== 参数设置 ====================
params = {
    'i1': 0.3,
    'i2': 0.3,
    'beta_J1': 1.2,
    'beta_J2': 1.2,
    'k1': 0.05,
    'k2': 0.05,
    'sigma1': 0.02,
    'sigma2': 0.02
}

i1 = params['i1']
i2 = params['i2']
beta1 = params['beta_J1']
beta2 = params['beta_J2']
kappa1 = params['k1']
kappa2 = params['k2']
sigma1 = params['sigma1']
sigma2 = params['sigma2']

T = 400.0
N = 15000
dt = T / N
ts = torch.linspace(0, T, N)
y0 = torch.tensor([[0.0, 0.0, 0.0, 0.0]])

torch.manual_seed(42)
np.random.seed(42)

# ==================== SDE模型 ====================
class CoupledPhaseOscillators(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        phi1, v1, phi2, v2 = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        phi1_w = (phi1 + np.pi) % (2 * np.pi) - np.pi
        phi2_w = (phi2 + np.pi) % (2 * np.pi) - np.pi
        diff = phi2_w - phi1_w
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
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
print("Running simulation...")
ys = torchsde.sdeint(sde, y0, ts, method='euler', dt=dt)

phi1 = ys[:, 0, 0].detach().numpy()
v1 = ys[:, 0, 1].detach().numpy()
phi2 = ys[:, 0, 2].detach().numpy()
v2 = ys[:, 0, 3].detach().numpy()
ts_np = ts.numpy()

phi1_w = (phi1 + np.pi) % (2 * np.pi) - np.pi
phi2_w = (phi2 + np.pi) % (2 * np.pi) - np.pi
diff_continuous = phi1 - phi2
diff_w = (diff_continuous + np.pi) % (2 * np.pi) - np.pi

print(f"Simulation completed.")

# ==================== 创建大图 (3x3布局) ====================
fig = plt.figure(figsize=(25, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.2, wspace=0.2)

# ========== 第0行：2D图1-3 ==========
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(ts_np, phi1_w, label='φ₁', color='#1f77b4', alpha=0.8, linewidth=1.2)
ax0.plot(ts_np, phi2_w, label='φ₂', color='#ff7f0e', alpha=0.8, linewidth=1.2)
ax0.set_title('Phase Trajectories [-π, π]', fontsize=14, fontweight='bold')
ax0.set_xlabel('Time t')
ax0.set_ylabel('Phase (rad)')
ax0.legend()
ax0.set_ylim([-np.pi, np.pi])
ax0.grid(True, alpha=0.3)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(ts_np, v1, label='v₁', color='#2ca02c', alpha=0.8, linewidth=1.2)
ax1.plot(ts_np, v2, label='v₂', color='#d62728', alpha=0.8, linewidth=1.2)
ax1.set_title('Angular Velocity', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time t')
ax1.set_ylabel('Velocity v')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(phi1_w, v1, c=ts_np, cmap='Blues', s=2, alpha=0.6, label='JJ 1')
ax2.scatter(phi2_w, v2, c=ts_np, cmap='Oranges', s=2, alpha=0.6, label='JJ 2')
ax2.set_title('Phase-Space (φ vs v)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Phase φ [-π, π]')
ax2.set_ylabel('Velocity v')
ax2.set_xlim([-np.pi, np.pi])
ax2.legend()
ax2.grid(True, alpha=0.3)

# ========== 第1行：2D图4-5 + 空位 ==========
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(phi1_w, phi2_w, color='#9467bd', alpha=0.6, linewidth=1)
ax3.scatter(phi1_w[0], phi2_w[0], color='green', s=80, marker='o', 
           label='Start', zorder=5, edgecolors='black')
ax3.scatter(phi1_w[-1], phi2_w[-1], color='red', s=80, marker='s', 
           label='End', zorder=5, edgecolors='black')
ax3.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', alpha=0.3, linewidth=1, label='Sync')
ax3.set_title('Phase-Phase Relationship', fontsize=14, fontweight='bold')
ax3.set_xlabel('φ₁ [-π, π]')
ax3.set_ylabel('φ₂ [-π, π]')
ax3.set_xlim([-np.pi, np.pi])
ax3.set_ylim([-np.pi, np.pi])
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')

ax4 = fig.add_subplot(gs[1, 1])
ax_twin = ax4.twinx()
line1 = ax4.plot(ts_np, diff_w, color='#8c564b', linewidth=1.5, label='Wrapped')
line2 = ax_twin.plot(ts_np, diff_continuous, color='#e377c2', alpha=0.5, 
                    linestyle='--', linewidth=1, label='Continuous')
ax4.set_title('Phase Difference', fontsize=14, fontweight='bold')
ax4.set_xlabel('Time t')
ax4.set_ylabel('Wrapped [rad]', color='#8c564b')
ax_twin.set_ylabel('Continuous [rad]', color='#e377c2')
ax4.tick_params(axis='y', labelcolor='#8c564b', labelsize=8)
ax_twin.tick_params(axis='y', labelcolor='#e377c2', labelsize=8)
ax4.set_xlim([0, T])
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
lines = line1 + line2
labs = [l.get_label() for l in lines]
ax4.legend(lines, labs, loc='upper right', fontsize=8)

ax_empty = fig.add_subplot(gs[1, 2])
ax_empty.axis('off')

# ========== 第2行：3D环面三个视角（修改后的版本） ==========
# 环面参数（扁平）
R_torus = 2.5
r_torus = 0.35
flatten_factor = 0.6

phi1_torus = phi1_w + np.pi  # [0, 2π]
phi2_torus = phi2_w + np.pi  # [0, 2π]

# 环面坐标
x_torus = (R_torus + r_torus * np.cos(phi2_torus)) * np.cos(phi1_torus)
y_torus = (R_torus + r_torus * np.cos(phi2_torus)) * np.sin(phi1_torus)
z_torus = r_torus * flatten_factor * np.sin(phi2_torus)

# 同步线
phi_sync = np.linspace(0, 2*np.pi, 100)
x_sync = (R_torus + r_torus * np.cos(phi_sync)) * np.cos(phi_sync)
y_sync = (R_torus + r_torus * np.cos(phi_sync)) * np.sin(phi_sync)
z_sync = r_torus * flatten_factor * np.sin(phi_sync)

# 准备网格数据（用于表面）
phi1_grid = np.linspace(0, 2*np.pi, 50)
phi2_grid = np.linspace(0, 2*np.pi, 30)
phi1_grid, phi2_grid = np.meshgrid(phi1_grid, phi2_grid)
x_grid = (R_torus + r_torus * np.cos(phi2_grid)) * np.cos(phi1_grid)
y_grid = (R_torus + r_torus * np.cos(phi2_grid)) * np.sin(phi1_grid)
z_grid = r_torus * flatten_factor * np.sin(phi2_grid)

# 准备轨迹线段数据
points = np.array([x_torus, y_torus, z_torus]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 6. 3D主视角
ax5 = fig.add_subplot(gs[2, 0], projection='3d')
ax5.plot_surface(x_grid, y_grid, z_grid, alpha=0.08, color='lightgray', 
                linewidth=0, antialiased=False, rstride=1, cstride=1)
# 使用Line3DCollection绘制轨迹（RdBu颜色映射）
lc = Line3DCollection(segments, cmap='RdBu', alpha=0.95, linewidth=1)
lc.set_array(ts_np)
ax5.add_collection3d(lc)
ax5.scatter(x_torus[0], y_torus[0], z_torus[0], color='green', s=150, marker='o', 
           edgecolors='black', linewidth=2, zorder=10)
ax5.scatter(x_torus[-1], y_torus[-1], z_torus[-1], color='red', s=150, marker='s', 
           edgecolors='black', linewidth=2, zorder=10)
ax5.plot(x_sync, y_sync, z_sync, 'k--', alpha=0.9, linewidth=1, label='Sync line', zorder=20)
ax5.set_xlim([-3, 3])
ax5.set_ylim([-3, 3])
ax5.set_zlim([-0.5, 0.5])
ax5.set_title('Torus T² (Main View)', fontsize=14, fontweight='bold')
ax5.set_axis_off()

# 7. 3D俯视图
ax6 = fig.add_subplot(gs[2, 1], projection='3d')
ax6.plot_surface(x_grid, y_grid, z_grid, alpha=0.08, color='lightgray', linewidth=0)
segments_top = np.concatenate([points[:-1], points[1:]], axis=1)
lc_top = Line3DCollection(segments_top, cmap='RdBu', alpha=0.95, linewidth=1)
lc_top.set_array(ts_np)
ax6.add_collection3d(lc_top)
ax6.scatter(x_torus[0], y_torus[0], z_torus[0], color='green', s=150, marker='o', 
           edgecolors='black', linewidth=2, zorder=10)
ax6.scatter(x_torus[-1], y_torus[-1], z_torus[-1], color='red', s=150, marker='s', 
           edgecolors='black', linewidth=2, zorder=10)
ax6.plot(x_sync, y_sync, z_sync, 'k--', alpha=0.9, linewidth=1, zorder=20)
ax6.view_init(elev=90, azim=-90)
ax6.set_xlim([-3, 3])
ax6.set_ylim([-3, 3])
ax6.set_zlim([-0.5, 0.5])
ax6.set_title('Torus (Top View)', fontsize=14, fontweight='bold')
ax6.set_axis_off()

# 8. 3D侧视图
ax7 = fig.add_subplot(gs[2, 2], projection='3d')
ax7.plot_surface(x_grid, y_grid, z_grid, alpha=0.08, color='lightgray', linewidth=0)
segments_side = np.concatenate([points[:-1], points[1:]], axis=1)
lc_side = Line3DCollection(segments_side, cmap='RdBu', alpha=0.95, linewidth=1)
lc_side.set_array(ts_np)
ax7.add_collection3d(lc_side)
ax7.scatter(x_torus[0], y_torus[0], z_torus[0], color='green', s=150, marker='o', 
           edgecolors='black', linewidth=2, zorder=10)
ax7.scatter(x_torus[-1], y_torus[-1], z_torus[-1], color='red', s=150, marker='s', 
           edgecolors='black', linewidth=2, zorder=10)
ax7.plot(x_sync, y_sync, z_sync, 'k--', alpha=0.9, linewidth=1, zorder=20)
ax7.view_init(elev=0, azim=0)
ax7.set_xlim([-3, 3])
ax7.set_ylim([-3, 3])
ax7.set_zlim([-0.5, 0.5])
ax7.set_title('Torus (Side View)', fontsize=14, fontweight='bold')
ax7.set_axis_off()

plt.suptitle(f'Coupled Phase Oscillators: Complete Dynamics Analysis\n{params}', 
             fontsize=16, fontweight='bold')

plt.savefig('coupled_oscillators_complete.png', dpi=300, 
            bbox_inches='tight', facecolor='white')
plt.close()

print("All 8 plots arranged in 3x3 grid with updated 3D torus visualization.")