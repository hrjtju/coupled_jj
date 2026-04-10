#!/usr/bin/env python3
"""
Train a neural-parameterized energy landscape for the bistable SDE
dX = (X - X**3) dt + sigma dW

Implements EnergyLandscapeSDE, data generation, training with two losses:
 - local finite-difference drift regression
 - equilibrium distribution (histogram) matching

Run: python bistable_energy_sde.py
"""
import os
import random
import math
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tqdm


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EnergyLandscapeSDE(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.U = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # learn log sigma (optional)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(1.0)))

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(x)

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1]
        x_req = x.requires_grad_(True)
        U = self.potential(x_req)
        
        # create_graph=True 允许梯度流回potential参数
        grad_U = torch.autograd.grad(
            U.sum(), 
            x_req, 
            create_graph=True,  # 关键：构建二阶计算图
            retain_graph=True   # 保留中间结果
        )[0]
        
        return -grad_U

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.drift(x), torch.exp(self.log_sigma)


def true_bistable_step(x: np.ndarray, dt: float, sigma: float) -> np.ndarray:
    # Euler-Maruyama for scalar SDE
    dW = np.random.normal(0.0, np.sqrt(dt), size=x.shape)
    
    return x + (x - x**3) * dt + sigma * dW # simplified Ginzburg-Landau problem
    # return x + (1 - x) * dt + sigma * dW # OU process
    # return x + np.sin(x) * dt + sigma * dW # sin


def generate_bistable_data(n_trajectories: int = 1000, n_steps: int = 10000, dt: float = 0.01, sigma: float = 1.0, x0=1.0):
    # Return array shape (n_trajectories, n_steps+1)
    trajs = np.zeros((n_trajectories, n_steps + 1), dtype=np.float32)
    # trajs[:, 0] = x0 * np.sign(np.random.randn(n_trajectories))  # start near ±1 randomly
    # trajs[:, 0] = x0 * np.random.randint(-10, 10, n_trajectories)  + np.random.randn(n_trajectories)
    trajs[:, 0] = np.random.randn(n_trajectories)
    for t in range(n_steps):
        trajs[:, t+1] = true_bistable_step(trajs[:, t], dt, sigma)
    return trajs


def equilibrium_distribution_loss(model: EnergyLandscapeSDE,
                                  data_flat: np.ndarray,
                                  device: torch.device,
                                  x_min: float = -10.0,
                                  x_max: float = 10.0,
                                  n_bins: int = 200,
                                  sigma_soft: float = 0.1):
    """
    Differentiable equilibrium distribution loss using Gaussian soft-binning
    - Build soft histogram of empirical data using Gaussian kernels on `n_bins` bin centers
    - Compute model equilibrium density at the same bin centers via p_model(x) ~ exp(-2 U(x) / sigma^2)
    - Use KL divergence between model (input as log-prob) and empirical (target prob)

    Returns a torch scalar (on `device`) with gradients flowing to model parameters.
    """
    # prepare bins
    bins = torch.linspace(x_min, x_max, n_bins, device=device)  # [n_bins]

    # empirical soft histogram (torch, no grad required for data)
    data_t = torch.from_numpy(data_flat).float().to(device).unsqueeze(1)  # [N,1]

    # compute distances [N, n_bins]
    dists = data_t - bins.unsqueeze(0)  # broadcasting -> [N, n_bins]
    weights_t = torch.exp(-0.5 * (dists / sigma_soft) ** 2)  # gaussian kernel
    hist_t = weights_t.sum(dim=0)  # [n_bins]
    prob_t = hist_t + 1e-12
    prob_t = prob_t / prob_t.sum()

    # model predicted probability at bin centers (differentiable)
    bins_in = bins.unsqueeze(-1)  # [n_bins,1]
    Uvals = model.potential(bins_in).squeeze()  # [n_bins]
    sigma_model = torch.exp(model.log_sigma)
    p_model = torch.exp(-2.0 * Uvals / (sigma_model ** 2 + 1e-12))
    prob_p = p_model + 1e-12
    prob_p = prob_p / prob_p.sum()

    # KL divergence: input is log_prob, target is prob
    kl = nn.KLDivLoss(reduction='batchmean')
    loss = kl(prob_p.log().unsqueeze(0), prob_t.unsqueeze(0))
    return loss


def find_local_extrema(y: np.ndarray):
    dy = np.gradient(y)
    ddy = np.gradient(dy)
    minima = np.where((np.hstack([dy[0], dy[:-1]]) > 0) & (np.hstack([dy[1:], dy[-1]]) < 0))[0]
    maxima = np.where((np.hstack([dy[0], dy[:-1]]) < 0) & (np.hstack([dy[1:], dy[-1]]) > 0))[0]
    return minima, maxima


def extract_bistable_structure(model: EnergyLandscapeSDE, device: torch.device):
    x_fine = np.linspace(-3, 3, 10001)
    with torch.no_grad():
        xs = torch.from_numpy(x_fine).float().unsqueeze(-1).to(device)
        Uvals = model.potential(xs).squeeze().cpu().numpy()

    # find minima and maxima
    # use simple finite-diff sign-change method
    dy = np.gradient(Uvals, x_fine)
    # minima where derivative crosses 0 from negative to positive
    minima_idx = np.where((np.hstack([dy[0], dy[:-1]]) < 0) & (np.hstack([dy[1:], dy[-1]]) > 0))[0]
    maxima_idx = np.where((np.hstack([dy[0], dy[:-1]]) > 0) & (np.hstack([dy[1:], dy[-1]]) < 0))[0]

    minima = x_fine[minima_idx].tolist()
    maxima = x_fine[maxima_idx].tolist()

    barrier_height = None
    if len(minima_idx) >= 2 and len(maxima_idx) >= 1:
        # pick largest-min and central max
        U_min_vals = Uvals[minima_idx]
        U_max_vals = Uvals[maxima_idx]
        barrier_height = float(U_max_vals.min() - U_min_vals.max())

    return {
        'stable_states': minima,
        'barrier_locations': maxima,
        'barrier_height': barrier_height,
        'x_fine': x_fine,
        'Uvals': Uvals,
    }


def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data params
    n_traj = 1000
    n_steps = 800  # length of each trajectory
    dt = 0.01
    sigma_true = 0.5

    print("Generating data...")
    trajs = generate_bistable_data(n_trajectories=n_traj, n_steps=n_steps, dt=dt, sigma=sigma_true)
    print("done.",end="\n\n")
    
    # flatten pairs (x_t, x_{t+1}) across all trajectories
    X = trajs[:, :-1].reshape(-1, 1)
    X_next = trajs[:, 1:].reshape(-1, 1)

    # create dataset for drift regression
    X_tensor = torch.from_numpy(X).float()
    Xn_tensor = torch.from_numpy(X_next).float()
    dataset = TensorDataset(X_tensor, Xn_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = EnergyLandscapeSDE(hidden=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 200
    alpha_energy = 1.0  # weight for equilibrium histogram loss

    # compute flattened empirical distribution once (all states)
    data_flat = trajs.flatten()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loader_len = len(loader)
        for idx, (xb, xb_next) in enumerate(loader):
            xb = xb.to(device)
            xb_next = xb_next.to(device)

            # empirical drift estimate (finite difference)
            empirical_drift = (xb_next - xb) / dt

            model_drift = model.drift(xb)

            # loss_drift = F.mse_loss(model_drift, empirical_drift)
            loss_drift = torch.tensor([0])
            
            loss_energy = equilibrium_distribution_loss(model, data_flat, device)

            loss = loss_energy
            
            # print("=== Gradient Flow Check ===")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         print(f"{name}: grad_norm = {grad_norm:.6f}")
            #     else:
            #         print(f"{name}: ❌ NO GRADIENT!")
            
            # # 关键：检查potential层是否有梯度
            # potential_params = [p for n, p in model.named_parameters() 
            #                 if 'potential' in n]
            # if potential_params:
            #     has_grad = all(p.grad is not None for p in potential_params)
            #     print(f"\nPotential parameters have gradients: {has_grad}")

            
            print(f'[{idx}/{loader_len}] (drift={loss_drift.item():.3e}, energy={loss_energy.item():.3e})', end='\r')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_energy < 1e-4:
                break
            
            total_loss += float(loss.item())
            
        avg = total_loss / len(loader)
        print(f'Epoch {epoch}/{epochs}  Loss: {avg:.3e}  (drift={loss_drift.item():.3e}, energy={loss_energy.item():.3e})')

        # evaluation
        print('\nExtracting learned structure...')
        info = extract_bistable_structure(model, device)
        print('Stable states (learned minima):', info['stable_states'][:5])
        print('Barrier locations (learned maxima):', info['barrier_locations'][:5])
        print('Barrier height:', info['barrier_height'])

        # plot potential and empirical histogram
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(info['x_fine'], info['Uvals'])
        ax1.set_title('Learned potential U(x)')

        ax2 = plt.subplot(1, 2, 2)
        ax2.hist(data_flat, bins=100, density=True, alpha=0.6, color='C1')
        # plot model equilibrium density
        xs = torch.from_numpy(info['x_fine']).float().unsqueeze(-1).to(device)
        with torch.no_grad():
            Uvals_t = model.potential(xs).squeeze().cpu().numpy()
            sigma_est = float(torch.exp(model.log_sigma).item())
            p_model = np.exp(-2.0 * Uvals_t / (sigma_est**2 + 1e-12))
            p_model = p_model / (p_model.sum() * (info['x_fine'][1] - info['x_fine'][0]))
        ax2.plot(info['x_fine'], p_model, '-k', lw=2, label='model eq')
        ax2.set_title('Empirical histogram vs model equilibrium')
        ax2.legend()
        plt.tight_layout()
        plt.show()

        # save model
        torch.save(model.state_dict(), 'energy_sde_model.pth')
        print('Model saved to energy_sde_model.pth')
            
        if loss_energy < 1e-4:
            break


if __name__ == '__main__':
    main()
