"""
# Generates Data Trajectories Under Different Sets of Parameters.

d phi_1 = v_1 dt
d v1 = [i_1 - beta_1 * v_1 - sin(phi_1) + kappa_1 (phi_2 - phi_1)] dt + sigma_1 dW_1
d phi_2 = v_2 dt
d v2 = [i_2 - beta_2 * v_2 - sin(phi_2) + kappa_2 (phi_1 - phi_2)] dt + sigma_2 dW_2

The parameters to-be-determined are:
    i_1, i_2, beta_1, beta_2, kappa_1, kappa_2, sigma_1, sigma_2

The possible range of each parameter is:
    i_1, i_2: [0.01, 2]
    beta_1, beta_2: [0.01, 2]
    kappa_1, kappa_2: [0.01, 2]
    sigma_1, sigma_2: [0.01, 0.5]

The time range is fixed to be [0, 100], and dt is fixed to be 0.05.

For fixed M, M random trajectories are sampled w.r.t. the same set of parameters. Namely for each parameter theta, M random trajectories are sampled, and forms a couple (theta, {trajectories}).
"""

import os
import sys
import random
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Optional, List, Union
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import torchsde

from utils import seed_everything

class PhysicalParamsConfig:
    """
    物理参数的采样范围配置。
    按照 PENN 论文的权重设计：损失权重与参数范围成反比
    """

    # 8 个物理参数的采样范围 (min, max)
    PARAM_RANGES = {
        'beta1': (0.05, 0.5),    # 阻尼系数
        'beta2': (0.05, 0.5),    # 阻尼系数
        'i1': (0.3, 1.5),        # 归一化偏置电流
        'i2': (0.3, 1.5),        # 归一化偏置电流
        'kappa1': (0.0, 0.15),   # 耦合强度
        'kappa2': (0.0, 0.15),   # 耦合强度
        'sigma1': (0.01, 0.05),  # 噪声强度（避免零噪声）
        'sigma2': (0.01, 0.05),  # 噪声强度
    }

    @classmethod
    def sample_params(cls, n: int = 1, device: str = 'cpu') -> Tuple[torch.Tensor, List[Dict]]:
        """
        Sample `n` different sets of parameters
        """
        params_dicts = []
        params_list = []

        for _ in range(n):
            params_dict = {}
            for name, (min_val, max_val) in cls.PARAM_RANGES.items():
                params_dict[name] = random.uniform(min_val, max_val)
            params_dicts.append(params_dict)

            params_list.append([
                params_dict['beta1'], params_dict['beta2'],
                params_dict['i1'], params_dict['i2'],
                params_dict['kappa1'], params_dict['kappa2'],
                params_dict['sigma1'], params_dict['sigma2'],
            ])

        params_tensor = torch.tensor(params_list, dtype=torch.float64, device=device)
        return params_tensor, params_dicts


class JosephsonJunctionSDE(nn.Module):
    """
    Coupled Josephson Junction SDE Simulator. 
    
    d phi_1 = v_1 dt
    d v1 = [i_1 - beta_1 * v_1 - sin(phi_1) + kappa_1 (phi_2 - phi_1)] dt + sigma_1 dW_1
    d phi_2 = v_2 dt
    d v2 = [i_2 - beta_2 * v_2 - sin(phi_2) + kappa_2 (phi_1 - phi_2)] dt + sigma_2 dW_2
    """

    def __init__(self, params: torch.Tensor):
        """
        beta_1, beta_2, i_1, i_2, kappa_1, kappa_2, sigma_1, sigma_2
        """
        super().__init__()

        self.register_buffer('params', params.clone())
        self.sde_type = "ito"
        self.noise_type = "diagonal"

    @property
    def beta1(self):
        return self.params[0]

    @property
    def beta2(self):
        return self.params[1]

    @property
    def i1(self):
        return self.params[2]

    @property
    def i2(self):
        return self.params[3]

    @property
    def kappa1(self):
        return self.params[4]

    @property
    def kappa2(self):
        return self.params[5]

    @property
    def sigma1(self):
        return self.params[6]

    @property
    def sigma2(self):
        return self.params[7]

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift term of SDE
        """
        phi1, v1, phi2, v2 = y[..., 0] % (torch.pi * 2), y[..., 1], y[..., 2] % (torch.pi * 2), y[..., 3]

        d_phi1 = v1
        dv1 = (self.i1 - self.beta1 * v1 - torch.sin(phi1) + self.kappa1 * (phi2 - phi1))
        d_phi2 = v2
        dv2 = (self.i2 - self.beta2 * v2 - torch.sin(phi2) + self.kappa2 * (phi1 - phi2))

        return torch.stack([d_phi1, dv1, d_phi2, dv2], dim=-1)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion Term"""
        batch_size = y.shape[0]
        device = y.device

        noise = torch.zeros(batch_size, 4, device=device)
        noise[..., 1] = self.sigma1
        noise[..., 3] = self.sigma2

        return noise

    def simulate(self, T: float, dt: float, initial_state: torch.Tensor,
                 n_samples: int = 1, method: str = 'euler') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate SDE Trajectory using `torchsde`
        """
        device = initial_state.device if torch.is_tensor(initial_state) else torch.device('cpu')
        times = torch.linspace(0, T, int(T / dt), device=device)

        if initial_state.dim() == 1:
            y0 = initial_state.unsqueeze(0).expand(n_samples, -1)
        else:
            y0 = initial_state

        with torch.no_grad():
            trajectories = torchsde.sdeint(
                sde=self,
                y0=y0,
                ts=times,
                dt=dt,
                method=method
            )

        trajectories = trajectories.permute(1, 0, 2)

        return trajectories

def _simulate_single(args):
    """
    Worker function for generating a single trajectory sample in parallel.
    Must be defined at the top level to be picklable by multiprocessing.
    """
    i, params, n_trajs_per_param_set, T, dt, device, seed = args
    
    if seed is not None:
        seed_everything(seed + i)
    
    sde = JosephsonJunctionSDE(params).to(device)
    initial_state = torch.zeros(4, device=device)
    
    trajectories = sde.simulate(
        T=T,
        dt=dt,
        initial_state=initial_state,
        n_samples=n_trajs_per_param_set,
        method='euler'
    )
    
    ys = [[params[0].item(), params[1].item(), params[2].item(), params[3].item(),
          params[4].item(), params[5].item(), params[6].item(), params[7].item()] for _ in range(n_trajs_per_param_set)]
    
    return trajectories, ys


def generate_trajectory_data(n_samples: int, n_trajs_per_param_set: int = 64, seed: Optional[int] = None, 
                             device: str = 'cpu', n_workers: Optional[int] = None) -> Dict:
    """
    生成轨迹数据（PENN 原始格式）。

    返回格式：[xs, T_list, ys]
    - xs: 轨迹列表，每个元素形状 [N, 4]
    - ys: 参数列表
    """
    if seed is not None:
        seed_everything(seed)

    print(f"Generating {n_samples} trajectory samples...")

    params_tensor, params_dicts = PhysicalParamsConfig.sample_params(n_samples, device='cpu')

    time_start = time.time()

    # Prepare arguments for parallel workers
    args_list = [
        (i, params_tensor[i].to(device), n_trajs_per_param_set, 100, 0.05, device, seed)
        for i in range(n_samples)
    ]

    # Parallelize CPU-bound SDE simulations across multiple processes.
    # On macOS the default multiprocessing start method is 'spawn', which is safe for PyTorch.
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    from tqdm.contrib.concurrent import process_map
    results = process_map(
        _simulate_single,
        args_list,
        max_workers=n_workers,
        chunksize=1,
        desc="Generating trajectories"
    )

    xs = [r[0] for r in results]
    ys = [r[1] for r in results]

    time_end = time.time()
    print(f"Data generation completed in {time_end - time_start:.1f}s. Total samples: {len(xs)}")
    return {'X': xs, 'Y': ys}

def save_data_npz(data: Dict, filename: str):
    """
    Save data to numpy file
    """
    np.savez(filename, **data)
    
    # try to read data
    data_loaded = np.load(filename)
    print(f"{data_loaded.keys()=}")
    print(f"{data_loaded['X'].shape=}")
    print(f"{data_loaded['Y'].shape=}")
    print(f"{data_loaded['X'][0].shape=}")
    print(f"{data_loaded['Y'][0].shape=}")

if __name__ == "__main__":
    data = generate_trajectory_data(n_samples=100, n_trajs_per_param_set=64, seed=42)
    save_data_npz(data, 'trajectory_data.npz')
