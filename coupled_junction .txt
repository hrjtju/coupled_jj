from numbers import Number
from typing import List, Sequence, Tuple, Optional
from matplotlib import axes, pyplot as plt
import numpy as np
from numba import njit

# coupled_junction.py

@njit(fastmath=True, parallel=True)
def core_step(state, linear_ode_matrix, drift_term, diffution_term, dt):
    """单步更新，纯 NumPy 计算，无 Python 对象"""

    return state + (linear_ode_matrix @ state + drift_term) * dt + diffution_term

class CoupledJunctionParams:
    def __init__(
        self,
        beta1: float,
        beta2: float,
        i1: float,
        i2: float,
        kappa1: float,
        kappa2: float,
        sigma1: float,
        sigma2: float,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.i1 = i1
        self.i2 = i2
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def readout(self) -> List[float]:
        return [
            self.beta1,
            self.beta2,
            self.i1,
            self.i2,
            self.kappa1,
            self.kappa2,
            self.sigma1,
            self.sigma2,
        ]

class CoupledJunctionInitCond:
    def __init__(
        self,
        phi1_0: float,
        v1_0: float,
        phi2_0: float,
        v2_0: float,
    ):
        self.phi1_0 = phi1_0
        self.v1_0 = v1_0
        self.phi2_0 = phi2_0
        self.v2_0 = v2_0
    
    def readout(self) -> List[float]:
        return [
            self.phi1_0,
            self.v1_0,
            self.phi2_0,
            self.v2_0,
        ]


class CoupledJunction:
    def __init__(self, params: CoupledJunctionParams, seed: Number|None=None):
        self.params = params
        self.state = None
        self.history = None
        self.seed = seed
    
    def init_state(self, init_cond: CoupledJunctionInitCond) -> None:
        if self.state is None:
            self.state = np.array(init_cond.readout(), dtype=np.float64)[:, None]
        
        self.history = None
    
    def update_history(self, i: int, t: float):
        self.history[:, i] = np.array([t, *self.state.flatten()], dtype=np.float64)
    
    def simulate_hand(self, tspan: float, dt: float = 1e-4):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        params = self.params
        
        linear_ode_matrix = np.array([
            [0, 1, 0, 0],
            [-params.kappa1, -params.beta1, params.kappa1, 0],
            [0, 0, 0, 1],
            [params.kappa2, 0, -params.kappa2, -params.beta2],
        ], dtype=np.float64)
        
        t = 0
        t_linspace = np.arange(0, tspan, dt)
        history_steps = len(t_linspace) // 100 + 1
        print(history_steps)
        
        self.history = np.empty((5, history_steps), dtype=np.float64)
        
        for i, t in enumerate(t_linspace):
            if i % 100 == 0:
                self.update_history(i//100, t)
            
            diffusion_term = np.array([[0], 
                                       [params.sigma1*np.random.normal(0, 1)], 
                                       [0], 
                                       [params.sigma2*np.random.normal(0, 1)]], dtype=np.float64) * np.sqrt(dt)
            
            drift_term = np.array([[0], 
                                   [-np.sin(self.state[0, 0]) + params.i1], 
                                   [0], 
                                   [-np.sin(self.state[2, 0]) + params.i2]], dtype=np.float64)

            self.state = core_step(self.state, 
                                   linear_ode_matrix, 
                                   drift_term, 
                                   diffusion_term, dt)
            
            print("\rProgress: {:.2f}%".format(t / tspan * 100), end="")
            
    def export_simulation(self):
        # print(self.history.shape)
        t, phi1, v1, phi2, v2 = self.history[:, :-1]
        
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=400)
        axs: List[axes.Axes] = axs.flatten()
        
        axs[0].plot(t, phi1, label='$\phi_1$')
        axs[0].plot(t, phi2, label='$\phi_2$')
        axs[0].grid(); axs[0].legend()
        axs[0].set_xlabel("Time"); axs[0].set_ylabel("Phase $\phi$")
        axs[0].set_title("Phase Evolution")

        axs[1].plot(t, v1, label='$v_1$')
        axs[1].plot(t, v2, label='$v_2$')
        axs[1].grid(); axs[1].legend()
        axs[1].set_xlabel("Time"); axs[1].set_ylabel("Velocity $v$")
        axs[1].set_title("Velocity Evolution")

        axs[2].plot(phi1, v1, label='Junction 1')
        axs[2].plot(phi2, v2, label='Junction 2')
        axs[2].grid(); axs[2].legend()
        axs[2].set_xlabel("$\phi$"); axs[2].set_ylabel("$v$")
        axs[2].set_title("Phase Space Trajectories")

        axs[3].plot(phi1, phi2)
        axs[3].grid()
        axs[3].set_xlabel("$\phi_1$"); axs[3].set_ylabel("$\phi_2$")
        axs[3].set_title("Phase Correlation")
        
        axs[4].plot(t, (phi1 - phi2))
        axs[4].grid()
        axs[4].set_xlabel("Time"); axs[4].set_ylabel("$\phi_1 - \phi_2$")
        axs[4].set_title("Phase Difference")

        # axs[5].close()
        
        # hist2d(phi1, phi2, bins=256, cmap="Blues")
        # axs[5].set_xlabel("$\phi_1$"); axs[5].set_ylabel("$\phi_2$")
        # axs[5].set_title("KDE Estimate of Joint Phase Distribution")
        
        fig.tight_layout()
        fig.savefig("coupled_junction_simulation.png", transparent=False)
        
        
        
if __name__ == "__main__":
    params = CoupledJunctionParams(
        beta1=0.1,
        beta2=0.1,
        i1=0.8,
        i2=0.8,
        kappa1=0.05,
        kappa2=0.05,
        sigma1=0.01,
        sigma2=0.01,
    )
    
    init_cond = CoupledJunctionInitCond(
        phi1_0=0.0,
        v1_0=0.0,
        phi2_0=0.0,
        v2_0=0.0,
    )
    
    cj = CoupledJunction(params)
    cj.init_state(init_cond)
    # cj.simulate_hand(tspan=100.0, dt=1e-4)
    cj.simulate_hand(tspan=100.0, dt=1e-6)
    # import cProfile
    # cProfile.run('cj.simulate_hand(tspan=100.0, dt=5e-4)', "stats")
    
    cj.export_simulation()
