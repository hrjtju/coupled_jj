"""
Neural SDE for Ornstein-Uhlenbeck (OU) Process with Random Pointwise KL Divergence

This script implements a Neural Stochastic Differential Equation (Neural SDE) model
for learning the Ornstein-Uhlenbeck process. It includes various KL divergence-based
loss functions, including the random pointwise KL divergence for better distributional
matching.

The OU process is defined by the SDE:
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t

where:
    - theta: mean reversion rate
    - mu: long-term mean
    - sigma: volatility
    - W_t: Wiener process (Brownian motion)

Reference:
    - Neural SDE paper: https://arxiv.org/pdf/2402.14989
"""

import os
import random
from typing import Any, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchcde
import torchsde
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Utility Functions
# =============================================================================

def seed_everything(seed: Any) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The seed value to use for all random number generators.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device: The device to use for tensor operations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Generation: Ornstein-Uhlenbeck Process
# =============================================================================

def simulate_ou_process(
    T: float, 
    N: int, 
    theta: float, 
    mu: float, 
    sigma: float, 
    X0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a 1D Ornstein-Uhlenbeck process using Euler-Maruyama discretization.
    
    The OU process follows the SDE:
        dX = theta * (mu - X) * dt + sigma * dW
    
    Args:
        T (float): Total simulation time.
        N (int): Number of time steps.
        theta (float): Rate of mean reversion.
        mu (float): Long-term mean value.
        sigma (float): Volatility parameter.
        X0 (float): Initial value of the process.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Time array and simulated OU process values.
    """
    dt = T / N
    t = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = X0

    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW

    return t, X


def generate_ou_dataset(
    num_samples: int, 
    T: float, 
    N: int, 
    theta: float, 
    mu: float, 
    sigma: float, 
    X0: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multiple samples of the OU process and compute CDE coefficients.
    
    Args:
        num_samples (int): Number of trajectories to generate.
        T (float): Total time.
        N (int): Number of time steps.
        theta (float): Rate of mean reversion.
        mu (float): Long-term mean.
        sigma (float): Volatility.
        X0 (float): Initial value.
    
    Returns:
        Tuple containing:
            - total_data (torch.Tensor): Tensor of shape [Batch size, Length, Dimension].
            - coeffs (torch.Tensor): Coefficients for CDE interpolation.
            - times (torch.Tensor): Time tensor.
    """
    data_list = []
    for _ in range(num_samples):
        t, X = simulate_ou_process(T, N, theta, mu, sigma, X0)
        data_list.append([t, X])

    # Convert to tensor and permute to [Batch size, Length, Dimension]
    total_data = torch.Tensor(np.array(data_list))
    total_data = total_data.permute(0, 2, 1)

    max_len = total_data.shape[1]
    times = torch.linspace(0, 1, max_len)
    
    # Compute coefficients of cubic Hermite spline interpolation
    # as discrete sequences are treated as continuous paths
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(total_data, times)

    return total_data, coeffs, times


# =============================================================================
# Dataset and DataLoader
# =============================================================================

class OUDataset(Dataset):
    """
    PyTorch Dataset for OU process data with precomputed CDE coefficients.
    """
    
    def __init__(self, data: torch.Tensor, coeffs: torch.Tensor):
        """
        Initialize the dataset.
        
        Args:
            data (torch.Tensor): The trajectory data.
            coeffs (torch.Tensor): The CDE interpolation coefficients.
        """
        self.data = data
        self.coeffs = coeffs

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.
        
        Args:
            idx (int): The index of the sample.
        
        Returns:
            Tuple containing the data sample and its coefficients.
        """
        return (
            self.data[idx, ...],
            self.coeffs[idx, ...],
        )


def split_data(
    data: torch.Tensor, 
    coeffs: torch.Tensor, 
    train_ratio: float = 0.8
) -> Tuple[torch.Tensor, ...]:
    """
    Split data into training and testing sets.
    
    Args:
        data (torch.Tensor): The trajectory data.
        coeffs (torch.Tensor): The CDE coefficients.
        train_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
    
    Returns:
        Tuple containing train_data, train_coeffs, test_data, test_coeffs.
    """
    total_size = len(data)
    train_size = int(total_size * train_ratio)

    train_idx = np.random.choice(range(total_size), train_size, replace=False)
    test_idx = np.array([i for i in range(total_size) if i not in train_idx])

    train_data = data[train_idx, ...]
    test_data = data[test_idx, ...]
    train_coeffs = coeffs[train_idx, ...]
    test_coeffs = coeffs[test_idx, ...]

    return train_data, train_coeffs, test_data, test_coeffs


def create_data_loaders(
    train_data: torch.Tensor, 
    train_coeffs: torch.Tensor,
    test_data: torch.Tensor, 
    test_coeffs: torch.Tensor, 
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.
    
    Args:
        train_data (torch.Tensor): Training trajectory data.
        train_coeffs (torch.Tensor): Training CDE coefficients.
        test_data (torch.Tensor): Testing trajectory data.
        test_coeffs (torch.Tensor): Testing CDE coefficients.
        batch_size (int, optional): Batch size for data loading. Defaults to 16.
    
    Returns:
        Tuple containing the training and testing DataLoaders.
    """
    train_dataset = OUDataset(train_data, train_coeffs)
    test_dataset = OUDataset(test_data, test_coeffs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# Neural Network Components
# =============================================================================

class LipSwish(nn.Module):
    """
    LipSwish activation function: 0.909 * silu(x)
    
    This is a Lipschitz-continuous approximation of the Swish activation,
    which helps with stability in Neural SDE training.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LipSwish activation."""
        return 0.909 * torch.nn.functional.silu(x)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable activation and optional Tanh output.
    """
    
    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        hidden_dim: int, 
        num_layers: int, 
        tanh: bool = False, 
        activation: str = 'lipswish'
    ):
        """
        Initialize the MLP.
        
        Args:
            in_size (int): Input dimension.
            out_size (int): Output dimension.
            hidden_dim (int): Hidden layer dimension.
            num_layers (int): Number of hidden layers.
            tanh (bool, optional): Whether to apply Tanh at output. Defaults to False.
            activation (str, optional): Activation function type ('lipswish' or 'relu').
        """
        super().__init__()

        if activation == 'lipswish':
            activation_fn = LipSwish()
        else:
            activation_fn = nn.ReLU()

        model = [nn.Linear(in_size, hidden_dim), activation_fn]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(activation_fn)
        model.append(nn.Linear(hidden_dim, out_size))
        if tanh:
            model.append(nn.Tanh())
        self._model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self._model(x)


# =============================================================================
# Neural SDE Model
# =============================================================================

class NeuralSDEFunc(nn.Module):
    """
    Neural SDE function defining the drift (f) and diffusion (g) terms.
    
    The SDE is defined as:
        dz(t) = f(t, z(t)) dt + g(t, z(t)) dW_t
    
    where f is the drift term and g is the diffusion term.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        hidden_hidden_dim: int, 
        num_layers: int, 
        activation: str = 'lipswish'
    ):
        """
        Initialize the Neural SDE function.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden state dimension.
            hidden_hidden_dim (int): Dimension for inner MLP layers.
            num_layers (int): Number of layers in inner MLPs.
            activation (str, optional): Activation function type. Defaults to 'lipswish'.
        """
        super(NeuralSDEFunc, self).__init__()
        self.sde_type = "ito"
        self.noise_type = "diagonal"  # Can also be "scalar"

        # Drift network (f)
        self.linear_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.f_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Diffusion network (g)
        self.noise_in = nn.Linear(hidden_dim + 1, hidden_dim)
        self.g_net = MLP(hidden_dim, hidden_dim, hidden_hidden_dim, num_layers, activation=activation)

    def set_X(self, coeffs: torch.Tensor, times: torch.Tensor) -> None:
        """
        Set the control path for the SDE using cubic spline interpolation.
        
        Args:
            coeffs (torch.Tensor): CDE interpolation coefficients.
            times (torch.Tensor): Time points.
        """
        self.coeffs = coeffs
        self.times = times
        self.X = torchcde.CubicSpline(self.coeffs, self.times)
    
    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift term of the SDE.
        
        Args:
            t (torch.Tensor): Current time.
            y (torch.Tensor): Current state.
        
        Returns:
            torch.Tensor: The drift value.
        """
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.linear_in(torch.cat((t, y), dim=-1))
        return self.f_net(yy)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion term of the SDE.
        
        Args:
            t (torch.Tensor): Current time.
            y (torch.Tensor): Current state.
        
        Returns:
            torch.Tensor: The diffusion value.
        """
        if t.dim() == 0:
            t = torch.full_like(y[:, 0], fill_value=t).unsqueeze(-1)
        yy = self.noise_in(torch.cat((t, y), dim=-1))
        return self.g_net(yy)


class NDE_model(nn.Module):
    """
    Neural Differential Equation model for time series prediction.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int, 
        activation: str = 'lipswish', 
        vector_field: Any = None
    ):
        """
        Initialize the NDE model.
        
        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden state dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers in the vector field.
            activation (str, optional): Activation function. Defaults to 'lipswish'.
            vector_field (Any, optional): The vector field class to use. Defaults to NeuralSDEFunc.
        """
        super(NDE_model, self).__init__()
        if vector_field is None:
            vector_field = NeuralSDEFunc
        self.func: NeuralSDEFunc = vector_field(input_dim, hidden_dim, hidden_dim, num_layers, activation=activation)
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, coeffs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Neural SDE.
        
        Args:
            coeffs (torch.Tensor): CDE interpolation coefficients.
            times (torch.Tensor): Time points for integration.
        
        Returns:
            torch.Tensor: Decoded output at all time points.
        """
        # Set control module
        self.func.set_X(coeffs, times)

        # Initialize hidden state from control path
        y0 = self.func.X.evaluate(times[0])
        y0 = self.initial(y0)

        # Integrate the SDE
        z = torchsde.sdeint(
            sde=self.func,
            y0=y0,
            ts=times,
            dt=0.05,
            method='euler'
        )
        
        # Permute and decode
        z = z.permute(1, 0, 2)
        return self.decoder(z)


# =============================================================================
# KL Divergence Loss Functions
# =============================================================================

class SoftHistogramKL(nn.Module):
    """
    Soft Histogram-based KL Divergence Loss.
    
    This loss computes KL divergence between the distributions of predicted
    and true values at specific time points using soft binning with Gaussian
    kernels for differentiability.
    """
    
    def __init__(self, n_bins: int = 10, sigma: float = 0.1):
        """
        Initialize the Soft Histogram KL loss.
        
        Args:
            n_bins (int, optional): Number of histogram bins. Defaults to 10.
            sigma (float, optional): Bandwidth for soft binning. Larger values
                produce smoother histograms. Defaults to 0.1.
        """
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.kl = nn.KLDivLoss(reduction='batchmean')
        
    def forward(
        self, 
        pred: torch.Tensor, 
        true: torch.Tensor, 
        ks: List[int]
    ) -> torch.Tensor:
        """
        Compute the soft histogram KL divergence at specified time points.
        
        Args:
            pred (torch.Tensor): Predicted values [batch_size, seq_len].
            true (torch.Tensor): True values [batch_size, seq_len].
            ks (List[int]): List of time step indices to evaluate KL divergence.
        
        Returns:
            torch.Tensor: The averaged KL divergence loss.
        """
        loss = torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        for k in ks:
            pred_k = pred[:, k]  # [batch_size]
            true_k = true[:, k]  # [batch_size]
            
            # Dynamically determine bin range covering both pred and true
            vmin = min(pred_k.min().item(), true_k.min().item()) - 0.1
            vmax = max(pred_k.max().item(), true_k.max().item()) + 0.1
            bins = torch.linspace(vmin, vmax, self.n_bins, device=pred.device)
            
            # Soft binning (differentiable): use Gaussian kernel to compute
            # weight of each sample belonging to each bin
            dist_p = torch.abs(pred_k.unsqueeze(1) - bins.unsqueeze(0))  # [batch, n_bins]
            dist_t = torch.abs(true_k.unsqueeze(1) - bins.unsqueeze(0))
            
            # Gaussian soft binning
            weights_p = torch.exp(-dist_p.pow(2) / (2 * self.sigma**2))
            weights_t = torch.exp(-dist_t.pow(2) / (2 * self.sigma**2))
            
            # Normalize to probability distributions (sum over batch to get bin frequencies)
            hist_p = weights_p.sum(dim=0)  # [n_bins]
            hist_t = weights_t.sum(dim=0)  # [n_bins]
            
            # Add smoothing to avoid log(0) and normalize
            prob_p = (hist_p + 1e-6) / (hist_p.sum() + 1e-6 * self.n_bins)
            prob_t = (hist_t + 1e-6) / (hist_t.sum() + 1e-6 * self.n_bins)
            
            # KL divergence: input must be log_prob, target is prob
            loss = loss + self.kl(prob_p.log(), prob_t) * max(1, k**0.5)
        
        return loss / len(ks)


def kl_timeslice_gaussian(
    pred: torch.Tensor, 
    true: torch.Tensor, 
    k: int
) -> torch.Tensor:
    """
    Compute KL divergence at a specific time slice assuming Gaussian distributions.
    
    This computes KL(N(mean_p, var_p) || N(mean_t, var_t)) at time index k.
    
    Args:
        pred (torch.Tensor): Predicted trajectories [n_samples, seq_len].
        true (torch.Tensor): True trajectories [n_samples, seq_len].
        k (int): Time step index (vertical slice position).
    
    Returns:
        torch.Tensor: The KL divergence value.
    """
    # Vertical slice: extract values at time k
    pred_k = pred[:, k]
    true_k = true[:, k]
    
    # Compute statistics (unbiased estimates)
    mu_p, var_p = pred_k.mean(), pred_k.var(unbiased=True)
    mu_t, var_t = true_k.mean(), true_k.var(unbiased=True)
    
    # Numerical stability: clamp variance to avoid division by zero
    var_t = torch.clamp(var_t, min=1e-6)
    var_p = torch.clamp(var_p, min=1e-6)
    
    # KL(N(mean_p, var_p) || N(mean_t, var_t))
    kl = 0.5 * (
        var_p/var_t + 
        (mu_p - mu_t).pow(2)/var_t - 
        1 + 
        torch.log(var_t/var_p)
    )
    
    return kl


def pathwise_kl_loss(
    pred_trajectories: torch.Tensor, 
    true_trajectories: torch.Tensor, 
    batch_size: int
) -> torch.Tensor:
    """
    Compute pathwise KL loss using Energy Distance between trajectory distributions.
    
    This measures the distributional difference between predicted and true
    trajectory ensembles without assuming Gaussianity.
    
    Args:
        pred_trajectories (torch.Tensor): Predicted trajectories [n_samples, batch, T].
        true_trajectories (torch.Tensor): True trajectories [batch, T].
        batch_size (int): Batch size.
    
    Returns:
        torch.Tensor: The averaged energy distance loss.
    """
    # Flatten paths to vectors [n_samples, batch, T]
    pred_flat = pred_trajectories.squeeze(-1)
    
    def energy_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute energy distance between two sets of paths.
        
        Args:
            X (torch.Tensor): First set of paths [n, d].
            Y (torch.Tensor): Second set of paths [m, d].
        
        Returns:
            torch.Tensor: The energy distance.
        """
        n, m = X.size(0), Y.size(0)
        XX = torch.pdist(X, p=2).pow(2).mean()
        YY = torch.pdist(Y, p=2).pow(2).mean()
        XY = torch.cdist(X, Y, p=2).pow(2).mean()
        return 2*XY - XX - YY
    
    total_ed = 0
    for b in range(batch_size):
        total_ed += energy_distance(pred_flat[:, b], true_trajectories[b:b+1, :])
    
    return total_ed / batch_size


# =============================================================================
# Random Pointwise KL Divergence (The Key Feature)
# =============================================================================

def random_pointwise_kl_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    kl_div_module: SoftHistogramKL,
    num_points: Optional[int] = None
) -> torch.Tensor:
    """
    Compute KL divergence at randomly selected time points.
    
    This is the "random pointwise KL divergence" that randomly samples
    time points to evaluate the distributional match, providing a
    computationally efficient way to enforce distributional consistency
    across the trajectory.
    
    Args:
        pred (torch.Tensor): Predicted values [batch_size, seq_len].
        true (torch.Tensor): True values [batch_size, seq_len].
        kl_div_module (SoftHistogramKL): The KL divergence computation module.
        num_points (int, optional): Number of random points to sample.
            Defaults to sqrt(seq_len).
    
    Returns:
        torch.Tensor: The computed KL divergence loss.
    """
    seq_len = pred.shape[1]
    if num_points is None:
        # Default: sample sqrt(seq_len) points, at least 2
        num_points = int(max(seq_len**0.5, 2))
    
    # Randomly select time points
    random_indices = random.choices(list(range(seq_len)), k=num_points)
    
    return kl_div_module(pred, true, random_indices)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    use_kl_loss: bool = True,
    kl_weight: float = 1.0
) -> None:
    """
    Train the Neural SDE model.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Testing data loader.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (torch.device): Device to train on.
        use_kl_loss (bool, optional): Whether to use KL divergence loss. Defaults to True.
        kl_weight (float, optional): Weight for KL loss. Defaults to 1.0.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    kl_div = SoftHistogramKL()
    
    num_samples = 5  # For visualization

    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)

            optimizer.zero_grad()
            true = batch[0][:, :, 1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            
            loss_mse = criterion(pred, true)
            
            if use_kl_loss:
                # Use random pointwise KL divergence
                loss_kl = random_pointwise_kl_loss(pred, true, kl_div)
                loss = loss_mse + kl_weight * loss_kl
                print(f"loss_mse={loss_mse:.4f}, loss_kl={loss_kl:.4f}", end='\r')
            else:
                loss = loss_mse
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            optimizer.step()

            total_loss += loss.item()

        if epoch % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'\nEpoch {epoch}, Loss: {avg_loss}')
            evaluate_model(model, test_loader, device, kl_div, num_samples)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    kl_div: SoftHistogramKL,
    num_samples: int = 5
) -> float:
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to evaluate on.
        kl_div (SoftHistogramKL): KL divergence module for loss computation.
        num_samples (int, optional): Number of samples to visualize. Defaults to 5.
    
    Returns:
        float: The average test loss.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_trues = []
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            coeffs = batch[1].to(device)
            times = torch.linspace(0, 1, batch[0].shape[1]).to(device)
    
            true = batch[0][:, :, 1].to(device)
            pred = model(coeffs, times).squeeze(-1)
            
            loss_mse = criterion(pred, true)
            loss_kl = kl_div(pred, true, [0, 5, 7, 10, 13, 15])
            loss = loss_mse + loss_kl
            total_loss += loss.item()
    
            all_preds.append(pred.cpu())
            all_trues.append(true.cpu())
    
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')
    
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    # Visualization
    visualize_predictions(all_trues, all_preds, num_samples)
    
    return avg_loss


def visualize_predictions(
    all_trues: torch.Tensor,
    all_preds: torch.Tensor,
    num_samples: int = 5
) -> None:
    """
    Visualize model predictions against true values.
    
    Args:
        all_trues (torch.Tensor): All true trajectories.
        all_preds (torch.Tensor): All predicted trajectories.
        num_samples (int, optional): Number of samples to plot. Defaults to 5.
    """
    plt.figure(figsize=(8, 4))
    for i in range(num_samples * 10):
        plt.plot(all_trues[i].numpy(), color='r', alpha=0.1)
        plt.plot(all_preds[i].numpy(), color='b', alpha=0.1)
    
    plt.plot(all_trues.mean(0).numpy(), color='r', label="true values")
    plt.plot(all_preds.mean(0).numpy(), color='b', label="predictions")
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.ylim(-0.75, 1.25)
    plt.title('Model Predictions vs True Values')
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Configuration
    config = {
        'num_samples': 2000,
        'T': 20.0,
        'N': 50,
        'theta': 0.2,
        'mu': 0.0,
        'sigma': 0.1,
        'X0': 1.0,
        'train_ratio': 0.8,
        'batch_size': 64,
        'seed': 42,
    }
    
    # Model hyperparameters
    model_config = {
        'input_dim': 2,
        'output_dim': 1,
        'hidden_dim': 32,
        'num_layers': 1,
    }
    
    # Training hyperparameters
    num_epochs = 100
    lr = 1e-3

    # Set device and seed
    device = get_device()
    seed_everything(config['seed'])

    # Generate data
    print("Generating OU process data...")
    total_data, coeffs, times = generate_ou_dataset(
        config['num_samples'], 
        config['T'], 
        config['N'], 
        config['theta'], 
        config['mu'], 
        config['sigma'], 
        config['X0']
    )
    print(f"Data shape: {total_data.shape}, Coeffs shape: {coeffs.shape}, Times shape: {times.shape}")

    # Split data
    train_data, train_coeffs, test_data, test_coeffs = split_data(
        total_data, 
        coeffs, 
        config['train_ratio']
    )

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_data, 
        train_coeffs, 
        test_data, 
        test_coeffs, 
        config['batch_size']
    )

    # Initialize model
    print("Initializing Neural SDE model...")
    model = NDE_model(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_layers=model_config['num_layers'],
        vector_field=NeuralSDEFunc
    ).to(device)

    # Train model
    print("Starting training...")
    train_model(
        model, 
        train_loader, 
        test_loader, 
        num_epochs, 
        lr, 
        device,
        use_kl_loss=True,
        kl_weight=1.0
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
