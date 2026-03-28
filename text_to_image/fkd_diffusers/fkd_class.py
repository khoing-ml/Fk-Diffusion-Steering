"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple, Union
import logging


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"
    EVOLUTION = "evolution"


def _to_numpy_1d(values: Union[torch.Tensor, np.ndarray, list, tuple]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float().numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def evolution_steering_binary_rewards(
    *,
    rewards: Union[torch.Tensor, np.ndarray, list, tuple],
    threshold_fn: Optional[Callable[[np.ndarray], float]] = None,
    threshold: Optional[float] = None,
) -> list[int]:
    """
    Convert scalar rewards into binary good/bad labels for evolution steering.

    A sample is labeled good (1) when its reward is strictly above the threshold.
    When neither `threshold` nor `threshold_fn` is provided, the median reward is
    used as the default split point.
    """
    rewards_np = _to_numpy_1d(rewards)
    if rewards_np.size == 0:
        return []

    if threshold is None:
        threshold_fn = threshold_fn or (lambda values: np.percentile(values, 50))
        threshold = float(threshold_fn(rewards_np))

    return [int(reward > threshold) for reward in rewards_np]


def _safe_resampling_weights(
    rewards: Union[torch.Tensor, np.ndarray, list, tuple],
    *,
    device: torch.device,
) -> torch.Tensor:
    weights = torch.as_tensor(rewards, dtype=torch.float32, device=device).flatten()
    if weights.numel() == 0:
        raise ValueError("Cannot resample from an empty particle population.")

    if torch.all(weights <= 0):
        return torch.ones_like(weights) / weights.numel()

    weights = torch.clamp(weights, min=0.0)
    return weights / weights.sum()


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF kernel matrix between two sets of points.
    
    Args:
        x: Points of shape (n, d)
        y: Points of shape (m, d)
        sigma: Bandwidth parameter for RBF kernel
    
    Returns:
        Kernel matrix of shape (n, m)
    """
    # Compute pairwise squared distances
    sq_dist = torch.cdist(x, y, p=2.0) ** 2
    return torch.exp(-sq_dist / (2 * sigma ** 2))


def _median_pairwise_distance(x: torch.Tensor) -> float:
    """
    Compute median pairwise distance for automatic bandwidth selection.
    Uses Silverman's rule for bandwidth selection.
    
    Args:
        x: Points of shape (n, d)
    
    Returns:
        Bandwidth parameter sigma
    """
    # Compute pairwise distances
    sq_dist = torch.cdist(x, x, p=2.0) ** 2
    # Get upper triangular part (excluding diagonal)
    distances = torch.triu(sq_dist, diagonal=1)
    # Remove zero entries and compute median
    distances_flat = distances[distances > 0]
    if distances_flat.numel() == 0:
        return 1.0
    median_dist = torch.sqrt(torch.median(distances_flat))
    return float(median_dist)


def compute_svgd_vector_field(
    particles: torch.Tensor,
    binary_rewards: list[int],
    *,
    sigma: Optional[float] = None,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Compute the Stein Variational Gradient Descent (SVGD) vector field
    that steers particles from bad regions (y=0) to good regions (y=1).
    
    The vector field is computed as:
    ϕ*(x) = (1/n) Σ_j [k(x_j, x) ∇_x_j log p̂(x_j, y=1) + ∇_x_j k(x_j, x)]
    
    However, since we don't have explicit access to gradients of log p(x, y=1),
    we approximate by using kernel-weighted movement from bad to good particles.
    
    Args:
        particles: Particle population of shape (num_particles, *spatial_dims)
        binary_rewards: Binary labels (0=bad, 1=good) for each particle
        sigma: Bandwidth for RBF kernel. If None, uses median pairwise distance.
        device: Device for computation
    
    Returns:
        Vector field (velocities) of shape (num_particles, *spatial_dims)
    """
    num_particles = particles.shape[0]
    particle_shape = particles.shape
    
    # Flatten particles for distance computation
    particles_flat = particles.reshape(num_particles, -1).to(device)
    
    # Separate good and bad particle indices
    bad_indices = torch.tensor([i for i, y in enumerate(binary_rewards) if y == 0], device=device)
    good_indices = torch.tensor([i for i, y in enumerate(binary_rewards) if y == 1], device=device)
    
    if good_indices.numel() == 0:
        # No good particles, return zero vector field
        return torch.zeros_like(particles)
    
    if sigma is None:
        sigma = _median_pairwise_distance(particles_flat)
    
    # Compute vector field
    vector_field = torch.zeros_like(particles_flat, device=device)
    
    if bad_indices.numel() > 0:
        # Get good and bad particles
        good_particles = particles_flat[good_indices]  # (n_good, d)
        bad_particles = particles_flat[bad_indices]    # (n_bad, d)
        
        # Compute kernel between bad and good particles
        # Shape: (n_bad, n_good)
        k_bg = _rbf_kernel(bad_particles, good_particles, sigma=sigma)
        
        # Compute mean direction from bad particles towards good particles
        # Direction: good_j - bad_i for each pair
        # Shape: (n_bad, n_good, d)
        directions = good_particles.unsqueeze(0) - bad_particles.unsqueeze(1)  # (n_bad, n_good, d)
        
        # Weight directions by kernel and average
        # Normalize kernel weights for each bad particle
        k_weights = k_bg / (k_bg.sum(dim=1, keepdim=True) + 1e-8)  # (n_bad, n_good)
        
        # Compute weighted mean direction
        mean_direction = torch.einsum('ij,ijd->id', k_weights, directions)  # (n_bad, d)
        
        # Assign direction to bad particles
        vector_field[bad_indices] = mean_direction
    
    # Reshape back to original shape
    vector_field = vector_field.reshape(particle_shape)
    
    return vector_field


def apply_svgd_steering(
    particles: torch.Tensor,
    binary_rewards: list[int],
    step_size: float = 0.1,
    sigma: Optional[float] = None,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Apply SVGD steering to particles by moving them along the computed vector field.
    
    Args:
        particles: Particle population of shape (num_particles, *spatial_dims)
        binary_rewards: Binary labels (0=bad, 1=good) for each particle
        step_size: Step size for gradient update (default: 0.1)
        sigma: Bandwidth for RBF kernel. If None, uses median pairwise distance.
        device: Device for computation
    
    Returns:
        Steered particles of shape (num_particles, *spatial_dims)
    """
    vector_field = compute_svgd_vector_field(
        particles=particles,
        binary_rewards=binary_rewards,
        sigma=sigma,
        device=device,
    )
    
    # Normalize vector field to prevent explosions
    field_norm = torch.norm(vector_field.reshape(vector_field.shape[0], -1), dim=1, keepdim=True)
    field_norm = field_norm.reshape(-1, *([1] * (len(vector_field.shape) - 1)))
    field_norm = torch.clamp(field_norm, min=1e-8)
    normalized_field = vector_field / field_norm
    
    # Apply steering
    steered_particles = particles + step_size * normalized_field
    
    return steered_particles


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        svgd_step_size: Step size for SVGD vector field application in evolution steering (default: 0.1).
        svgd_sigma: Bandwidth for SVGD RBF kernel. If None, uses median pairwise distance.
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device('cuda'),
        svgd_step_size: float = 0.1,
        svgd_sigma: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        # SVGD parameters
        self.svgd_step_size = svgd_step_size
        self.svgd_sigma = svgd_sigma

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Includes binary scoring for evolution steering using existing reward functions.
        For evolution steering, applies SVGD to steer particles from bad to good regions.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Check if resampling is within the allowed range and conditions
        resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )

        if sampling_idx not in resampling_interval:
            return latents, None

        # Evolution steering should score the predicted clean sample x0.
        decoded_images = self.latent_to_decode_fn(x0_preds)

        # Compute rewards using the existing reward function
        rewards = self.reward_fn(decoded_images)

        if self.potential_type == PotentialType.EVOLUTION:
            # Get binary rewards for evolution steering
            binary_rewards = evolution_steering_binary_rewards(rewards=rewards)
            
            # Standard resampling based on binary rewards
            resampling_weights = _safe_resampling_weights(
                binary_rewards,
                device=latents.device,
            )
            
            resampled_indices = torch.multinomial(
                resampling_weights,
                num_samples=self.num_particles,
                replacement=True,
            )
            
            resampled_latents = latents[resampled_indices]
            resampled_images = (
                decoded_images[resampled_indices] if decoded_images is not None else None
            )
            
            # Apply SVGD steering to x0_preds to move particles from bad to good regions
            # This supplements the resampling with smooth transport
            resampled_x0_preds = x0_preds[resampled_indices]
            resampled_binary_rewards = [binary_rewards[i] for i in resampled_indices.tolist()]
            
            # Apply SVGD vector field steering
            steered_x0_preds = apply_svgd_steering(
                particles=resampled_x0_preds,
                binary_rewards=resampled_binary_rewards,
                step_size=self.svgd_step_size,
                sigma=self.svgd_sigma,
                device=latents.device,
            )
            
            return resampled_latents, steered_x0_preds
            
        else:
            resampling_weights = _safe_resampling_weights(rewards, device=latents.device)

            resampled_indices = torch.multinomial(
                resampling_weights,
                num_samples=self.num_particles,
                replacement=True,
            )

            resampled_latents = latents[resampled_indices]
            resampled_images = (
                decoded_images[resampled_indices] if decoded_images is not None else None
            )

            return resampled_latents, resampled_images


if __name__ == "__main__":

    # Demonstration of FKD resampling step
    import matplotlib.pyplot as plt
    import random

    # set seed
    random.seed(0)

    # 1x1 pixel images
    num_particles = 8
    x0s = torch.rand(num_particles, 1, 1)

    # reward darker images
    reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

    # Define the FKD steering mechanism
    fkds = FKD(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
        reward_fn=lambda x: reward_function(x),
        device=torch.device('cpu'),
    )

    # Define the sampling index
    sampling_idx = 0

    # Perform resampling
    resampled_latents, resampled_images = fkds.resample(
        sampling_idx=sampling_idx,
        latents=x0s,
        x0_preds=x0s,
    )

    plt.rc('text', usetex=True)
    fig, axs = plt.subplots(2, num_particles)

    axs[0, 0].set_title('Initial')
    axs[1, 0].set_title('Resampled')

    for i in range(num_particles):
        axs[0, i].imshow(x0s[i].detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[1, i].imshow(
            resampled_images[i].detach().numpy(), cmap='gray', vmin=0, vmax=1
        )

        axs[1, i].axis('off')
        axs[0, i].axis('off')

    out_path = 'resampled_examples.png'
    plt.savefig(out_path)
    print('Saved resampled examples to:', out_path)
