"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple, Union


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"
    EVOLUTION = "evolution"


VALID_RESAMPLE_STRATEGIES = {
    "multinomial",
    "systematic",
    "stratified",
    "residual",
    "none",
}


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


def _draw_multinomial_indices(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    return torch.multinomial(weights, num_samples=num_samples, replacement=True)


def _draw_systematic_indices(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    device = weights.device
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0

    u0 = torch.rand(1, device=device) / float(num_samples)
    positions = u0 + torch.arange(num_samples, device=device, dtype=weights.dtype) / float(num_samples)
    positions = torch.clamp(positions, max=1.0 - 1e-8)
    return torch.searchsorted(cdf, positions).to(torch.long)


def _draw_stratified_indices(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    device = weights.device
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0

    jitter = torch.rand(num_samples, device=device, dtype=weights.dtype)
    positions = (torch.arange(num_samples, device=device, dtype=weights.dtype) + jitter) / float(num_samples)
    positions = torch.clamp(positions, max=1.0 - 1e-8)
    return torch.searchsorted(cdf, positions).to(torch.long)


def _draw_residual_indices(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    device = weights.device
    expected = num_samples * weights
    deterministic = torch.floor(expected).to(torch.long)
    residual_count = int(num_samples - int(deterministic.sum().item()))

    base_indices = torch.repeat_interleave(
        torch.arange(weights.numel(), device=device, dtype=torch.long),
        deterministic,
    )

    if residual_count <= 0:
        if base_indices.numel() > num_samples:
            base_indices = base_indices[:num_samples]
        return base_indices

    residual = expected - deterministic.to(expected.dtype)
    residual_sum = residual.sum()
    if residual_sum <= 0:
        tail = _draw_systematic_indices(weights, residual_count)
    else:
        residual_weights = residual / residual_sum
        tail = _draw_systematic_indices(residual_weights, residual_count)

    out = torch.cat([base_indices, tail], dim=0)
    if out.numel() < num_samples:
        pad = _draw_systematic_indices(weights, num_samples - out.numel())
        out = torch.cat([out, pad], dim=0)
    elif out.numel() > num_samples:
        out = out[:num_samples]
    return out


def _draw_resampled_indices(
    *,
    weights: torch.Tensor,
    num_samples: int,
    strategy: str,
) -> torch.Tensor:
    if strategy == "multinomial":
        return _draw_multinomial_indices(weights, num_samples)
    if strategy == "systematic":
        return _draw_systematic_indices(weights, num_samples)
    if strategy == "stratified":
        return _draw_stratified_indices(weights, num_samples)
    if strategy == "residual":
        return _draw_residual_indices(weights, num_samples)
    raise ValueError(
        f"Unknown resample strategy '{strategy}'. "
        f"Expected one of: {sorted(VALID_RESAMPLE_STRATEGIES)}"
    )


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
    # torch.cdist on some CUDA builds does not support float16/bfloat16.
    # Compute distances in float32 for stability, then cast back.
    out_dtype = x.dtype
    x_cdist = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y_cdist = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y

    sq_dist = torch.cdist(x_cdist, y_cdist, p=2.0) ** 2
    kernel = torch.exp(-sq_dist / (2 * sigma ** 2))
    return kernel.to(dtype=out_dtype)


def _median_pairwise_distance(x: torch.Tensor) -> float:
    """
    Compute median pairwise distance for automatic bandwidth selection.
    Uses Silverman's rule for bandwidth selection.
    
    Args:
        x: Points of shape (n, d)
    
    Returns:
        Bandwidth parameter sigma
    """
    # torch.cdist on some CUDA builds does not support float16/bfloat16.
    x_cdist = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    sq_dist = torch.cdist(x_cdist, x_cdist, p=2.0) ** 2
    # Get upper triangular part (excluding diagonal)
    distances = torch.triu(sq_dist, diagonal=1)
    # Remove zero entries and compute median
    distances_flat = distances[distances > 0]
    if distances_flat.numel() == 0:
        return 1.0
    median_dist = torch.sqrt(torch.median(distances_flat))
    return float(median_dist)


def _unique_anchor_particles(anchors: torch.Tensor) -> torch.Tensor:
    """Drop exact duplicate anchors so cloned winners do not dominate the mixture score."""
    if anchors.numel() == 0 or anchors.shape[0] <= 1:
        return anchors

    flat = anchors.reshape(anchors.shape[0], -1)
    flat_unique = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
    unique_flat = torch.unique(flat_unique, dim=0)
    return unique_flat.to(device=anchors.device, dtype=anchors.dtype).reshape(
        -1, *anchors.shape[1:]
    )


def _clip_vector_field(
    vector_field: torch.Tensor,
    *,
    max_particle_norm: float = 1.0,
) -> torch.Tensor:
    """Clip overly large particle updates while preserving relative magnitudes."""
    if max_particle_norm <= 0:
        return torch.zeros_like(vector_field)

    field_flat = vector_field.reshape(vector_field.shape[0], -1)
    field_norm = torch.norm(field_flat, dim=1, keepdim=True)
    scale = torch.clamp(
        max_particle_norm / torch.clamp(field_norm, min=1e-8),
        max=1.0,
    )
    scale = scale.reshape(-1, *([1] * (len(vector_field.shape) - 1)))
    return vector_field * scale


def _score_xt_given_x0_mixture(
    *,
    xt_flat: torch.Tensor,
    x0_good_flat: torch.Tensor,
    alpha_bar_t: float,
) -> torch.Tensor: 
    """
    Compute ∇_{x_t} log q_t(x_t | y=1, c) under a uniform mixture over good x0 anchors.

    For VP-style forward diffusion,
        q(x_t | x0) = N(sqrt(alpha_bar_t) * x0, (1 - alpha_bar_t) I)
    and the mixture score is:
        ∇ log Σ_i q(x_t | x0_i) = Σ_i w_i(x_t) ∇ log q(x_t | x0_i)
    with posterior-like weights w_i.

    Args:
        xt_flat: Current particles x_t, shape (n, d).
        x0_good_flat: Good anchors x0, shape (m, d).
        alpha_bar_t: Forward cumulative alpha at timestep t.

    Returns:
        Mixture score evaluated at each x_t, shape (n, d).
    """
    
    # Clamp for numerical stability near t=0.
    alpha_bar_t = float(max(0.0, min(1.0, alpha_bar_t)))
    variance = max(1.0 - alpha_bar_t, 1e-6)
    sqrt_alpha = float(np.sqrt(alpha_bar_t))

    means = sqrt_alpha * x0_good_flat  # (m, d)
    diff = xt_flat.unsqueeze(1) - means.unsqueeze(0)  # (n, m, d) 
    

    # Unnormalized log-likelihoods for each mixture component (constant terms cancel in softmax).
    log_liks = -0.5 * (diff.pow(2).sum(dim=-1) / variance)  # (n, m)
    weights = torch.softmax(log_liks, dim=1)  # (n, m)

    # Score of each Gaussian component wrt x_t.
    component_scores = -(diff / variance)  # (n, m, d)
    return (weights.unsqueeze(-1) * component_scores).sum(dim=1)  # (n, d)


def _compute_svgd_vector_field_mixture_score(
    *,
    xt_particles: torch.Tensor,
    x0_good_anchors: torch.Tensor,
    sigma: Optional[float] = None,
    device: torch.device,
    alpha_bar_t: float,
) -> torch.Tensor:
    """
    Empirical SVGD field using a mixture-score estimator for q_t(x_t | y=1, c).

    φ(x_i) = (1/n) Σ_j [k(x_j, x_i) s(x_j) + ∇_{x_j} k(x_j, x_i)]
    where s(x_j) approximates ∇_{x_j} log q_t(x_j | y=1, c).
    """
    n = xt_particles.shape[0]
    particle_shape = xt_particles.shape
    x = xt_particles.reshape(n, -1).to(device)
    good = x0_good_anchors.reshape(x0_good_anchors.shape[0], -1).to(device)

    if good.shape[0] == 0:
        return torch.zeros_like(xt_particles)

    if sigma is None:
        sigma = _median_pairwise_distance(x)
    sigma = float(max(sigma, 1e-6))

    # Score estimate at particle locations.
    score = _score_xt_given_x0_mixture(
        xt_flat=x,
        x0_good_flat=good,
        alpha_bar_t=alpha_bar_t,
    )  # (n, d)

    # K[j, i] = k(x_j, x_i)
    K = _rbf_kernel(x, x, sigma=sigma)  # (n, n)

    # First term: Σ_j K[j, i] * score(x_j)
    first_term = K.transpose(0, 1) @ score  # (n, d)

    # Second term: Σ_j ∇_{x_j} k(x_j, x_i)
    # For RBF, ∇_{x_j}k = -k * (x_j - x_i) / sigma^2
    diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n_j, n_i, d)
    grad_j_k = -(K.unsqueeze(-1) * diff) / (sigma**2)  # (n, n, d)
    second_term = grad_j_k.sum(dim=0)  # (n_i, d)

    field = (first_term + second_term) / float(n)
    return field.reshape(particle_shape)


def _compute_svgd_vector_field_transport(
    *,
    particles: torch.Tensor,
    binary_rewards: list[int],
    sigma: Optional[float] = None,
    device: torch.device,
) -> torch.Tensor:
    """
    Score-free fallback: kernel-weighted transport from bad particles to good particles.
    """
    num_particles = particles.shape[0]
    particle_shape = particles.shape
    particles_flat = particles.reshape(num_particles, -1).to(device)

    bad_indices = torch.tensor(
        [i for i, y in enumerate(binary_rewards) if y == 0], device=device
    )
    good_indices = torch.tensor(
        [i for i, y in enumerate(binary_rewards) if y == 1], device=device
    )

    if good_indices.numel() == 0:
        return torch.zeros_like(particles)

    if sigma is None:
        sigma = _median_pairwise_distance(particles_flat)

    vector_field = torch.zeros_like(particles_flat, device=device)
    if bad_indices.numel() > 0:
        good_particles = particles_flat[good_indices]
        bad_particles = particles_flat[bad_indices]
        k_bg = _rbf_kernel(bad_particles, good_particles, sigma=sigma)
        directions = good_particles.unsqueeze(0) - bad_particles.unsqueeze(1)
        k_weights = k_bg / (k_bg.sum(dim=1, keepdim=True) + 1e-8)
        mean_direction = torch.einsum("ij,ijd->id", k_weights, directions)
        vector_field[bad_indices] = mean_direction

    return vector_field.reshape(particle_shape)


def compute_svgd_vector_field(
    particles: torch.Tensor,
    binary_rewards: list[int],
    *,
    x0_particles: Optional[torch.Tensor] = None,
    good_anchor_x0: Optional[torch.Tensor] = None,
    alpha_bar_t: Optional[float] = None,
    sigma: Optional[float] = None,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Compute the Stein Variational Gradient Descent (SVGD) vector field
    that steers particles from bad regions (y=0) to good regions (y=1).
    
    The vector field is computed as:
    ϕ*(x) = (1/n) Σ_j [k(x_j, x) ∇_x_j log p̂(x_j, y=1) + ∇_x_j k(x_j, x)]
    
        Primary path (when `x0_particles` and `alpha_bar_t` are provided):
        - approximates ∇ log q_t(x_t | y=1, c) via a mixture over good x0 anchors and
            applies the empirical SVGD formula.

        Fallback path:
        - uses kernel-weighted transport from bad to good particles when diffusion
            coefficients are unavailable.
    
    Args:
        particles: Particle population of shape (num_particles, *spatial_dims)
        binary_rewards: Binary labels (0=bad, 1=good) for each particle
        x0_particles: Predicted clean samples x0 aligned with particles, shape compatible with `particles`.
        good_anchor_x0: Optional explicit good-anchor set. Useful for evolution steering
            when the current particle population already contains resampled duplicates.
        alpha_bar_t: Forward diffusion alpha_bar at the current timestep.
        sigma: Bandwidth for RBF kernel. If None, uses median pairwise distance.
        device: Device for computation
    
    Returns:
        Vector field (velocities) of shape (num_particles, *spatial_dims)
    """
    good_indices = [i for i, y in enumerate(binary_rewards) if y == 1]

    if good_anchor_x0 is None and x0_particles is not None and len(good_indices) > 0:
        good_anchor_x0 = x0_particles[good_indices]

    if good_anchor_x0 is not None and alpha_bar_t is not None:
        good_anchor_x0 = _unique_anchor_particles(good_anchor_x0)
        return _compute_svgd_vector_field_mixture_score(
            xt_particles=particles,
            x0_good_anchors=good_anchor_x0,
            sigma=sigma,
            device=device,
            alpha_bar_t=float(alpha_bar_t),
        )

    if len(good_indices) == 0:
        return torch.zeros_like(particles)

    return _compute_svgd_vector_field_transport(
        particles=particles,
        binary_rewards=binary_rewards,
        sigma=sigma,
        device=device,
    )


def apply_svgd_steering(
    particles: torch.Tensor,
    binary_rewards: list[int],
    x0_particles: Optional[torch.Tensor] = None,
    good_anchor_x0: Optional[torch.Tensor] = None,
    alpha_bar_t: Optional[float] = None,
    step_size: float = 0.1,
    sigma: Optional[float] = None,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Apply SVGD steering to particles by moving them along the computed vector field.
    
    Args:
        particles: Particle population of shape (num_particles, *spatial_dims)
        binary_rewards: Binary labels (0=bad, 1=good) for each particle
        x0_particles: Predicted clean samples x0 used for score approximation.
        alpha_bar_t: Forward diffusion alpha_bar at the current timestep.
        step_size: Step size for gradient update (default: 0.1)
        sigma: Bandwidth for RBF kernel. If None, uses median pairwise distance.
        device: Device for computation
    
    Returns:
        Steered particles of shape (num_particles, *spatial_dims)
    """
    vector_field = compute_svgd_vector_field(
        particles=particles,
        binary_rewards=binary_rewards,
        x0_particles=x0_particles,
        good_anchor_x0=good_anchor_x0,
        alpha_bar_t=alpha_bar_t,
        sigma=sigma,
        device=device,
    )

    # Clip only oversized updates so weak fields stay weak instead of being
    # normalized to the same step size as strong fields.
    clipped_field = _clip_vector_field(vector_field, max_particle_norm=1.0)
    return particles + step_size * clipped_field


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to apply FKD updates.
        resampling_t_start: Timestep to start FKD updates.
        resampling_t_end: Timestep to stop FKD updates.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        svgd_step_size: Step size for SVGD vector field application in evolution steering (default: 0.1).
        svgd_sigma: Bandwidth for SVGD RBF kernel. If None, uses median pairwise distance.
        resample_strategy: Resampling strategy used when resampling is enabled.
            Supported: multinomial, systematic, stratified, residual, none.
        alpha_bar_fn: Optional callback mapping sampling index -> alpha_bar_t for score-based SVGD.
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
        resample_strategy: str = "multinomial",
        alpha_bar_fn: Optional[Callable[[int], Optional[float]]] = None,
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
        self.resample_strategy = str(resample_strategy).lower()
        if self.resample_strategy not in VALID_RESAMPLE_STRATEGIES:
            raise ValueError(
                f"Unknown resample strategy '{self.resample_strategy}'. "
                f"Expected one of: {sorted(VALID_RESAMPLE_STRATEGIES)}"
            )
        self.alpha_bar_fn = alpha_bar_fn

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
        Perform an FKD update when the current timestep is in the configured interval.
        For non-evolution potentials this is a particle resampling step.
        For evolution potential, optional resampling is followed by SVGD latent adjustment.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing updated latents and optionally decoded images.
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
            binary_rewards = evolution_steering_binary_rewards(rewards=rewards)
            good_indices = [i for i, y in enumerate(binary_rewards) if y == 1]
            good_anchor_x0 = (
                _unique_anchor_particles(x0_preds[good_indices])
                if good_indices
                else None
            )

            if self.resample_strategy != "none":
                resampling_weights = _safe_resampling_weights(
                    binary_rewards,
                    device=latents.device,
                )
                resampled_indices = _draw_resampled_indices(
                    weights=resampling_weights,
                    num_samples=self.num_particles,
                    strategy=self.resample_strategy,
                )
                latents = latents[resampled_indices]
                x0_preds = x0_preds[resampled_indices]
                decoded_images = (
                    decoded_images[resampled_indices]
                    if decoded_images is not None
                    else None
                )
                binary_rewards = [binary_rewards[i] for i in resampled_indices.tolist()]

            alpha_bar_t = None
            if self.alpha_bar_fn is not None:
                alpha_bar_t = self.alpha_bar_fn(sampling_idx)

            steered_latents = apply_svgd_steering(
                particles=latents,
                binary_rewards=binary_rewards,
                x0_particles=x0_preds,
                good_anchor_x0=good_anchor_x0,
                alpha_bar_t=alpha_bar_t,
                step_size=self.svgd_step_size,
                sigma=self.svgd_sigma,
                device=latents.device,
            )

            return steered_latents, decoded_images
            
        else:
            if self.resample_strategy == "none":
                return latents, decoded_images

            resampling_weights = _safe_resampling_weights(rewards, device=latents.device)
            resampled_indices = _draw_resampled_indices(
                weights=resampling_weights,
                num_samples=self.num_particles,
                strategy=self.resample_strategy,
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
