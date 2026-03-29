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


def _to_tensor_1d(
    values: Union[torch.Tensor, np.ndarray, list, tuple],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(device=device, dtype=torch.float32).flatten()
    return torch.as_tensor(values, dtype=torch.float32, device=device).flatten()


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


def _dedupe_anchor_particles(
    anchors: torch.Tensor,
    scores: torch.Tensor,
    *,
    keep_highest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse exact duplicate anchors while keeping the most informative score."""
    if anchors.numel() == 0 or anchors.shape[0] <= 1:
        return anchors, scores

    flat = anchors.reshape(anchors.shape[0], -1)
    flat_unique = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
    unique_flat, inverse = torch.unique(flat_unique, dim=0, return_inverse=True)

    reduced_scores = []
    for idx in range(unique_flat.shape[0]):
        matched_scores = scores[inverse == idx]
        reduced_scores.append(
            matched_scores.max() if keep_highest else matched_scores.min()
        )

    reduced_scores = torch.stack(reduced_scores).to(device=scores.device, dtype=scores.dtype)
    unique_anchors = unique_flat.to(device=anchors.device, dtype=anchors.dtype).reshape(
        -1, *anchors.shape[1:]
    )
    return unique_anchors, reduced_scores


def _prune_anchor_particles(
    anchors: torch.Tensor,
    scores: torch.Tensor,
    *,
    max_size: int,
    keep_highest: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep a bounded, diverse archive of anchors.

    Anchors are ordered by reward quality and then greedily diversified with
    farthest-point selection so one dense mode cannot monopolize the bank.
    """
    if anchors.numel() == 0 or anchors.shape[0] <= max_size:
        return anchors, scores

    order = torch.argsort(scores, descending=keep_highest)
    anchors = anchors[order]
    scores = scores[order]

    if max_size == 1:
        return anchors[:1], scores[:1]

    flat = anchors.reshape(anchors.shape[0], -1)
    flat = flat.float() if flat.dtype in (torch.float16, torch.bfloat16) else flat
    selected = [0]
    min_dists = torch.cdist(flat[:1], flat).squeeze(0)
    score_bonus = torch.linspace(1.0, 0.0, anchors.shape[0], device=flat.device)

    while len(selected) < max_size:
        candidate_value = min_dists + 1e-3 * score_bonus
        candidate_value[selected] = -1.0
        next_idx = int(torch.argmax(candidate_value).item())
        selected.append(next_idx)
        dist_to_new = torch.cdist(flat[next_idx : next_idx + 1], flat).squeeze(0)
        min_dists = torch.minimum(min_dists, dist_to_new)

    selected_idx = torch.tensor(selected, device=anchors.device, dtype=torch.long)
    return anchors[selected_idx], scores[selected_idx]


def _update_anchor_bank(
    *,
    bank_anchors: Optional[torch.Tensor],
    bank_scores: Optional[torch.Tensor],
    new_anchors: torch.Tensor,
    new_scores: torch.Tensor,
    max_size: int,
    keep_highest: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if new_anchors.numel() == 0 or new_scores.numel() == 0:
        return bank_anchors, bank_scores

    if bank_anchors is None or bank_scores is None:
        merged_anchors = new_anchors
        merged_scores = new_scores
    else:
        merged_anchors = torch.cat([bank_anchors, new_anchors], dim=0)
        merged_scores = torch.cat([bank_scores, new_scores], dim=0)

    merged_anchors, merged_scores = _dedupe_anchor_particles(
        merged_anchors,
        merged_scores,
        keep_highest=keep_highest,
    )
    merged_anchors, merged_scores = _prune_anchor_particles(
        merged_anchors,
        merged_scores,
        max_size=max_size,
        keep_highest=keep_highest,
    )
    return merged_anchors, merged_scores


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
    x0_bad_anchors: Optional[torch.Tensor] = None,
    bad_guidance_strength: float = 0.0,
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
    if x0_bad_anchors is not None and bad_guidance_strength > 0.0:
        bad = x0_bad_anchors.reshape(x0_bad_anchors.shape[0], -1).to(device)
        if bad.shape[0] > 0:
            bad_score = _score_xt_given_x0_mixture(
                xt_flat=x,
                x0_good_flat=bad,
                alpha_bar_t=alpha_bar_t,
            )
            score = score - float(bad_guidance_strength) * bad_score

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
    bad_anchor_x0: Optional[torch.Tensor] = None,
    alpha_bar_t: Optional[float] = None,
    bad_guidance_strength: float = 0.0,
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
        bad_anchor_x0: Optional explicit bad-anchor set for contrastive archive guidance.
        alpha_bar_t: Forward diffusion alpha_bar at the current timestep.
        bad_guidance_strength: Strength of bad-anchor repulsion in the mixture-score path.
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
        if bad_anchor_x0 is not None:
            bad_anchor_x0 = _unique_anchor_particles(bad_anchor_x0)
        return _compute_svgd_vector_field_mixture_score(
            xt_particles=particles,
            x0_good_anchors=good_anchor_x0,
            x0_bad_anchors=bad_anchor_x0,
            bad_guidance_strength=bad_guidance_strength,
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
    bad_anchor_x0: Optional[torch.Tensor] = None,
    alpha_bar_t: Optional[float] = None,
    step_size: float = 0.1,
    bad_guidance_strength: float = 0.0,
    sigma: Optional[float] = None,
    device: torch.device = torch.device('cuda'),
) -> torch.Tensor:
    """
    Apply SVGD steering to particles by moving them along the computed vector field.
    
    Args:
        particles: Particle population of shape (num_particles, *spatial_dims)
        binary_rewards: Binary labels (0=bad, 1=good) for each particle
        x0_particles: Predicted clean samples x0 used for score approximation.
        good_anchor_x0: Optional explicit good-anchor set.
        bad_anchor_x0: Optional explicit bad-anchor set for contrastive archive guidance.
        alpha_bar_t: Forward diffusion alpha_bar at the current timestep.
        step_size: Step size for gradient update (default: 0.1)
        bad_guidance_strength: Strength of bad-anchor repulsion in the mixture-score path.
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
        bad_anchor_x0=bad_anchor_x0,
        alpha_bar_t=alpha_bar_t,
        bad_guidance_strength=bad_guidance_strength,
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
        guidance_frequency: Frequency (in timesteps) for applying SVGD guidance in evolution mode.
            If None, defaults to `resample_frequency` to preserve the previous behavior.
        use_anchor_archive: Whether to accumulate good/bad x0 anchors before guidance.
        archive_size: Maximum number of good and bad anchors retained in the archive.
        archive_good_quantile: Reward quantile used to admit good anchors into the archive.
        archive_bad_quantile: Reward quantile used to admit bad anchors into the archive.
        archive_burn_in_steps: Number of initial evolution steps used only for collection.
        min_good_anchors: Minimum number of archived good anchors required before archive guidance starts.
        min_bad_anchors: Minimum number of archived bad anchors required before contrastive bad guidance starts.
        bad_guidance_strength: Strength of archived bad-anchor repulsion in the mixture-score path.
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
        guidance_frequency: Optional[int] = None,
        use_anchor_archive: bool = False,
        archive_size: int = 64,
        archive_good_quantile: float = 0.75,
        archive_bad_quantile: float = 0.25,
        archive_burn_in_steps: int = 0,
        min_good_anchors: int = 8,
        min_bad_anchors: int = 0,
        bad_guidance_strength: float = 0.0,
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
        self.guidance_frequency = (
            self.resample_frequency if guidance_frequency is None else int(guidance_frequency)
        )
        if self.guidance_frequency <= 0:
            raise ValueError("guidance_frequency must be a positive integer.")
        self.use_anchor_archive = bool(use_anchor_archive)
        self.archive_size = int(archive_size)
        self.archive_good_quantile = float(archive_good_quantile)
        self.archive_bad_quantile = float(archive_bad_quantile)
        self.archive_burn_in_steps = int(archive_burn_in_steps)
        self.min_good_anchors = int(min_good_anchors)
        self.min_bad_anchors = int(min_bad_anchors)
        self.bad_guidance_strength = float(max(0.0, bad_guidance_strength))
        if self.archive_size <= 0:
            raise ValueError("archive_size must be a positive integer.")
        if not (0.0 <= self.archive_bad_quantile < self.archive_good_quantile <= 1.0):
            raise ValueError(
                "Expected 0 <= archive_bad_quantile < archive_good_quantile <= 1."
            )
        if self.archive_burn_in_steps < 0:
            raise ValueError("archive_burn_in_steps must be >= 0.")
        if self.min_good_anchors < 0 or self.min_bad_anchors < 0:
            raise ValueError("min_good_anchors and min_bad_anchors must be >= 0.")
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
        self.archive_good_anchors: Optional[torch.Tensor] = None
        self.archive_good_scores: Optional[torch.Tensor] = None
        self.archive_bad_anchors: Optional[torch.Tensor] = None
        self.archive_bad_scores: Optional[torch.Tensor] = None

    def _update_evolution_archive(
        self,
        *,
        x0_preds: torch.Tensor,
        rewards: Union[torch.Tensor, np.ndarray, list, tuple],
    ) -> None:
        if not self.use_anchor_archive:
            return

        reward_tensor = _to_tensor_1d(rewards, device=x0_preds.device)
        if reward_tensor.numel() == 0:
            return

        good_threshold = torch.quantile(reward_tensor, self.archive_good_quantile)
        bad_threshold = torch.quantile(reward_tensor, self.archive_bad_quantile)

        good_mask = reward_tensor >= good_threshold
        bad_mask = reward_tensor <= bad_threshold

        if good_mask.any():
            self.archive_good_anchors, self.archive_good_scores = _update_anchor_bank(
                bank_anchors=self.archive_good_anchors,
                bank_scores=self.archive_good_scores,
                new_anchors=x0_preds[good_mask].detach().cpu(),
                new_scores=reward_tensor[good_mask].detach().cpu(),
                max_size=self.archive_size,
                keep_highest=True,
            )

        if bad_mask.any():
            self.archive_bad_anchors, self.archive_bad_scores = _update_anchor_bank(
                bank_anchors=self.archive_bad_anchors,
                bank_scores=self.archive_bad_scores,
                new_anchors=x0_preds[bad_mask].detach().cpu(),
                new_scores=reward_tensor[bad_mask].detach().cpu(),
                max_size=self.archive_size,
                keep_highest=False,
            )

    def _get_archive_guidance_anchors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        good_anchor_x0 = None
        bad_anchor_x0 = None

        num_good = 0 if self.archive_good_anchors is None else int(self.archive_good_anchors.shape[0])
        num_bad = 0 if self.archive_bad_anchors is None else int(self.archive_bad_anchors.shape[0])

        if num_good >= self.min_good_anchors and self.archive_good_anchors is not None:
            good_anchor_x0 = self.archive_good_anchors.to(device=device, dtype=dtype)

        need_bad = self.bad_guidance_strength > 0.0
        required_bad = max(self.min_bad_anchors, 1) if need_bad else 0
        if need_bad and num_bad >= required_bad and self.archive_bad_anchors is not None:
            bad_anchor_x0 = self.archive_bad_anchors.to(device=device, dtype=dtype)

        return good_anchor_x0, bad_anchor_x0

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
        if self.potential_type == PotentialType.EVOLUTION:
            in_collection_window = (
                self.resampling_t_start <= sampling_idx <= self.resampling_t_end
            )
            resampling_interval = np.arange(
                self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
            )
            guidance_interval = np.arange(
                self.resampling_t_start, self.resampling_t_end + 1, self.guidance_frequency
            )
            should_collect = self.use_anchor_archive and in_collection_window
            should_resample = (
                self.resample_strategy != "none" and sampling_idx in resampling_interval
            )
            should_guide = sampling_idx in guidance_interval

            if not should_collect and not should_resample and not should_guide:
                return latents, None

            # Evolution steering scores the predicted clean sample x0 and can
            # use the same rewards for both selection and SVGD transport.
            decoded_images = self.latent_to_decode_fn(x0_preds)
            rewards = self.reward_fn(decoded_images)
            if should_collect:
                self._update_evolution_archive(x0_preds=x0_preds, rewards=rewards)
            binary_rewards = evolution_steering_binary_rewards(rewards=rewards)
            good_indices = [i for i, y in enumerate(binary_rewards) if y == 1]
            good_anchor_x0 = (
                _unique_anchor_particles(x0_preds[good_indices])
                if good_indices
                else None
            )
            bad_anchor_x0 = None

            if should_resample:
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

            if not should_guide:
                return latents, decoded_images

            if self.use_anchor_archive:
                enough_burn_in = (
                    sampling_idx - self.resampling_t_start
                ) >= self.archive_burn_in_steps
                if not enough_burn_in:
                    return latents, decoded_images

                archive_good_anchor_x0, archive_bad_anchor_x0 = self._get_archive_guidance_anchors(
                    device=latents.device,
                    dtype=x0_preds.dtype,
                )
                need_bad_archive = self.bad_guidance_strength > 0.0
                if archive_good_anchor_x0 is None:
                    return latents, decoded_images
                if need_bad_archive and archive_bad_anchor_x0 is None:
                    return latents, decoded_images

                good_anchor_x0 = archive_good_anchor_x0
                bad_anchor_x0 = archive_bad_anchor_x0

            alpha_bar_t = None
            if self.alpha_bar_fn is not None:
                alpha_bar_t = self.alpha_bar_fn(sampling_idx)

            steered_latents = apply_svgd_steering(
                particles=latents,
                binary_rewards=binary_rewards,
                x0_particles=x0_preds,
                good_anchor_x0=good_anchor_x0,
                bad_anchor_x0=bad_anchor_x0,
                alpha_bar_t=alpha_bar_t,
                step_size=self.svgd_step_size,
                bad_guidance_strength=self.bad_guidance_strength,
                sigma=self.svgd_sigma,
                device=latents.device,
            )

            return steered_latents, decoded_images
            
        else:
            # Check if resampling is within the allowed range and conditions
            resampling_interval = np.arange(
                self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
            )

            if sampling_idx not in resampling_interval:
                return latents, None

            decoded_images = self.latent_to_decode_fn(x0_preds)
            rewards = self.reward_fn(decoded_images)

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
