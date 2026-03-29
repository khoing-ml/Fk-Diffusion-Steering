"""Archive-based evolution guidance independent from FKD internals."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import torch


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


def quantile_good_bad_masks(
    rewards: Union[torch.Tensor, np.ndarray, list, tuple],
    *,
    good_quantile: float = 0.75,
    bad_quantile: float = 0.25,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return quantile-based good and bad masks for a reward vector."""
    reward_tensor = _to_tensor_1d(rewards, device=device)
    if reward_tensor.numel() == 0:
        empty = torch.zeros(0, dtype=torch.bool, device=reward_tensor.device)
        return empty, empty

    good_threshold = torch.quantile(reward_tensor, float(good_quantile))
    bad_threshold = torch.quantile(reward_tensor, float(bad_quantile))
    return reward_tensor >= good_threshold, reward_tensor <= bad_threshold


def quantile_good_bad_labels(
    rewards: Union[torch.Tensor, np.ndarray, list, tuple],
    *,
    good_quantile: float = 0.75,
    bad_quantile: float = 0.25,
    ignore_middle: bool = True,
) -> list[int]:
    """
    Convert rewards to quantile labels.

    Returns:
        `1` for good, `0` for bad, and `-1` for the ignored middle if
        `ignore_middle=True`.
    """
    rewards_np = _to_numpy_1d(rewards)
    if rewards_np.size == 0:
        return []

    good_threshold = float(np.quantile(rewards_np, good_quantile))
    bad_threshold = float(np.quantile(rewards_np, bad_quantile))

    labels: list[int] = []
    for reward in rewards_np:
        if reward >= good_threshold:
            labels.append(1)
        elif reward <= bad_threshold:
            labels.append(0)
        else:
            labels.append(-1 if ignore_middle else 0)
    return labels


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    out_dtype = x.dtype
    x_cdist = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y_cdist = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
    sq_dist = torch.cdist(x_cdist, y_cdist, p=2.0) ** 2
    kernel = torch.exp(-sq_dist / (2.0 * sigma * sigma))
    return kernel.to(dtype=out_dtype)


def _median_pairwise_distance(x: torch.Tensor) -> float:
    x_cdist = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    sq_dist = torch.cdist(x_cdist, x_cdist, p=2.0) ** 2
    distances = torch.triu(sq_dist, diagonal=1)
    distances_flat = distances[distances > 0]
    if distances_flat.numel() == 0:
        return 1.0
    return float(torch.sqrt(torch.median(distances_flat)))


def _unique_anchor_particles(anchors: torch.Tensor) -> torch.Tensor:
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


def _normalize_anchor_weights(
    scores: Optional[torch.Tensor],
    *,
    device: torch.device,
    num_items: int,
    prefer_highest: bool,
) -> Optional[torch.Tensor]:
    if scores is None:
        return None

    weights = scores.detach().to(device=device, dtype=torch.float32).flatten()
    if weights.numel() != num_items:
        return None

    if not prefer_highest:
        weights = -weights

    weights = weights - weights.mean()
    weights = torch.softmax(weights, dim=0)
    return weights / torch.clamp(weights.sum(), min=1e-8)


def _score_xt_given_x0_mixture(
    *,
    xt_flat: torch.Tensor,
    x0_anchor_flat: torch.Tensor,
    alpha_bar_t: float,
    anchor_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Approximate the score of q(x_t | archive anchors) under a VP mixture."""
    alpha_bar_t = float(max(0.0, min(1.0, alpha_bar_t)))
    variance = max(1.0 - alpha_bar_t, 1e-6)
    sqrt_alpha = float(np.sqrt(alpha_bar_t))

    means = sqrt_alpha * x0_anchor_flat
    diff = xt_flat.unsqueeze(1) - means.unsqueeze(0)
    log_liks = -0.5 * (diff.pow(2).sum(dim=-1) / variance)

    if anchor_weights is not None:
        anchor_weights = anchor_weights.to(device=xt_flat.device, dtype=torch.float32)
        log_liks = log_liks + torch.log(torch.clamp(anchor_weights, min=1e-8)).unsqueeze(0)

    posterior_weights = torch.softmax(log_liks, dim=1)
    component_scores = -(diff / variance)
    return (posterior_weights.unsqueeze(-1) * component_scores).sum(dim=1)


def _compute_mixture_guidance_vector_field(
    *,
    xt_particles: torch.Tensor,
    x0_good_anchors: torch.Tensor,
    good_anchor_scores: Optional[torch.Tensor] = None,
    x0_bad_anchors: Optional[torch.Tensor] = None,
    bad_anchor_scores: Optional[torch.Tensor] = None,
    bad_guidance_strength: float = 0.0,
    sigma: Optional[float] = None,
    device: torch.device,
    alpha_bar_t: float,
) -> torch.Tensor:
    n = xt_particles.shape[0]
    particle_shape = xt_particles.shape
    x = xt_particles.reshape(n, -1).to(device)
    good = x0_good_anchors.reshape(x0_good_anchors.shape[0], -1).to(device)

    if good.shape[0] == 0:
        return torch.zeros_like(xt_particles)

    if sigma is None:
        sigma = _median_pairwise_distance(x)
    sigma = float(max(sigma, 1e-6))

    good_weights = _normalize_anchor_weights(
        good_anchor_scores,
        device=device,
        num_items=good.shape[0],
        prefer_highest=True,
    )
    score = _score_xt_given_x0_mixture(
        xt_flat=x,
        x0_anchor_flat=good,
        alpha_bar_t=alpha_bar_t,
        anchor_weights=good_weights,
    )

    if x0_bad_anchors is not None and bad_guidance_strength > 0.0:
        bad = x0_bad_anchors.reshape(x0_bad_anchors.shape[0], -1).to(device)
        if bad.shape[0] > 0:
            bad_weights = _normalize_anchor_weights(
                bad_anchor_scores,
                device=device,
                num_items=bad.shape[0],
                prefer_highest=False,
            )
            bad_score = _score_xt_given_x0_mixture(
                xt_flat=x,
                x0_anchor_flat=bad,
                alpha_bar_t=alpha_bar_t,
                anchor_weights=bad_weights,
            )
            score = score - float(bad_guidance_strength) * bad_score

    kernel = _rbf_kernel(x, x, sigma=sigma)
    first_term = kernel.transpose(0, 1) @ score

    diff = x.unsqueeze(1) - x.unsqueeze(0)
    grad_j_kernel = -(kernel.unsqueeze(-1) * diff) / (sigma**2)
    second_term = grad_j_kernel.sum(dim=0)

    field = (first_term + second_term) / float(n)
    return field.reshape(particle_shape)


def _weighted_transport_direction(
    *,
    particles_flat: torch.Tensor,
    anchor_flat: torch.Tensor,
    anchor_scores: Optional[torch.Tensor],
    sigma: float,
    prefer_highest: bool,
) -> torch.Tensor:
    kernel = _rbf_kernel(particles_flat, anchor_flat, sigma=sigma)
    if anchor_scores is not None:
        score_weights = _normalize_anchor_weights(
            anchor_scores,
            device=particles_flat.device,
            num_items=anchor_flat.shape[0],
            prefer_highest=prefer_highest,
        )
        if score_weights is not None:
            kernel = kernel * score_weights.unsqueeze(0)

    directions = anchor_flat.unsqueeze(0) - particles_flat.unsqueeze(1)
    kernel_weights = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-8)
    return torch.einsum("ij,ijd->id", kernel_weights, directions)


def _compute_transport_guidance_vector_field(
    *,
    particles: torch.Tensor,
    good_anchor_x0: torch.Tensor,
    good_anchor_scores: Optional[torch.Tensor] = None,
    bad_anchor_x0: Optional[torch.Tensor] = None,
    bad_anchor_scores: Optional[torch.Tensor] = None,
    bad_guidance_strength: float = 0.0,
    sigma: Optional[float] = None,
    device: torch.device,
) -> torch.Tensor:
    particle_shape = particles.shape
    particles_flat = particles.reshape(particles.shape[0], -1).to(device)
    good_anchor_flat = good_anchor_x0.reshape(good_anchor_x0.shape[0], -1).to(device)

    if good_anchor_flat.shape[0] == 0:
        return torch.zeros_like(particles)

    if sigma is None:
        sigma = _median_pairwise_distance(particles_flat)
    sigma = float(max(sigma, 1e-6))

    field = _weighted_transport_direction(
        particles_flat=particles_flat,
        anchor_flat=good_anchor_flat,
        anchor_scores=good_anchor_scores,
        sigma=sigma,
        prefer_highest=True,
    )

    if bad_anchor_x0 is not None and bad_guidance_strength > 0.0:
        bad_anchor_flat = bad_anchor_x0.reshape(bad_anchor_x0.shape[0], -1).to(device)
        if bad_anchor_flat.shape[0] > 0:
            bad_field = _weighted_transport_direction(
                particles_flat=particles_flat,
                anchor_flat=bad_anchor_flat,
                anchor_scores=bad_anchor_scores,
                sigma=sigma,
                prefer_highest=False,
            )
            field = field - float(bad_guidance_strength) * bad_field

    return field.reshape(particle_shape)


def compute_evo_vector_field(
    particles: torch.Tensor,
    *,
    good_anchor_x0: torch.Tensor,
    good_anchor_scores: Optional[torch.Tensor] = None,
    bad_anchor_x0: Optional[torch.Tensor] = None,
    bad_anchor_scores: Optional[torch.Tensor] = None,
    alpha_bar_t: Optional[float] = None,
    bad_guidance_strength: float = 0.0,
    sigma: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute the Evo guidance field from archived anchors."""
    device = device or particles.device
    good_anchor_x0 = _unique_anchor_particles(good_anchor_x0)
    if bad_anchor_x0 is not None:
        bad_anchor_x0 = _unique_anchor_particles(bad_anchor_x0)

    if alpha_bar_t is not None:
        return _compute_mixture_guidance_vector_field(
            xt_particles=particles,
            x0_good_anchors=good_anchor_x0,
            good_anchor_scores=good_anchor_scores,
            x0_bad_anchors=bad_anchor_x0,
            bad_anchor_scores=bad_anchor_scores,
            bad_guidance_strength=bad_guidance_strength,
            sigma=sigma,
            device=device,
            alpha_bar_t=float(alpha_bar_t),
        )

    return _compute_transport_guidance_vector_field(
        particles=particles,
        good_anchor_x0=good_anchor_x0,
        good_anchor_scores=good_anchor_scores,
        bad_anchor_x0=bad_anchor_x0,
        bad_anchor_scores=bad_anchor_scores,
        bad_guidance_strength=bad_guidance_strength,
        sigma=sigma,
        device=device,
    )


def apply_evo_guidance(
    particles: torch.Tensor,
    *,
    good_anchor_x0: torch.Tensor,
    good_anchor_scores: Optional[torch.Tensor] = None,
    bad_anchor_x0: Optional[torch.Tensor] = None,
    bad_anchor_scores: Optional[torch.Tensor] = None,
    alpha_bar_t: Optional[float] = None,
    step_size: float = 0.1,
    bad_guidance_strength: float = 0.0,
    sigma: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    vector_field = compute_evo_vector_field(
        particles=particles,
        good_anchor_x0=good_anchor_x0,
        good_anchor_scores=good_anchor_scores,
        bad_anchor_x0=bad_anchor_x0,
        bad_anchor_scores=bad_anchor_scores,
        alpha_bar_t=alpha_bar_t,
        bad_guidance_strength=bad_guidance_strength,
        sigma=sigma,
        device=device or particles.device,
    )
    clipped_field = _clip_vector_field(vector_field, max_particle_norm=1.0)
    return particles + float(step_size) * clipped_field


class EvoGuidance:
    """
    Archive-based evolution guidance controller.

    The controller collects good/bad anchors across denoising steps and applies a
    delayed guidance field during the configured update window.
    """

    def __init__(
        self,
        *,
        reward_fn: Callable[[torch.Tensor], Union[torch.Tensor, np.ndarray, list, tuple]],
        num_particles: int,
        guidance_frequency: int,
        update_t_start: int,
        update_t_end: int,
        archive_size: int = 64,
        archive_good_quantile: float = 0.75,
        archive_bad_quantile: float = 0.25,
        archive_burn_in_steps: int = 0,
        min_good_anchors: int = 8,
        min_bad_anchors: int = 0,
        step_size: float = 0.1,
        sigma: Optional[float] = None,
        bad_guidance_strength: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        alpha_bar_fn: Optional[Callable[[int], Optional[float]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.reward_fn = reward_fn
        self.num_particles = int(num_particles)
        self.guidance_frequency = int(guidance_frequency)
        self.update_t_start = int(update_t_start)
        self.update_t_end = int(update_t_end)
        self.archive_size = int(archive_size)
        self.archive_good_quantile = float(archive_good_quantile)
        self.archive_bad_quantile = float(archive_bad_quantile)
        self.archive_burn_in_steps = int(archive_burn_in_steps)
        self.min_good_anchors = int(min_good_anchors)
        self.min_bad_anchors = int(min_bad_anchors)
        self.step_size = float(step_size)
        self.sigma = sigma
        self.bad_guidance_strength = float(max(0.0, bad_guidance_strength))
        self.latent_to_decode_fn = latent_to_decode_fn
        self.alpha_bar_fn = alpha_bar_fn
        self.device = device

        if self.num_particles <= 0:
            raise ValueError("num_particles must be a positive integer.")
        if self.guidance_frequency <= 0:
            raise ValueError("guidance_frequency must be a positive integer.")
        if self.update_t_end < self.update_t_start:
            raise ValueError("update_t_end must be >= update_t_start.")
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

        self.archive_good_anchors: Optional[torch.Tensor] = None
        self.archive_good_scores: Optional[torch.Tensor] = None
        self.archive_bad_anchors: Optional[torch.Tensor] = None
        self.archive_bad_scores: Optional[torch.Tensor] = None

    def _in_update_window(self, sampling_idx: int) -> bool:
        return self.update_t_start <= sampling_idx <= self.update_t_end

    def _on_guidance_cadence(self, sampling_idx: int) -> bool:
        if not self._in_update_window(sampling_idx):
            return False
        return (sampling_idx - self.update_t_start) % self.guidance_frequency == 0

    def _update_archive(
        self,
        *,
        x0_preds: torch.Tensor,
        rewards: Union[torch.Tensor, np.ndarray, list, tuple],
    ) -> None:
        reward_tensor = _to_tensor_1d(rewards, device=x0_preds.device)
        if reward_tensor.numel() == 0:
            return

        good_mask, bad_mask = quantile_good_bad_masks(
            reward_tensor,
            good_quantile=self.archive_good_quantile,
            bad_quantile=self.archive_bad_quantile,
            device=x0_preds.device,
        )

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

    def _get_guidance_anchors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        good_anchor_x0 = None
        good_anchor_scores = None
        bad_anchor_x0 = None
        bad_anchor_scores = None

        num_good = 0 if self.archive_good_anchors is None else int(self.archive_good_anchors.shape[0])
        num_bad = 0 if self.archive_bad_anchors is None else int(self.archive_bad_anchors.shape[0])

        if num_good >= self.min_good_anchors and self.archive_good_anchors is not None:
            good_anchor_x0 = self.archive_good_anchors.to(device=device, dtype=dtype)
            if self.archive_good_scores is not None:
                good_anchor_scores = self.archive_good_scores.to(device=device, dtype=torch.float32)

        if num_bad >= self.min_bad_anchors and self.archive_bad_anchors is not None:
            bad_anchor_x0 = self.archive_bad_anchors.to(device=device, dtype=dtype)
            if self.archive_bad_scores is not None:
                bad_anchor_scores = self.archive_bad_scores.to(device=device, dtype=torch.float32)

        return good_anchor_x0, good_anchor_scores, bad_anchor_x0, bad_anchor_scores

    def step(
        self,
        *,
        sampling_idx: int,
        latents: torch.Tensor,
        x0_preds: torch.Tensor,
        alpha_bar_t: Optional[float] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run one Evo update.

        Returns updated latents and optionally decoded images used for scoring.
        """
        if not self._in_update_window(sampling_idx):
            return latents, None

        decoded_images = self.latent_to_decode_fn(x0_preds)
        rewards = self.reward_fn(decoded_images)
        self._update_archive(x0_preds=x0_preds, rewards=rewards)

        if not self._on_guidance_cadence(sampling_idx):
            return latents, decoded_images

        if (sampling_idx - self.update_t_start) < self.archive_burn_in_steps:
            return latents, decoded_images

        (
            good_anchor_x0,
            good_anchor_scores,
            bad_anchor_x0,
            bad_anchor_scores,
        ) = self._get_guidance_anchors(
            device=latents.device,
            dtype=x0_preds.dtype,
        )

        if good_anchor_x0 is None:
            return latents, decoded_images

        if alpha_bar_t is None and self.alpha_bar_fn is not None:
            alpha_bar_t = self.alpha_bar_fn(sampling_idx)

        updated_latents = apply_evo_guidance(
            particles=latents,
            good_anchor_x0=good_anchor_x0,
            good_anchor_scores=good_anchor_scores,
            bad_anchor_x0=bad_anchor_x0,
            bad_anchor_scores=bad_anchor_scores,
            alpha_bar_t=alpha_bar_t,
            step_size=self.step_size,
            bad_guidance_strength=self.bad_guidance_strength,
            sigma=self.sigma,
            device=latents.device,
        )
        return updated_latents, decoded_images
