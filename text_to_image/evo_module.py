"""Standalone EVO guidance module for base diffusion pipelines.

This module intentionally does not depend on FKD classes/pipelines.
It provides a callback-based latent adjustment mechanism that can be used
with vanilla diffusers Stable Diffusion pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from fkd_diffusers.rewards import get_reward_function


@dataclass
class EvoConfig:
    guidance_reward_fn: str = "Clip-Score"
    step_size: float = 0.1
    sigma: Optional[float] = None
    update_frequency: int = 10
    update_t_start: int = 10
    update_t_end: int = 45


def _to_numpy_1d(values: torch.Tensor | np.ndarray | list | tuple) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().float().numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def binary_good_bad_labels(
    rewards: torch.Tensor | np.ndarray | list | tuple,
    threshold: Optional[float] = None,
) -> list[int]:
    rewards_np = _to_numpy_1d(rewards)
    if rewards_np.size == 0:
        return []
    if threshold is None:
        threshold = float(np.percentile(rewards_np, 50))
    return [int(reward > threshold) for reward in rewards_np]


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    x_cdist = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y_cdist = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
    sq_dist = torch.cdist(x_cdist, y_cdist, p=2.0) ** 2
    return torch.exp(-sq_dist / (2.0 * sigma * sigma)).to(dtype=x.dtype)


def _median_pairwise_distance(x: torch.Tensor) -> float:
    x_cdist = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    sq_dist = torch.cdist(x_cdist, x_cdist, p=2.0) ** 2
    distances = torch.triu(sq_dist, diagonal=1)
    distances_flat = distances[distances > 0]
    if distances_flat.numel() == 0:
        return 1.0
    return float(torch.sqrt(torch.median(distances_flat)))


def _compute_transport_vector_field(
    particles: torch.Tensor,
    binary_labels: list[int],
    sigma: Optional[float] = None,
) -> torch.Tensor:
    n = particles.shape[0]
    particle_shape = particles.shape
    x = particles.reshape(n, -1)

    device = x.device
    bad_indices = torch.tensor([i for i, y in enumerate(binary_labels) if y == 0], device=device)
    good_indices = torch.tensor([i for i, y in enumerate(binary_labels) if y == 1], device=device)

    if good_indices.numel() == 0:
        return torch.zeros_like(particles)

    if sigma is None:
        sigma = _median_pairwise_distance(x)
    sigma = float(max(sigma, 1e-6))

    vector_field = torch.zeros_like(x)
    if bad_indices.numel() > 0:
        good_particles = x[good_indices]
        bad_particles = x[bad_indices]
        k_bg = _rbf_kernel(bad_particles, good_particles, sigma=sigma)
        directions = good_particles.unsqueeze(0) - bad_particles.unsqueeze(1)
        k_weights = k_bg / (k_bg.sum(dim=1, keepdim=True) + 1e-8)
        mean_direction = torch.einsum("ij,ijd->id", k_weights, directions)
        vector_field[bad_indices] = mean_direction

    return vector_field.reshape(particle_shape)


def apply_evo_adjustment(
    latents: torch.Tensor,
    rewards: torch.Tensor | np.ndarray | list | tuple,
    *,
    step_size: float,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    labels = binary_good_bad_labels(rewards)
    vector_field = _compute_transport_vector_field(latents, labels, sigma=sigma)

    field_norm = torch.norm(vector_field.reshape(vector_field.shape[0], -1), dim=1, keepdim=True)
    field_norm = field_norm.reshape(-1, *([1] * (len(vector_field.shape) - 1)))
    field_norm = torch.clamp(field_norm, min=1e-8)
    normalized_field = vector_field / field_norm

    return latents + step_size * normalized_field


def decode_latents_to_pil(pipe, latents: torch.Tensor):
    if latents.dtype != pipe.vae.dtype and torch.backends.mps.is_available():
        pipe.vae = pipe.vae.to(latents.dtype)

    has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        latents_dec = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
    else:
        latents_dec = latents / pipe.vae.config.scaling_factor

    image = pipe.vae.decode(latents_dec, return_dict=False)[0]
    return list(pipe.image_processor.postprocess(image, output_type="pil"))


def build_evo_step_callback(
    *,
    pipe,
    prompts: list[str],
    evo_cfg: EvoConfig,
) -> Callable:
    """Return a callback_on_step_end function for vanilla diffusers pipelines."""

    def _callback(_pipe, step_idx: int, _timestep, callback_kwargs: dict):
        if step_idx < evo_cfg.update_t_start or step_idx > evo_cfg.update_t_end:
            return callback_kwargs
        if evo_cfg.update_frequency <= 0 or (step_idx - evo_cfg.update_t_start) % evo_cfg.update_frequency != 0:
            return callback_kwargs

        latents = callback_kwargs.get("latents", None)
        if latents is None:
            return callback_kwargs

        with torch.no_grad():
            pil_images = decode_latents_to_pil(pipe, latents)
            rewards = get_reward_function(
                evo_cfg.guidance_reward_fn,
                images=pil_images,
                prompts=prompts,
                metric_to_chase=None,
            )
            latents = apply_evo_adjustment(
                latents,
                rewards,
                step_size=evo_cfg.step_size,
                sigma=evo_cfg.sigma,
            )

        callback_kwargs["latents"] = latents
        return callback_kwargs

    return _callback
