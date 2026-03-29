"""Stable Diffusion wrapper that adds archive-based Evo steering."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

try:
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion, latent_to_decode
except ModuleNotFoundError:
    from fkd_pipeline_sd import FKDStableDiffusion, latent_to_decode

try:
    from .evo_class import EvoGuidance
    from .rewards import get_reward_function
except ImportError:
    from evo_class import EvoGuidance
    from rewards import get_reward_function


class EvoStableDiffusion(FKDStableDiffusion):
    """FKD-compatible SD pipeline that can run Evo guidance via `evo_args`."""

    _callback_tensor_inputs = FKDStableDiffusion._callback_tensor_inputs + ["x0_preds"]

    def _alpha_bar_from_timestep(self, timestep) -> Optional[float]:
        if not hasattr(self.scheduler, "alphas_cumprod"):
            return None

        timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        alphas_cumprod = self.scheduler.alphas_cumprod
        if not isinstance(alphas_cumprod, torch.Tensor):
            return None

        idx = max(0, min(timestep_value, alphas_cumprod.shape[0] - 1))
        return float(alphas_cumprod[idx].detach().cpu().item())

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], None] = None,
        evo_args: Optional[Dict[str, Any]] = None,
        fkd_args: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        **kwargs,
    ):
        if evo_args is not None and fkd_args is not None:
            raise ValueError("Pass either `evo_args` or `fkd_args`, not both.")

        if evo_args is None:
            return super().__call__(
                prompt=prompt,
                fkd_args=fkd_args,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs or ["latents"],
                **kwargs,
            )

        evo_config = dict(evo_args)
        reward_name = evo_config.pop("guidance_reward_fn", None)
        metric_to_chase = evo_config.pop("metric_to_chase", None)
        reward_fn = evo_config.pop("reward_fn", None)
        latent_to_decode_fn = evo_config.pop("latent_to_decode_fn", None)
        alpha_bar_fn = evo_config.pop("alpha_bar_fn", None)

        if reward_fn is None:
            if reward_name is None:
                raise ValueError(
                    "`evo_args` must include either `reward_fn` or `guidance_reward_fn`."
                )

            def reward_fn(decoded_images: torch.Tensor):
                pil_images = list(
                    self.image_processor.postprocess(decoded_images, output_type="pil")
                )
                rewards = get_reward_function(
                    reward_name,
                    images=pil_images,
                    prompts=prompt,
                    metric_to_chase=metric_to_chase,
                )
                return torch.as_tensor(rewards, device=decoded_images.device)

        if latent_to_decode_fn is None:
            latent_to_decode_fn = lambda x: latent_to_decode(
                model=self, output_type="pil", latents=x
            )

        evo = EvoGuidance(
            reward_fn=reward_fn,
            latent_to_decode_fn=latent_to_decode_fn,
            alpha_bar_fn=alpha_bar_fn,
            **evo_config,
        )

        requested_inputs = list(callback_on_step_end_tensor_inputs or ["latents"])
        if "x0_preds" not in requested_inputs:
            requested_inputs.append("x0_preds")

        user_callback = callback_on_step_end

        def evo_callback(_pipe, step_idx: int, timestep, callback_kwargs: Dict[str, Any]):
            resolved_alpha_bar = (
                alpha_bar_fn(step_idx)
                if alpha_bar_fn is not None
                else self._alpha_bar_from_timestep(timestep)
            )
            updated_latents, decoded_images = evo.step(
                sampling_idx=step_idx,
                latents=callback_kwargs["latents"],
                x0_preds=callback_kwargs["x0_preds"],
                alpha_bar_t=resolved_alpha_bar,
            )
            callback_kwargs["latents"] = updated_latents

            if decoded_images is not None and "particle_previews" in callback_kwargs:
                callback_kwargs["particle_previews"] = decoded_images

            if user_callback is None:
                return callback_kwargs
            return user_callback(_pipe, step_idx, timestep, callback_kwargs)

        return super().__call__(
            prompt=prompt,
            fkd_args=None,
            callback_on_step_end=evo_callback,
            callback_on_step_end_tensor_inputs=requested_inputs,
            **kwargs,
        )
