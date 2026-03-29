"""Utilities to visualize diffusion generation from noise to image.

This module captures intermediate states via ``callback_on_step_end`` and
offers plotting/GIF helpers for quick notebook inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image


@dataclass
class TrajectoryFrame:
	step_idx: int
	timestep: int
	image: Image.Image


@dataclass
class TrajectoryResult:
	final_images: list[Image.Image]
	frames: list[TrajectoryFrame]


def _to_pil_from_tensor(pipe, image_tensor: torch.Tensor) -> list[Image.Image]:
	"""Convert decoded image tensor to PIL images via the pipeline image processor."""
	images = pipe.image_processor.postprocess(image_tensor, output_type="pil")
	return list(images)


def _decode_latents_to_pil(pipe, latents: torch.Tensor) -> list[Image.Image]:
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
	return _to_pil_from_tensor(pipe, image)


def _first_preview_image(pipe, callback_kwargs: dict[str, Any], fallback_latents: torch.Tensor) -> Image.Image:
	previews = callback_kwargs.get("particle_previews", None)
	if isinstance(previews, list) and previews and isinstance(previews[0], Image.Image):
		return previews[0]

	if isinstance(previews, torch.Tensor):
		pil = _to_pil_from_tensor(pipe, previews)
		if pil:
			return pil[0]

	return _decode_latents_to_pil(pipe, fallback_latents[:1])[0]


@torch.no_grad()
def run_and_capture_trajectory(
	*,
	pipe,
	prompt: str | list[str],
	num_inference_steps: int = 40,
	seed: Optional[int] = 0,
	guidance_scale: float = 7.5,
	fkd_args: Optional[dict[str, Any]] = None,
	num_images_per_prompt: int = 1,
	capture_every: int = 1,
) -> TrajectoryResult:
	"""Run generation while capturing intermediate decoded images.

	Notes:
	- Uses ``particle_previews`` when available (x0-style preview in this repo).
	- Falls back to decoding current latents if previews are unavailable.
	"""
	if capture_every <= 0:
		raise ValueError("capture_every must be >= 1")

	if seed is not None:
		generator = torch.Generator(device=pipe.device).manual_seed(seed)
	else:
		generator = None

	frames: list[TrajectoryFrame] = []

	def _capture_callback(_pipe, step_idx: int, timestep, callback_kwargs: dict[str, Any]):
		if step_idx % capture_every != 0:
			return callback_kwargs

		latents = callback_kwargs.get("latents", None)
		if latents is None:
			return callback_kwargs

		timestep_int = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
		image0 = _first_preview_image(pipe, callback_kwargs, latents)
		frames.append(
			TrajectoryFrame(
				step_idx=int(step_idx),
				timestep=timestep_int,
				image=image0.copy(),
			)
		)
		return callback_kwargs

	result = pipe(
		prompt,
		num_inference_steps=num_inference_steps,
		guidance_scale=guidance_scale,
		num_images_per_prompt=num_images_per_prompt,
		generator=generator,
		fkd_args=fkd_args,
		output_type="pil",
		callback_on_step_end=_capture_callback,
		callback_on_step_end_tensor_inputs=["latents", "particle_previews"],
	)

	final_images = result.images if hasattr(result, "images") else result[0]
	final_images = list(final_images)

	if not frames and final_images:
		frames.append(TrajectoryFrame(step_idx=0, timestep=0, image=final_images[0]))

	return TrajectoryResult(final_images=final_images, frames=frames)


def show_trajectory_grid(
	frames: list[TrajectoryFrame],
	*,
	max_frames: int = 12,
	cols: int = 4,
	title: str = "Diffusion trajectory (noise -> image)",
):
	"""Render a sampled grid of trajectory frames."""
	if not frames:
		raise ValueError("No frames to display.")

	if max_frames < 2:
		selected = [frames[-1]]
	elif len(frames) <= max_frames:
		selected = frames
	else:
		idxs = torch.linspace(0, len(frames) - 1, steps=max_frames).long().tolist()
		selected = [frames[i] for i in idxs]

	n = len(selected)
	cols = max(1, cols)
	rows = (n + cols - 1) // cols

	fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), dpi=130)
	if rows == 1 and cols == 1:
		axes = [[axes]]
	elif rows == 1:
		axes = [axes]
	elif cols == 1:
		axes = [[ax] for ax in axes]

	flat_axes = [ax for row_axes in axes for ax in row_axes]
	for ax, frame in zip(flat_axes, selected):
		ax.imshow(frame.image)
		ax.set_title(f"step={frame.step_idx} | t={frame.timestep}")
		ax.axis("off")

	for ax in flat_axes[len(selected):]:
		ax.axis("off")

	fig.suptitle(title, fontsize=13)
	fig.tight_layout()
	plt.show()


def save_trajectory_gif(
	frames: list[TrajectoryFrame],
	output_path: str | Path,
	*,
	fps: int = 8,
	loop: int = 0,
) -> Path:
	"""Save trajectory frames as an animated GIF."""
	if not frames:
		raise ValueError("No frames to write.")

	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	duration_ms = int(1000 / max(1, fps))

	images = [f.image for f in frames]
	images[0].save(
		output,
		save_all=True,
		append_images=images[1:],
		duration=duration_ms,
		loop=loop,
		optimize=False,
	)
	return output
