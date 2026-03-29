#!/usr/bin/env python3
"""Compare base diffusion vs standalone EVO guidance (no FKD pipeline/classes)."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

from evo_module import EvoConfig, build_evo_step_callback
from fkd_diffusers.rewards import get_reward_function


MODEL_MAP = {
    "stable-diffusion-2-1": ("sd2-community/stable-diffusion-2-1", "sd"),
    "stable-diffusion-v1-5": ("runwayml/stable-diffusion-v1-5", "sd"),
    "stable-diffusion-v1-4": ("CompVis/stable-diffusion-v1-4", "sd"),
    "stable-diffusion-xl": ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl"),
}


def _get_pipe(model_name: str, dtype: torch.dtype):
    model_id, family = MODEL_MAP[model_name]
    if family == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe


def _make_generators(seed: int, n: int, device: str):
    return [torch.Generator(device=device).manual_seed(seed + i) for i in range(n)]


def _score_images(images, prompts, reward_name: str):
    arr = np.asarray(
        get_reward_function(
            reward_name,
            images=images,
            prompts=prompts,
            metric_to_chase=None,
        ),
        dtype=np.float32,
    )
    return arr


def _save_grid(images, scores, out_path: Path, title: str):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=160)
    if n == 1:
        axes = [axes]
    for i, (ax, image, score) in enumerate(zip(axes, images, scores)):
        ax.imshow(image)
        ax.set_title(f"#{i + 1} score={score:.3f}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base diffusion vs standalone EVO guidance")
    p.add_argument("--model-name", default="stable-diffusion-2-1", choices=sorted(MODEL_MAP.keys()))
    p.add_argument("--prompt", default="a cinematic photo of a corgi astronaut on mars, ultra detailed")
    p.add_argument("--num-particles", type=int, default=4)
    p.add_argument("--time-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--guidance-reward-fn", default="Clip-Score")

    p.add_argument("--evo-step-size", type=float, default=0.12)
    p.add_argument("--evo-sigma", type=float, default=None)
    p.add_argument("--evo-update-frequency", type=int, default=10)
    p.add_argument("--evo-update-start", type=int, default=10)
    p.add_argument("--evo-update-end", type=int, default=45)

    p.add_argument("--output-dir", default="text_to_image/output")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = _get_pipe(args.model_name, dtype=dtype).to(device)

    prompts = [args.prompt] * args.num_particles

    base_generators = _make_generators(args.seed, args.num_particles, device)
    base_images = pipe(
        prompt=prompts,
        num_inference_steps=args.time_steps,
        eta=1.0,
        generator=base_generators,
    ).images
    base_scores = _score_images(base_images, prompts, args.guidance_reward_fn)

    evo_cfg = EvoConfig(
        guidance_reward_fn=args.guidance_reward_fn,
        step_size=args.evo_step_size,
        sigma=args.evo_sigma,
        update_frequency=args.evo_update_frequency,
        update_t_start=args.evo_update_start,
        update_t_end=args.evo_update_end,
    )
    evo_callback = build_evo_step_callback(pipe=pipe, prompts=prompts, evo_cfg=evo_cfg)

    evo_generators = _make_generators(args.seed, args.num_particles, device)
    evo_images = pipe(
        prompt=prompts,
        num_inference_steps=args.time_steps,
        eta=1.0,
        generator=evo_generators,
        callback_on_step_end=evo_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images
    evo_scores = _score_images(evo_images, prompts, args.guidance_reward_fn)

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"evo_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_order = np.argsort(base_scores)[::-1]
    evo_order = np.argsort(evo_scores)[::-1]
    base_sorted = [base_images[i] for i in base_order]
    evo_sorted = [evo_images[i] for i in evo_order]

    _save_grid(
        base_sorted,
        base_scores[base_order],
        out_dir / "base.png",
        title=f"Base | mean={base_scores.mean():.4f} | std={base_scores.std():.4f}",
    )
    _save_grid(
        evo_sorted,
        evo_scores[evo_order],
        out_dir / "evo.png",
        title=f"Evo (standalone) | mean={evo_scores.mean():.4f} | std={evo_scores.std():.4f}",
    )

    with (out_dir / "scores.txt").open("w", encoding="utf-8") as f:
        f.write("Base scores:\n")
        f.write(", ".join(f"{x:.6f}" for x in base_scores.tolist()) + "\n")
        f.write("Evo scores:\n")
        f.write(", ".join(f"{x:.6f}" for x in evo_scores.tolist()) + "\n")
        f.write(f"Base mean/std: {base_scores.mean():.6f} / {base_scores.std():.6f}\n")
        f.write(f"Evo mean/std:  {evo_scores.mean():.6f} / {evo_scores.std():.6f}\n")

    print(f"Saved comparison to: {out_dir}")
    print(f"Base mean/std: {base_scores.mean():.4f} / {base_scores.std():.4f}")
    print(f"Evo  mean/std: {evo_scores.mean():.4f} / {evo_scores.std():.4f}")


if __name__ == "__main__":
    main()
