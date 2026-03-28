#!/usr/bin/env python3
"""Run FKD text-to-image steering experiments from one script.

Example:
    python text_to_image/evo_steering_playground.py \
      --model-name stable-diffusion-2-1 \
      --prompt "a cinematic photo of a corgi astronaut on mars, ultra detailed" \
      --modes base,diff,max,evolution \
      --num-particles 4 \
      --time-steps 50 \
      --lmbda 2.0 \
      --svgd-step-size 0.12 \
      --auto-fix-deps
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


VALID_MODES = {"base", "diff", "max", "evolution"}


def _version(mod_name: str) -> str | None:
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def ensure_dependency_compat(*, auto_fix: bool) -> None:
    """Ensure `transformers` is compatible with the installed diffusers version.

    Some environments have older `transformers` that miss `EncoderDecoderCache`,
    which breaks diffusers pipelines at import time.
    """
    transformers_ok = False
    try:
        import transformers  # type: ignore

        transformers_ok = hasattr(transformers, "EncoderDecoderCache")
    except Exception:
        transformers_ok = False

    print("Dependency check:")
    print(f"  diffusers    = {_version('diffusers')}")
    print(f"  transformers = {_version('transformers')}")
    print(f"  has EncoderDecoderCache = {transformers_ok}")

    if transformers_ok:
        return

    if not auto_fix:
        raise RuntimeError(
            "Incompatible transformers/diffusers stack. "
            "Re-run with --auto-fix-deps to install a compatible transformers build."
        )

    print("Installing compatible transformers/tokenizers...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "transformers>=4.45.2,<5",
            "tokenizers>=0.20,<0.22",
        ]
    )

    # Re-exec so the current process sees upgraded packages.
    print("Dependencies updated. Restarting script process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def resolve_paths() -> Tuple[Path, Path]:
    """Return (repo_root, text_to_image_root) from script location."""
    script_path = Path(__file__).resolve()
    text_to_image_root = script_path.parent
    repo_root = text_to_image_root.parent

    if not (text_to_image_root / "fkd_diffusers").exists():
        raise FileNotFoundError(f"Expected fkd_diffusers under {text_to_image_root}")

    return repo_root, text_to_image_root


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_fkd_args(
    *,
    potential_type: str,
    num_particles: int,
    time_steps: int,
    lmbda: float,
    use_smc: bool,
    guidance_reward_fn: str,
    svgd_step_size: float,
    svgd_sigma: float | None,
) -> Dict:
    args = {
        "lmbda": lmbda,
        "use_smc": use_smc,
        "adaptive_resampling": True,
        "resample_frequency": 10,
        "resampling_t_start": 10,
        "resampling_t_end": max(10, time_steps - 5),
        "time_steps": time_steps,
        "num_particles": num_particles,
        "guidance_reward_fn": guidance_reward_fn,
        "metric_to_chase": None,
    }

    if use_smc:
        args["potential_type"] = potential_type

    if potential_type == "evolution":
        args["svgd_step_size"] = svgd_step_size
        args["svgd_sigma"] = svgd_sigma

    return args


def show_or_save_images(
    *,
    images: Sequence,
    rewards: np.ndarray,
    title: str,
    out_path: Path | None,
    show: bool,
) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=160)
    if n == 1:
        axes = [axes]

    for i, (ax, im, r) in enumerate(zip(axes, images, rewards)):
        ax.imshow(im)
        ax.set_title(f"#{i + 1} IR={r:.3f}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_mode(
    *,
    pipe,
    do_eval,
    prompt: str,
    mode: str,
    seed: int,
    num_particles: int,
    time_steps: int,
    lmbda: float,
    svgd_step_size: float,
    svgd_sigma: float | None,
    guidance_reward_fn: str,
):
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    seed_everything(seed)

    if mode == "base":
        fkd_args = build_fkd_args(
            potential_type="diff",
            num_particles=num_particles,
            time_steps=time_steps,
            lmbda=lmbda,
            use_smc=False,
            guidance_reward_fn=guidance_reward_fn,
            svgd_step_size=svgd_step_size,
            svgd_sigma=svgd_sigma,
        )
    else:
        fkd_args = build_fkd_args(
            potential_type=mode,
            num_particles=num_particles,
            time_steps=time_steps,
            lmbda=lmbda,
            use_smc=True,
            guidance_reward_fn=guidance_reward_fn,
            svgd_step_size=svgd_step_size,
            svgd_sigma=svgd_sigma,
        )

    prompts = [prompt] * num_particles
    images = pipe(
        prompts,
        num_inference_steps=time_steps,
        eta=1.0,
        fkd_args=fkd_args,
    )[0]

    results = do_eval(prompt=prompts, images=images, metrics_to_compute=["ImageReward"])
    rewards = np.array(results["ImageReward"]["result"])

    order = np.argsort(rewards)[::-1]
    images_sorted = [images[i] for i in order]
    rewards_sorted = rewards[order]

    return images_sorted, rewards_sorted, fkd_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FKD Evo steering playground script")

    p.add_argument("--model-name", default="stable-diffusion-2-1")
    p.add_argument(
        "--prompt",
        default="a cinematic photo of a corgi astronaut on mars, ultra detailed",
    )
    p.add_argument("--modes", default="base,diff,max,evolution")

    p.add_argument("--num-particles", type=int, default=4)
    p.add_argument("--time-steps", type=int, default=50)
    p.add_argument("--lmbda", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--svgd-step-size", type=float, default=0.12)
    p.add_argument("--svgd-sigma", type=float, default=None)
    p.add_argument("--guidance-reward-fn", default="ImageReward")

    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--auto-fix-deps", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("--modes is empty")
    invalid = [m for m in modes if m not in VALID_MODES]
    if invalid:
        raise ValueError(f"Invalid mode(s): {invalid}. Valid modes: {sorted(VALID_MODES)}")

    ensure_dependency_compat(auto_fix=args.auto_fix_deps)

    repo_root, text_to_image_root = resolve_paths()
    fkd_diffusers_root = text_to_image_root / "fkd_diffusers"
    sys.path.insert(0, str(text_to_image_root))
    sys.path.insert(0, str(fkd_diffusers_root))

    from fks_utils import do_eval, get_model

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (text_to_image_root / "output" / f"evo_script_{timestamp}")
    )

    print("Run config:")
    print(f"  repo_root      = {repo_root}")
    print(f"  text_to_image  = {text_to_image_root}")
    print(f"  model          = {args.model_name}")
    print(f"  modes          = {modes}")
    print(f"  device         = {device}")
    print(f"  output_dir     = {output_dir}")

    pipe = get_model(args.model_name).to(device)

    all_scores: Dict[str, Dict] = {}
    for mode in modes:
        print(f"\nRunning mode: {mode}")
        images, rewards, fkd_args = run_mode(
            pipe=pipe,
            do_eval=do_eval,
            prompt=args.prompt,
            mode=mode,
            seed=args.seed,
            num_particles=args.num_particles,
            time_steps=args.time_steps,
            lmbda=args.lmbda,
            svgd_step_size=args.svgd_step_size,
            svgd_sigma=args.svgd_sigma,
            guidance_reward_fn=args.guidance_reward_fn,
        )

        out_path = output_dir / f"{mode}.png"
        show_or_save_images(
            images=images,
            rewards=rewards,
            title=f"{args.model_name} | mode={mode} | prompt={args.prompt}",
            out_path=out_path,
            show=args.show,
        )

        all_scores[mode] = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "best": float(rewards.max()),
            "args": fkd_args,
        }

    print("\nImageReward summary:")
    for mode in modes:
        s = all_scores[mode]
        print(
            f"  {mode:10s} mean={s['mean']:.4f} std={s['std']:.4f} best={s['best']:.4f}"
        )


if __name__ == "__main__":
    main()
