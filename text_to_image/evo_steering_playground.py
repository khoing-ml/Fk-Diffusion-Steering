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
import csv
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
from PIL import Image


VALID_MODES = {"base", "diff", "max", "evolution"}
VALID_REWARD_FNS = {
    "ImageReward",
    "Clip-Score",
    "Clip-Score-only",
    "HumanPreference",
    "LLMGrader",
}


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
    resample_strategy: str,
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
        "resample_strategy": resample_strategy,
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
    resample_strategy: str,
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
            resample_strategy=resample_strategy,
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
            resample_strategy=resample_strategy,
        )

    prompts = [prompt] * num_particles
    images = pipe(
        prompts,
        num_inference_steps=time_steps,
        eta=1.0,
        fkd_args=fkd_args,
    )[0]

    metric_name = guidance_reward_fn
    if metric_name not in {"ImageReward", "Clip-Score", "HumanPreference", "LLMGrader", "Clip-Score-only"}:
        metric_name = "ImageReward"

    results = do_eval(prompt=prompts, images=images, metrics_to_compute=[metric_name])
    rewards = np.array(results[metric_name]["result"])

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
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for sweep, e.g. 1,2,3,4,5",
    )

    p.add_argument("--svgd-step-size", type=float, default=0.12)
    p.add_argument(
        "--svgd-step-sizes",
        default=None,
        help="Comma-separated evo step sizes for sweep, e.g. 0.03,0.06,0.1,0.15",
    )
    p.add_argument("--svgd-sigma", type=float, default=None)
    p.add_argument(
        "--resample-strategy",
        default="multinomial",
        choices=["multinomial", "systematic", "stratified", "residual", "none"],
        help="Particle resampling strategy for SMC/FKD updates.",
    )
    p.add_argument(
        "--resample-strategies",
        default=None,
        help=(
            "Comma-separated resampling strategies to sweep, e.g. "
            "multinomial,systematic,stratified,residual,none. "
            "If set, overrides --resample-strategy."
        ),
    )
    p.add_argument("--guidance-reward-fn", default="ImageReward")
    p.add_argument(
        "--fallback-reward-fn",
        default="Clip-Score",
        help="Fallback reward function when primary reward backend is unavailable.",
    )
    p.add_argument(
        "--disable-reward-fallback",
        action="store_true",
        help="Disable automatic fallback when guidance reward backend fails.",
    )

    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--auto-fix-deps", action="store_true")

    return p.parse_args()


def _parse_int_csv(raw: str | None, *, fallback: int) -> List[int]:
    if not raw:
        return [fallback]
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        return [fallback]
    return [int(v) for v in vals]


def _parse_float_csv(raw: str | None, *, fallback: float) -> List[float]:
    if not raw:
        return [fallback]
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        return [fallback]
    return [float(v) for v in vals]


def _parse_str_csv(raw: str | None, *, fallback: str) -> List[str]:
    if not raw:
        return [fallback]
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        return [fallback]
    return vals


def resolve_guidance_reward_fn(*, reward_fn: str, fallback_reward_fn: str, allow_fallback: bool) -> str:
    """Preflight-check reward backend and optionally fallback.

    This specifically protects against ImageReward import crashes from broken
    wandb/protobuf stacks in managed notebook runtimes.
    """
    if reward_fn not in VALID_REWARD_FNS:
        raise ValueError(f"Unknown guidance reward fn: {reward_fn}")
    if fallback_reward_fn not in VALID_REWARD_FNS:
        raise ValueError(f"Unknown fallback reward fn: {fallback_reward_fn}")

    if reward_fn != "ImageReward":
        return reward_fn

    try:
        from fkd_diffusers.rewards import do_image_reward

        # Tiny smoke test to trigger backend import early.
        _ = do_image_reward(images=[Image.new("RGB", (224, 224))], prompts=["test"])
        return reward_fn
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "ImageReward backend is unavailable and reward fallback is disabled."
            ) from exc

        print(
            "Warning: ImageReward backend failed to initialize. "
            f"Falling back to {fallback_reward_fn}."
        )
        print(f"Reason: {exc}")
        return fallback_reward_fn


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

    args.guidance_reward_fn = resolve_guidance_reward_fn(
        reward_fn=args.guidance_reward_fn,
        fallback_reward_fn=args.fallback_reward_fn,
        allow_fallback=not args.disable_reward_fallback,
    )

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

    seeds = _parse_int_csv(args.seeds, fallback=args.seed)
    svgd_step_sizes = _parse_float_csv(args.svgd_step_sizes, fallback=args.svgd_step_size)
    resample_strategies = _parse_str_csv(
        args.resample_strategies,
        fallback=args.resample_strategy,
    )
    valid_resample_strategies = {"multinomial", "systematic", "stratified", "residual", "none"}
    bad_strategies = [s for s in resample_strategies if s not in valid_resample_strategies]
    if bad_strategies:
        raise ValueError(
            f"Invalid resample strategy(ies): {bad_strategies}. "
            f"Valid options: {sorted(valid_resample_strategies)}"
        )
    print(f"  seeds          = {seeds}")
    print(f"  svgd_step_sizes= {svgd_step_sizes}")
    print(f"  resample_strategies = {resample_strategies}")

    pipe = get_model(args.model_name).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for resample_strategy in resample_strategies:
        for svgd_step_size in svgd_step_sizes:
            for seed in seeds:
                print(
                    "\n=== Sweep run: "
                    f"strategy={resample_strategy}, seed={seed}, svgd_step_size={svgd_step_size} ==="
                )
                for mode in modes:
                    print(f"Running mode: {mode}")
                    images, rewards, fkd_args = run_mode(
                        pipe=pipe,
                        do_eval=do_eval,
                        prompt=args.prompt,
                        mode=mode,
                        seed=seed,
                        num_particles=args.num_particles,
                        time_steps=args.time_steps,
                        lmbda=args.lmbda,
                        svgd_step_size=svgd_step_size,
                        svgd_sigma=args.svgd_sigma,
                        guidance_reward_fn=args.guidance_reward_fn,
                        resample_strategy=resample_strategy,
                    )

                    out_path = output_dir / (
                        f"{mode}_rs-{resample_strategy}_seed{seed}_step{svgd_step_size:.4f}.png"
                    )
                    show_or_save_images(
                        images=images,
                        rewards=rewards,
                        title=(
                            f"{args.model_name} | mode={mode} | rs={resample_strategy} | "
                            f"seed={seed} | svgd_step_size={svgd_step_size}"
                        ),
                        out_path=out_path,
                        show=args.show,
                    )

                    rows.append(
                        {
                            "mode": mode,
                            "resample_strategy": resample_strategy,
                            "seed": seed,
                            "svgd_step_size": svgd_step_size,
                            "mean": float(rewards.mean()),
                            "std": float(rewards.std()),
                            "best": float(rewards.max()),
                            "lmbda": args.lmbda,
                            "num_particles": args.num_particles,
                            "time_steps": args.time_steps,
                            "reward_fn": args.guidance_reward_fn,
                            "fkd_args": str(fkd_args),
                        }
                    )

    detail_csv = output_dir / "scores_detail.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "resample_strategy",
                "seed",
                "svgd_step_size",
                "mean",
                "std",
                "best",
                "lmbda",
                "num_particles",
                "time_steps",
                "reward_fn",
                "fkd_args",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[Tuple[str, float], List[Dict]] = {}
    for row in rows:
        key = (
            row["mode"],
            row["resample_strategy"],
            float(row["svgd_step_size"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (mode, resample_strategy, step), vals in grouped.items():
        means = np.array([v["mean"] for v in vals], dtype=np.float32)
        bests = np.array([v["best"] for v in vals], dtype=np.float32)
        summary_rows.append(
            {
                "mode": mode,
                "resample_strategy": resample_strategy,
                "svgd_step_size": step,
                "n_seeds": len(vals),
                "mean_of_mean": float(means.mean()),
                "std_of_mean": float(means.std()),
                "mean_of_best": float(bests.mean()),
            }
        )

    summary_rows.sort(key=lambda r: r["mean_of_mean"], reverse=True)

    summary_csv = output_dir / "scores_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "resample_strategy",
                "svgd_step_size",
                "n_seeds",
                "mean_of_mean",
                "std_of_mean",
                "mean_of_best",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{args.guidance_reward_fn} sweep summary (sorted by mean_of_mean):")
    for row in summary_rows:
        print(
            "  "
            f"mode={row['mode']:10s} "
            f"rs={row['resample_strategy']:11s} "
            f"step={row['svgd_step_size']:.4f} "
            f"n={row['n_seeds']} "
            f"mean={row['mean_of_mean']:.4f} "
            f"std={row['std_of_mean']:.4f} "
            f"best_mean={row['mean_of_best']:.4f}"
        )

    print(f"\nSaved detail CSV: {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
