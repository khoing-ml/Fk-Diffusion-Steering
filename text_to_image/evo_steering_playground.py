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


def _require_numpy():
    import numpy as np

    return np


def _require_torch():
    import torch

    return torch


def _require_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _require_pil_image():
    from PIL import Image

    return Image


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
    np = _require_numpy()
    torch = _require_torch()
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
    update_t_start: int,
    update_t_end: int,
    lmbda: float,
    use_smc: bool,
    guidance_reward_fn: str,
    svgd_step_size: float,
    svgd_sigma: float | None,
    guidance_frequency: int | None,
    use_anchor_archive: bool,
    archive_size: int,
    archive_good_quantile: float,
    archive_bad_quantile: float,
    archive_burn_in_steps: int,
    min_good_anchors: int,
    min_bad_anchors: int,
    bad_guidance_strength: float,
    resample_strategy: str,
) -> Dict:
    args = {
        "lmbda": lmbda,
        "use_smc": use_smc,
        "adaptive_resampling": True,
        "resample_frequency": 10,
        "resampling_t_start": update_t_start,
        "resampling_t_end": update_t_end,
        "time_steps": time_steps,
        "num_particles": num_particles,
        "guidance_reward_fn": guidance_reward_fn,
        "metric_to_chase": None,
        "resample_strategy": resample_strategy,
    }

    if use_smc:
        args["potential_type"] = potential_type

    if potential_type == "evolution":
        effective_guidance_frequency = (
            args["resample_frequency"]
            if guidance_frequency is None
            else guidance_frequency
        )
        args["svgd_step_size"] = svgd_step_size
        args["svgd_sigma"] = svgd_sigma
        args["guidance_frequency"] = effective_guidance_frequency
        args["use_anchor_archive"] = use_anchor_archive
        args["archive_size"] = archive_size
        args["archive_good_quantile"] = archive_good_quantile
        args["archive_bad_quantile"] = archive_bad_quantile
        args["archive_burn_in_steps"] = archive_burn_in_steps
        args["min_good_anchors"] = min_good_anchors
        args["min_bad_anchors"] = min_bad_anchors
        args["bad_guidance_strength"] = bad_guidance_strength

    return args


def build_evo_args(
    *,
    num_particles: int,
    update_t_start: int,
    update_t_end: int,
    guidance_reward_fn: str,
    step_size: float,
    sigma: float | None,
    guidance_frequency: int | None,
    archive_size: int,
    archive_good_quantile: float,
    archive_bad_quantile: float,
    archive_burn_in_steps: int,
    min_good_anchors: int,
    min_bad_anchors: int,
    bad_guidance_strength: float,
) -> Dict:
    effective_guidance_frequency = 10 if guidance_frequency is None else guidance_frequency
    return {
        "num_particles": num_particles,
        "update_t_start": update_t_start,
        "update_t_end": update_t_end,
        "guidance_reward_fn": guidance_reward_fn,
        "step_size": step_size,
        "sigma": sigma,
        "guidance_frequency": effective_guidance_frequency,
        "archive_size": archive_size,
        "archive_good_quantile": archive_good_quantile,
        "archive_bad_quantile": archive_bad_quantile,
        "archive_burn_in_steps": archive_burn_in_steps,
        "min_good_anchors": min_good_anchors,
        "min_bad_anchors": min_bad_anchors,
        "bad_guidance_strength": bad_guidance_strength,
    }


def show_or_save_images(
    *,
    images: Sequence,
    rewards: np.ndarray,
    title: str,
    out_path: Path | None,
    show: bool,
) -> None:
    plt = _require_pyplot()
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
    update_t_start: int,
    update_t_end: int,
    lmbda: float,
    svgd_step_size: float,
    svgd_sigma: float | None,
    guidance_frequency: int | None,
    use_anchor_archive: bool,
    archive_size: int,
    archive_good_quantile: float,
    archive_bad_quantile: float,
    archive_burn_in_steps: int,
    min_good_anchors: int,
    min_bad_anchors: int,
    bad_guidance_strength: float,
    guidance_reward_fn: str,
    resample_strategy: str,
    evolution_resample_strategy: str,
):
    np = _require_numpy()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    seed_everything(seed)

    effective_resample_strategy = (
        evolution_resample_strategy if mode == "evolution" else resample_strategy
    )

    if mode == "evolution":
        steering_args = build_evo_args(
            num_particles=num_particles,
            update_t_start=update_t_start,
            update_t_end=update_t_end,
            guidance_reward_fn=guidance_reward_fn,
            step_size=svgd_step_size,
            sigma=svgd_sigma,
            guidance_frequency=guidance_frequency,
            archive_size=archive_size,
            archive_good_quantile=archive_good_quantile,
            archive_bad_quantile=archive_bad_quantile,
            archive_burn_in_steps=archive_burn_in_steps,
            min_good_anchors=min_good_anchors,
            min_bad_anchors=min_bad_anchors,
            bad_guidance_strength=bad_guidance_strength,
        )
        pipe_kwargs = {"evo_args": steering_args}
    elif mode == "base":
        fkd_args = build_fkd_args(
            potential_type="diff",
            num_particles=num_particles,
            time_steps=time_steps,
            update_t_start=update_t_start,
            update_t_end=update_t_end,
            lmbda=lmbda,
            use_smc=False,
            guidance_reward_fn=guidance_reward_fn,
            svgd_step_size=svgd_step_size,
            svgd_sigma=svgd_sigma,
            guidance_frequency=guidance_frequency,
            use_anchor_archive=use_anchor_archive,
            archive_size=archive_size,
            archive_good_quantile=archive_good_quantile,
            archive_bad_quantile=archive_bad_quantile,
            archive_burn_in_steps=archive_burn_in_steps,
            min_good_anchors=min_good_anchors,
            min_bad_anchors=min_bad_anchors,
            bad_guidance_strength=bad_guidance_strength,
            resample_strategy=effective_resample_strategy,
        )
        steering_args = fkd_args
        pipe_kwargs = {"fkd_args": steering_args}
    else:
        fkd_args = build_fkd_args(
            potential_type=mode,
            num_particles=num_particles,
            time_steps=time_steps,
            update_t_start=update_t_start,
            update_t_end=update_t_end,
            lmbda=lmbda,
            use_smc=True,
            guidance_reward_fn=guidance_reward_fn,
            svgd_step_size=svgd_step_size,
            svgd_sigma=svgd_sigma,
            guidance_frequency=guidance_frequency,
            use_anchor_archive=use_anchor_archive,
            archive_size=archive_size,
            archive_good_quantile=archive_good_quantile,
            archive_bad_quantile=archive_bad_quantile,
            archive_burn_in_steps=archive_burn_in_steps,
            min_good_anchors=min_good_anchors,
            min_bad_anchors=min_bad_anchors,
            bad_guidance_strength=bad_guidance_strength,
            resample_strategy=effective_resample_strategy,
        )
        steering_args = fkd_args
        pipe_kwargs = {"fkd_args": steering_args}

    prompts = [prompt] * num_particles
    images = pipe(
        prompts,
        num_inference_steps=time_steps,
        eta=1.0,
        **pipe_kwargs,
    )[0]

    metric_name = guidance_reward_fn
    if metric_name not in {"ImageReward", "Clip-Score", "HumanPreference", "LLMGrader", "Clip-Score-only"}:
        metric_name = "ImageReward"

    results = do_eval(prompt=prompts, images=images, metrics_to_compute=[metric_name])
    rewards = np.array(results[metric_name]["result"])

    order = np.argsort(rewards)[::-1]
    images_sorted = [images[i] for i in order]
    rewards_sorted = rewards[order]

    return images_sorted, rewards_sorted, steering_args


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
    p.add_argument("--update-t-start", type=int, default=10)
    p.add_argument(
        "--update-t-end",
        type=int,
        default=None,
        help="Last timestep index to allow FKD/Evo updates. Defaults to time_steps - 10.",
    )
    p.add_argument("--lmbda", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for sweep, e.g. 1,2,3,4,5",
    )

    p.add_argument("--svgd-step-size", type=float, default=0.12)
    p.add_argument(
        "--guidance-frequency",
        type=int,
        default=None,
        help=(
            "Frequency for SVGD guidance in evolution mode. "
            "If omitted, evolution guidance follows --resample-frequency's effective cadence "
            "(currently the built-in default of 10 in this script)."
        ),
    )
    p.add_argument(
        "--use-anchor-archive",
        action="store_true",
        help=(
            "Legacy FKD evolution option. The new `evo_diffusers` path is "
            "archive-based by design, so this flag is ignored there."
        ),
    )
    p.add_argument(
        "--archive-size",
        type=int,
        default=64,
        help="Maximum number of archived good anchors and bad anchors to retain.",
    )
    p.add_argument(
        "--archive-good-quantile",
        type=float,
        default=0.75,
        help="Reward quantile for admitting good anchors into the archive.",
    )
    p.add_argument(
        "--archive-bad-quantile",
        type=float,
        default=0.25,
        help="Reward quantile for admitting bad anchors into the archive.",
    )
    p.add_argument(
        "--archive-burn-in-steps",
        type=int,
        default=0,
        help="Number of early evolution steps used only for archive collection.",
    )
    p.add_argument(
        "--min-good-anchors",
        type=int,
        default=8,
        help="Minimum archived good anchors required before archive guidance starts.",
    )
    p.add_argument(
        "--min-bad-anchors",
        type=int,
        default=0,
        help="Minimum archived bad anchors required before contrastive bad guidance starts.",
    )
    p.add_argument(
        "--bad-guidance-strength",
        type=float,
        default=0.0,
        help="Strength of bad-anchor repulsion in archive-based evolution guidance.",
    )
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
        "--evolution-resample-strategy",
        default="none",
        choices=["multinomial", "systematic", "stratified", "residual", "none"],
        help=(
            "Legacy FKD evolution resampling strategy. The new `evo_diffusers` "
            "module does not resample and keeps this option only for backward compatibility."
        ),
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
        Image = _require_pil_image()
        from evo_diffusers.rewards import do_image_reward

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
    np = _require_numpy()
    torch = _require_torch()
    args = parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("--modes is empty")
    invalid = [m for m in modes if m not in VALID_MODES]
    if invalid:
        raise ValueError(f"Invalid mode(s): {invalid}. Valid modes: {sorted(VALID_MODES)}")
    if args.archive_size <= 0:
        raise ValueError("--archive-size must be > 0")
    if not (0.0 <= args.archive_bad_quantile < args.archive_good_quantile <= 1.0):
        raise ValueError(
            "Expected 0 <= --archive-bad-quantile < --archive-good-quantile <= 1"
        )
    if args.archive_burn_in_steps < 0:
        raise ValueError("--archive-burn-in-steps must be >= 0")
    if args.min_good_anchors < 0 or args.min_bad_anchors < 0:
        raise ValueError("--min-good-anchors and --min-bad-anchors must be >= 0")
    if args.bad_guidance_strength < 0:
        raise ValueError("--bad-guidance-strength must be >= 0")

    effective_update_t_end = (
        max(args.update_t_start, args.time_steps - 10)
        if args.update_t_end is None
        else args.update_t_end
    )
    if args.update_t_start < 0:
        raise ValueError("--update-t-start must be >= 0")
    if effective_update_t_end < args.update_t_start:
        raise ValueError("--update-t-end must be >= --update-t-start")

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
    print(f"  update_window  = [{args.update_t_start}, {effective_update_t_end}]")

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
    print(f"  guidance_frequency = {args.guidance_frequency}")
    print(f"  use_anchor_archive = {args.use_anchor_archive}")
    print(f"  archive_size   = {args.archive_size}")
    print(f"  archive_good_q = {args.archive_good_quantile}")
    print(f"  archive_bad_q  = {args.archive_bad_quantile}")
    print(f"  archive_burn_in_steps = {args.archive_burn_in_steps}")
    print(f"  min_good_anchors = {args.min_good_anchors}")
    print(f"  min_bad_anchors  = {args.min_bad_anchors}")
    print(f"  bad_guidance_strength = {args.bad_guidance_strength}")
    print(f"  resample_strategies = {resample_strategies}")
    print(f"  evolution_resample_strategy = {args.evolution_resample_strategy}")

    pipe = get_model(args.model_name, pipeline_family="evo").to(device)

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
                    images, rewards, steering_args = run_mode(
                        pipe=pipe,
                        do_eval=do_eval,
                        prompt=args.prompt,
                        mode=mode,
                        seed=seed,
                        num_particles=args.num_particles,
                        time_steps=args.time_steps,
                        update_t_start=args.update_t_start,
                        update_t_end=effective_update_t_end,
                        lmbda=args.lmbda,
                        svgd_step_size=svgd_step_size,
                        svgd_sigma=args.svgd_sigma,
                        guidance_frequency=args.guidance_frequency,
                        use_anchor_archive=args.use_anchor_archive,
                        archive_size=args.archive_size,
                        archive_good_quantile=args.archive_good_quantile,
                        archive_bad_quantile=args.archive_bad_quantile,
                        archive_burn_in_steps=args.archive_burn_in_steps,
                        min_good_anchors=args.min_good_anchors,
                        min_bad_anchors=args.min_bad_anchors,
                        bad_guidance_strength=args.bad_guidance_strength,
                        guidance_reward_fn=args.guidance_reward_fn,
                        resample_strategy=resample_strategy,
                        evolution_resample_strategy=args.evolution_resample_strategy,
                    )

                    effective_resample_strategy = steering_args.get("resample_strategy", "none")
                    guidance_tag = ""
                    if mode == "evolution":
                        guidance_value = steering_args.get("guidance_frequency")
                        archive_tag = "_archive"
                        guidance_tag = f"_gfreq{guidance_value}{archive_tag}"
                    out_path = output_dir / (
                        f"{mode}_rs-{effective_resample_strategy}{guidance_tag}_seed{seed}_step{svgd_step_size:.4f}.png"
                    )
                    uses_archive = mode == "evolution" or steering_args.get("use_anchor_archive", False)
                    show_or_save_images(
                        images=images,
                        rewards=rewards,
                        title=(
                            f"{args.model_name} | mode={mode} | rs={effective_resample_strategy} | "
                            f"seed={seed} | svgd_step_size={svgd_step_size} | "
                            f"guidance_frequency={steering_args.get('guidance_frequency')} | "
                            f"archive={uses_archive}"
                        ),
                        out_path=out_path,
                        show=args.show,
                    )

                    rows.append(
                        {
                            "mode": mode,
                            "resample_strategy": effective_resample_strategy,
                            "seed": seed,
                            "svgd_step_size": svgd_step_size,
                            "guidance_frequency": steering_args.get("guidance_frequency"),
                            "use_anchor_archive": uses_archive,
                            "mean": float(rewards.mean()),
                            "std": float(rewards.std()),
                            "best": float(rewards.max()),
                            "lmbda": args.lmbda,
                            "num_particles": args.num_particles,
                            "time_steps": args.time_steps,
                            "reward_fn": args.guidance_reward_fn,
                            "steering_args": str(steering_args),
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
                "guidance_frequency",
                "use_anchor_archive",
                "mean",
                "std",
                "best",
                "lmbda",
                "num_particles",
                "time_steps",
                "reward_fn",
                "steering_args",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[Tuple[str, str, float, int | None, bool], List[Dict]] = {}
    for row in rows:
        key = (
            row["mode"],
            row["resample_strategy"],
            float(row["svgd_step_size"]),
            row["guidance_frequency"],
            bool(row["use_anchor_archive"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (mode, resample_strategy, step, guidance_frequency, use_anchor_archive), vals in grouped.items():
        means = np.array([v["mean"] for v in vals], dtype=np.float32)
        bests = np.array([v["best"] for v in vals], dtype=np.float32)
        summary_rows.append(
            {
                "mode": mode,
                "resample_strategy": resample_strategy,
                "svgd_step_size": step,
                "guidance_frequency": guidance_frequency,
                "use_anchor_archive": use_anchor_archive,
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
                "guidance_frequency",
                "use_anchor_archive",
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
            f"gfreq={row['guidance_frequency']} "
            f"archive={row['use_anchor_archive']} "
            f"n={row['n_seeds']} "
            f"mean={row['mean_of_mean']:.4f} "
            f"std={row['std_of_mean']:.4f} "
            f"best_mean={row['mean_of_best']:.4f}"
        )

    print(f"\nSaved detail CSV: {detail_csv}")
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
