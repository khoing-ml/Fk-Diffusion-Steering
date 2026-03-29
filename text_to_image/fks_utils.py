"""
Utility functions for the FKD pipeline.
"""
import torch
from diffusers import DDIMScheduler

try:
    # Preferred when `text_to_image` is on sys.path.
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    from evo_diffusers.evo_pipeline_sdxl import EvoStableDiffusionXL
    from evo_diffusers.evo_pipeline_sd import EvoStableDiffusion
except ModuleNotFoundError:
    # Backward-compatible fallback for older layouts.
    from fkd_pipeline_sdxl import FKDStableDiffusionXL
    from fkd_pipeline_sd import FKDStableDiffusion
    try:
        from evo_diffusers.evo_pipeline_sdxl import EvoStableDiffusionXL
        from evo_diffusers.evo_pipeline_sd import EvoStableDiffusion
    except ModuleNotFoundError:
        from evo_pipeline_sdxl import EvoStableDiffusionXL
        from evo_pipeline_sd import EvoStableDiffusion

from fkd_diffusers.rewards import (
    do_clip_score,
    do_clip_score_diversity,
    do_image_reward,
    do_human_preference_score,
    do_llm_grading
)


def get_model(model_name, *, pipeline_family="fkd"):
    """
    Get a steering-aware pipeline based on the model name and wrapper family.
    """
    if pipeline_family == "fkd":
        sdxl_cls = FKDStableDiffusionXL
        sd_cls = FKDStableDiffusion
    elif pipeline_family == "evo":
        sdxl_cls = EvoStableDiffusionXL
        sd_cls = EvoStableDiffusion
    else:
        raise ValueError(f"Unknown pipeline family: {pipeline_family}")

    if model_name == "stable-diffusion-xl":
        pipeline = sdxl_cls.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        )
    elif model_name == "stable-diffusion-v1-5":
        pipeline = sd_cls.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        )
    elif model_name == "stable-diffusion-v1-4":
        pipeline = sd_cls.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        )
    elif model_name == "stable-diffusion-2-1":
        pipeline = sd_cls.from_pretrained(
            "sd2-community/stable-diffusion-2-1",
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    return pipeline



def do_eval(*, prompt, images, metrics_to_compute):
    """
    Compute the metrics for the given images and prompt.
    """
    results = {}
    for metric in metrics_to_compute:
        if metric == "Clip-Score":
            results[metric] = {}
            (
                results[metric]["result"],
                results[metric]["diversity"],
            ) = do_clip_score_diversity(images=images, prompts=prompt)
            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "ImageReward":
            results[metric] = {}
            results[metric]["result"] = do_image_reward(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "Clip-Score-only":
            results[metric] = {}
            results[metric]["result"] = do_clip_score(images=images, prompts=prompt)

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()
        elif metric == "HumanPreference":
            results[metric] = {}
            results[metric]["result"] = do_human_preference_score(
                images=images, prompts=prompt
            )

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        elif metric == "LLMGrader":
            results[metric] = {}
            out = do_llm_grading(images=images, prompts=prompt)
            print(out)
            results[metric]["result"] = out

            results_arr = torch.tensor(results[metric]["result"])

            results[metric]["mean"] = results_arr.mean().item()
            results[metric]["std"] = results_arr.std().item()
            results[metric]["max"] = results_arr.max().item()
            results[metric]["min"] = results_arr.min().item()

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results
