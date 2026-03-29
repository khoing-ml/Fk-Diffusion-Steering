"""Archive-based Evo steering package for text-to-image pipelines."""

from .evo_class import (
    EvoGuidance,
    apply_evo_guidance,
    compute_evo_vector_field,
    quantile_good_bad_labels,
    quantile_good_bad_masks,
)

__all__ = [
    "EvoGuidance",
    "apply_evo_guidance",
    "compute_evo_vector_field",
    "quantile_good_bad_labels",
    "quantile_good_bad_masks",
    "EvoStableDiffusion",
    "EvoStableDiffusionXL",
]


def __getattr__(name: str):
    if name == "EvoStableDiffusion":
        from .evo_pipeline_sd import EvoStableDiffusion

        return EvoStableDiffusion
    if name == "EvoStableDiffusionXL":
        from .evo_pipeline_sdxl import EvoStableDiffusionXL

        return EvoStableDiffusionXL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
