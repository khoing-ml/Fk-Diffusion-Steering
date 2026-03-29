"""Evo-side shim for ImageReward loading."""

try:
    from fkd_diffusers.image_reward_utils import rm_load
except ModuleNotFoundError:
    from text_to_image.fkd_diffusers.image_reward_utils import rm_load

__all__ = ["rm_load"]
