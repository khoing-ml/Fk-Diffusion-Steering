"""Compatibility shim for importing the playground module from repo root."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TEXT_TO_IMAGE_ROOT = REPO_ROOT / "text_to_image"
FKD_DIFFUSERS_ROOT = TEXT_TO_IMAGE_ROOT / "fkd_diffusers"

for path in (TEXT_TO_IMAGE_ROOT, FKD_DIFFUSERS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from text_to_image.evo_steering_playground import *  # noqa: F401,F403
