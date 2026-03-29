import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from evo_diffusers.evo_class import EvoGuidance, compute_evo_vector_field


def _mock_reward_fn(_images):
    return torch.tensor([0.2, 0.8, 0.5, 0.9], dtype=torch.float32)


def test_evo_archive_burn_in_delays_guidance():
    evo = EvoGuidance(
        reward_fn=_mock_reward_fn,
        num_particles=4,
        guidance_frequency=1,
        update_t_start=0,
        update_t_end=10,
        archive_size=8,
        archive_good_quantile=0.75,
        archive_bad_quantile=0.25,
        archive_burn_in_steps=2,
        min_good_anchors=1,
        step_size=0.2,
        latent_to_decode_fn=lambda x: x,
        device=torch.device("cpu"),
    )

    latents = torch.randn(4, 3, 8, 8)
    x0_preds = torch.randn(4, 3, 8, 8)

    warmup_latents, warmup_images = evo.step(
        sampling_idx=0,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert torch.allclose(warmup_latents, latents)
    assert warmup_images is not None
    assert evo.archive_good_anchors is not None

    guided_latents, guided_images = evo.step(
        sampling_idx=2,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert not torch.allclose(guided_latents, latents)
    assert guided_images is not None


def test_evo_guidance_respects_cadence():
    evo = EvoGuidance(
        reward_fn=_mock_reward_fn,
        num_particles=4,
        guidance_frequency=3,
        update_t_start=0,
        update_t_end=10,
        archive_size=8,
        archive_burn_in_steps=0,
        min_good_anchors=1,
        step_size=0.15,
        latent_to_decode_fn=lambda x: x,
        device=torch.device("cpu"),
    )

    latents = torch.randn(4, 3, 8, 8)
    x0_preds = torch.randn(4, 3, 8, 8)

    skipped_latents, skipped_images = evo.step(
        sampling_idx=1,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert torch.allclose(skipped_latents, latents)
    assert skipped_images is not None

    guided_latents, _ = evo.step(
        sampling_idx=3,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert not torch.allclose(guided_latents, latents)


def test_evo_skips_when_not_enough_good_anchors():
    evo = EvoGuidance(
        reward_fn=_mock_reward_fn,
        num_particles=4,
        guidance_frequency=1,
        update_t_start=0,
        update_t_end=10,
        archive_size=4,
        archive_burn_in_steps=0,
        min_good_anchors=10,
        step_size=0.2,
        latent_to_decode_fn=lambda x: x,
        device=torch.device("cpu"),
    )

    latents = torch.randn(4, 3, 8, 8)
    x0_preds = torch.randn(4, 3, 8, 8)

    updated_latents, decoded_images = evo.step(
        sampling_idx=0,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert torch.allclose(updated_latents, latents)
    assert decoded_images is not None


def test_evo_falls_back_to_good_only_when_bad_archive_is_missing():
    evo = EvoGuidance(
        reward_fn=_mock_reward_fn,
        num_particles=4,
        guidance_frequency=1,
        update_t_start=0,
        update_t_end=10,
        archive_size=8,
        archive_burn_in_steps=0,
        min_good_anchors=1,
        min_bad_anchors=10,
        bad_guidance_strength=1.0,
        step_size=0.2,
        latent_to_decode_fn=lambda x: x,
        device=torch.device("cpu"),
    )

    latents = torch.randn(4, 3, 8, 8)
    x0_preds = torch.randn(4, 3, 8, 8)

    updated_latents, _ = evo.step(
        sampling_idx=0,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert not torch.allclose(updated_latents, latents)


def test_evo_does_not_run_outside_update_window():
    evo = EvoGuidance(
        reward_fn=_mock_reward_fn,
        num_particles=4,
        guidance_frequency=1,
        update_t_start=2,
        update_t_end=4,
        archive_size=8,
        step_size=0.2,
        latent_to_decode_fn=lambda x: x,
        device=torch.device("cpu"),
    )

    latents = torch.randn(4, 3, 8, 8)
    x0_preds = torch.randn(4, 3, 8, 8)

    updated_latents, decoded_images = evo.step(
        sampling_idx=0,
        latents=latents.clone(),
        x0_preds=x0_preds.clone(),
        alpha_bar_t=0.7,
    )
    assert torch.allclose(updated_latents, latents)
    assert decoded_images is None
    assert evo.archive_good_anchors is None


def test_evo_vector_field_has_expected_shape_and_is_finite():
    particles = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.2, 0.1],
            [0.1, 0.3],
        ],
        dtype=torch.float32,
    )
    good_anchors = torch.tensor(
        [
            [3.0, 3.0],
            [2.8, 3.2],
        ],
        dtype=torch.float32,
    )
    bad_anchors = torch.tensor(
        [
            [-3.0, -3.0],
            [-2.8, -3.1],
        ],
        dtype=torch.float32,
    )

    field = compute_evo_vector_field(
        particles=particles,
        good_anchor_x0=good_anchors,
        bad_anchor_x0=bad_anchors,
        alpha_bar_t=0.7,
        bad_guidance_strength=0.5,
        device=torch.device("cpu"),
    )

    assert field.shape == particles.shape
    assert torch.isfinite(field).all()
    assert field.mean().item() > 0
