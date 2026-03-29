import pathlib
import sys

import torch
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from fkd_diffusers.fkd_class import (
    FKD,
    PotentialType,
    evolution_steering_binary_rewards,
    compute_svgd_vector_field,
    apply_svgd_steering,
    _rbf_kernel,
    _median_pairwise_distance,
)

def test_evolution_steering():
    """
    Test the evolution steering mechanism.
    """
    # Mock reward function
    def mock_reward_fn(images):
        return torch.tensor([0.2, 0.8, 0.5, 0.9])

    # Initialize FKD instance
    fkd = FKD(
        potential_type=PotentialType.EVOLUTION,
        lmbda=1.0,
        num_particles=4,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=0,
        resampling_t_end=10,
        time_steps=10,
        reward_fn=mock_reward_fn,
        device=torch.device("cpu"),
    )

    # Mock latents and predictions
    latents = torch.randn(4, 3, 64, 64)  # 4 particles, 3 channels, 64x64 resolution
    x0_preds = torch.randn(4, 3, 64, 64)

    # Perform resampling
    resampled_latents, resampled_images = fkd.resample(
        sampling_idx=5, latents=latents, x0_preds=x0_preds
    )

    # Assert resampled latents shape
    assert resampled_latents.shape == latents.shape, "Resampled latents shape mismatch"
    
    # For evolution steering, optional images may be returned by decoder path.
    # In this unit test latent_to_decode_fn is identity, so output shape should match.
    assert resampled_images is not None, "Evolution steering should return decoded/resampled images"
    assert resampled_images.shape == x0_preds.shape, "Resampled image tensor shape mismatch"

    # Test binary rewards
    rewards = [0.2, 0.8, 0.5, 0.9]
    binary_rewards = evolution_steering_binary_rewards(
        rewards=rewards, threshold_fn=lambda r: np.percentile(r, 50)
    )
    assert binary_rewards == [0, 1, 0, 1], "Binary rewards mismatch"


def test_evolution_guidance_frequency():
    """
    Evolution mode should only apply SVGD on the configured guidance cadence.
    """
    def mock_reward_fn(images):
        return torch.tensor([0.2, 0.8, 0.5, 0.9])

    fkd = FKD(
        potential_type=PotentialType.EVOLUTION,
        lmbda=1.0,
        num_particles=4,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=0,
        resampling_t_end=10,
        time_steps=10,
        reward_fn=mock_reward_fn,
        guidance_frequency=3,
        resample_strategy="none",
        device=torch.device("cpu"),
    )

    latents = torch.randn(4, 3, 16, 16)
    x0_preds = torch.randn(4, 3, 16, 16)

    skipped_latents, skipped_images = fkd.resample(
        sampling_idx=1, latents=latents.clone(), x0_preds=x0_preds.clone()
    )
    assert torch.allclose(skipped_latents, latents), "Latents should be unchanged off guidance cadence"
    assert skipped_images is None, "No decoded images should be produced when evolution update is skipped"

    guided_latents, guided_images = fkd.resample(
        sampling_idx=3, latents=latents.clone(), x0_preds=x0_preds.clone()
    )
    assert not torch.allclose(guided_latents, latents), "Latents should change on guidance cadence"
    assert guided_images is not None, "Decoded images should be returned when guidance runs"


def test_rbf_kernel():
    """
    Test RBF kernel computation.
    """
    # Create simple test points
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    
    kernel = _rbf_kernel(x, y, sigma=1.0)
    
    # Check shape
    assert kernel.shape == (2, 2), "Kernel shape mismatch"
    
    # Diagonal should be close to 1 (distance 0)
    assert torch.allclose(torch.diag(kernel), torch.ones(2), atol=1e-5), "Diagonal values should be ~1"
    
    # Kernel should be symmetric
    assert torch.allclose(kernel, kernel.T, atol=1e-5), "Kernel should be symmetric"
    
    print("RBF kernel test passed.")


def test_median_pairwise_distance():
    """
    Test median pairwise distance computation for bandwidth selection.
    """
    # Create simple grid of points
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    sigma = _median_pairwise_distance(x)
    
    # Sigma should be positive
    assert sigma > 0, "Sigma should be positive"
    
    # Sigma should be reasonable (between 0 and 2 for unit square)
    assert 0 < sigma < 2, "Sigma out of reasonable range"
    
    print(f"Median pairwise distance: {sigma:.4f}")
    print("Median pairwise distance test passed.")


def test_svgd_vector_field():
    """
    Test SVGD vector field computation.
    """
    device = torch.device("cpu")
    num_particles = 8
    particle_dim = 2
    
    # Create simple particle distribution
    # Bad particles: clustered around (0, 0)
    # Good particles: clustered around (5, 5)
    bad_particles = torch.randn(num_particles // 2, particle_dim) * 0.5
    good_particles = torch.randn(num_particles // 2, particle_dim) * 0.5 + 5.0
    particles = torch.cat([bad_particles, good_particles], dim=0)
    
    # Binary rewards: first half bad (0), second half good (1)
    binary_rewards = [0] * (num_particles // 2) + [1] * (num_particles // 2)
    
    # Compute vector field
    vector_field = compute_svgd_vector_field(
        particles=particles,
        binary_rewards=binary_rewards,
        sigma=None,  # Auto-select sigma
        device=device,
    )
    
    # Check shape
    assert vector_field.shape == particles.shape, "Vector field shape mismatch"
    
    # For bad particles, vector field should point towards good region (positive on average)
    bad_field = vector_field[:num_particles // 2]
    good_field = vector_field[num_particles // 2:]
    
    # Bad particles should have non-zero vector field (pointing towards good)
    bad_field_magnitude = torch.norm(bad_field, dim=1)
    assert bad_field_magnitude.mean() > 0, "Bad particles should have steering velocity"
    
    # Good particles should have near-zero vector field
    good_field_magnitude = torch.norm(good_field, dim=1)
    assert good_field_magnitude.mean() < bad_field_magnitude.mean(), "Good particles should have less steering"
    
    print("SVGD vector field test passed.")


def test_svgd_vector_field_mixture_score():
    """
    Test score-based SVGD path using q(x_t | x0) mixture approximation.
    """
    device = torch.device("cpu")
    num_particles = 10
    dim = 4

    # Construct two clusters in x0 space: bad near 0, good near +4.
    bad_x0 = torch.randn(num_particles // 2, dim) * 0.2
    good_x0 = torch.randn(num_particles // 2, dim) * 0.2 + 4.0
    x0_particles = torch.cat([bad_x0, good_x0], dim=0)

    # Build x_t by forward noising from corresponding x0.
    alpha_bar_t = 0.7
    noise_scale = np.sqrt(max(1.0 - alpha_bar_t, 1e-6))
    xt_particles = np.sqrt(alpha_bar_t) * x0_particles + noise_scale * torch.randn_like(x0_particles)

    binary_rewards = [0] * (num_particles // 2) + [1] * (num_particles // 2)

    field = compute_svgd_vector_field(
        particles=xt_particles,
        binary_rewards=binary_rewards,
        x0_particles=x0_particles,
        alpha_bar_t=alpha_bar_t,
        sigma=None,
        device=device,
    )

    assert field.shape == xt_particles.shape, "Mixture-score field shape mismatch"

    steered = apply_svgd_steering(
        particles=xt_particles,
        binary_rewards=binary_rewards,
        x0_particles=x0_particles,
        alpha_bar_t=alpha_bar_t,
        step_size=0.1,
        sigma=None,
        device=device,
    )
    assert steered.shape == xt_particles.shape, "Mixture-score steering shape mismatch"

    # Bad particles should become closer to good x_t center on average.
    good_center = xt_particles[num_particles // 2:].mean(dim=0)
    bad_before = xt_particles[: num_particles // 2]
    bad_after = steered[: num_particles // 2]
    dist_before = torch.norm(bad_before - good_center, dim=1).mean()
    dist_after = torch.norm(bad_after - good_center, dim=1).mean()
    assert dist_after <= dist_before + 1e-5, "Mixture-score SVGD should move bad particles toward good region"

    print("SVGD mixture-score test passed.")


def test_svgd_vector_field_dedupes_good_anchors():
    """
    Duplicated good anchors from resampling should not change the mixture score.
    """
    device = torch.device("cpu")
    xt_particles = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ]
    )
    good_anchor = torch.tensor([[2.0, 2.0]])
    duplicated_good_anchors = torch.cat([good_anchor, good_anchor], dim=0)

    field_single = compute_svgd_vector_field(
        particles=xt_particles,
        binary_rewards=[0, 0, 1],
        good_anchor_x0=good_anchor,
        alpha_bar_t=0.8,
        sigma=None,
        device=device,
    )
    field_dup = compute_svgd_vector_field(
        particles=xt_particles,
        binary_rewards=[0, 0, 1],
        good_anchor_x0=duplicated_good_anchors,
        alpha_bar_t=0.8,
        sigma=None,
        device=device,
    )

    assert torch.allclose(field_single, field_dup, atol=1e-5), (
        "Duplicated anchors should not overweight the mixture-score target"
    )

    print("SVGD good-anchor dedupe test passed.")


def test_apply_svgd_steering():
    """
    Test SVGD steering application to particles.
    """
    device = torch.device("cpu")
    num_particles = 8
    particle_dim = 2
    
    # Create simple particle distribution
    bad_particles = torch.randn(num_particles // 2, particle_dim) * 0.5
    good_particles = torch.randn(num_particles // 2, particle_dim) * 0.5 + 5.0
    particles = torch.cat([bad_particles, good_particles], dim=0)
    
    binary_rewards = [0] * (num_particles // 2) + [1] * (num_particles // 2)
    
    # Apply SVGD steering
    steered_particles = apply_svgd_steering(
        particles=particles,
        binary_rewards=binary_rewards,
        step_size=0.1,
        sigma=None,
        device=device,
    )
    
    # Check shape
    assert steered_particles.shape == particles.shape, "Steered particles shape mismatch"
    
    # Bad particles should move towards good region
    bad_particles_orig = particles[:num_particles // 2]
    bad_particles_steered = steered_particles[:num_particles // 2]
    
    # Distance to good region should decrease on average
    good_particles_center = particles[num_particles // 2:].mean(dim=0)
    
    dist_before = torch.norm(bad_particles_orig - good_particles_center, dim=1).mean()
    dist_after = torch.norm(bad_particles_steered - good_particles_center, dim=1).mean()
    
    # Check if steering moved bad particles closer to good region
    # (This is a probabilistic check, so we just verify it didn't move them further)
    assert dist_after <= dist_before + 1e-5, "Steering should move particles towards good region"
    
    print(f"Distance before steering: {dist_before:.4f}")
    print(f"Distance after steering: {dist_after:.4f}")
    print("SVGD steering test passed.")


if __name__ == "__main__":
    test_evolution_steering()
    print("✓ Evolution steering test passed.")
    
    test_rbf_kernel()
    print("✓ RBF kernel test passed.")
    
    test_median_pairwise_distance()
    print("✓ Median pairwise distance test passed.")
    
    test_svgd_vector_field()
    print("✓ SVGD vector field test passed.")

    test_svgd_vector_field_mixture_score()
    print("✓ SVGD mixture-score test passed.")

    test_svgd_vector_field_dedupes_good_anchors()
    print("✓ SVGD good-anchor dedupe test passed.")
    
    test_apply_svgd_steering()
    print("✓ SVGD steering test passed.")
    
    print("\n✅ All evolution steering tests passed!")
