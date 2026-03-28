import torch
import numpy as np
from fkd_diffusers.fkd_class import FKD, PotentialType
from discrete_diffusion.fk_diffusion import evolution_steering_binary_rewards

def test_evolution_steering():
    """
    Test the evolution steering mechanism.
    """
    # Mock reward function
    def mock_reward_fn(images):
        return torch.tensor([0.2, 0.8, 0.5, 0.9])

    # Initialize FKD instance
    fkd = FKD(
        potential_type=PotentialType.MAX,
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
    resampled_latents, _ = fkd.resample(
        sampling_idx=5, latents=latents, x0_preds=x0_preds
    )

    # Assert resampled latents shape
    assert resampled_latents.shape == latents.shape, "Resampled latents shape mismatch"

    # Test binary rewards
    rewards = [0.2, 0.8, 0.5, 0.9]
    binary_rewards = evolution_steering_binary_rewards(
        rewards=rewards, threshold_fn=lambda r: np.percentile(r, 50)
    )
    assert binary_rewards == [0, 1, 0, 1], "Binary rewards mismatch"

if __name__ == "__main__":
    test_evolution_steering()
    print("Evolution steering test passed.")