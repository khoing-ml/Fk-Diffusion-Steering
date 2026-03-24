import os
import sys

# Ensure sibling package imports work when this file is run as a script.
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TEST_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from fkd_diffusers.image_reward_utils import rm_load
from fkd_diffusers.rewards import do_image_reward
import torch



# path to image for test 
image_path = "test/blue_knife.png"
# prompt = "A photo of a blue cat sitting on a red windowsill."
prompt = "A brown knife with blue donut"
# add noise to image to test the reward model's robustness
from PIL import Image
import numpy as np
def add_noise_to_image(image_path, noise_level=0.1):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, image_array.shape)
    noisy_image_array = np.clip(image_array + noise, 0, 1) * 255
    noisy_image = Image.fromarray(noisy_image_array.astype(np.uint8))
    return noisy_image
noisy_image = add_noise_to_image(image_path, 0.5)
noisy_image_path = "test/noisy_blue_knife.png"
noisy_image.save(noisy_image_path)
model = rm_load("ImageReward-v1.0")

with torch.no_grad():
    image_reward_result = model.score(prompt, noisy_image_path)

print(image_reward_result)