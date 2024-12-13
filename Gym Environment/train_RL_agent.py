import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gymnasium as gym

# Add the path to the custom environment
sys.path.insert(0, "homework")

import kartEnv  # Your custom environment

# Register custom environment
from gymnasium.envs.registration import register, registry

if "kartEnv-v0" not in registry:
    register(
        id="kartEnv-v0",
        entry_point="kartEnv:kartEnv",
        max_episode_steps=100000
    )

# Define a custom feature extractor for the PPO model
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.items():
            if key == "image":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(7 * 7 * 64, 512),
                    nn.ReLU()
                )
                total_concat_size += 512
            elif key == "speed":
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                )
                total_concat_size += 64

        super().__init__(observation_space, features_dim=total_concat_size)
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations):
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

# Initialize environment
env = gym.make("kartEnv-v0")
check_env(env)  # Check if the environment is valid

# Convert to vectorized environment
venv = DummyVecEnv([lambda: env])
venv = VecMonitor(venv)  # For monitoring the environment
venv = VecNormalize(venv, training=True, norm_obs=True, norm_reward=True)

# Define PPO policy kwargs
policy_kwargs = {
    'features_extractor_class': CustomCombinedExtractor,
    'features_extractor_kwargs': {},
}

# Initialize PPO model
model = PPO(
    "MultiInputPolicy",
    venv,
    verbose=1,
    tensorboard_log="./ppo_kart_tensorboard/",
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="auto",
)

# Train the model
model.learn(
    total_timesteps=1000000,  # Set desired timesteps
    log_interval=1,
    tb_log_name="ppo_kart",
)

# Save the model and environment
model.save("ppo_kart_model")
venv.save("venv.pkl")
print("Model and environment saved.")

# Test and render the trained model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()  # Ensure render logic is implemented in kartEnv
    if done:
        obs = env.reset()

print("Testing complete.")
