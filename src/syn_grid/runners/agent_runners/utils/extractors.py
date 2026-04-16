import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Any
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# TODO: look over these. TinyGridCNN works and can train CnnPolicy, but I haven't checked the logic
# yet. GroupedMetaExtractor is for learning the agent to process a dict in its logically structured
# keys instead of flattening everything into a 1D vector and then process it


class GroupedMetaExtractor(BaseFeaturesExtractor):
    @classmethod
    def get_agent_hyperparameters(cls) -> dict[str, Any]:
        return {
            "policy": "MultiInputPolicy",
            "device": "cpu",
            "ent_coef": 0.02,
            "policy_kwargs": {"features_extractor_class": cls},
        }

    def __init__(self, observation_space: spaces.Dict):
        # We'll calculate the final dimension later
        super().__init__(observation_space, features_dim=1)

        # 1. Define a small network for EACH of your logical groups
        # Get the dimensions of your groups. These are just examples.
        droid_dim = observation_space.spaces["droid_meta"].shape[0]
        orb_dim = observation_space.spaces["orb_meta"].shape[0]
        world_dim = observation_space.spaces["world_meta"].shape[0]

        # Each group gets its own feature processor (e.g., a simple MLP)
        # The output size (e.g., 32, 64) is a hyperparameter you can tune.
        self.droid_net = nn.Sequential(
            nn.Linear(droid_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Compress to a 32-dim feature vector
        )

        self.orb_net = nn.Sequential(
            nn.Linear(orb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Compress to a 32-dim feature vector
        )

        self.world_net = nn.Sequential(
            nn.Linear(world_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Compress to a 64-dim feature vector
        )

        # 2. Calculate the total feature dimension after processing
        total_concat_size = 32 + 32 + 64  # droid + orb + world features
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations is a dictionary of tensors
        droid_features = self.droid_net(observations["droid_meta"])
        orb_features = self.orb_net(observations["orb_meta"])
        world_features = self.world_net(observations["world_meta"])

        # Concatenate the processed feature vectors
        return th.cat([droid_features, orb_features, world_features], dim=1)


class TinyGridCNN(BaseFeaturesExtractor):
    @classmethod
    def get_agent_hyperparameters(cls) -> dict[str, Any]:
        return {
            "policy": "CnnPolicy",
            "device": "cpu",
            "ent_coef": 0.02,
            "policy_kwargs": {
                "features_extractor_class": cls,
                "features_extractor_kwargs": {"features_dim": 64},
                "normalize_images": False,
            },
        }

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]  # channels last: (H, W, C)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape after conv layers
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            # SB3 expects channels last, but Conv2d expects channels first
            sample = sample.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Convert from channels-last (H, W, C) to channels-first (C, H, W)
        x = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(x))
