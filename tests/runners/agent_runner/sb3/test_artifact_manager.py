import gymnasium as gym
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from syn_grid.runners.agent_runners.sb3.artifact_manager import ArtifactManager


def _make_env() -> DummyVecEnv:
    return DummyVecEnv([lambda: gym.make("CartPole-v1")])


@pytest.mark.parametrize("training", [True, False])
def test_loaded_normalization_wrapper_preserves_requested_mode(
    tmp_path, training: bool
):
    stats_path = tmp_path / "vec_normalize.pkl"
    VecNormalize(_make_env()).save(str(stats_path))

    artifact_manager = ArtifactManager(
        algorithm=PPO,
        hyper_parameters={},
        model_dir=tmp_path,
        vec_norm_stats_dir=tmp_path,
        find_latest_saved_path=lambda _: stats_path,
    )

    normalized_env = artifact_manager.load_normalize_wrapper(
        _make_env(), training=training
    )

    assert normalized_env.training is training
    normalized_env.close()
