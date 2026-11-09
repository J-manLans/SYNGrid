"""How an SB3 run's model and its normalization statistics are created, saved and loaded.

A small PPO on Gymnasium's CartPole keeps the saves and loads fast and away from SYNGrid's own env, which
the SB3 runner's tests cover.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from syn_grid.runners.agent_runners.sb3.artifact_manager import ArtifactManager

# ================= #
#     Constants     #
# ================= #


RUN_ID = "a_run"
SEED = 3
# The runner resolves the policy from the perception and hands it in with the rest
HYPER_PARAMETERS = {
    "policy": "MlpPolicy",
    "device": "cpu",
    "n_steps": 16,
    "batch_size": 16,
    "ent_coef": 0.5,
}


# ================= #
#      Helpers      #
# ================= #


def _only_file(folder: Path) -> Path:
    """Stands in for the runner's search for the latest save: each test saves once."""

    [path] = folder.iterdir()
    return path


def _manager(tmp_path: Path) -> ArtifactManager:
    model_dir, stats_dir = tmp_path / "models", tmp_path / "vec_norm_stats"
    model_dir.mkdir()
    stats_dir.mkdir()
    return ArtifactManager(
        algorithm=PPO,
        hyper_parameters=HYPER_PARAMETERS,
        model_dir=model_dir,
        vec_norm_stats_dir=stats_dir,
        find_latest_saved_path=_only_file,
    )


def _plain_env() -> DummyVecEnv:
    return DummyVecEnv([lambda: gym.make("CartPole-v1")])


def _normalized_env_that_has_seen_steps(manager: ArtifactManager) -> VecNormalize:
    """A normalized env whose statistics have moved away from their start."""

    env = manager.create_normalize_wrapper(_plain_env())
    env.seed(SEED)
    env.reset()
    for _ in range(20):
        env.step(np.array([env.action_space.sample()]))
    return env


# ================= #
#       Tests       #
# ================= #


class TestNormalization:
    """
    The running statistics a fresh run's env is wrapped in, and how a later run gets them back.
    """

    def test_a_fresh_run_normalizes_its_observations_and_not_its_rewards(
        self, tmp_path: Path
    ):
        env = _manager(tmp_path).create_normalize_wrapper(_plain_env())

        assert env.norm_obs
        assert not env.norm_reward

    def test_a_later_run_gets_the_saved_statistics_and_updates_them_only_to_train(
        self, tmp_path: Path
    ):
        manager = _manager(tmp_path)
        env = _normalized_env_that_has_seen_steps(manager)
        manager.save_model(manager.create_model(env, None, SEED), env, RUN_ID)

        training = manager.load_normalize_wrapper(_plain_env(), training=True)
        evaluating = manager.load_normalize_wrapper(_plain_env(), training=False)

        # CartPole's observation is one array, so each env holds one set of statistics, not a dict
        assert isinstance(saved := env.obs_rms, RunningMeanStd)
        assert isinstance(loaded := training.obs_rms, RunningMeanStd)
        assert np.array_equal(loaded.mean, saved.mean)
        assert training.training
        assert not evaluating.training


class TestModel:
    """The model a fresh run creates, and what a save and a load carry."""

    def test_a_fresh_model_has_the_runs_seed_tensorboard_folder_and_hyper_parameters(
        self, tmp_path: Path
    ):
        board = str(tmp_path / "board")

        model = _manager(tmp_path).create_model(_plain_env(), board, SEED)

        assert isinstance(model, PPO)
        assert model.seed == SEED
        assert model.tensorboard_log == board
        assert model.ent_coef == HYPER_PARAMETERS["ent_coef"]

    def test_a_save_is_named_by_its_steps_and_the_run_and_carries_its_statistics(
        self, tmp_path: Path
    ):
        manager = _manager(tmp_path)
        env = _normalized_env_that_has_seen_steps(manager)
        model = manager.create_model(env, None, SEED)
        model.learn(total_timesteps=16)

        manager.save_model(model, env, RUN_ID)

        steps = model.num_timesteps
        assert _only_file(tmp_path / "models").name == f"{steps}_{RUN_ID}.zip"
        assert _only_file(tmp_path / "vec_norm_stats").name == f"{steps}_{RUN_ID}.pkl"

    def test_a_model_without_normalization_saves_no_statistics(self, tmp_path: Path):
        manager = _manager(tmp_path)
        env = _plain_env()

        manager.save_model(manager.create_model(env, None, SEED), env, RUN_ID)

        assert list((tmp_path / "vec_norm_stats").iterdir()) == []

    def test_a_loaded_model_has_the_saved_weights_and_works_on_the_env_it_is_given(
        self, tmp_path: Path
    ):
        manager = _manager(tmp_path)
        saved_env = _plain_env()
        saved = manager.create_model(saved_env, None, SEED)
        manager.save_model(saved, saved_env, RUN_ID)
        new_env = _plain_env()

        loaded = manager.load_model(new_env, SEED)

        for before, after in zip(
            saved.policy.parameters(), loaded.policy.parameters(), strict=True
        ):
            assert torch.equal(before, after)
        assert loaded.get_env() is new_env
        assert loaded.device == torch.device("cpu")
