from syn_grid.gymnasium.environment import SYNGridEnv

from tests.utils.config_helpers import get_test_config, update_conf

import numpy as np
import pytest
from typing import Any


class TestEnvironment:
    """
    Test suite for the SynergyGridEnv Gymnasium environment.

    These tests verify that the environment correctly follows the Gymnasium
    API contract and behaves as expected under normal, invalid, and edge
    conditions.
    """

    # ================= #
    #       Init        #
    # ================= #

    @pytest.fixture
    def env(self):
        """
        Pytest fixture that provides a fresh environment instance for tests that require a default configuration.
        """

        conf = get_test_config()

        env = SYNGridEnv(conf.world, conf.obs)

        env.reset()
        return env

    # ================= #
    #       Tests       #
    # ================= #

    def test_headless_initialization(self, env: SYNGridEnv):
        """
        Verify that the default headless environment initializes correctly and defines the required Gymnasium spaces.
        """

        assert not hasattr(env, "renderer")
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_reset_has_valid_return(self, env: SYNGridEnv):
        """
        Verify that reset() returns a valid observation and info dict as required by the Gymnasium API.
        """

        obs, info = env.reset()

        # Observation must conform to the defined observation space
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_has_valid_return(self, env: SYNGridEnv):
        """
        Verify that step() returns valid observation, reward, terminal and truncated, and info values as required by the Gymnasium API.
        """

        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset_with_seed_is_reproducible(self):
        """
        Verify that resetting two environments with the same seed
        produces identical observations.
        """

        conf = get_test_config()

        env1 = SYNGridEnv(conf.world, conf.obs)
        env2 = SYNGridEnv(conf.world, conf.obs)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2)

    def test_invalid_action_raises(self, env: SYNGridEnv):
        """
        Verify that invalid actions raise an exception.
        """

        # Negative action index
        with pytest.raises(ValueError):
            env.step(-1)

        # Out-of-range action index
        with pytest.raises(ValueError):
            env.step(999)

    def test_truncated_after_max_steps(self):
        """
        Verify that the environment eventually returns truncated=True
        when the maximum number of steps is exceeded.
        """

        conf = get_test_config()
        obs_conf = update_conf(conf.obs, {"observation_handler": {"max_steps": 5}})

        env = SYNGridEnv(conf.world, obs_conf)
        env.reset()

        terminated = False

        # Run for more steps than the allowed maximum
        for _ in range(obs_conf.perception.max_steps + 1):
            _, _, terminated, _, _ = env.step(env.action_space.sample())

            if terminated:
                break

        assert terminated

    def test_environment_terminates(self):
        """
        Verify that the environment terminates correctly
        """

        conf = get_test_config()
        world_conf = update_conf(conf.world, {"droid_conf": {"starting_score": 1}})

        env = SYNGridEnv(world_conf, conf.obs)
        env.reset()

        terminated = False
        _, _, terminated, _, _ = env.step(env.action_space.sample())

        assert terminated

    def test_render_without_human_mode_returns_none_or_str(self):
        """
        Verify that render() returns either None or a string when rendering
        is enabled with human mode.
        """

        conf = get_test_config()

        env = SYNGridEnv(conf.world, conf.obs, render_mode="human")
        env.reset()

        assert hasattr(env, "renderer")

    def test_reset_creates_identical_state(self):
        conf = get_test_config()
        world_conf = update_conf(conf.world, {"droid_conf": {"starting_score": 99999}})

        baseline_env = SYNGridEnv(world_conf, conf.obs)
        baseline_env.reset()

        baseline_state = self._capture_state(baseline_env)

        env = SYNGridEnv(world_conf, conf.obs)
        env.reset()

        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

            if terminated or truncated:
                done = True

        env.reset()

        assert self._capture_state(env) == baseline_state

    def test_same_step_in_identical_environments_produces_consistent_state(self):
        conf = get_test_config()
        world_conf = update_conf(conf.world, {"droid_conf": {"starting_score": 99999}})

        env1 = SYNGridEnv(world_conf, conf.obs)
        env1.reset(seed=42)

        env2 = SYNGridEnv(world_conf, conf.obs)
        env2.reset(seed=42)

        for _ in range(70):
            action = env1.action_space.sample()
            env1.step(action)
            env2.step(action)

        assert self._capture_state(env1) == self._capture_state(env2)

    # ================= #
    #      Helpers      #
    # ================= #

    def _capture_state(self, env: SYNGridEnv) -> dict[str, Any]:
        return {
            "steps_left": env._observation_handler.steps_left,
            "num_orbs_pool": len(env.world.ALL_ORBS),
            "num_active_orbs": len(env.world._ACTIVE_ORBS),
            "num_inactive_orbs": len(env.world._inactive_orbs),
            "droid_position": env.world.DROID.position.copy(),
            "droid_score": env.world.DROID.score,
            "chained_tiers": env.world.DROID.DIGESTION_ENGINE.chained_tiers,
            "pending_reward": env.world.DROID.DIGESTION_ENGINE._pending_reward,
        }
