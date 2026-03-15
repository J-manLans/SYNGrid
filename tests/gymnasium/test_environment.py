import numpy as np
import pytest
from synergygrid.core import AgentAction
from synergygrid.gymnasium.environment import SYNGridEnv


class TestEnvironment:
    """
    Test suite for the SynergyGridEnv Gymnasium environment.

    These tests verify that the environment correctly follows the Gymnasium
    API contract and behaves as expected under normal, invalid, and edge
    conditions.
    """

    @pytest.fixture
    def env(self):
        """
        Pytest fixture that provides a fresh environment instance
        for tests that require a default configuration.
        """
        return SYNGridEnv()

    def test_initialization(self, env):
        """
        Verify that the environment initializes correctly and defines
        the required Gymnasium spaces.
        """
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_custom_initialization(self):
        """
        Verify that custom initialization parameters are correctly
        applied to the environment.
        """
        env = SYNGridEnv(
            max_active_resources=5, grid_rows=7, grid_cols=6, max_steps=50
        )

        # Check that configuration parameters were stored correctly
        assert env.max_active_resources == 5
        assert env.grid_rows == 7
        assert env.grid_cols == 6

        # Ensure required Gym spaces exist
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_reset_returns_valid_observation(self, env):
        """
        Verify that reset() returns a valid observation and info dictionary
        as required by the Gymnasium API.
        """
        obs, info = env.reset()

        # Observation must conform to the defined observation space
        assert env.observation_space.contains(obs)

        # Info must be a dictionary (Gymnasium contract)
        assert isinstance(info, dict)

    def test_reset_with_seed_is_reproducible(self):
        """
        Verify that resetting two environments with the same seed
        produces identical observations.
        """
        env1 = SYNGridEnv()
        env2 = SYNGridEnv()

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Both observations should have the same structure
        assert obs1.keys() == obs2.keys()

        # Each observation component should be identical
        for key in obs1:
            assert np.array_equal(obs1[key], obs2[key])

    def test_step_returns_gym_contract(self, env):
        """
        Verify that step() returns values that follow the Gymnasium API:
        (observation, reward, terminated, truncated, info).
        """
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Observation must remain valid
        assert env.observation_space.contains(obs)

        # Reward must be numeric
        assert isinstance(reward, (int, float))

        # Termination flags must be booleans
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        # Info must be a dictionary
        assert isinstance(info, dict)

    def test_invalid_action_raises(self, env):
        """
        Verify that invalid actions raise an exception.
        """
        env.reset()

        # Negative action index
        with pytest.raises((ValueError, KeyError, AssertionError)):
            env.step(-1)

        # Out-of-range action index
        with pytest.raises((ValueError, KeyError, AssertionError)):
            env.step(999)

    def test_truncated_after_max_steps(self):
        """
        Verify that the environment eventually returns truncated=True
        when the maximum number of steps is exceeded.
        """
        env = SYNGridEnv(max_steps=5)
        env.reset()

        truncated = False

        # Run for more steps than the allowed maximum
        for _ in range(20):
            _, _, _, truncated, _ = env.step(env.action_space.sample())

            if truncated:
                break

        assert truncated

    def test_environment_runs_until_termination_or_truncation(self, env):
        """
        Verify that the environment eventually reaches either a terminated
        or truncated state during normal operation.
        """
        env.reset()

        terminated = False
        truncated = False

        # Run environment for a bounded number of steps
        for _ in range(500):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())

            if terminated or truncated:
                break

        assert terminated or truncated

    def test_reder_without_human_mode_returns_none_or_str(self):
        """
        Verify that render() returns either None or a string when rendering
        is enabled with human mode.
        """
        env = SYNGridEnv(render_mode="human")
        env.reset()

        result = env.render()

        assert result is None or isinstance(result, str)
