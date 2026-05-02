from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.runners.agent_runners.agent_registry import ALGORITHMS

from tests.utils.config_helpers import get_test_config, update_conf

from pathlib import Path
import pytest


class TestAgentRunner:
    """
    This test suite verifies that the `AgentRunner` class functions correctly in various scenarios. Specifically, it tests:
    - The initialization of an `AgentRunner` object.
    - Handling of invalid algorithms.
    - Behavior when model steps are not provided.
    - Behavior when no matching model is found for the specified steps.

    Tests are designed to check edge cases, error handling, and correct interactions with the underlying model-loading functionality.
    """

    @pytest.fixture
    def agent_runner(self):
        """
        Fixture to create and return an instance of the `AgentRunner` class for testing.

        Returns:
            AgentRunner: A configured `AgentRunner` instance with a specified environment ("synergy_grid-v0")
                         and algorithm ("PPO").
        """

        full_conf = get_test_config()

        full_conf = update_conf(
            full_conf, {
                'agent': {
                    'global_agent_conf': {
                        "alg": "PPO"
                    }
                }
            }
        )

        run_conf = full_conf.world
        obs_conf = full_conf.obs

        return ALGORITHMS[full_conf.agent.global_agent_conf.alg](
            full_conf.agent, obs_conf, run_conf
        )

    def test_initialization_with_invalid_algorithm(self):
        """
        Tests the behavior when an invalid algorithm is passed to the `AgentRunner` constructor.
        Verifies that a `KeyError` is raised when an unsupported algorithm is provided.
        This helps ensure that only supported algorithms are used in the `AgentRunner`.

        Raises:
            KeyError: If the algorithm is invalid (e.g., "invalid_algorithm").
        """

        full_conf = get_test_config()

        full_conf = update_conf(
            full_conf, {
                'agent': {
                    'global_agent_conf': {
                        "alg": "Ajja_bajja"
                    }
                }
            }
        )
        run_conf = full_conf.world
        obs_conf = full_conf.obs

        with pytest.raises(KeyError):
            ALGORITHMS[full_conf.agent.global_agent_conf.alg](full_conf.agent, obs_conf, run_conf)

    def test_get_model_with_no_agent_steps(self, agent_runner: BaseAgentRunner):
        """
        Tests the behavior when no agent steps are provided to the `get_model` method.
        Verifies that the program exits with an error (SystemExit) when no agent steps are specified.
        This test ensures that the `get_model` method handles the missing steps case correctly and prevents further execution.

        Raises:
            SystemExit: If no agent steps are provided.
        """

        agent_runner._conf.agent_steps = ""

        with pytest.raises(SystemExit):
            agent_runner._get_saved_path(Path("null_path"))


