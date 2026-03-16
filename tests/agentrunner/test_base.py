import pytest
from synergygrid.agentrunner import AgentRunner
from stable_baselines3 import A2C
from unittest.mock import patch


class TestAgentRunnerBase:
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
                         and algorithm ("A2C").
        """
        return AgentRunner(2)

    def test_initialization(self, agent_runner):
        """
        Tests the correct initialization of the `AgentRunner` object.

        Verifies that:
            - The `environment` attribute is set to "synergy_grid-v0".
            - The `model` attribute is initially `None`.
            - The `algorithm` attribute is set to "A2C".
            - The `AlgorithmClass` is set to the `A2C` class.

        Args:
            agent_runner (AgentRunner): The `AgentRunner` instance to test.
        """
        assert agent_runner.environment == "synergy_grid-v0"
        assert agent_runner.model == None
        assert agent_runner.algorithm == "A2C"
        assert agent_runner.AlgorithmClass == A2C

    def test_initialization_with_invalid_algorithm(self):
        """
        Tests the behavior when an invalid algorithm is passed to the `AgentRunner` constructor.

        Verifies that a `ValueError` is raised when an unsupported algorithm is provided.

        This helps ensure that only supported algorithms are used in the `AgentRunner`.

        Raises:
            ValueError: If the algorithm is invalid (e.g., "invalid_algorithm").
        """
        with pytest.raises(IndexError):
            AgentRunner(4)

    def test_get_model_with_no_agent_steps(self, agent_runner):
        """
        Tests the behavior when no agent steps are provided to the `get_model` method.

        Verifies that the program exits with an error (SystemExit) when no agent steps are specified.

        This test ensures that the `get_model` method handles the missing steps case correctly and prevents further execution.

        Raises:
            SystemExit: If no agent steps are provided.
        """
        with pytest.raises(SystemExit):
            agent_runner.get_model("", "env")

    def test_get_model_with_no_matching_model(self, agent_runner):
        """
        Tests the behavior when the `get_model` method is called with valid agent steps but no matching model is found.

        Verifies that an `IndexError` is raised when the `glob` method returns an empty list (indicating no matching models).

        This ensures that the system correctly handles cases where no models match the specified agent steps.

        Raises:
            IndexError: If no matching model is found for the specified agent steps.
        """
        valid_agent_steps = "1000"
        mock_env = "some_env"

        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = []

            with pytest.raises(IndexError):
                agent_runner.get_model(valid_agent_steps, mock_env)
