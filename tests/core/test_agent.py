import pytest
from synergygrid.core.agent.synergy_agent import AgentAction, SynergyAgent


class DummyPositiveResource:
    """
    Minimal stub resource for testing.

    Simulates a resource that always returns a fixed positive reward
    when consumed. Used to isolate agent logic from resource logic.
    """

    def consume(self):
        return 5


class DummyNegativeResource:
    """
    Minimal stub resource for testing.

    Simulates a resource that always returns a fixed negative reward
    when consumed. Used to test score deduction logic.
    """

    def consume(self):
        return -5


class TestAgent:
    """
    Unit tests for the SynergyAgent class.

    Tests cover:
    - Agent initialization and score
    - Movement in all directions
    - Boundary handling
    - Reset behavior
    - Resource consumption (positive and negative)
    - Edge cases (minimal grid, invalid inputs)
    """

    @pytest.fixture
    def agent(self):
        """
        Returns a SynergyAgent instance on a 6x6 grid with a starting score of 10,
        reset to its initial state. Used as a reusable fixture for most tests.
        """
        agent = SynergyAgent(grid_rows=6, grid_cols=6, starting_score=10)
        agent.reset()

        return agent

    @pytest.mark.parametrize(
        "y, x, expected_position",
        [
            (6, 6, [3, 3]),
            (5, 5, [2, 2]),
            (1, 1, [0, 0]),
        ],
    )
    def test_initial_positions(self, y, x, expected_position):
        """
        Check that the agent starts at the center of the grid when reset,
        for various grid sizes.
        """
        agent = SynergyAgent(grid_rows=y, grid_cols=x)
        agent.reset()

        assert agent.position == expected_position

    @pytest.mark.parametrize("score", [100, 0, -100])
    def test_initial_score(self, score):
        """
        Ensure that the agent's score is initialized to the starting value
        after creation.
        """
        agent = SynergyAgent(1, 1, score)

        assert agent.score == score

    @pytest.mark.parametrize(
        "action, expected_position",
        [
            (AgentAction.RIGHT, [3, 4]),
            (AgentAction.LEFT, [3, 2]),
            (AgentAction.UP, [2, 3]),
            (AgentAction.DOWN, [4, 3]),
        ],
    )
    def test_movement(self, agent, action, expected_position):
        """
        Verify that each movement action correctly updates the agent's position
        when not at a boundary.
        """
        initial_score = agent.score
        agent.perform_action(action)

        assert agent.position == expected_position
        assert initial_score == agent.score + 1

    @pytest.mark.parametrize("action", [None, "LEFT"])
    def test_invalid_movement(self, agent, action):
        """
        Ensure that passing an invalid value to perform_action
        (anything other than an AgentAction enum) raises a TypeError.
        """
        with pytest.raises(TypeError):
            agent.perform_action(action)

    @pytest.mark.parametrize(
        "position, action, expected_position",
        [
            ([0, 0], AgentAction.LEFT, [0, 0]),
            ([0, 5], AgentAction.RIGHT, [0, 5]),
            ([0, 0], AgentAction.UP, [0, 0]),
            ([5, 0], AgentAction.DOWN, [5, 0]),
        ],
    )
    def test_boundaries(self, agent, position, action, expected_position):
        """
        Ensure the agent does not move outside the grid boundaries
        when attempting moves at edges.
        """
        agent.position = position.copy()
        agent.perform_action(action)

        assert agent.position == expected_position

    @pytest.mark.parametrize(
        "action",
        [AgentAction.LEFT, AgentAction.RIGHT, AgentAction.UP, AgentAction.DOWN],
    )
    def test_edge_case_boundaries(self, action):
        """
        Verify that on a minimal 1x1 grid, the agent cannot move
        in any direction and remains at [0, 0].
        """
        agent = SynergyAgent(1, 1)
        agent.reset()
        agent.perform_action(action)

        assert agent.position == [0, 0]

    def test_reset_restores_position(self, agent):
        """
        Ensure that calling reset() restores the agent's position
        to the center of the grid.
        """
        starting_position = agent.position.copy()
        agent.position = [0, 0]
        agent.reset()

        assert agent.position == starting_position

    def test_reset_restores_score(self, agent):
        """
        Ensure that calling reset() restores the agent's score
        to its original starting value.
        """
        starting_score = agent.score
        agent.score += 10
        agent.reset()

        assert agent.score == starting_score

    def test_consume_resource_adds_score(self, agent):
        """
        Verify that consuming a positive resource increases the agent's score
        and returns the correct reward value.
        """
        resource = DummyPositiveResource()
        reward = agent.consume_resource(resource)

        assert reward == 5
        assert agent.score == 15

    def test_consume_negative_resource_reduces_score(self, agent):
        """
        Verify that consuming a negative resource decreases the agent's score
        and returns the correct negative reward value.
        """
        resource = DummyNegativeResource()
        reward = agent.consume_resource(resource)

        assert reward == -5
        assert agent.score == 5
