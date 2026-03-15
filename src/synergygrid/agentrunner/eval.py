from synergygrid.agentrunner import AgentRunner
import gymnasium as gym


def evaluate_agent(runner: AgentRunner, agent_steps: str, trained_model: bool):
    """
    Run the benchmark with the specified model.

    :param agent_steps: The specific checkpoint steps of the model to run the benchmark with.
    """

    # Create the environment with human rendering and load the model
    env = gym.make(runner.environment, render_mode="human")

    # Define get_action() depending on type of model
    if trained_model:
        model = runner.get_model(agent_steps, env)

        def get_action(obs):
            # Predict action from the model
            action, _ = model.predict(obs)
            return action

    else:

        def get_action(obs):
            # Sample a random action
            action = env.action_space.sample()
            return action

    # Reset the environment
    obs, _ = env.reset()

    done = False
    while not done:
        try:
            obs, _, terminated, truncated, _ = env.step(get_action(obs))
        except Exception as e:
            print(f"System crashed: {e}")
            return  # exit function gracefully
        finally:
            env.close()  # cleanup

        # Exit environment if terminated or truncated.
        done = truncated or terminated

    env.close()
