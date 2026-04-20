from syn_grid.config.models import EvalAgentConf
from syn_grid.runners.agent_runners.agent_runner import AgentRunner
from syn_grid.gymnasium.env_factory import make


def evaluate_agent(runner: AgentRunner, conf: EvalAgentConf):
    """
    Run the benchmark with the specified model.

    :param agent_steps: The specific checkpoint steps of the model to run the benchmark with.
    """

    # Create the environment with human rendering and load the model
    env = make("human", runner.run_conf, runner.obs_conf)

    # Define get_action() depending on type of model
    if conf.trained_model:
        model = runner.get_model(env)

        def predict_action_from_model(obs):
            # Predict action from the model
            action, _ = model.predict(obs)
            return action

        get_action = predict_action_from_model
    else:

        def sample_random_action(obs):
            # Sample a random action
            action = env.action_space.sample()
            return action

        get_action = sample_random_action

    # Reset the environment
    obs, _ = env.reset(seed=42)

    done = False
    total_rew = 0.0
    while not done:
        try:
            obs, rew, terminated, truncated, _ = env.step(get_action(obs))
            total_rew += float(rew)
        except Exception as e:
            print(f"System crashed: {e}")
            return  # exit function gracefully
        finally:
            env.close()  # cleanup

        # Exit environment if terminated or truncated.
        done = truncated or terminated

    print(total_rew)
    env.close()
