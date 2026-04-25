from syn_grid.config.models import EvalAgentConf
from syn_grid.runners.agent_runners.agent_runner import AgentRunner
from syn_grid.gymnasium.env_factory import make

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


def evaluate_agent(runner: AgentRunner, conf: EvalAgentConf):
    """
    Run the benchmark with the specified model.

    :param agent_steps: The specific checkpoint steps of the model to run the benchmark with.
    """

    # Create the environment with human rendering and load the model
    raw_env = make(None if conf.time_env else "human", runner.run_conf, runner.obs_conf)

    # Reset the environment
    raw_env.reset(seed=42)

    # Wrap it for lstm
    env = DummyVecEnv([lambda: raw_env])

    # Define get_action() depending on type of model
    if conf.trained_model:
        model = runner.load_model(env)

        if runner.algorithm == "RPPO":

            def predict_action_from_lstm_model(
                obs, lstm_states=None, episode_starts=None
            ):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                return action, lstm_states

            get_action = predict_action_from_lstm_model
        else:

            def predict_action_from_model(obs, lstm_states=None, episode_starts=None):
                # Predict action from the model
                action, _ = model.predict(obs)
                return action, lstm_states

            get_action = predict_action_from_model
    else:

        def sample_random_action(obs, lstm_states=None, episode_starts=None):
            # Sample a random action
            action = env.action_space.sample()
            return action, lstm_states

        get_action = sample_random_action

    obs = env.reset()

    # if I want to time the speed of my raw environment, to ensure its not the bottleneck.
    # TODO: remove this after project completion
    if conf.time_env:
        _environment_timer_test(obs, get_action, env)

    # TODO: for LSTM, remember to refactor into a strategy pattern or something instead of having
    # all these if else states here
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)

    done = False
    try:
        while not done:
            if runner.algorithm == "RPPO":
                action, lstm_states = get_action(obs, lstm_states, episode_starts)
            else:
                action, _ = get_action(obs)

            obs, _, dones, _ = env.step(action)

            # Exit environment if terminated or truncated.
            done = bool(dones[0])
            episode_starts = dones
    except Exception as e:
        print(f"System crashed: {e}")
        raise  # exit function gracefully

    env.close()


def _environment_timer_test(obs, get_action, env):
    import time

    count = 0
    start_time = time.perf_counter()

    for _ in range(10000):
        action = get_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        count += 1

        if terminated or truncated:
            obs, _ = env.reset()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    fps = 10000 / total_time

    print(f"Total time: {total_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"env steps: {count}")
