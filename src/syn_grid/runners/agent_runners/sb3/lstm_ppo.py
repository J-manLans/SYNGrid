from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

import numpy as np
from sb3_contrib import RecurrentPPO


class LstmPPO(BaseSB3Runner[RecurrentPPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        policy = self.get_policy_from_perception(obs_conf.observation_handler.perception, True)
        hyper_parameters = {
            "policy": policy,
            "device": "cpu",
            "ent_coef": 0.05,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 5,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "policy_kwargs": {
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "shared_lstm": False,
            },
        }
        super().__init__(
            conf,
            obs_conf,
            run_conf,
            hyper_parameters,
            RecurrentPPO,
            hyper_parameters["policy_kwargs"]["lstm_hidden_size"],
        )

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._make_wrapped_dummy_vec_env(self._train_conf.render_mode)
        model = self._get_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._make_wrapped_dummy_vec_env("human")
        model = self._load_model(env)

        # prep lstm variables
        lstm_states = None
        num_envs = env.num_envs
        episode_starts = np.ones((num_envs,), dtype=bool)

        # Each environment will have a list of rewards and lengths for completed episodes
        all_rewards = []
        all_lengths = []
        current_rewards = np.zeros(num_envs)
        current_lengths = np.zeros(num_envs, dtype=int)
        episode_counts = np.zeros(num_envs, dtype=int)

        # start the eval loop
        obs = env.reset()
        total_episodes_to_collect = self._eval_conf.num_eval_episodes * num_envs
        try:
            while len(all_rewards) < total_episodes_to_collect:
                action, lstm_states = model.predict(
                    obs,  # type: ignore[arg-type]
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, rewards, dones, infos = env.step(action)

                current_rewards += rewards
                current_lengths += 1

                for i in range(num_envs):
                    if dones[i]:
                        # Episode finished – store it
                        all_rewards.append(current_rewards[i])
                        all_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                        # Reset accumulators for this environment
                        current_rewards[i] = 0.0
                        current_lengths[i] = 0

                episode_starts = dones
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()

        # Compute averages
        avg_reward = np.mean(all_rewards)
        avg_length = np.mean(all_lengths)
        print(
            f"Eval over {len(all_rewards)} episodes: average reward = {avg_reward:.2f}, average length = {avg_length:.1f}"
        )
