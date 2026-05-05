from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

import numpy as np
from stable_baselines3 import PPO


class StatelessPPO(BaseSB3Runner[PPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        policy = self._get_policy_from_perception(
            obs_conf.observation_handler.perception
        )
        hyper_parameters = {
            "policy": policy,
            "device": "cpu",
            "ent_coef": 0.02,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 8,
        }
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, PPO)

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._make_wrapped_dummy_vec_env(self._train_conf.render_mode)
        env = self._get_normalized_env(env)
        model = self._get_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._make_wrapped_dummy_vec_env(self._eval_conf.render_mode)
        env = self._get_normalized_env(env)
        model = self._load_model(env)

        # stores total reward and episode length for each evaluation episode
        all_rewards = []
        episode_lengths = []
        episode_rewards = []

        try:
            for i in range(self._eval_conf.num_eval_episodes):
                # start the eval loop
                obs = env.reset()
                step_count = 0
                episode_rewards = []
                if i == 1:
                    pass

                while True:
                    action, states = model.predict(
                        obs, deterministic=True  # type: ignore[arg-type]
                    )
                    obs, reward_arr, done_arr, info = env.step(action)

                    episode_rewards.append(info[0].get("reward"))
                    step_count += 1

                    print(
                        f'Episode {i}, reward: {episode_rewards[-1]}, droid score: {info[0]["score"]}, sum of rewards: {np.sum(episode_rewards)}'
                    )
                    if done_arr[0]:
                        print()
                        break

                episode_lengths.append(step_count)
                all_rewards.extend(episode_rewards)
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()

        avg_length = sum(episode_lengths) / len(episode_lengths)
        sum_rew = sum(r for r in all_rewards)
        avg_rew = sum_rew / self._eval_conf.num_eval_episodes
        max_tier_reached = sum(1 for r in all_rewards if r == 9)
        avg_max_tier = max_tier_reached / self._eval_conf.num_eval_episodes
        num_tier_out_of_order = sum(1 for r in all_rewards if r == -2)
        average_tier_out_of_order = (
            num_tier_out_of_order / self._eval_conf.num_eval_episodes
        )

        print(
            f"Eval over {self._eval_conf.num_eval_episodes} episodes\n"
            f"average episode length = {avg_length:.1f}\n"
            f"Rewards collected: {sum_rew}, avg: {avg_rew:.2f}\n"
            f"Max Tier reached {max_tier_reached} times, avg: {avg_max_tier}\n"
            f"Tiers out of order: {num_tier_out_of_order}, avg: {average_tier_out_of_order}"
        )
