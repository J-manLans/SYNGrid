from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

from stable_baselines3 import PPO


class StatelessPPO(BaseSB3Runner[PPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        policy = self._get_policy_from_perception(
            obs_conf.observation_handler.perception
        )
        hyper_parameters = {"policy": policy, "device": "cpu", "ent_coef": 0.02}
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, PPO)

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._make_wrapped_dummy_vec_env(self._train_conf.render_mode)
        env = self._get_vec_env(env)
        model = self._get_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._make_wrapped_dummy_vec_env(self._eval_conf.render_mode)
        env = self._get_vec_env(env)
        model = self._load_model(env)

        # stores total reward and episode length for each evaluation episode
        episode_rewards = []
        episode_lengths = []

        try:
            # ==========================
            #   Debug
            # ==========================

            rewards = []
            chains = []
            scores = []
            steps = []

            # ==========================
            #   Debug END
            # ==========================

            for i in range(self._eval_conf.num_eval_episodes):
                # start the eval loop
                obs = env.reset()
                total_reward = 0.0
                step_count = 0

                # ==========================
                #   Debug
                # ==========================

                # ==========================
                #   Debug END
                # ==========================

                while True:
                    assert isinstance(obs, dict)
                    action, states = model.predict(obs, deterministic=True)
                    obs, reward_arr, done_arr, info = env.step(action)

                    # ==========================
                    #   Debug
                    # ==========================

                    rewards.append(info[0].get("reward"))
                    steps.append(info[0].get("steps_left"))
                    scores.append(info[0].get("score"))
                    chains.append(info[0].get("chain"))

                    # ==========================
                    #   Debug END
                    # ==========================

                    total_reward += reward_arr[0]
                    step_count += 1
                    if done_arr[0]:
                        break

                episode_rewards.append(total_reward)
                episode_lengths.append(step_count)
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        print(
            f"Eval over {self._eval_conf.num_eval_episodes} episodes: average reward = {avg_reward:.2f}, average length = {avg_length:.1f}"
        )

        # ==========================
        #   Debug
        # ==========================

        num_max_tier_reached = sum(1 for r in rewards if r == 9)
        average_max_tier = num_max_tier_reached / self._eval_conf.num_eval_episodes

        print(f"Max tier reached: {num_max_tier_reached} times")
        print(f"Average: {average_max_tier:.2f}")

        # ==========================
        #   Debug END
        # ==========================
