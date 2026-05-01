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
        env = self._get_normalized_env(env)
        model = self._get_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._make_wrapped_dummy_vec_env(self._eval_conf.render_mode)
        env = self._get_normalized_env(env)
        model = self._load_model(env)

        # stores total reward and episode length for each evaluation episode
        episode_rewards = []
        episode_lengths = []
        rewards = []  # stores all rewards

        try:
            for i in range(self._eval_conf.num_eval_episodes):
                # start the eval loop
                obs = env.reset()
                total_reward = 0.0
                step_count = 0

                while True:
                    assert isinstance(obs, dict)
                    action, states = model.predict(obs, deterministic=True)
                    obs, reward_arr, done_arr, info = env.step(action)

                    rewards.append(info[0].get("reward"))
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
        num_max_tier_reached = sum(1 for r in rewards if r == 19.8)
        num_tier_out_of_order = sum(1 for r in rewards if r == -0.5)
        average_max_tier = num_max_tier_reached / self._eval_conf.num_eval_episodes
        average_tier_out_of_order = (
            num_tier_out_of_order / self._eval_conf.num_eval_episodes
        )

        print(
            f"Eval over {self._eval_conf.num_eval_episodes} episodes:"
            f"average reward = {avg_reward:.2f}, average length = {avg_length:.1f}\n"
            f"Max tier reached: {num_max_tier_reached} times, average: {average_max_tier:.2f}\n"
            f"Tier orbs out of order: {num_tier_out_of_order}, avg: {average_tier_out_of_order:.2f}"
        )
