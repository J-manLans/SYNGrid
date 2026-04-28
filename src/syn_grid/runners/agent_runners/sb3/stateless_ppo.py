from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

from stable_baselines3 import PPO


class StatelessPPO(BaseSB3Runner[PPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        policy = self.get_policy_from_perception(obs_conf.observation_handler.perception)
        hyper_parameters = {"policy": policy, "device": "cpu", "ent_coef": 0.02}
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, PPO)

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._make_env(self.train_conf.render_mode)
        model = self._get_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._make_raw_env("human")
        model = self._load_model(env)

        # stores total reward and episode length for each evaluation episode
        episode_rewards = []
        episode_lengths = []

        try:
            for _ in range(self.eval_conf.num_eval_episodes):
                # start the eval loop
                obs, _ = env.reset()
                done = False
                total_reward = 0.0
                step_count = 0
                while not done:
                    action, states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                    total_reward += float(reward)
                    step_count += 1
                    done = truncated or terminated

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
            f"Eval over {self.eval_conf.num_eval_episodes} episodes: average reward = {avg_reward:.2f}, average length = {avg_length:.1f}"
        )
