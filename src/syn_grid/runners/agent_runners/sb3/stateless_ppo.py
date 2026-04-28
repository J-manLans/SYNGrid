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
        env = self._make_raw_env(self.eval_conf.render_mode)
        model = self._load_model(env)

        # stores total reward and episode length for each evaluation episode
        episode_rewards = []
        episode_lengths = []

        # ======================================
        # logging for debugging the env
        # ======================================

        _GLOBAL_KEY = 'global_data'
        _DROID_KEY = 'droid_data'
        _ORB_KEY = 'orb_data'

        # ======================================
        # END
        # ======================================

        try:
            for i in range(self.eval_conf.num_eval_episodes):
                # start the eval loop
                obs, _ = env.reset()

                # ======================================
                # logging for debugging the env
                # ======================================

                rewards = []
                steps = []
                chained_tiers = []
                scores = []
                droid_pos = []
                orb_positions = []
                global_min = []
                droid_min = []
                orb_min = []

                # ======================================
                # END
                # ======================================

                done = False
                total_reward = 0.0
                step_count = 0

                print(f'EVAL: {i}\n----------------')
                while not done:
                    action, states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)

                    # ======================================
                    # logging for debugging the env
                    # ======================================

                    global_min.append(obs[_GLOBAL_KEY].min())
                    d_pos = f'[{obs[_DROID_KEY][0]}, {obs[_DROID_KEY][1]}]'
                    droid_pos.append(d_pos)
                    droid_min.append(obs[_DROID_KEY].min())
                    scores.append(obs[_DROID_KEY][2])
                    orb_min.append(obs[_ORB_KEY].min())

                    pass

                    # ======================================
                    # END
                    # ======================================

                    total_reward += float(reward)
                    step_count += 1
                    done = truncated or terminated

                episode_rewards.append(total_reward)
                episode_lengths.append(step_count)

                # ======================================
                # logging for debugging the env
                # ======================================

                for i in range(1, len(global_min)):
                    if global_min[i] < 0:
                        print(f'global min wrong: {global_min[i]}')
                    if droid_min[i] < 0:
                        print(f'droid min wrong: {droid_min[i]}\nposition: {droid_pos[i]}\nscore: {scores[i]}')
                    if orb_min[i] < -1:
                        print(f'orb min wrong: {orb_min[i]}')

                # ======================================
                # END
                # ======================================
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
