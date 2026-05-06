from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig
from syn_grid.gymnasium.utils.episode_stats_wrapper import EpisodeStatsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


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
        env = self._make_wrapped_dummy_vec_env(
            self._train_conf.render_mode, self._TRAIN
        )
        env = self._get_normalized_env(env)
        model = self._get_model(env, self._TRAIN)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._make_wrapped_dummy_vec_env(self._eval_conf.render_mode, self._EVAL)
        env = self._get_normalized_env(env)
        model = self._load_model(env)

        try:
            for i in range(self._eval_conf.num_eval_episodes):
                # start the eval loop
                obs = env.reset()
                while True:
                    action, states = model.predict(
                        obs, deterministic=True  # type: ignore[arg-type]
                    )
                    obs, reward_arr, done_arr, info = env.step(action)

                    if done_arr[0]:
                        print(info[0]["episode"])
                        break
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()
