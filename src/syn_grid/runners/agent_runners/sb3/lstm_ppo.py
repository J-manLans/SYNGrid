from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

import numpy as np
from sb3_contrib import RecurrentPPO


class LstmPPO(BaseSB3Runner[RecurrentPPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        policy = self._get_policy_from_perception(
            obs_conf.observation_handler.perception, True
        )
        hyper_parameters = {
            "policy": policy,
            "device": "cpu",
            "ent_coef": 0.025,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 1e-4,
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

        # prep lstm variables
        lstm_states = None
        num_envs = env.num_envs
        episode_starts = np.ones((num_envs,), dtype=bool)

        try:
            for _ in range(self._eval_conf.num_eval_episodes):
                # start the eval loop
                obs = env.reset()
                lstm_states = None
                episode_starts = np.ones((num_envs,), dtype=bool)
                while True:
                    action, lstm_states = model.predict(
                        obs,  # type: ignore[arg-type]
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    obs, rewards, dones, infos = env.step(action)

                    episode_starts = dones
                    if dones[0]:
                        break
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()
