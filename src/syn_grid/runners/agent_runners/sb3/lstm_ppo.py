import numpy as np
from sb3_contrib import RecurrentPPO

from syn_grid.runners.agent_runners.agent_bundle import AgentBundle
from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner


class LstmPPO(BaseSB3Runner[RecurrentPPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, agent_bundle: AgentBundle):
        policy = super()._get_policy_from_perception(
            agent_bundle.obs_conf.observation_handler.perception, True
        )
        hyper_parameters = {
            "policy": policy,
            "device": "cuda",
            "ent_coef": 0.025,
            "n_steps": 128,
            "batch_size": 128,
            "n_epochs": 4,
            "policy_kwargs": {
                "lstm_hidden_size": 265,
                "n_lstm_layers": 1,
                "shared_lstm": False,
            },
        }
        super().__init__(
            agent_bundle,
            hyper_parameters,
            RecurrentPPO,
        )
        print("Initializing RecurrentPPO...")

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = super()._make_wrapped_dummy_vec_env(
            self._train_conf.render_mode, self._TRAIN
        )
        env = super()._get_normalized_env(env)
        model = super()._get_model(env, self._TRAIN)

        super()._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = super()._make_wrapped_dummy_vec_env(
            self._eval_conf.render_mode, self._EVAL
        )
        env = super()._get_normalized_env(env)
        model = super()._load_model(env)

        # prep lstm variables
        lstm_states = None
        num_envs = env.num_envs
        episode_starts = np.ones((num_envs,), dtype=bool)

        # start the eval loop
        obs = env.reset()
        try:
            for _ in range(self._eval_conf.num_eval_episodes):
                lstm_states = None
                episode_starts = np.ones((num_envs,), dtype=bool)
                while True:
                    action, lstm_states = model.predict(
                        obs,  # type: ignore[arg-type]
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    obs, _, dones, _ = env.step(action)

                    episode_starts = dones
                    if dones[0]:
                        break
        except Exception as e:
            print(f"System crashed: {e}")
            raise
        finally:
            env.close()
