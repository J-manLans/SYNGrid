"""How a human training run's window input reaches SB3: after every step, and quitting ends `learn`."""

from typing import Any

from syn_grid.runners.agent_runners.sb3.run_controls_callback import (
    RunControlsCallback,
)

# ================= #
#     Stand-ins     #
# ================= #


class _HumanEnv:
    """Stands in for a human training run's single env, which holds the window's renderer."""

    def __init__(self):
        self.renderer = object()

    def get_attr(self, name: str, indices: int) -> list[Any]:
        assert indices == 0
        return [getattr(self, name)]


class _Model:
    """Stands in for the SB3 model the callback is attached to."""

    num_timesteps = 0

    def __init__(self, env: _HumanEnv):
        self._env = env

    def get_env(self) -> _HumanEnv:
        return self._env


# ================= #
#      Helpers      #
# ================= #


def _started(answers: list[bool]) -> tuple[RunControlsCallback, _HumanEnv, list[Any]]:
    """A callback whose run controls give `answers` in turn, attached and started as `learn` does it.
    Also the env, and the renderers the run controls were handed."""

    handed: list[Any] = []
    replies = iter(answers)

    def run_controls(renderer: Any) -> bool:
        handed.append(renderer)
        return next(replies)

    env = _HumanEnv()
    callback = RunControlsCallback(run_controls)
    callback.init_callback(_Model(env))  # type: ignore[arg-type]
    callback.on_training_start({}, {})

    return callback, env, handed


# ================= #
#       Tests       #
# ================= #


class TestEachStep:
    """What the callback does after every training step."""

    def test_it_reads_the_window_with_the_envs_renderer_and_lets_training_go_on(self):
        callback, env, handed = _started([False, False])

        assert callback.on_step()
        assert callback.on_step()

        assert handed == [env.renderer, env.renderer]
        assert not callback.stopped

    def test_quitting_ends_learn_and_says_so(self):
        callback, _, _ = _started([False, True])

        callback.on_step()

        assert not callback.on_step()
        assert callback.stopped
