"""
The run controls of a training run with a human renderer. Used for debugging:
watching early training after a reward or rule change, and pausing when the
agent starts exploiting something.

An eval loop calls the runner's run controls itself between two steps. Training
runs inside SB3's `model.learn()`, and a callback is its way in, so this module
hands the same run controls to SB3.
"""

from collections.abc import Callable

from stable_baselines3.common.callbacks import BaseCallback

from syn_grid.rendering.pygame_renderer import PygameRenderer


class RunControlsCallback(BaseCallback):
    """
    Reads the window's input after every training step, through the runner's run controls.

    For a training run with a human renderer, which has a single env. The renderer is taken
    from that env when training starts.
    """

    def __init__(self, handle_run_controls: Callable[[PygameRenderer], None]):
        super().__init__()
        self._handle_run_controls = handle_run_controls
        self._renderer: PygameRenderer

    def _on_training_start(self) -> None:
        self._renderer = self.training_env.get_attr("renderer", indices=0)[0]

    def _on_step(self) -> bool:
        self._handle_run_controls(self._renderer)
        return True
