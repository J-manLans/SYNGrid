"""
When a single tier chain episode ends.

This is the single tier chain implementation of ``TerminationRules``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.scenario.blocks.termination import EpisodeOutcome

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


# #################### #
# Termination strategy #
# #################### #


@dataclass(frozen=True)
class SingleTierChainTermination:
    """Ends a single tier chain episode, and states what the ending costs.

    An episode ends when the droid ends the task, or when the steps run out. The droid ends the
    task when it is out of energy, when the chain is broken, or when the chain is completed.

    The ending costs the timeout penalty when the steps run out and the droid did not end the
    task, and what `_task_ending_reward` says otherwise, which is nothing here. What the running
    chain is owed is digestion's, and the droid settles it.

    A variant of the family changes what ends the task and what that costs by overriding
    `_droid_ended_task` and `_task_ending_reward`. The deadline is not its to change.

    Attributes:
        timeout_penalty: what the clock running out costs.
    """

    timeout_penalty: float

    def evaluate(self, world: GridWorld) -> EpisodeOutcome:
        droid_ended_task = self._droid_ended_task(world)
        out_of_steps = world.steps_left <= 0

        if not (droid_ended_task or out_of_steps):
            return EpisodeOutcome(terminated=False)

        # The clock is charged only when it alone ended the episode. A task ended on the last step
        # is not also a timeout.
        if not droid_ended_task:
            return EpisodeOutcome(terminated=True, ending_reward=self.timeout_penalty)

        return EpisodeOutcome(
            terminated=True, ending_reward=self._task_ending_reward(world)
        )

    # ================= #
    #      Helpers      #
    # ================= #

    def _droid_ended_task(self, world: GridWorld) -> bool:
        """Whether the droid ended the task this step: out of energy, or a chain broken or
        completed."""

        tier_digester = world.droid.get_digester(TierDigester)

        return (
            world.droid.energy <= 0
            or tier_digester.chains_broken > 0
            or tier_digester.chains_completed > 0
        )

    def _task_ending_reward(self, world: GridWorld) -> float:
        """What the ending costs or pays when the droid ended the task."""

        return 0.0


@dataclass(frozen=True)
class DelayTermination(SingleTierChainTermination):
    """Ends a delay episode the way the legacy delay scenario did.

    A broken chain does not end the episode. The droid ends the task when it is out of energy,
    when the chain is completed, or when it has eaten every tier orb. Eating every tier orb
    without completing the chain costs half the timeout penalty.

    Attributes:
        max_tier: how many tier orbs the chain has, each eaten at most once.
    """

    # TODO(delay-ablation): each override is one of legacy's choices. The ablations remove them
    # one at a time; if the family's termination does as well, delay uses it and this goes.

    max_tier: int

    # ================= #
    #      Helpers      #
    # ================= #

    def _droid_ended_task(self, world: GridWorld) -> bool:
        tier_digester = world.droid.get_digester(TierDigester)

        return (
            world.droid.energy <= 0
            or tier_digester.chains_broken > 0
            or tier_digester.chains_completed > 0
            # or self._every_tier_orb_eaten(tier_digester)
        )

    def _task_ending_reward(self, world: GridWorld) -> float:
        tier_digester = world.droid.get_digester(TierDigester)

        if tier_digester.chains_completed == 0 and self._every_tier_orb_eaten(
            tier_digester
        ):
            return self.timeout_penalty / 2

        return 0.0

    # === Global === #

    def _every_tier_orb_eaten(self, tier_digester: TierDigester) -> bool:
        # Each eaten tier orb is one progress, break or completion. The field cannot tell: a frozen
        # tier orb is off it just as an eaten one is
        eaten = (
            tier_digester.chain_progresses
            + tier_digester.chains_broken
            + tier_digester.chains_completed
        )

        return eaten >= self.max_tier
