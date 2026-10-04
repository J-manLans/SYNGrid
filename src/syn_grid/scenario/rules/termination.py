"""
When an episode ends, and what the last step is worth.

Termination is the clearest case of scenario knowledge sitting in infrastructure:
the old implementation branched on ``world._conf.single_chain_mode``,
``world._conf.max_tier_scoring`` and ``world._conf.curriculum_training`` from
outside, and reached into ``digestion_engine._pending_reward`` while doing it.
All three decisions now belong to whoever defines the scenario, so a rule is
constructed with them.

There are two implementations because the two bodies are genuinely disjoint --
a goal scenario has a chain to break and a ceiling to hit, a continuous one has
neither. There is no shared base class beyond the score check, because that is
all they actually share, and inventing a hierarchy over two classes that happen
to be siblings is the pattern-chasing this refactor is meant to avoid.

``truncated`` is always False. Running out of steps is a punishment and sets
``terminated``, so no value function is bootstrapped at the horizon. That is a
deliberate decision with a real cost, pinned by a test so it is not later
"fixed" by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from syn_grid.config.models.scenario.orb_models import ScoringMode

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld

# Reward paid on completing a chain outside curriculum training, where the
# consumption reward is replaced by a fixed ceiling. The engine's own reward is
# unbounded across a run; the ceiling is what makes one completed chain worth
# a predictable amount.
COMPLETION_CEILING: float = 10.0

# When a delay scenario runs out of orbs with steps to spare, the episode ends
# early and pays half the timeout penalty. Half rather than all of it, because
# finishing the field is a partial success, not a timeout.
EXHAUSTED_FIELD_REWARD_SHARE: float = 0.5


@dataclass(frozen=True)
class EpisodeOutcome:
    terminated: bool
    truncated: bool
    reward: float


class TerminationRules(Protocol):
    def evaluate(
        self, world: GridWorld, steps_left: int, reward: float
    ) -> EpisodeOutcome: ...


@dataclass(frozen=True)
class GoalTermination:
    """Ends a goal episode: a broken chain, a finished chain, the clock, or an
    exhausted delay field.

    Attributes:
        timeout_penalty: paid for reaching the step limit unfinished.
        delay: whether the scenario runs the delay mechanic, which both
            suppresses the chain-break termination and changes what a timeout
            pays.
        scoring: the world's scoring mode, which decides whether a timeout
            settles a partially accumulated reward.
        curriculum: under curriculum training a completed chain keeps the
            engine's own reward instead of being replaced by the ceiling.
    """

    timeout_penalty: float
    delay: bool
    scoring: ScoringMode
    curriculum: bool

    def evaluate(
        self, world: GridWorld, steps_left: int, reward: float
    ) -> EpisodeOutcome:
        engine = world.droid.digestion_engine
        terminated = world.droid.score <= 0

        if engine.chains_broken > 0 and not self.delay:
            # A delay scenario lets a broken chain be recovered from, so the
            # break does not end the episode there.
            terminated = True

        elif steps_left <= 0:
            reward = self._timeout_reward(world)
            terminated = True

        elif engine.chains_completed > 0:
            if not self.curriculum and self.scoring is ScoringMode.MAX_TIER:
                reward = COMPLETION_CEILING
            terminated = True

        elif self.delay and len(world.active_orbs) == 0:
            # Delay mode with the field emptied and the clock still running.
            # Not a timeout -- the objective ran out of material first.
            terminated = True
            reward = self.timeout_penalty * EXHAUSTED_FIELD_REWARD_SHARE

        return EpisodeOutcome(terminated=terminated, truncated=False, reward=reward)

    def _timeout_reward(self, world: GridWorld) -> float:
        """Settle the terminal reward for reaching the step limit.

        Max-tier scoring never accumulates a partial reward, so there is
        nothing to settle up and the timeout pays the configured penalty. A
        delay scenario pays it too, whatever the scoring mode: with the field
        on cooldown the partial reward is not something the agent earned
        towards the objective.
        """

        if self.scoring is ScoringMode.MAX_TIER:
            return self.timeout_penalty

        reward = world.droid.digestion_engine.pending_reward

        if self.delay:
            return self.timeout_penalty

        return reward


@dataclass(frozen=True)
class ContinuousTermination:
    """Ends a continuous episode: only the score running out, or the clock.

    There is no chain to break and nothing to complete, so the clock is the
    only interesting boundary. A timeout still settles a partially accumulated
    reward unless the scoring mode is max-tier, which never accumulates one.
    """

    scoring: ScoringMode

    def evaluate(
        self, world: GridWorld, steps_left: int, reward: float
    ) -> EpisodeOutcome:
        terminated = world.droid.score <= 0

        if steps_left <= 0:
            if self.scoring is not ScoringMode.MAX_TIER:
                reward = world.droid.digestion_engine.pending_reward
            terminated = True

        return EpisodeOutcome(terminated=terminated, truncated=False, reward=reward)
