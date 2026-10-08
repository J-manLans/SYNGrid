"""Orb-field rules, tested against a real GridWorld.

The old code had three conditionals in `perform_droid_action` and one in
`reset`, each branching on a scenario flag, plus a `delay_mode` check buried in
the consumption path. They are the three decisions `SpawningRules` now states
outright, so these drive a real world and watch the field rather than asserting
on the rule object's attributes -- an attribute test would pass unchanged if the
rule stopped being called.
"""

import numpy as np

from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.action_space import DroidAction
from syn_grid.scenario.registry import build_scenario
from syn_grid.scenario.rules.population import WeightedPopulation
from syn_grid.scenario.rules.spawning import (
    LeaveOrbFieldAlone,
    ReactivateAllOrbs,
    RefillOrbField,
    SpawningRules,
)
from tests.utils.config_helpers import get_scenario, get_test_config


def _world(scenario=None, seed: int = 0, spawning=None) -> GridWorld:
    """A world with a seeded rng, reset exactly once.

    ``GridWorld.reset()`` falls back to an unseeded ``default_rng()``, so any
    second reset silently re-rolls the orb layout. Pass ``spawning`` to install
    rules before the single reset rather than resetting again afterwards.
    """

    conf = get_test_config()
    world = GridWorld(
        scenario or get_scenario(conf),
        conf.world.grid_conf,
        conf.world.orb_factory_conf,
        conf.world.droid_conf,
        conf.world.negative_orb_conf,
        conf.world.tier_orb_conf,
    )
    if spawning is not None:
        world._spawning = spawning
    world.reset(np.random.default_rng(seed))
    return world


def _spawning(**overrides) -> SpawningRules:
    base = {
        "fill_pool_on_reset": False,
        "max_active_orbs": 3,
        "tier_orb_expires": False,
        "delay_on_consume": False,
        "after_action": LeaveOrbFieldAlone(),
    }
    return SpawningRules(**{**base, **overrides})


# (droid row offset, droid col offset, action that steps from there onto the orb)
_STEP_ONTO = (
    (-1, 0, DroidAction.DOWN),
    (1, 0, DroidAction.UP),
    (0, -1, DroidAction.RIGHT),
    (0, 1, DroidAction.LEFT),
)


def _park_droid(world: GridWorld) -> None:
    """Hold the droid in a corner.

    A step ticks every orb timer whether or not the droid moves, so pinning it
    against a wall is enough to observe a timer without also risking a
    consumption -- which sets the timer to the orb's cooldown and would make an
    "it aged by one" assertion read as an increase. No orb can be sitting in the
    corner either: ``_empty_spawn_cell`` refuses to spawn on the droid.
    """

    world.droid.position = [0, 0]


def _count_spawns(world: GridWorld) -> list[int]:
    """Record every spawn attempt the world makes from now on."""

    calls: list[int] = []
    original = world.spawn_orb_if_ready

    def counting_spawn() -> None:
        calls.append(1)
        original()

    world.spawn_orb_if_ready = counting_spawn
    return calls


def _consume_one_orb(world: GridWorld):
    """Stand next to an active orb and step onto it, so a real consumption happens.

    Writing ``droid.position`` is not enough: a step moves the droid first and
    only then checks for a collision, so a consumption needs a real move onto
    the cell. Returns the orb that was consumed.
    """

    rows, cols = world._world_conf.grid_rows, world._world_conf.grid_cols
    for orb in list(world.active_orbs):
        r, c = orb.position
        for dr, dc, action in _STEP_ONTO:
            start = (r + dr, c + dc)
            if not (0 <= start[0] < rows and 0 <= start[1] < cols):
                continue
            if any(list(o.position) == list(start) for o in world.active_orbs):
                continue
            world.droid.position = [start[0], start[1]]
            world.perform_droid_action(action)
            return orb

    raise AssertionError("no active orb had a reachable neighbour")


class TestOnReset:
    def test_a_single_orb_is_placed_by_default(self):
        world = _world()

        assert len(world.active_orbs) == 1

    def test_filling_the_pool_places_one_orb_per_slot(self):
        world = _world(spawning=_spawning(fill_pool_on_reset=True, max_active_orbs=3))

        assert len(world.active_orbs) == 3

    def test_a_field_larger_than_the_pool_places_what_there_is(self):
        world = _world(spawning=_spawning(fill_pool_on_reset=True, max_active_orbs=99))

        assert len(world.active_orbs) == len(world.ALL_ORBS)

    def test_orbs_are_never_stacked_on_one_cell(self):
        world = _world(spawning=_spawning(fill_pool_on_reset=True, max_active_orbs=5))

        cells = [tuple(o.position) for o in world.active_orbs]
        assert len(set(cells)) == len(cells)
        assert tuple(world.droid.position) not in cells


class TestTierExpiry:
    """Whether tier orbs age on their timer. A chain holds its sequence still; a
    continuous world may not, and that is a difficulty knob."""

    def _tier_orb(self, world: GridWorld):
        # Fill the field first: a continuous world starts with a single orb, and
        # that one may well be a negative orb.
        return next(o for o in world.active_orbs if o.META.TIER != 0)

    def _world_with_tier_orb(self, **overrides) -> tuple[GridWorld, object]:
        conf = get_test_config()
        pool = len(
            WeightedPopulation(
                conf.world.orb_factory_conf,
                conf.world.negative_orb_conf,
                conf.world.tier_orb_conf,
            ).create()
        )
        world = _world(
            spawning=_spawning(
                fill_pool_on_reset=True,
                max_active_orbs=pool,
                after_action=LeaveOrbFieldAlone(),
                **overrides,
            )
        )
        _park_droid(world)
        return world, self._tier_orb(world)

    def test_a_tier_orb_does_not_age_when_expiry_is_off(self):
        world, orb = self._world_with_tier_orb(tier_orb_expires=False)
        before = orb.TIMER.remaining

        world.perform_droid_action(DroidAction.UP)

        assert orb.TIMER.remaining == before

    def test_a_tier_orb_ages_when_expiry_is_on(self):
        world, orb = self._world_with_tier_orb(tier_orb_expires=True)
        before = orb.TIMER.remaining

        world.perform_droid_action(DroidAction.UP)

        assert orb.TIMER.remaining == before - 1

    def test_a_direct_orb_always_ages(self):
        """Only tier orbs are exempt, and only from their own timer. A direct orb
        ages either way, or it would never leave the board."""

        world, _ = self._world_with_tier_orb(tier_orb_expires=False)
        direct = next(o for o in world.active_orbs if o.META.TIER == 0)
        before = direct.TIMER.remaining

        world.perform_droid_action(DroidAction.UP)

        assert direct.TIMER.remaining == before - 1

    def test_a_tier_orb_de_spawns_once_its_timer_runs_out(self):
        """The consequence of expiry, and the reason a chain forbids it: the orb
        leaves the board and the sequence has a hole in it. Asserted on the orb
        rather than on the field size -- direct orbs age in the same step, so the
        count can drop by more than the one under test."""

        world, orb = self._world_with_tier_orb(tier_orb_expires=True)
        assert orb.is_active
        orb.TIMER.set(1)

        world.perform_droid_action(DroidAction.UP)

        assert not orb.is_active, "the orb leaves the board rather than waiting"
        assert orb.TIMER.remaining > 0, "and goes on cooldown before it can return"


class TestDelayOnConsume:
    def test_consumption_leaves_the_field_alone_without_delay(self):
        world = _world(
            spawning=_spawning(delay_on_consume=False, after_action=RefillOrbField())
        )
        orb = _consume_one_orb(world)

        assert not orb.is_active, "the orb should have been consumed"
        # Its cooldown is its own, not the whole field's: another orb is still up.
        assert any(o.is_active for o in world.active_orbs)

    def test_delay_puts_the_whole_field_on_cooldown(self):
        """The defining property of the delay mechanic: one consumption takes the
        entire field off the board, not just the orb that was taken."""

        world = _world(
            spawning=_spawning(
                fill_pool_on_reset=True,
                max_active_orbs=3,
                delay_on_consume=True,
                after_action=LeaveOrbFieldAlone(),
            )
        )
        assert len(world.active_orbs) == 3

        _consume_one_orb(world)

        assert not any(o.is_active for o in world.active_orbs)


class TestAfterAction:
    def test_leaving_the_field_alone_changes_nothing(self):
        world = _world(
            spawning=_spawning(after_action=LeaveOrbFieldAlone(), max_active_orbs=3)
        )
        before = len(world.active_orbs)

        world.perform_droid_action(DroidAction.UP)

        assert len(world.active_orbs) == before

    def test_refilling_spawns_at_most_one_orb_per_step(self):
        """One spawn attempt per step, not a loop that fills the field at once.

        Counted as calls rather than as field size: the droid can eat an orb on
        the same step, so the field's size can stay flat while a spawn happened,
        and a size-based assertion would pass against a loop.
        """

        world = _world(
            spawning=_spawning(after_action=RefillOrbField(), max_active_orbs=5)
        )

        calls = _count_spawns(world)

        world.perform_droid_action(DroidAction.UP)

        assert len(calls) == 1

    def test_refilling_grows_the_field_and_stops_at_its_size(self):
        world = _world(
            spawning=_spawning(after_action=RefillOrbField(), max_active_orbs=3)
        )
        _park_droid(world)
        assert len(world.active_orbs) == 1

        for _ in range(20):
            world.perform_droid_action(DroidAction.UP)

        assert len(world.active_orbs) == 3

    def test_a_delay_field_is_reactivated_once_the_cooldown_lapses(self):
        world = _world(
            spawning=_spawning(
                fill_pool_on_reset=True,
                max_active_orbs=3,
                delay_on_consume=True,
                after_action=ReactivateAllOrbs(),
            )
        )
        _consume_one_orb(world)
        assert not any(o.is_active for o in world.active_orbs)

        # Driven directly rather than by stepping: the droid would wander, and
        # whether it walks into an orb during the cooldown depends on the layout,
        # which depends on which test happened to set the shared orb lifespan
        # last. See AGENTS.md, "Global mutable state".
        for orb in world.active_orbs:
            orb.TIMER.set(0)

        world._spawning.after_step(world)

        assert all(o.is_active for o in world.active_orbs)

    def test_reactivation_leaves_orbs_alone_while_they_are_cooling_down(self):
        world = _world(
            spawning=_spawning(
                fill_pool_on_reset=True,
                max_active_orbs=3,
                delay_on_consume=True,
                after_action=ReactivateAllOrbs(),
            )
        )
        _consume_one_orb(world)

        world._spawning.after_step(world)

        assert not any(o.is_active for o in world.active_orbs)

    def test_after_step_is_what_invokes_reactivation(self):
        """The rule is reached through the world's step, not by the test calling it
        directly. Asserted with a spy so the wiring is covered even though the
        cooldown itself cannot be advanced deterministically."""

        world = _world(
            spawning=_spawning(
                fill_pool_on_reset=True,
                max_active_orbs=3,
                after_action=ReactivateAllOrbs(),
            )
        )

        calls = _count_spawns(world)
        original = world.reactivate_all_orbs

        def counting_reactivate() -> None:
            calls.append(1)
            original()

        world.reactivate_all_orbs = counting_reactivate

        world.perform_droid_action(DroidAction.UP)

        assert calls, "after_step did not reach the world"

    def test_reactivation_returns_orbs_to_where_they_were(self):
        """Not to fresh random cells. A delay scenario's cooldown freezes the
        field in place, which is what makes the layout learnable. The consumed orb
        is not part of this: it left the field and is waiting on its own."""

        world = _world(
            spawning=_spawning(
                fill_pool_on_reset=True,
                max_active_orbs=3,
                delay_on_consume=True,
                after_action=ReactivateAllOrbs(),
            )
        )
        consumed = _consume_one_orb(world)
        expected = sorted(
            tuple(o.position) for o in world.active_orbs if o is not consumed
        )
        assert expected, "expected the field to still have orbs on it"

        for orb in world.active_orbs:
            orb.TIMER.set(0)
        world._spawning.after_step(world)

        assert sorted(tuple(o.position) for o in world.active_orbs) == expected


class TestScenarioWiring:
    """The rules a scenario actually ships with, checked against the world they
    are handed. A mismatch here is a scenario that lies about itself."""

    def test_a_goal_scenario_starts_with_its_whole_chain_on_the_field(self):
        conf = get_test_config()
        world_conf = conf.world.model_copy(
            update={
                "grid_world_conf": conf.world.grid_conf.model_copy(
                    update={"max_tier": 3, "max_active_orbs": 3}
                )
            }
        )
        obs_conf = conf.obs
        scenario = build_scenario("goal_tier_chain_spatial", world_conf, obs_conf)

        world = GridWorld(
            scenario,
            world_conf.grid_conf,
            world_conf.orb_factory_conf,
            world_conf.droid_conf,
            world_conf.negative_orb_conf,
            world_conf.tier_orb_conf,
        )
        world.reset(np.random.default_rng(0))

        assert len(world.active_orbs) == 3
        assert sorted(o.META.TIER for o in world.active_orbs) == [1, 2, 3]

    def test_a_continuous_scenario_starts_with_a_single_orb(self):
        world = _world()

        assert len(world.active_orbs) == 1
