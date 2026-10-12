"""
How a scenario makes its worlds.

A scenario is built once and makes many worlds, and a world holds episode
state, so a scenario holds a way to make a world and never a world. This is
that way, shared by every scenario family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from syn_grid.config.models.scenario.common_models import WorldConf
from syn_grid.scenario.blocks.orb_bundle import OrbBundle

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld

# ################## #
#    World Recipe    #
# ################## #


@dataclass(frozen=True)
class WorldRecipe:
    """Makes a new world each time it is called.

    It is what a `Scenario` holds as `build_world`. It is called once per
    environment, so everything that holds episode state, the digesters, the
    droid and the orbs, is created in the call and never shared through the
    scenario. The orbs are every bundle's orbs joined and the digesters are
    one per bundle; nothing asks which kinds they are.

    Only what every family's world has is read from the config: the grid, the
    droid's settings and the step limit. What a family adds to its world
    config reaches the world through the bundles.

    Frozen so that two recipes with the same pieces compare equal.

    Attributes:
        world_conf: the scenario's world config.
        orb_bundles: one bundle per orb kind in the world.
        max_active_orbs: how many orbs the field may hold at once. In a fixed
            field it is every orb the bundles make.
    """

    world_conf: WorldConf
    orb_bundles: tuple[OrbBundle, ...]
    max_active_orbs: int

    def __call__(self) -> GridWorld:
        # orbs = [orb for bundle in self.orb_bundles for orb in bundle.population.create()]
        # digesters = [bundle.make_digester() for bundle in self.orb_bundles]
        # grid = (self.world_conf.grid_conf.grid_rows, self.world_conf.grid_conf.grid_cols)
        # TODO(phase 3): build the droid from the digesters, the grid and `droid_conf`, then return
        # GridWorld(grid, droid, orbs, self.max_active_orbs).
        raise NotImplementedError("WorldRecipe waits for core to take pieces (phase 3)")
