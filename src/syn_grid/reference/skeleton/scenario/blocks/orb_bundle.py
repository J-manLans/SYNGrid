"""
One orb kind, bundled: the orbs and the digester that scores them.

A scenario's orb side is a tuple of bundles, one per kind. Adding a kind to a
scenario is adding its bundle; the world builder joins the orbs and collects the
digesters without asking which kinds they are. That keeps the two from drifting
apart: there is no orb without a digester and no digester without orbs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from syn_grid.core.digestion.digestion import OrbDigester
from syn_grid.scenario.blocks.population import OrbPopulation


@dataclass(frozen=True)
class OrbBundle:
    """Everything a world needs for one orb kind.

    Attributes:
        population: makes this kind's orbs. It holds no episode state and
            returns new orbs on every call, so it can be held directly.
        make_digester: returns a new digester for this kind. A way to make one
            and not an instance, because a digester holds episode state and one
            scenario builds many worlds.
    """

    population: OrbPopulation
    make_digester: Callable[[], OrbDigester]
