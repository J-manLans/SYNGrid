"""Digester for negative orbs."""

from __future__ import annotations

from syn_grid.core.digestion.digestion import DigestionResult, Event, OrbKind
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectType, OrbCategory


# #################### #
# OrbDigester Strategy #
# #################### #


class NegativeDigester:
    """A negative orb is worth exactly its own reward, and has no state."""

    kind: OrbKind = (OrbCategory.DIRECT, DirectType.NEGATIVE)

    def reset(self) -> None:
        pass

    def digest(self, orb: BaseOrb) -> DigestionResult:
        return DigestionResult(orb.REWARD)

    def notice(self, orb: BaseOrb) -> tuple[Event, ...]:
        return ()
