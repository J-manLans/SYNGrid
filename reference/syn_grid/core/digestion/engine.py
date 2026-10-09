"""Routes consumed orbs to the digester that owns their kind."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TypeVar

from syn_grid.core.digestion.digestion import (
    Event,
    OrbDigester,
    OrbKind,
    kind_of,
)
from syn_grid.core.orbs.base_orb import BaseOrb

D = TypeVar("D", bound=OrbDigester)


class DigestionEngine:
    """A router with a tally. It holds no scoring rules of its own.

    Per consumed orb it:
      1. hands the orb to the digester that owns its kind,
      2. lets every other digester notice that it happened,
      3. records the events from both.

    Effect orbs that reshape rewards (a converter, a synergy) are meant to
    plug in as reward modifiers applied to the digester's reward in step 1.
    That is deliberately not built yet: where they sit depends on whether a
    modifier should touch a delayed payout, which is still an open question.
    """

    def __init__(self, digesters: Iterable[OrbDigester]):
        self._digesters: dict[OrbKind, OrbDigester] = {}

        for digester in digesters:
            if digester.kind in self._digesters:
                raise ValueError(f"Two digesters registered for kind {digester.kind}")

            self._digesters[digester.kind] = digester

        self._counts: Counter[type[Event]] = Counter()

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        for digester in self._digesters.values():
            digester.reset()
        self._counts.clear()

    def digest(self, orb: BaseOrb) -> float:
        """Digest a consumed orb and return the resulting reward."""

        kind = kind_of(orb)
        owner = self._digesters.get(kind)

        if owner is None:
            raise KeyError(f"No digester registered for orb kind {kind}")

        digestion_result = owner.digest(orb)

        self._counts.update(type(event) for event in digestion_result.events)

        for digester in self._digesters.values():
            if digester is not owner:
                self._counts.update(type(event) for event in digester.notice(orb))

        return digestion_result.reward

    def count(self, event_type: type[Event]) -> int:
        """How many times this event has occurred since the last reset."""

        return self._counts[event_type]

    def get(self, digester_type: type[D]) -> D:
        """The registered digester of this type, for readers that need its state."""

        for digester in self._digesters.values():
            if isinstance(digester, digester_type):
                return digester
        raise KeyError(f"No {digester_type.__name__} registered")
