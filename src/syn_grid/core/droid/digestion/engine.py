"""
The digestion engine: the router that hands each consumed orb to the digester
of its kind.

The engine holds no scoring rules and no episode state of its own. Everything
it knows is which digester owns which orb kind.
"""

from collections.abc import Iterable
from typing import Final, TypeVar

from syn_grid.core.droid.digestion.digester import OrbDigester
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbKind

D = TypeVar("D", bound=OrbDigester)


class DigestionEngine:
    """Routes consumed orbs to their digesters, one digester per orb kind.

    A mistake in how the digesters were put together is an error, never a
    silent default: two digesters for one kind, an orb whose kind has no
    digester, and a reader asking for a digester that is not here all raise.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, digesters: Iterable[OrbDigester]):
        self._digesters: Final[dict[OrbKind, OrbDigester]] = {}

        for digester in digesters:
            if digester.KIND in self._digesters:
                raise ValueError(
                    f"Two digesters own the orb kind {digester.KIND}: "
                    f"{type(self._digesters[digester.KIND]).__name__} and "
                    f"{type(digester).__name__}"
                )

            self._digesters[digester.KIND] = digester

    def reset(self) -> None:
        """Clear every digester's episode state, at the start of an episode."""

        for digester in self._digesters.values():
            digester.reset()

    # ================= #
    #        API        #
    # ================= #

    def digest(self, orb: BaseOrb) -> float:
        """Hand a consumed orb to the digester of its kind and return the reward."""

        kind = orb.META.KIND

        if kind not in self._digesters:
            raise KeyError(f"No digester owns the orb kind {kind}")

        return self._digesters[kind].digest(orb)

    def get(self, digester_class: type[D]) -> D:
        """Return the digester that is an instance of this class.

        This is how a reader reaches what a digester knows. Asking for a base
        class finds the subclass that is registered, so a reader does not need
        to know which scoring mode a scenario uses.
        """

        for digester in self._digesters.values():
            if isinstance(digester, digester_class):
                return digester

        raise KeyError(f"No {digester_class.__name__} is registered")
