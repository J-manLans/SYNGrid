"""Digestion rules for scenario-specific orb consumption.

This module defines the contract used by core runtime components without
encoding any particular scenario's digestion mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from syn_grid.core.orbs.base_orb import BaseOrb


@dataclass(frozen=True)
class DigestionResult:
    """Outcome of digesting one orb."""

    reward: float
    chain_broken: bool = False
    chain_completed: bool = False


class DigestionRules(Protocol):
    """Scenario-specific rules for digesting consumed orbs."""

    def reset(self) -> None:
        """Reset episode-level digestion state."""
        ...

    def digest(self, orb: BaseOrb) -> DigestionResult:
        """Digest one consumed orb and return its scenario outcome."""
        ...
