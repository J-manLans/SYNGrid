"""Tier-chain digestion rules.

The concrete tier-chain implementation will be moved here from the legacy
DigestionEngine as the scenario refactor progresses.

This file intentionally contains only the new scenario-facing structure for
now; the existing digestion implementation remains untouched.
"""

from __future__ import annotations

from syn_grid.scenario.rules.digestion import DigestionResult, DigestionRules
from syn_grid.core.orbs.base_orb import BaseOrb


class TierDigestion(DigestionRules):
    """Scenario-specific digestion for tier-chain tasks."""

    def reset(self) -> None:
        """Reset tier-chain digestion state."""
        raise NotImplementedError

    def digest(self, orb: BaseOrb) -> DigestionResult:
        """Digest a consumed tier-chain orb."""
        raise NotImplementedError
