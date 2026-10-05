"""Builds a DigestionEngine from the orb and droid configs.

Which digesters exist follows from which orbs the config enables, not from
which scenario is being built. A scenario builder calls this the same way it
calls the orb factory.
"""

from __future__ import annotations

from syn_grid.config.models.common_models import DroidConf, OrbPoolConf
from syn_grid.config.models.tier_chain_models import TierDroidConf, TierOrbPoolConf
from syn_grid.core.digestion.new_digestion import OrbDigester
from syn_grid.core.digestion.new_engine import DigestionEngine
from syn_grid.core.digestion.new_negative_digester import NegativeDigester
from syn_grid.core.digestion.new_tier_digester import TierDigester


def build_digestion(orb_conf: OrbPoolConf, droid_conf: DroidConf) -> DigestionEngine:
    digesters: list[OrbDigester] = []

    if isinstance(orb_conf, TierOrbPoolConf):
        if not isinstance(droid_conf, TierDroidConf):
            raise TypeError("Tier orbs need a TierDroidConf for their penalties")
        digesters.append(
            TierDigester(
                scoring=orb_conf.tier.scoring,
                max_tier=orb_conf.tier.max_tier,
                tier_consumption_penalty=droid_conf.tier_consumption_penalty,
                reward_multiplier=droid_conf.reward_multiplier,
                chain_break_penalty=droid_conf.chain_break_penalty,
            )
        )

    if orb_conf.negative is not None:
        digesters.append(NegativeDigester())

    return DigestionEngine(digesters)
