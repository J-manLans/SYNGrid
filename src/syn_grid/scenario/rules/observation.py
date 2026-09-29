"""
What a scenario lets the agent see.

The scenario owns how many orb slots an observation is built to hold, because
that is a property of the world rather than of the perception encoding. Two
different counts are involved and they are not the same number, which is why
they are named rather than inlined:

* ``observation_slot_count`` sizes the observation vector. Under curriculum
  training a tier chain reserves more slots than there are orbs, so the agent
  sees a fixed-width field regardless of how far the world has progressed.
* ``sort_limit`` caps the distance sort that picks which orbs get written into
  those slots. This one has never followed the curriculum setting, so with
  curriculum on, the trailing slots are sized but left at zero. That is
  existing behaviour and the reason the two are not collapsed into one.

``include_timer`` stays in the observation config rather than moving here. It
is a choice about encoding, not about the world: only the perceptions that
have a spare column for it read it, and nothing forces the two to agree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationRules:
    """How many orb slots the agent's observation holds, and how many orbs
    compete for them.

    Attributes:
        observation_slot_count: sizes the observation vector.
        sort_limit: caps the distance sort that chooses which orbs are written
            into those slots.
        max_tier: the upper bound a tier value is encoded against, for the
            perceptions that give tier its own channel. Taken from the world
            rather than from the observation config, which used to carry a
            second copy that had to be kept in step by hand.
    """

    observation_slot_count: int
    sort_limit: int
    max_tier: int
