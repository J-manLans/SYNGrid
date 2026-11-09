"""What an orb meta accepts, and the identity it gives the agent to tell orbs apart by."""

import pytest

from syn_grid.core.orbs.orb_meta import (
    DirectKind,
    OrbCategory,
    OrbKind,
    OrbMeta,
    SynergyKind,
)

# ================= #
#      Helpers      #
# ================= #


def _identity(kind: OrbKind, tier: int | None = None) -> int:
    """The identity an orb of this kind and tier is given."""

    return OrbMeta(kind, tier).IDENTITY


# ================= #
#       Tests       #
# ================= #


class TestWhatItAccepts:
    """A tier for a tier orb, of at least 1, no tier for any other kind, and only kinds from the two enums."""

    def test_a_tier_orb_without_a_tier_or_below_tier_1_is_refused(self):
        with pytest.raises(ValueError):
            OrbMeta(SynergyKind.TIER)

        with pytest.raises(ValueError):
            OrbMeta(SynergyKind.TIER, tier=0)

    def test_any_other_kind_with_a_tier_is_refused(self):
        with pytest.raises(ValueError):
            OrbMeta(DirectKind.POSITIVE, tier=1)

    def test_a_kind_from_neither_enum_is_refused(self):
        with pytest.raises(TypeError):
            OrbMeta(1)  # pyright: ignore[reportArgumentType]


class TestEnumValues:
    """The values the categories and kinds are numbered by."""

    def test_they_start_at_1_so_0_is_left_for_no_orb(self):
        values = [member.value for member in (*OrbCategory, *DirectKind, *SynergyKind)]

        assert min(values) == 1


class TestIdentity:
    """What the agent tells orbs apart by."""

    def test_orbs_of_the_same_kind_and_tier_share_one(self):
        assert _identity(SynergyKind.TIER, 2) == _identity(SynergyKind.TIER, 2)

    def test_orbs_that_differ_in_kind_or_tier_get_different_ones(self):
        identities = [
            _identity(DirectKind.NEGATIVE),
            _identity(DirectKind.POSITIVE),
            *(_identity(SynergyKind.TIER, tier) for tier in (1, 2, 3, 4)),
        ]

        # A set drops repeats, so it comes out shorter than the list if two orbs share an identity
        assert len(set(identities)) == len(identities)
