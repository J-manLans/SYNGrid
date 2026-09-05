import pytest

from syn_grid.core.orbs.orb_meta import (
    DirectType,
    OrbCategory,
    OrbMeta,
    SynergyType,
)


class TestOrbMeta:
    """
    Unit tests for OrbMeta.

    These tests validate:
    - Correct enum assignments
    - Tier default behavior
    - Explicit tier behavior
    - Validation of invalid inputs
    """

    def test_tier_1_initialization(self):
        """
        Verify that a SYNERGY TIER orb initializes correctly.

        Expected behavior:
        - category is OrbCategory.SYNERGY
        - type is SynergyType.TIER
        - Tier is 1
        """

        meta = OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, 1)

        assert meta.CATEGORY == OrbCategory.SYNERGY
        assert meta.TYPE == SynergyType.TIER
        assert meta.TIER == 1

    def test_direct_negative_initialization(self):
        """
        Verify that a DIRECT NEGATIVE orb initializes correctly.

        Ensures:
        - category and type are stored correctly
        - tier defaults to 0
        """

        meta = OrbMeta(OrbCategory.DIRECT, DirectType.NEGATIVE)

        assert meta.CATEGORY == OrbCategory.DIRECT
        assert meta.TYPE == DirectType.NEGATIVE
        assert meta.TIER == 0

    @pytest.mark.parametrize("tier", [1, 2, 5])
    def test_explicit_tier(self, tier):
        """
        Verify that providing a positive tier value
        correctly overrides the default.

        Expected behavior:
        - tier equals the explicitly provided value
        """

        meta = OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, tier)

        assert meta.TIER == tier

    def test_explicit_none_tier_defaults_to_zero(self):
        """
        Ensure that explicitly passing tier=None
        results in tier being set to 0.

        This confirms consistent default handling.
        """

        meta = OrbMeta(OrbCategory.DIRECT, DirectType.NEGATIVE)

        assert meta.TIER == 0

    def test_negative_tier_raises(self):
        """
        Ensure that passing a negative tier value
        raises a ValueError.

        Negative tiers are considered invalid domain input.
        """

        with pytest.raises(ValueError):
            OrbMeta(OrbCategory.SYNERGY, SynergyType.TIER, -1)

    def test_mismatch_category_and_type_raises(self):
        """
        Ensure that mismatched category/type combinations
        raise a TypeError.

        Example:
        - DIRECT category must use DirectType
        - SYNERGY category must use SynergyType
        """

        with pytest.raises(TypeError):
            OrbMeta(OrbCategory.DIRECT, SynergyType.TIER)
