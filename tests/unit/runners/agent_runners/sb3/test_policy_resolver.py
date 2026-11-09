"""Which SB3 policy a perception gets: by the kind its name starts with, as the algorithm names it."""

from typing import ClassVar

import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

from syn_grid.runners.agent_runners.sb3.policy_resolver import resolve_policy

# ================= #
#     Stand-ins     #
# ================= #


class _TwoMlpPolicies:
    """An algorithm that offers two policies for a vector perception, so neither can be picked."""

    __name__ = "TwoMlpPolicies"
    policy_aliases: ClassVar[dict[str, type]] = {
        "MlpPolicy": object,
        "MlpLstmPolicy": object,
    }


# ================= #
#       Tests       #
# ================= #


class TestResolve:
    """A perception's kind, the part of its name before the first `_`, picks the policy."""

    @pytest.mark.parametrize(
        ("perception", "policy"),
        [
            ("vector_fog_of_war", "MlpPolicy"),
            ("composite_markovian", "MultiInputPolicy"),
            ("grid_pixel", "CnnPolicy"),
        ],
    )
    def test_each_kind_gets_its_policy(self, perception: str, policy: str):
        assert resolve_policy(perception, PPO) == policy

    def test_the_policy_is_named_as_the_algorithm_names_it(self):
        assert resolve_policy("vector_markovian", RecurrentPPO) == "MlpLstmPolicy"

    def test_a_kind_with_no_policy_is_refused(self):
        with pytest.raises(ValueError, match="pixel_markovian"):
            resolve_policy("pixel_markovian", PPO)

    def test_an_algorithm_offering_two_policies_for_the_kind_is_refused(self):
        with pytest.raises(ValueError, match="exactly one"):
            resolve_policy("vector_markovian", _TwoMlpPolicies)  # type: ignore[arg-type]
