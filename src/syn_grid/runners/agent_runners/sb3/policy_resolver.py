"""Resolve SB3 policy strings from SYNGrid's perception configuration."""

from stable_baselines3.common.base_class import BaseAlgorithm

# Maps a perception name's leading segment (before the first "_") to its
# base SB3 policy. e.g. "vector_markovian_easy" -> "vector" -> "Mlp".
_POLICY_PREFIXES: dict[str, str] = {
    "vector": "Mlp",
    "composite": "MultiInput",
    "grid": "Cnn",
}


def resolve_policy(perception: str, algorithm: type[BaseAlgorithm]) -> str:
    """
    Map a perception name to the SB3 policy string the algorithm offers for it.

    The name is taken from the algorithm's own `policy_aliases`: the one that
    starts with the perception's base policy.

    Args:
        perception: The configured perception name, e.g. "vector_fog_of_war".
        algorithm: The SB3 algorithm class the policy is for.

    Returns:
        An SB3 policy string, e.g. "MlpPolicy" for PPO or "MlpLstmPolicy" for
        RecurrentPPO.

    Raises:
        ValueError: If the perception's leading segment isn't a known prefix,
            or if the algorithm doesn't offer exactly one policy for it.
    """

    prefix = perception.split("_", 1)[0]
    base_policy = _POLICY_PREFIXES.get(prefix)

    if base_policy is None:
        raise ValueError(
            f"No SB3 policy mapping found for perception '{perception}'. "
            f"Known prefixes: {list(_POLICY_PREFIXES)}"
        )

    # Get the policy for the algorithm in the SB3 package
    policies = [
        name for name in algorithm.policy_aliases if name.startswith(base_policy)
    ]

    # Make sure we only got one match
    if len(policies) != 1:
        raise ValueError(
            f"Expected exactly one '{base_policy}' policy in {algorithm.__name__} "
            f"for perception '{perception}', found {policies}. "
            f"It offers: {list(algorithm.policy_aliases)}"
        )

    return policies[0]
