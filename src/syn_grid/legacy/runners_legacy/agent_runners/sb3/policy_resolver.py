"""Resolve SB3 policy strings from SYNGrid's perception configuration."""

# Maps a perception name's leading segment (before the first "_") to its
# base SB3 policy. e.g. "vector_markovian_easy" -> "vector" -> "Mlp".
_POLICY_PREFIXES: dict[str, str] = {
    "vector": "Mlp",
    "composite": "MultiInput",
    "grid": "Cnn",
}


def resolve_policy(perception: str, use_lstm: bool = False) -> str:
    """
    Map a perception name to its SB3 policy string.

    Args:
        perception: The configured perception name, e.g. "vector_fog_of_war".
        use_lstm: Whether to resolve the recurrent variant of the policy.

    Returns:
        An SB3 policy string, e.g. "MlpPolicy" or "MlpLstmPolicy".

    Raises:
        ValueError: If the perception's leading segment isn't a known prefix.
    """

    prefix = perception.split("_", 1)[0]
    base_policy = _POLICY_PREFIXES.get(prefix)

    if base_policy is None:
        raise ValueError(
            f"No SB3 policy mapping found for perception '{perception}'. "
            f"Known prefixes: {list(_POLICY_PREFIXES)}"
        )

    suffix = "LstmPolicy" if use_lstm else "Policy"
    return base_policy + suffix
