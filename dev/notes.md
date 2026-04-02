## Issues
-

## Implementation thoughts
- Might need to change from stable baselines Monitor to:
    `from gymnasium.wrappers import RecordEpisodeStatistics`
    Then I might be able to add it as part of the BaseAgent abstract contract.
- Might extract `_create_orbs()` from `GridWorld` to a python module. I have a feeling as more orbs are added this method might explode...but we'll see. No need to jump the gun on it.