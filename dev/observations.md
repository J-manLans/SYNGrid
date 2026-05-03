```python
# =======================
# CompositeGridMarkovian
# =======================

spaces.Dict({
    "grid": spaces.Box(
        low=low,
        high=high,
        shape=(H, W, C),
        dtype=np.float32
    ),
    "globals": spaces.Box(
        low=0,
        high=1,
        shape=(3,),
        dtype=np.float32
    ),
})

'''
C = [
    droid_presence, # binary
    is_direct, # binary
    is_synergy, # binary
    is_negative, # binary
    is_tier, # binary
    is_converter, # binary
    ...
    tier, # continuous
    timer_remaining # continuous
]

globals = [steps_left, score, chained_tiers]
'''

# ========================================================
# CompositeMarkovian (future - requires custom extractor)
# ========================================================

spaces.Dict({
    "orbs": spaces.Tuple([
        spaces.Dict({
            "is_active": spaces.MultiBinary(1)
            "identity": spaces.MultiDiscrete([n_categories, n_types, n_tiers]),
            "position": spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32),
            "timer": spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32),
        })
    ] * max_active_orbs),
    "agent": spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32),
    "globals": spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32),
})

'''
identity = [category, type, tier]  # MultiDiscrete axes
position = [y, x]
globals = [steps_left, score, chained_tiers]
'''
```
