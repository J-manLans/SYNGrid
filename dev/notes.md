## Issues
-

## Implementation thoughts
- Might need to change from stable baselines Monitor to:
    `from gymnasium.wrappers import RecordEpisodeStatistics`
    Then I might be able to add it as part of the BaseAgent abstract contract.
- Might extract `_create_orbs()` from `GridWorld` to a python module. I have a feeling as more orbs are added this method might explode...but we'll see. No need to jump the gun on it.
- So returning a (H, W) shaped obs would look something like this:

```python
def _build_grid_observation(self):
    grid = np.zeros((5, 5), dtype=np.float32)

    # example: agent
    grid[self.agent_x, self.agent_y] = 1

    # example: orb
    grid[2, 2] = 3

    # example: other orb
    grid[1, 3] = 2

    return grid

    # and this is what that could look like
    obs = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 2, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
```

And the setup would look like this:

```python
spaces.Box(
    low=0,
    high=3,
    shape=(5,5),
    dtype=np.float32
    )
```

- A HWC shape is a 3D array with shape (H, W, C). The spatial structure (H, W) is preserved, but each cell is no longer a single value. Instead, each cell contains a vector of length C, representing multiple features of that location, for example:

```python
obs = [
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
    [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]],
]
```

And the setup would look like this:

```python
spaces.Box(
    # low and high is what C can contain, while rows and cols determine
    # the dimension of the space
    low=0,
    high=3,
    shape=(grid_rows, grid_cols, C),
    dtype=np.float32
)
```

And the observation the agent would get is:

```python
def _build_grid_observation(self):
    grid = np.zeros((5, 5, 11), dtype=np.float32)

    # example: agent
    grid[self.agent_x, self.agent_y, 0] = 23
    grid[self.agent_x, self.agent_y, 1] = 2

    # example: orb 1
    grid[self.orb1_x, self.orb1_y, 2] = 1
    grid[self.orb1_x, self.orb1_y, 3] = 2
    grid[self.orb1_x, self.orb1_y, 4] = 1

    # example: orb 2
    grid[self.orb2_x, self.orb2_y, 5] = 1
    grid[self.orb2_x, self.orb2_y, 6] = 1
    grid[self.orb2_x, self.orb2_y, 7] = 3

    # example: orb 3
    grid[self.orb3_x, self.orb3_y, 8] = 2
    grid[self.orb3_x, self.orb3_y, 9] = 2
    grid[self.orb3_x, self.orb3_y, 10] = 1

    return grid
```

- So for my HWC observation for the hard difficulty I want the channels to contain:

```python
[is_agent, orb_category, orb_type, orb_tier]

# empty cell:
[0, 0, 0, 0]

# agent:
[1, 0, 0, 0]

# negative orb
[0, 1, 1, 0]

# tier1 orb
[0, 2, 1, 1]

# tier3 orb
[0, 2, 1, 3]
```
