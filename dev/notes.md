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

obs[0, 2, 0] = 1
obs[0, 2, 1] = 23
obs[0, 2, 2] = 23
obs = [
    [[0,0,0], [0,0,0], [1,23,23], [0,0,0], [0,0,0]],
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

# Example with global values steps left, score and chained tiers at the start [33, 54, 1, ..., 0]:
obs = [
    [[33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0]],
    [[33,54,1,0,0,0,0,0], [33,54,1,1,0,0,0,0], [33,54,1,0,1,1,0,5], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0]],
    [[33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0]],
    [[33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,2,1,2,3], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0]],
    [[33,54,1,0,2,1,1,8], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0], [33,54,1,0,0,0,0,0]],
]
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

## n_step and batch_size hyper parameters for RecurrentPPO
In my environment the maximum length of an episode is 100 steps then the episode terminates, the other deciding factor for termination is if the droid's score reaches 0. Each step in the env cost 1 score and the agent starts with 50. So effective length of an episode is [50-100]

### n_step = 128, batch_size = 32 (or whatever that bigger than the actual collection of episodes, which will be maximum 2)
Collect phase:
- Run environment for n_steps=128 steps
- Gives 2 complete episodes (30-100,0-58 steps each)

Update phase:
- Only have 2 episode to work with
- Epoch 1:
    - Since batch_size is so big we update on all episodes at once
- Epoch 2: same update
- Epoch 3: same
- Epoch 4: same
- Total: 4 gradient updates

Takeaway:
The agent sees only partial episodes, no chance to learn long term dynamics, this is catastrophic for the agents learning

### n_step = 512, batch_size = 64 (or whatever that bigger than the actual collection of episodes, which will be maximum 2)
Collect phase:
- Run environment for n_steps=512 steps
- Gives 5 complete episodes
- 5 / 64 < 0 so we will train on the whole batch in one go

Update phase:
- Only have 5 episode to work with, plus whatever leftover
- Epoch 1:
    - Since batch_size is so big we update on all episodes at once
- Epoch 2: same update
- Epoch 3: same
- Epoch 4: same
- Total: 4 gradient updates

Takeaway:
The agent have a reasonable dataset to work with, it updates on the whole one right away without batching it up. Not catastrophic, but could presumably be better with longer n_step to better catch long term dynamics.

### n_step = 1024, batch_size = 5
Collect phase:
- Run environment for n_steps=1024 steps
- Gives ~10 complete episodes

Update phase:
- Shuffle the 10 episodes into 2 minibatches of 5
- Epoch 1:
    - Gradient update on minibatch 1 (episodes 1-5)
    - Gradient update on minibatch 2 (episodes 6-10)
- Epoch 2: same 2 updates
- Epoch 3: same
- Epoch 4: same
- Total: 8 gradient updates

Takeaway:
The agent updates on a batch of ca 512 steps each, one after the other, then repeats. Then go out for new experiences, so it's 2 updates before updating the policy compared to the 512 version. I have a hard time to visualize how this affects learning to be honest