# Code Style Guidelines for SynergyGrid

This document outlines the coding conventions and best practices for the SynergyGrid project.

## 1. Python Type Annotations
- **Function Arguments & Returns**: All functions and methods should include type annotations for arguments and return values.

```python
def get_orb_meta(self, only_active: bool) -> list[OrbMeta]:
```

- **Variable Annotations**: Variables should be annotated when the type is not immediately obvious or for clarity in complex data structures.

```python
from typing import Final

_chained_tiers: Final[list[int]] = []
```

## 2. Variable and Function Naming
- Use snake_case as above for:
    - Variables
    - Functions
    - Methods
- Private variables and methods shall be prefixed with a "_"
- Class names should use PascalCase.

## 3. Automatic Formatting
- Use Black for code formatting.
- Run Black before committing:

```python
black path/to/file.py
# or
black src/synergygrid
```

## 4. Docstrings and Comments
- Every public function or class should have a docstring describing:
    - Purpose
    - Input arguments
    - Return values
    - Side effects (if any)

```python
def train_agent(
    runner: AgentRunner,
    continue_training=False,
    agent_steps="",
    timesteps=20000,
    iterations=10,
) -> None:
    """
    Train an agent, either from scratch or by continuing from a saved checkpoint.

    :param continue_training: If True, the training continues from an existing model checkpoint.
    :type continue_training: bool
    :param agent_steps: The specific checkpoint steps of the model to continue training from.
    :type agent_steps: str
    :param timesteps: Number of steps to train before saving the agent.
    :type timesteps: int
    :param iterations: Number of training loops, each consisting of `timesteps` steps.
    :type iterations: int
    """
```

- Private helper methods should be prefixed with a "_" as mentioned before and don't need a docstring, clarifying comments is enough.

```python
def _empty_spawn_cell(self, position: list[np.int64]) -> bool:
        # Check against agent
        if position == self._agent.position:
            return False

        # If there are no active orbs we can spawn right away
        if len(self._active_orbs) == 0:
            return True

        # Else check against all active orbs
        for r in self._active_orbs:
            if position == r.position:
                return False

        return True
```

- Use inline comments sparingly, only for clarification, not obvious code.

## 5. Headings and structure
For all classes, use headings to improve readability and navigation, as of now 3 main sections are used:

```python
# ================= #
#       Init        #
# ================= #

# ================= #
#        API        #
# ================= #

# ================= #
#      Helpers      #
# ================= #
```

If subsections are needed write them like this:

```python
# === Logic === #

# === Getters === #

# ...
```

For the helpers section, provide subheadings indicating what section the helpers belong to:

```python
# ================= #
#      Helpers      #
# ================= #

# === Init === #

# === API === #

# === Global === #
```

---

Following these conventions will help maintain readability, consistency, and ensure the codebase remains clean and understandable for all contributors.