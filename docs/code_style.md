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

- **Overrides**: A method that replaces one inherited from a parent class, ours or a library's, is marked
  `@override`, so a type checker flags it when the parent's method is renamed or gone. `reportImplicitOverride` in
  `pyproject.toml`'s `[tool.pyright]` has Pyright flag a missing one, in the editor through Pylance and in CI. A class
  that only matches a `Protocol` without inheriting from it overrides nothing, and its methods are not marked.
- **Protocols**: A class that implements a `Protocol` matches it and does not inherit from it. A subclass that forgets
  a method inherits the protocol's empty one and nothing flags it; a class that only matches is checked by Pyright
  wherever it is handed over as the protocol. The class diagrams draw these as realization arrows
  (`scripts/diagrams/add_protocol_edges.py`), since a diagram tool sees only inheritance.

```python
from typing import override

class VectorMarkovian(BasePerception):
    @override
    def get_observation(self, state: GridWorld) -> np.ndarray:
```

## 2. Variable and Function Naming
- Use snake_case as above for:
    - Variables
    - Functions
    - Methods
- Private variables and methods shall be prefixed with a "_"
- Class names should use PascalCase.

## 3. Automatic Formatting
- Use Ruff for code formatting.
- Run Ruff before committing, or use `scripts/tidy.sh` to format and lint everything
  at once:

```bash
ruff format path/to/file.py
# or, for the whole project (matches what CI enforces):
./scripts/tidy.sh
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
- A comment starts with a capital letter. A comment of one sentence has no period; a comment of more than one
  sentence ends each sentence with one, the last included: `# Reset the step clock`, but
  `# Spawn every orb. The field is full from the start.`

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


## 6. Tests
- **As few tests as cover the behaviour.** A test earns its place only if breaking the code it guards turns it red
  and no other test already does. No tests of the language or of a library, such as a dataclass comparing equal, and
  no parametrizing over values that take the same path. Asserts that check one behaviour share one test.
- **Unit tests** in `tests/unit/` mirror `src/syn_grid/`, one test file per source file: the tests for
  `core/grid_world.py` are in `tests/unit/core/test_grid_world.py`. A unit test asks whether one file keeps its
  promise, and may use real collaborators to do it.
- **Integration tests** in `tests/integration/` check that parts across packages hand off to each other, such as the
  environment's step through the world, the watcher and termination.
- **Regression tests** in `tests/regression/` check that each locked scenario still plays the replay tape as its
  baseline in `tests/regression/baselines/` recorded it. A locked scenario without a baseline fails.
- Inside a file, one class per method or behaviour under test, named `Test<Thing>`, with a one-line docstring.
  Fixtures and helpers stay at module level, so every class in the file can use them.
- **A test plays what the code would really see,** values and sequences the world can produce, even when a shortcut
  would be a few lines shorter. An orb that expires is removed, not eaten; a lifespan is one a grid can give. A step
  the world never takes raises more questions than it saves.
- A value that a test puts in and expects back unchanged, such as an orb's reward or a penalty, is a constant, so
  the two cannot drift. A value the code works out, such as a chain's worth, is written out in the test as the
  worked example it is, not computed with the code's own formula.
- A test builds a locked scenario from its base config, or from a config it writes itself, never from the run YAMLs a
  person edits to train. Editing a run to train must not turn the suite red.
- **Setup that tests share is written once,** as the rest of the code is. A world, a config or a helper that several
  tests build is a shared fixture or helper, not a copy in each. The tier-chain digester tests share their orbs and
  a way to eat them through one `conftest.py`, the unit tests share `tick` through theirs, and tests build a locked
  scenario's config from its `ScenarioEntry.base_conf`. A helper that can go in a `conftest.py` and is used often
  enough goes there, as a fixture, even across folders; how often is enough is the agent's judgment. Setup used by
  one or two tests of one file stays in that file.
- Fixtures that several test files in a folder share go in a `conftest.py` there, which nothing imports. A value
  such a fixture depends on is a fixture too, so a test file's own fixtures can take it and the two cannot drift.
- A test file's module level is in sections, with the headings of section 5, each used only when the file has
  something for it, in this order:
  - **Constants:** module-level values, such as a scenario name or a penalty the tests share.
  - **Stand-ins:** classes that play a collaborator, such as a recorder, a fake or a user's own runner, and what
    they record into.
  - **Helpers:** functions that build or drive what a test needs, and Hypothesis strategies. A helper that serves
    one test class goes under a subheading named for it, such as `# === Invariants === #`, after the shared ones.
  - **Fixtures:** `@pytest.fixture` functions only.
  - **Tests:** the test classes, with no headings between them.

  Stand-ins and helpers come before the fixtures because Python reads a function's annotations when it is defined,
  so a fixture that returns a stand-in needs the stand-in above it.

```python
class TestPerformDroidAction:
    """What one tick of the world reports: the orb eaten, and the reward."""

    def test_eating_nothing_reports_none(self, world: GridWorld): ...
```

---

Following these conventions will help maintain readability, consistency, and ensure the codebase remains clean and understandable for all contributors.