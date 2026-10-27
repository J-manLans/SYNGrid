# Contributing to SYNGrid

Thanks for wanting to help. This file says how to set up, what a change has to pass and how to send it. How the code
is written is in [`docs/code_style.md`](docs/code_style.md).

## Setup

SYNGrid needs Python 3.10.

```bash
git clone https://github.com/J-manLans/SYNGrid.git
cd SYNGrid
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On a machine without a display or an audio device, set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.

## Before you send a change

CI runs these on every push and pull request. Run them first:

```bash
pytest
ruff check src/syn_grid tests
ruff format --check src/syn_grid tests
```

If you touched the environment, the world or a scenario, also run `python3 -m syn_grid.check_env`, the Gymnasium
contract check.

`./scripts/tidy.sh` fixes what ruff can fix and formats everything.

## Branches

Branch from `main` and name the branch by its kind:

- `feature/...` for something new
- `bugfix/...` for a fix
- `refactor/...` for a change in shape that keeps behaviour
- `docs/...` for documentation
- `experiment/...` for something you are trying out

## Commits

Keep commits small and focused on one thing. A commit message says what changed, why, and what you ran to check it.

Mark work that is still to do with a `# TODO` or `# NOTE` comment, so it can be searched for. Put a tag in brackets
when several belong together, as in `# TODO(delay):`.

## Pull requests

Rebase onto `main` before you open one:

```bash
git fetch
git rebase origin/main
git push --force-with-lease
```

Open the pull request against `main`. Keep it to one feature or fix, and say what it changes and how you checked it.
