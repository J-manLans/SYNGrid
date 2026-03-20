# SynergyGrid

The project SYNGrid concerns the design and implementation of a lightweight benchmark environment
for evaluating Long-Term Credit Assignment in Artificial Intelligence agents. The
benchmark is implemented as a grid-based environment in Python and follows the Gymnasium API
standard, allowing reinforcement learning agents to interact
with the environment in a consistent and reproducible manner.

---

## Requirements

* Python 3.10 – 3.12
* pip

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/J-manLans/SynergyGrid.git
cd synergygrid
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

* macOS/Linux:

```bash
source .venv/bin/activate
```

* Windows:

```bash
.venv\Scripts\activate
```

### 3. Install the project

Install in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

---

## ⚠️ Important Usage Note

Do **not** run the project without arguments:

```bash
python -m synergygrid
```

This will attempt to evaluate a non-existing model and result in an error.

---

## Quick Start

### 1. Train an agent

```bash
python -m synergygrid --timesteps <T> --iterations <I>
```

This will:

* Train an agent using the default algorithm
* Save model checkpoints during training
* Train the agent for ``<T>`` timesteps
    * ``<T>`` should be a multiple of 2048, since each training batch contains 2048 steps
* Train the agent for ``<I>`` iterations

Example:

```bash
python -m synergygrid --timesteps 2048 --iterations 10
```

---

### 2. Evaluate a trained agent

```bash
python -m synergygrid --run --steps <N>
```

Replace ``<N>`` with the number of steps of a saved model.

Example:

```bash
python -m synergygrid --run --steps 20480
```

---

### 3. Continue training from a saved model

```bash
python -m synergygrid --cont --steps 20480
```

---

### 4. Human control mode

```bash
python -m synergygrid --human_controls
```

Play the environment manually using keyboard controls.

---

### 5. Run without a trained agent (random actions)

```bash
python -m synergygrid --run --no-agent
```

---

## Command Line Arguments

| Argument           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `--alg {0,1,2}`    | Select algorithm index                                       |
| `--run`            | Run/evaluate a trained agent (default is training)           |
| `--no-agent`       | Use random actions instead of a trained agent                |
| `--cont`           | Continue training from a saved model                         |
| `--steps <N>`      | Model checkpoint to load (required with `--run` or `--cont`) |
| `--timesteps <N>`  | Number of timesteps per training iteration                   |
| `--iterations <N>` | Number of training iterations                                |
| `--human_controls` | Enable manual control                                        |

---

