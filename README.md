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

## Quick Start

```bash
python3 -m syn_grid
```

This will:

* Train or evaluate an agent or start a human play-testing run depending on the settings in the configs.yaml file. For more detailed information read the [config file](/src/syn_grid/config/configs.yaml) itself
