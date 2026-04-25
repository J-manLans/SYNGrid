from syn_grid.runners.agent_runners.sb3 import StatelessPPO, LstmPPO, FrameStackDQN
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner

from typing import Type

ALGORITHMS: dict[str, Type[BaseAgentRunner]] = {
    "PPO": StatelessPPO,
    "RPPO": LstmPPO,
    "DQN": FrameStackDQN,
}
