from enum import Enum


class LogKey(str, Enum):
    EPISODE = "episode"
    REWARD = "reward"
    LENGTH = "length"
    CHAINS_BROKEN = "chains_broken"
    CHAIN_PROGRESSED = "chain_progressed"
    CHAINS_COMPLETED = "chains_completed"
