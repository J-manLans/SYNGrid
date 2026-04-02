from enum import Enum


class DroidAction(Enum):
    """Actions the droid is capable of performing i.e. go in a certain direction"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
