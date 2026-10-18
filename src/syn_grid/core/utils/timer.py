class Timer:
    """A countdown in steps.

    A timer set to `None` never completes and never ticks. It is how an orb
    stays off the field for the rest of the episode.
    """

    def __init__(self):
        self.remaining: int | None = 0

    def reset(self) -> None:
        self.remaining = 0

    def is_completed(self) -> bool:
        if self.remaining is None:
            return False

        return self.remaining <= 0

    def set(self, duration: int | None) -> None:
        self.remaining = duration

    def tick(self) -> None:
        if self.remaining is not None and self.remaining > 0:
            self.remaining -= 1
