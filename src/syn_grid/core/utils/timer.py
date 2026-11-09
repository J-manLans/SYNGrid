class Timer:
    """A countdown in steps.

    A timer set to `None` never completes and never ticks. It is how an orb
    stays off the field for the rest of the episode.
    """

    def __init__(self):
        self.remaining: int | None = 0

    def reset(self) -> None:
        """Set the timer to completed."""

        self.remaining = 0

    def is_completed(self) -> bool:
        """Whether the countdown has run out. A timer set to `None` is never completed."""

        if self.remaining is None:
            return False

        return self.remaining <= 0

    def set(self, duration: int | None) -> None:
        """Start a countdown of `duration` steps, or one that never completes if it is `None`."""

        self.remaining = duration

    def tick(self) -> None:
        """Count one step down. A completed timer and a timer set to `None` stay as they are."""

        if self.remaining is not None and self.remaining > 0:
            self.remaining -= 1
