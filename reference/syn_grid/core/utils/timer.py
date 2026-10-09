class Timer:
    def __init__(self):
        self.remaining = 0

    def reset(self) -> None:
        self.remaining = 0

    def is_completed(self) -> bool:
        return self.remaining <= 0

    def set(self, duration: int) -> None:
        self.remaining = duration

    def tick(self) -> None:
        if self.remaining > 0:
            self.remaining -= 1
