from .config import Limits
from .mathutil import clamp


class Scheduler:
    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.attempt: int = 0

    def backoff_s(self) -> int:
        return clamp(2 ** self.attempt, 1, self.limits.timeout_s)

    def note_failure(self) -> None:
        self.attempt += 1
        if self.attempt > self.limits.max_retries:
            self.attempt = self.limits.max_retries

    def restore(self, snapshot: dict[str, str]) -> None:
        self.attempt = int(snapshot.get("attempt", "0"))
