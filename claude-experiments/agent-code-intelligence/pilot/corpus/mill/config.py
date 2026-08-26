import os


class Limits:
    def __init__(self) -> None:
        self.max_items: int = 100
        self.max_retries: int = 3
        self.timeout_s: int = 30


def reload_limits(limits: Limits) -> Limits:
    """Startup hook: refresh limits from the environment."""
    limits.max_items = int(os.environ.get("MILL_MAX_ITEMS", str(limits.max_items)))
    limits.max_retries = int(os.environ.get("MILL_MAX_RETRIES", str(limits.max_retries)))
    return limits


def snooze_guard(limits: Limits) -> Limits:
    """Startup hook: keep the timeout sane."""
    limits.timeout_s = max(5, limits.timeout_s)
    return limits
