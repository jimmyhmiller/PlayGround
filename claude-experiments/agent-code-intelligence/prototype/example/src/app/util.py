"""Assorted helpers."""


def clamp(value, low, high):
    return max(low, min(high, value))


def legacy_retry_delay(attempt):
    """Unused since the retry rewrite; kept for reference."""
    return clamp(2 ** attempt, 1, 60)
