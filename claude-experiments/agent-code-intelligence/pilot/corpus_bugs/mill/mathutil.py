def checked_div(num: int, den: int) -> int:
    if den == 0:
        raise ValueError("division by zero")
    return num // den


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def legacy_norm(values: list[int]) -> list[int]:
    """Old normalization pass; superseded by transform.apply_rules."""
    top = max(values) if values else 1
    return [v * 100 // top for v in values]
