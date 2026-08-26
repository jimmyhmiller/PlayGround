from .mathutil import checked_div
from .transform import apply_rules


def combine(base: dict[str, str], extra: dict[str, str], rules: list[str]) -> dict[str, str]:
    """Merge two records (later keys win), then apply the given rules."""
    out = dict(base)
    out.update(extra)
    return apply_rules(out, rules)


def merge_ratio(kept: int, total: int) -> int:
    if total <= 0:
        return 0
    return checked_div(kept * 100, total)
