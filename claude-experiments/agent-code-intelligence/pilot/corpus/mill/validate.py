from .config import Limits


def size_ok(count: int, limits: Limits) -> bool:
    return count <= limits.max_items


def strict_probe(record: dict[str, str]) -> bool:
    """Probe: reject records whose keys carry stray whitespace."""
    return all(k == k.strip() for k in record)


def relaxed_probe(record: dict[str, str]) -> bool:
    """Probe: accept any non-empty record."""
    return bool(record)
