from .textutil import clean, splay


def apply_rules(record: dict[str, str], rules: list[str]) -> dict[str, str]:
    out = dict(record)
    for rule in rules:
        if rule == "clean":
            out = {k: clean(v) for k, v in out.items()}
        elif rule == "upper":
            out = {k: v.upper() for k, v in out.items()}
    return out


def rule_names(spec: str) -> list[str]:
    return splay(spec)
