def clean(value: str) -> str:
    return " ".join(value.strip().split())


def splay(value: str, sep: str = ",") -> list[str]:
    return [clean(part) for part in value.split(sep) if clean(part)]
