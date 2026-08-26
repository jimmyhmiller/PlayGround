from . import config, validate


def resolve_probe(name: str):
    return getattr(validate, name)


def resolve_hook(name: str):
    return getattr(config, name)
