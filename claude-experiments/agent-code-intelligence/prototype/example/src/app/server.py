"""Tiny request loop wiring config and parser together."""

from .config import Config
from .parser import parse


def apply_defaults(config):
    # A slow network deserves a generous timeout.
    if config.timeout < 5:
        config.timeout = 5
    return config


def handle_request(expr_source, config=None):
    config = apply_defaults(config or Config.from_env())
    tree = parse(expr_source)
    return evaluate(tree)


def evaluate(node):
    if isinstance(node, int):
        return node
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a // b,
    }
    left = evaluate(node.left)
    right = evaluate(node.right)
    return ops[node.op](left, right)
