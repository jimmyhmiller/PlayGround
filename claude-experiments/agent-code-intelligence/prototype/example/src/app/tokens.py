"""Token definitions for the expression language."""


class Token:
    def __init__(self, kind: str, text: str) -> None:
        self.kind = kind
        self.text = text

    def __repr__(self) -> str:
        return f"Token({self.kind!r}, {self.text!r})"


def tokenize(source: str) -> list[Token]:
    """Split source into number and operator tokens."""
    tokens = []
    for part in source.replace("(", " ( ").replace(")", " ) ").split():
        if part.isdigit():
            tokens.append(Token("num", part))
        elif part in "+-*/()":
            tokens.append(Token("op", part))
        else:
            raise ValueError(f"unexpected token: {part}")
    return tokens
