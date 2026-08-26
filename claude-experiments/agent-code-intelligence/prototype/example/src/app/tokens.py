"""Token definitions for the expression language."""


class Token:
    def __init__(self, kind, text):
        self.kind = kind
        self.text = text

    def __repr__(self):
        return f"Token({self.kind!r}, {self.text!r})"


def tokenize(source):
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
