"""A small Pratt parser over arithmetic expressions."""

from .tokens import Token, tokenize

BINDING_POWER = {"+": 10, "-": 10, "*": 20, "/": 20}


class Expr:
    def __init__(self, op: str, left: "Expr | int", right: "Expr | int") -> None:
        self.op = op
        self.left = left
        self.right = right


class Parser:
    def __init__(self, source: str) -> None:
        self.tokens = tokenize(source)
        self.pos = 0

    def peek_token(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def bump(self) -> Token | None:
        tok = self.peek_token()
        self.pos += 1
        return tok

    def parse_expression(self, min_bp: int) -> "Expr | int":
        """Pratt-parse a single expression."""
        lhs: Expr | int = self.parse_atom()
        while True:
            tok = self.peek_token()
            if tok is None or tok.kind != "op" or tok.text in "()":
                break
            bp = BINDING_POWER[tok.text]
            if bp < min_bp:
                break
            self.bump()
            rhs = self.parse_expression(bp + 1)
            lhs = Expr(tok.text, lhs, rhs)
        return lhs

    def parse_atom(self) -> "Expr | int":
        tok = self.bump()
        if tok is None:
            raise ValueError("unexpected end of input")
        if tok.kind == "num":
            return int(tok.text)
        if tok.text == "(":
            inner = self.parse_expression(0)
            self.bump()
            return inner
        raise ValueError(f"unexpected token: {tok.text!r}")


def parse(source: str) -> "Expr | int":
    return Parser(source).parse_expression(0)
