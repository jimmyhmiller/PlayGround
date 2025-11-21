package com.jsparser.ast;

public record ThrowStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression argument
) implements Statement {
    @Override
    public String type() {
        return "ThrowStatement";
    }
}
