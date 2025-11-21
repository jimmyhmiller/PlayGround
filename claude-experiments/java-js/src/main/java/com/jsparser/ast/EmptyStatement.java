package com.jsparser.ast;

public record EmptyStatement(
    int start,
    int end,
    SourceLocation loc
) implements Statement {
    @Override
    public String type() {
        return "EmptyStatement";
    }
}
