package com.jsparser.ast;

public record Super(
    int start,
    int end,
    SourceLocation loc
) implements Expression {
    @Override
    public String type() {
        return "Super";
    }
}
