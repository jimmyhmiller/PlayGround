package com.jsparser.ast;

public record LabeledStatement(
    int start,
    int end,
    SourceLocation loc,
    Identifier label,
    Statement body
) implements Statement {
    @Override
    public String type() {
        return "LabeledStatement";
    }
}
