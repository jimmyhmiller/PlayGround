package com.jsparser.ast;

public record DebuggerStatement(
    int start,
    int end,
    SourceLocation loc
) implements Statement {
    @Override
    public String type() {
        return "DebuggerStatement";
    }
}
