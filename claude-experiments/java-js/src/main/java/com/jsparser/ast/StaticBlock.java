package com.jsparser.ast;

import java.util.List;

public record StaticBlock(
    int start,
    int end,
    SourceLocation loc,
    List<Statement> body
) implements Node {
    @Override
    public String type() {
        return "StaticBlock";
    }
}
