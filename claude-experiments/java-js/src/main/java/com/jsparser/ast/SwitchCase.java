package com.jsparser.ast;

import java.util.List;

public record SwitchCase(
    int start,
    int end,
    SourceLocation loc,
    Expression test,  // null for default case
    List<Statement> consequent
) implements Node {
    @Override
    public String type() {
        return "SwitchCase";
    }
}
