package com.jsparser.ast;

import java.util.List;

public record SwitchStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression discriminant,
    List<SwitchCase> cases
) implements Statement {
    @Override
    public String type() {
        return "SwitchStatement";
    }
}
