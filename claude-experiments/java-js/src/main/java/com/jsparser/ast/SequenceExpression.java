package com.jsparser.ast;

import java.util.List;

public record SequenceExpression(
    int start,
    int end,
    SourceLocation loc,
    List<Expression> expressions
) implements Expression {
    @Override
    public String type() {
        return "SequenceExpression";
    }
}
