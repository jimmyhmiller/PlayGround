package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record YieldExpression(
    int start,
    int end,
    SourceLocation loc,
    boolean delegate,  // true for yield*, false for yield
    Expression argument  // null for standalone yield
) implements Expression {
    public YieldExpression(boolean delegate, Expression argument) {
        this(0, 0, null, delegate, argument);
    }

    @Override
    public String type() {
        return "YieldExpression";
    }
}
