package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record BinaryExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression left,
    String operator,
    Expression right
) implements Expression {
    public BinaryExpression(String operator, Expression left, Expression right) {
        this(0, 0, null, left, operator, right);
    }

    @Override
    public String type() {
        return "BinaryExpression";
    }
}
