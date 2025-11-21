package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record UnaryExpression(
    int start,
    int end,
    SourceLocation loc,
    String operator,  // "!", "-", "+", "~", "typeof", "void", "delete"
    boolean prefix,   // Always true for prefix unary operators
    Expression argument
) implements Expression {
    // Constructor with prefix defaulting to true
    public UnaryExpression(SourceLocation loc, String operator, Expression argument) {
        this(0, 0, loc, operator, true, argument);
    }

    @Override
    public String type() {
        return "UnaryExpression";
    }
}
