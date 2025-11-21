package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record AssignmentExpression(
    int start,
    int end,
    SourceLocation loc,
    String operator,
    Node left,  // Can be Expression or Pattern (for destructuring)
    Expression right
) implements Expression {
    public AssignmentExpression(String operator, Node left, Expression right) {
        this(0, 0, null, operator, left, right);
    }

    @Override
    public String type() {
        return "AssignmentExpression";
    }
}
