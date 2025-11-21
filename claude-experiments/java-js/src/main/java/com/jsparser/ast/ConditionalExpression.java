package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ConditionalExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression test,
    Expression consequent,
    Expression alternate
) implements Expression {
    @Override
    public String type() {
        return "ConditionalExpression";
    }
}
