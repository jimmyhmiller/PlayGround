package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ChainExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression expression
) implements Expression {
    @Override
    public String type() {
        return "ChainExpression";
    }
}
