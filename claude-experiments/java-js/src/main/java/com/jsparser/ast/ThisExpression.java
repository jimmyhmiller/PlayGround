package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ThisExpression(
    int start,
    int end,
    SourceLocation loc
) implements Expression {
    @Override
    public String type() {
        return "ThisExpression";
    }
}
