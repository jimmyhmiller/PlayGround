package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record LogicalExpression(
    int start,
    int end,
    SourceLocation loc,
    String operator,  // "&&" | "||" | "??"
    Expression left,
    Expression right
) implements Expression {
    @Override
    public String type() {
        return "LogicalExpression";
    }
}
