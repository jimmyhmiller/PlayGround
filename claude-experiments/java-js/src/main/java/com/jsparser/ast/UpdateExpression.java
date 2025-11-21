package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record UpdateExpression(
    int start,
    int end,
    SourceLocation loc,
    String operator,  // "++" | "--"
    boolean prefix,   // true for ++x, false for x++
    Expression argument
) implements Expression {
    @Override
    public String type() {
        return "UpdateExpression";
    }
}
