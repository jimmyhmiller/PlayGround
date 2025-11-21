package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ExpressionStatement(
    int start,
    int end,
    SourceLocation loc,
    Expression expression,
    @JsonInclude(JsonInclude.Include.NON_NULL) String directive
) implements Statement {
    public ExpressionStatement(Expression expression) {
        this(0, 0, null, expression, null);
    }

    public ExpressionStatement(int start, int end, SourceLocation loc, Expression expression) {
        this(start, end, loc, expression, null);
    }

    @Override
    public String type() {
        return "ExpressionStatement";
    }
}
