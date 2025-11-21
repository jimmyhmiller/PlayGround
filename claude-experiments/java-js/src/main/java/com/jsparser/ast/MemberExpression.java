package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record MemberExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression object,
    Expression property,
    boolean computed,
    boolean optional
) implements Expression, Pattern {
    public MemberExpression(Expression object, Expression property, boolean computed) {
        this(0, 0, null, object, property, computed, false);
    }

    @Override
    public String type() {
        return "MemberExpression";
    }
}
