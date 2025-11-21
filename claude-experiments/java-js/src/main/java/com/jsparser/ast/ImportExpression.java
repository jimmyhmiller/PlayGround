package com.jsparser.ast;

public record ImportExpression(
    int start,
    int end,
    SourceLocation loc,
    Expression source,
    Expression options  // Second argument for import attributes: { with: { type: "json" } }
) implements Expression {
    @Override
    public String type() {
        return "ImportExpression";
    }
}
