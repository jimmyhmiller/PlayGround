package com.jsparser.ast;

public record ClassExpression(
    int start,
    int end,
    SourceLocation loc,
    Identifier id,         // Class name (can be null for anonymous class)
    Expression superClass, // Can be null if no extends
    ClassBody body
) implements Expression {
    @Override
    public String type() {
        return "ClassExpression";
    }
}
