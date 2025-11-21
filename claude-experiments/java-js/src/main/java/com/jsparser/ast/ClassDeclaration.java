package com.jsparser.ast;

public record ClassDeclaration(
    int start,
    int end,
    SourceLocation loc,
    Identifier id,         // Class name
    Expression superClass, // Can be null if no extends
    ClassBody body
) implements Statement {
    @Override
    public String type() {
        return "ClassDeclaration";
    }
}
