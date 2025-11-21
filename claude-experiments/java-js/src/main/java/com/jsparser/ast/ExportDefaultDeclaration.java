package com.jsparser.ast;

public record ExportDefaultDeclaration(
    int start,
    int end,
    SourceLocation loc,
    Node declaration  // Can be Expression or Declaration
) implements Statement {
    @Override
    public String type() {
        return "ExportDefaultDeclaration";
    }
}
