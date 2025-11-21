package com.jsparser.ast;

public record ExportSpecifier(
    int start,
    int end,
    SourceLocation loc,
    Node local,     // The local name (Identifier or Literal)
    Node exported   // The exported name (Identifier or Literal)
) implements Node {
    @Override
    public String type() {
        return "ExportSpecifier";
    }
}
