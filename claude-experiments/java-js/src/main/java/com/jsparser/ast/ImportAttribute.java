package com.jsparser.ast;

public record ImportAttribute(
    int start,
    int end,
    SourceLocation loc,
    Node key,      // Identifier or Literal
    Literal value  // Always a Literal (string)
) implements Node {
    @Override
    public String type() {
        return "ImportAttribute";
    }
}
