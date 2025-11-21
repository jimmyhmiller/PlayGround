package com.jsparser.ast;

public record ImportSpecifier(
    int start,
    int end,
    SourceLocation loc,
    Node imported,        // The name in the module (Identifier or Literal)
    Identifier local      // The local binding name (always Identifier)
) implements Node {
    @Override
    public String type() {
        return "ImportSpecifier";
    }
}
