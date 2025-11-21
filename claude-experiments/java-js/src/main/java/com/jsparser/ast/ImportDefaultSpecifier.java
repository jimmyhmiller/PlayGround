package com.jsparser.ast;

public record ImportDefaultSpecifier(
    int start,
    int end,
    SourceLocation loc,
    Identifier local  // The local binding name for the default import
) implements Node {
    @Override
    public String type() {
        return "ImportDefaultSpecifier";
    }
}
