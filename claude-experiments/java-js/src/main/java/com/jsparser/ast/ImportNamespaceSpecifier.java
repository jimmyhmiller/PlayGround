package com.jsparser.ast;

public record ImportNamespaceSpecifier(
    int start,
    int end,
    SourceLocation loc,
    Identifier local  // The local binding name for the namespace
) implements Node {
    @Override
    public String type() {
        return "ImportNamespaceSpecifier";
    }
}
