package com.jsparser.ast;

import java.util.List;

public record ImportDeclaration(
    int start,
    int end,
    SourceLocation loc,
    List<Node> specifiers,  // ImportSpecifier, ImportDefaultSpecifier, or ImportNamespaceSpecifier
    Literal source,         // String literal for the module path
    List<ImportAttribute> attributes  // Import attributes (with { type: 'json' })
) implements Statement {
    @Override
    public String type() {
        return "ImportDeclaration";
    }
}
