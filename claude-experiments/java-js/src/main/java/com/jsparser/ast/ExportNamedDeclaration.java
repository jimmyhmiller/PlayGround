package com.jsparser.ast;

import java.util.List;

public record ExportNamedDeclaration(
    int start,
    int end,
    SourceLocation loc,
    Statement declaration,      // Can be null if using specifiers
    List<Node> specifiers,      // List of ExportSpecifier
    Literal source,             // Can be null if not re-exporting
    List<ImportAttribute> attributes  // Import attributes (for re-exports)
) implements Statement {
    @Override
    public String type() {
        return "ExportNamedDeclaration";
    }
}
