package com.jsparser.ast;

public record ExportAllDeclaration(
    int start,
    int end,
    SourceLocation loc,
    Literal source,      // String literal for the module
    Node exported,       // Can be null for "export * from 'mod'", or Identifier/Literal for "export * as ns from 'mod'"
    java.util.List<ImportAttribute> attributes  // Import attributes
) implements Statement {
    @Override
    public String type() {
        return "ExportAllDeclaration";
    }
}
