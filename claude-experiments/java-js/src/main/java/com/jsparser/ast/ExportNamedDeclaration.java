package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record ExportNamedDeclaration(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Statement declaration,      // Can be null if using specifiers
    List<Node> specifiers,      // List of ExportSpecifier
    Literal source,             // Can be null if not re-exporting
    List<ImportAttribute> attributes  // Import attributes (for re-exports)
) implements Statement {
    @JsonCreator
    public ExportNamedDeclaration(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("declaration") Statement declaration,
        @JsonProperty("specifiers") List<Node> specifiers,
        @JsonProperty("source") Literal source,
        @JsonProperty("attributes") List<ImportAttribute> attributes
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             declaration,
             specifiers,
             source,
             attributes);
    }

    @Override
    @JsonProperty("loc")
    public SourceLocation loc() {
        return new SourceLocation(
            new SourceLocation.Position(startLine, startCol),
            new SourceLocation.Position(endLine, endCol)
        );
    }

    @Override
    public String type() {
        return "ExportNamedDeclaration";
    }
}
