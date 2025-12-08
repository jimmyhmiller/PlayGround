package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record ExportAllDeclaration(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Literal source,      // String literal for the module
    Node exported,       // Can be null for "export * from 'mod'", or Identifier/Literal for "export * as ns from 'mod'"
    java.util.List<ImportAttribute> attributes  // Import attributes
) implements Statement {
    @JsonCreator
    public ExportAllDeclaration(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("source") Literal source,
        @JsonProperty("exported") Node exported,
        @JsonProperty("attributes") java.util.List<ImportAttribute> attributes
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             source,
             exported,
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
        return "ExportAllDeclaration";
    }
}
