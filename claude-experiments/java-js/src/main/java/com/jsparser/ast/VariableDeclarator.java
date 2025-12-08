package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record VariableDeclarator(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Pattern id,
    Expression init     // Can be null
) {
    @JsonCreator
    public VariableDeclarator(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("id") Pattern id,
        @JsonProperty("init") Expression init
    ) {
        this(start, end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             id, init);
    }

    @JsonProperty("loc")
    public SourceLocation loc() {
        return new SourceLocation(
            new SourceLocation.Position(startLine, startCol),
            new SourceLocation.Position(endLine, endCol)
        );
    }

    @JsonProperty(value = "type", index = 0)
    public String type() {
        return "VariableDeclarator";
    }
}
