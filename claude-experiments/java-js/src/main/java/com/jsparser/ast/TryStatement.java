package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record TryStatement(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    BlockStatement block,
    CatchClause handler,      // Can be null
    BlockStatement finalizer  // Can be null
) implements Statement {
    @JsonCreator
    public TryStatement(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("block") BlockStatement block,
        @JsonProperty("handler") CatchClause handler,
        @JsonProperty("finalizer") BlockStatement finalizer
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             block,
             handler,
             finalizer);
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
        return "TryStatement";
    }
}
