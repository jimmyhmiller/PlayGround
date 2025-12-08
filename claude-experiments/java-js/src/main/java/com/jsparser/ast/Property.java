package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record Property(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    boolean method,
    boolean shorthand,
    boolean computed,
    @JsonDeserialize(as = Node.class) Node key,
    @JsonDeserialize(as = Node.class) Node value,
    String kind
) implements Node {
    public Property(int startLine, int startCol, int endLine, int endCol, Node key, Node value, String kind, boolean computed) {
        this(0, 0, startLine, startCol, endLine, endCol, false, false, computed, key, value, kind);
    }

    @JsonCreator
    public Property(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("method") boolean method,
        @JsonProperty("shorthand") boolean shorthand,
        @JsonProperty("computed") boolean computed,
        @JsonProperty("key") Node key,
        @JsonProperty("value") Node value,
        @JsonProperty("kind") String kind
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             method,
             shorthand,
             computed,
             key,
             value,
             kind);
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
        return "Property";
    }
}
