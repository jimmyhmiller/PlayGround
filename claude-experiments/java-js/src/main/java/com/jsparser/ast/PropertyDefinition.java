package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record PropertyDefinition(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Expression key,        // Property name (Identifier or PrivateIdentifier)
    Expression value,      // Can be null for class fields without initializer
    boolean computed,
    @JsonProperty("static") boolean isStatic
) implements Node {
    @JsonCreator
    public PropertyDefinition(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("key") Expression key,
        @JsonProperty("value") Expression value,
        @JsonProperty("computed") boolean computed,
        @JsonProperty("isStatic") boolean isStatic
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             key,
             value,
             computed,
             isStatic);
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
        return "PropertyDefinition";
    }
}
