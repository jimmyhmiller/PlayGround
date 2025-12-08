package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record MethodDefinition(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Expression key,        // Property name (Identifier or PrivateIdentifier)
    FunctionExpression value,
    String kind,          // "constructor" | "method" | "get" | "set"
    boolean computed,
    @JsonProperty("static") boolean isStatic
) implements Node {
    @JsonCreator
    public MethodDefinition(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("key") Expression key,
        @JsonProperty("value") FunctionExpression value,
        @JsonProperty("kind") String kind,
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
             kind,
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
        return "MethodDefinition";
    }
}
