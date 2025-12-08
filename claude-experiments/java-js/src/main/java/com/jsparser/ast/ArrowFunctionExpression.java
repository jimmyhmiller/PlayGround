package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record ArrowFunctionExpression(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Identifier id,       // Always null for arrow functions
    boolean expression,  // true if body is expression, false if block
    boolean generator,   // Always false for arrows
    boolean async,
    List<Pattern> params,
    Node body            // Can be Expression or BlockStatement
) implements Expression {
    @JsonCreator
    public ArrowFunctionExpression(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("id") Identifier id,
        @JsonProperty("expression") boolean expression,
        @JsonProperty("generator") boolean generator,
        @JsonProperty("async") boolean async,
        @JsonProperty("params") List<Pattern> params,
        @JsonProperty("body") Node body
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             id,
             expression,
             generator,
             async,
             params,
             body);
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
        return "ArrowFunctionExpression";
    }
}
