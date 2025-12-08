package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record ForOfStatement(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    boolean await,
    Node left,   // VariableDeclaration or Expression
    Expression right,
    Statement body
) implements Statement {
    @JsonCreator
    public ForOfStatement(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("await") boolean await,
        @JsonProperty("left") Node left,
        @JsonProperty("right") Expression right,
        @JsonProperty("body") Statement body
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             await,
             left,
             right,
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
        return "ForOfStatement";
    }
}
