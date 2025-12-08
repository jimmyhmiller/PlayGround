package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(value = {"startLine", "startCol", "endLine", "endCol"}, ignoreUnknown = true)
public record SwitchCase(
    int start,
    int end,
    int startLine,
    int startCol,
    int endLine,
    int endCol,
    Expression test,  // null for default case
    List<Statement> consequent
) implements Node {
    @JsonCreator
    public SwitchCase(
        @JsonProperty("start") int start,
        @JsonProperty("end") int end,
        @JsonProperty("loc") SourceLocation loc,
        @JsonProperty("test") Expression test,
        @JsonProperty("consequent") List<Statement> consequent
    ) {
        this(start,
             end,
             loc != null ? loc.start().line() : 0,
             loc != null ? loc.start().column() : 0,
             loc != null ? loc.end().line() : 0,
             loc != null ? loc.end().column() : 0,
             test,
             consequent);
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
        return "SwitchCase";
    }
}
