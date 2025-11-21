package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Program(
    int start,
    int end,
    SourceLocation loc,
    List<Statement> body,
    String sourceType
) implements Node {

    public Program(List<Statement> body, String sourceType) {
        this(0, 0, null, body, sourceType);
    }

    @Override
    public String type() {
        return "Program";
    }
}
