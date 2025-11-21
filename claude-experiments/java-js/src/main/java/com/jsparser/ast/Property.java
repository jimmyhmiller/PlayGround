package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Property(
    int start,
    int end,
    SourceLocation loc,
    boolean method,
    boolean shorthand,
    boolean computed,
    @JsonDeserialize(as = Node.class) Node key,
    @JsonDeserialize(as = Node.class) Node value,
    String kind
) implements Node {
    public Property(SourceLocation loc, Node key, Node value, String kind, boolean computed) {
        this(0, 0, loc, false, false, computed, key, value, kind);
    }

    @JsonProperty(value = "type", index = 0)
    public String type() {
        return "Property";
    }
}
