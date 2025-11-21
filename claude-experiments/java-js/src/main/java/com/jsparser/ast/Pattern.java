package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
    @JsonSubTypes.Type(value = Identifier.class, name = "Identifier"),
    @JsonSubTypes.Type(value = ObjectPattern.class, name = "ObjectPattern"),
    @JsonSubTypes.Type(value = ArrayPattern.class, name = "ArrayPattern"),
    @JsonSubTypes.Type(value = RestElement.class, name = "RestElement"),
    @JsonSubTypes.Type(value = AssignmentPattern.class, name = "AssignmentPattern"),
    @JsonSubTypes.Type(value = MemberExpression.class, name = "MemberExpression")
})
public sealed interface Pattern extends Node permits Identifier, ObjectPattern, ArrayPattern, RestElement, AssignmentPattern, MemberExpression {
}
