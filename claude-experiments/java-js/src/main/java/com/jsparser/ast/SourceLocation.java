package com.jsparser.ast;

public record SourceLocation(Position start, Position end) {
    public record Position(int line, int column) {}
}
