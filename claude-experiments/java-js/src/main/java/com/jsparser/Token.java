package com.jsparser;

public record Token(
    TokenType type,
    String lexeme,
    Object literal,
    int line,
    int column,
    int position,
    int endPosition,
    int endLine,      // For multi-line tokens: the line where the token ends
    int endColumn,    // For multi-line tokens: the column where the token ends
    String raw        // For template literals: the raw string value (with escape sequences, line continuations)
) {
    public Token(TokenType type, String lexeme, int line, int column, int position) {
        this(type, lexeme, null, line, column, position, position + (lexeme != null ? lexeme.length() : 0), line, column + (lexeme != null ? lexeme.length() : 0), null);
    }

    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position) {
        this(type, lexeme, literal, line, column, position, position + (lexeme != null ? lexeme.length() : 0), line, column + (lexeme != null ? lexeme.length() : 0), null);
    }

    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position, int endPosition) {
        this(type, lexeme, literal, line, column, position, endPosition, line, column + (lexeme != null ? lexeme.length() : 0), null);
    }

    // Constructor for multi-line tokens (like template literals)
    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position, int endPosition, int endLine, int endColumn, String raw) {
        this.type = type;
        this.lexeme = lexeme;
        this.literal = literal;
        this.line = line;
        this.column = column;
        this.position = position;
        this.endPosition = endPosition;
        this.endLine = endLine;
        this.endColumn = endColumn;
        this.raw = raw;
    }

    // Old constructor (for backwards compatibility with existing code)
    public Token(TokenType type, String lexeme, Object literal, int line, int column, int position, int endPosition, String raw) {
        this(type, lexeme, literal, line, column, position, endPosition, line, column + (lexeme != null ? lexeme.length() : 0), raw);
    }
}
