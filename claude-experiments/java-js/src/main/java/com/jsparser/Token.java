package com.jsparser;

public record Token(
    TokenType type,
    Object literal,
    int line,
    int column,
    int position,
    int endPosition,
    int endLine,      // For multi-line tokens: the line where the token ends
    int endColumn,    // For multi-line tokens: the column where the token ends
    String raw        // For template literals: the raw string value (with escape sequences, line continuations)
) {
    // Simple constructor for single-line tokens without literal
    public Token(TokenType type, int line, int column, int position, int endPosition) {
        this(type, null, line, column, position, endPosition, line, column + (endPosition - position), null);
    }

    // Constructor with literal for single-line tokens
    public Token(TokenType type, Object literal, int line, int column, int position, int endPosition) {
        this(type, literal, line, column, position, endPosition, line, column + (endPosition - position), null);
    }

    // Constructor for multi-line tokens (like template literals)
    public Token(TokenType type, Object literal, int line, int column, int position, int endPosition, int endLine, int endColumn, String raw) {
        this.type = type;
        this.literal = literal;
        this.line = line;
        this.column = column;
        this.position = position;
        this.endPosition = endPosition;
        this.endLine = endLine;
        this.endColumn = endColumn;
        this.raw = raw;
    }

    // Lazy lexeme creation - only allocates when actually needed
    public String lexeme(char[] source) {
        return new String(source, position, endPosition - position);
    }

    /**
     * Returns the identifier name for IDENTIFIER tokens.
     * For identifiers with unicode escapes, the resolved name is stored in literal.
     * For regular identifiers, the lexeme is the name.
     */
    public String identifierName(char[] source) {
        if (literal instanceof String s) {
            return s;
        }
        return new String(source, position, endPosition - position);
    }
}
