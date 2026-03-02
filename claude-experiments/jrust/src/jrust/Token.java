package jrust;

public class Token {
    public enum Kind {
        // Keywords
        FN, STRUCT, IMPL, LET, MUT, SELF, IMPORT, RETURN, TRUE, FALSE, IF, ELSE, WHILE,
        FOR, IN, MATCH, ENUM, CONST, NULL, BREAK, CONTINUE,

        // Identifiers and literals
        IDENT, INT_LIT, FLOAT_LIT, STRING_LIT, CHAR_LIT,

        // Operators
        PLUS, MINUS, STAR, SLASH, EQ, EQEQ, BANGEQ, LT, GT, LTEQ, GTEQ,
        DOT, DOTDOT, COLONCOLON, ARROW, FATARROW, AMP, AMPAMP, PIPE, PIPEPIPE, BANG, PERCENT,

        // Delimiters
        LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET, COMMA, COLON, SEMI, UNDERSCORE,

        // Special
        EOF
    }

    public final Kind kind;
    public final String text;
    public final int line;
    public final int col;

    public Token(Kind kind, String text, int line, int col) {
        this.kind = kind;
        this.text = text;
        this.line = line;
        this.col = col;
    }

    @Override
    public String toString() {
        return kind + "(" + text + ") at " + line + ":" + col;
    }
}
