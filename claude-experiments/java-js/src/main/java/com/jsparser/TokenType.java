package com.jsparser;

public enum TokenType {
    // Literals
    NUMBER,
    STRING,
    REGEX,
    TEMPLATE_LITERAL,
    TEMPLATE_HEAD,      // `hello ${
    TEMPLATE_MIDDLE,    // } world ${
    TEMPLATE_TAIL,      // } end`
    TRUE,
    FALSE,
    NULL,

    // Identifiers
    IDENTIFIER,

    // Operators
    PLUS,           // +
    MINUS,          // -
    STAR,           // *
    STAR_STAR,      // **
    SLASH,          // /
    PERCENT,        // %
    ASSIGN,         // =
    PLUS_ASSIGN,    // +=
    MINUS_ASSIGN,   // -=
    STAR_ASSIGN,    // *=
    STAR_STAR_ASSIGN, // **=
    SLASH_ASSIGN,   // /=
    PERCENT_ASSIGN, // %=
    LEFT_SHIFT_ASSIGN, // <<=
    RIGHT_SHIFT_ASSIGN, // >>=
    UNSIGNED_RIGHT_SHIFT_ASSIGN, // >>>=
    BIT_AND_ASSIGN, // &=
    BIT_OR_ASSIGN,  // |=
    BIT_XOR_ASSIGN, // ^=
    AND_ASSIGN,     // &&=
    OR_ASSIGN,      // ||=
    QUESTION_QUESTION_ASSIGN, // ??=
    EQ,             // ==
    NE,             // !=
    EQ_STRICT,      // ===
    NE_STRICT,      // !==
    LT,             // <
    LE,             // <=
    GT,             // >
    GE,             // >=
    BANG,           // !
    TILDE,          // ~
    AND,            // &&
    OR,             // ||
    BIT_AND,        // &
    BIT_OR,         // |
    BIT_XOR,        // ^
    LEFT_SHIFT,     // <<
    RIGHT_SHIFT,    // >>
    UNSIGNED_RIGHT_SHIFT, // >>>
    INCREMENT,      // ++
    DECREMENT,      // --
    ARROW,          // =>

    // Punctuation
    SEMICOLON,      // ;
    COMMA,          // ,
    DOT,            // .
    DOT_DOT_DOT,    // ...
    HASH,           // #
    AT,             // @
    COLON,          // :
    QUESTION,       // ?
    QUESTION_DOT,   // ?.
    QUESTION_QUESTION, // ??
    LPAREN,         // (
    RPAREN,         // )
    LBRACE,         // {
    RBRACE,         // }
    LBRACKET,       // [
    RBRACKET,       // ]

    // Keywords
    VAR,
    LET,
    CONST,
    FUNCTION,
    CLASS,
    ASYNC,
    AWAIT,
    RETURN,
    IF,
    ELSE,
    FOR,
    WHILE,
    DO,
    BREAK,
    CONTINUE,
    SWITCH,
    CASE,
    DEFAULT,
    TRY,
    CATCH,
    FINALLY,
    THROW,
    NEW,
    TYPEOF,
    VOID,
    DELETE,
    THIS,
    SUPER,
    IN,
    OF,
    INSTANCEOF,
    GET,
    SET,
    // Note: YIELD removed - now tokenized as IDENTIFIER, validated contextually by parser
    IMPORT,
    EXPORT,
    WITH,
    DEBUGGER,

    // Special
    EOF
}
