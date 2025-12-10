package com.jsparser;

import com.jsparser.ast.Literal;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Lexer {
    private final String source;
    private final char[] buf;
    private final int length;
    private int position = 0;
    private int line = 1;
    private int column = 0;
    private Stack<Integer> templateBraceDepthStack = new Stack<>();
    private TokenType lastTokenType = null;
    private boolean atLineStart = true;

    // Context stack for tracking statement vs expression contexts
    private Stack<LexerContext> contextStack = new Stack<>();
    // Whether an expression is allowed at the current position
    private boolean exprAllowed = true;
    private int generatorDepth = 0; // Track nesting level of generator functions
    private Stack<Boolean> functionIsGeneratorStack = new Stack<>(); // Track if each function level is a generator
    private int asyncDepth = 0; // Track nesting level of async functions
    private Stack<Boolean> functionIsAsyncStack = new Stack<>(); // Track if each function level is async


    private boolean strictMode;
    private boolean isModule; // Whether parsing in module mode (enables top-level await)

    public Lexer(String source) {
        this(source, false, false);
    }

    public Lexer(String source, boolean strictMode) {
        this(source, strictMode, false);
    }

    public Lexer(String source, boolean strictMode, boolean isModule) {
        this.source = source;
        this.buf = source.toCharArray();
        this.length = buf.length;
        this.strictMode = strictMode;
        this.isModule = isModule;
        // Initialize with statement context
        contextStack.push(LexerContext.B_STAT);
    }

    public List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();

        // Handle hashbang comment at the very start of the file: #!/usr/bin/env node
        if (position == 0 && !isAtEnd() && peek() == '#' && peekNext() == '!') {
            advance(); // consume #
            advance(); // consume !
            // Skip until line terminator (LF, CR, LS, PS)
            while (!isAtEnd()) {
                char c = peek();
                if (c == '\n' || c == '\r' || c == '\u2028' || c == '\u2029') {
                    break;
                }
                advance();
            }
        }

        while (!isAtEnd()) {
            // Skip whitespace first, before checking template continuation
            skipWhitespace();
            if (isAtEnd()) break;

            // When inside template, handle template continuation or expression tokens
            if (!templateBraceDepthStack.isEmpty()) {
                int depth = templateBraceDepthStack.peek();
                if (peek() == '}' && depth == 0) {
                    // End of interpolation, scan template continuation
                    Token token = scanTemplateContinuation();
                    if (token != null) {
                        tokens.add(token);
                        updateContext(lastTokenType, token);
                        lastTokenType = token.type();
                    }
                    continue;
                } else if (peek() == '{' && depth >= 0) {
                    // Only track braces if we're in interpolation (depth >= 0)
                    // depth == -1 means we haven't entered interpolation yet
                    templateBraceDepthStack.push(templateBraceDepthStack.pop() + 1);
                } else if (peek() == '}' && depth > 0) {
                    // Only decrement if depth > 0 (not the closing brace of ${...})
                    templateBraceDepthStack.push(templateBraceDepthStack.pop() - 1);
                }
            }

            Token token = nextToken();
            if (token != null) {
                tokens.add(token);
                updateContext(lastTokenType, token);
                lastTokenType = token.type();
                atLineStart = false;

            }
        }

        tokens.add(new Token(TokenType.EOF, "", line, column, position));
        return tokens;
    }

    /**
     * Updates the context stack and exprAllowed flag after reading a token.
     * Based on Acorn's updateContext approach.
     */
    private void updateContext(TokenType prevType, Token currentToken) {
        TokenType currentType = currentToken.type();
        // Handle specific token types that need custom context updates
        switch (currentType) {
            case RPAREN, RBRACE -> {
                // Pop context when closing delimiters
                if (contextStack.size() > 1) {
                    LexerContext out = contextStack.pop();
                    // Special case: if we close a block and parent is function, pop function too
                    if (out == LexerContext.B_STAT && !contextStack.isEmpty() &&
                        contextStack.peek().getToken().equals("function")) {
                        out = contextStack.pop();
                        // Pop the generator tracking stack and update depth
                        if (!functionIsGeneratorStack.isEmpty()) {
                            boolean wasGenerator = functionIsGeneratorStack.pop();
                            if (wasGenerator && generatorDepth > 0) {
                                generatorDepth--;
                            }
                        }
                        // Pop the async tracking stack and update depth
                        if (!functionIsAsyncStack.isEmpty()) {
                            boolean wasAsync = functionIsAsyncStack.pop();
                            if (wasAsync && asyncDepth > 0) {
                                asyncDepth--;
                            }
                        }
                    }
                    exprAllowed = !out.isExpr();
                } else {
                    exprAllowed = true;
                }
            }
            case LBRACE -> {
                // Determine if { starts a block statement or object expression
                boolean isBlock = braceIsBlock(prevType);
                contextStack.push(isBlock ? LexerContext.B_STAT : LexerContext.B_EXPR);
                exprAllowed = true;
            }
            case LPAREN -> {
                // Determine if ( is for statement (if, while, for) or expression
                boolean isStatementParen = prevType == TokenType.IF ||
                                          prevType == TokenType.FOR ||
                                          prevType == TokenType.WHILE ||
                                          prevType == TokenType.WITH;
                contextStack.push(isStatementParen ? LexerContext.P_STAT : LexerContext.P_EXPR);
                exprAllowed = true;
            }
            case FUNCTION, CLASS -> {
                // Determine if function/class is declaration or expression
                boolean isExpr = prevType != null && TokenTypeProperties.beforeExpr(prevType) &&
                                prevType != TokenType.ELSE &&
                                !(prevType == TokenType.SEMICOLON && !contextStack.isEmpty() &&
                                  contextStack.peek() == LexerContext.P_STAT) &&
                                !(prevType == TokenType.RETURN && atLineStart) &&
                                !((prevType == TokenType.COLON || prevType == TokenType.LBRACE) &&
                                  !contextStack.isEmpty() && contextStack.peek() == LexerContext.B_STAT);
                contextStack.push(isExpr ? LexerContext.F_EXPR : LexerContext.F_STAT);
                exprAllowed = false;
                // Mark this function as non-generator by default (will be updated if we see *)
                functionIsGeneratorStack.push(false);
                // Mark this function as async if preceded by 'async' keyword
                boolean isAsync = prevType == TokenType.ASYNC;
                functionIsAsyncStack.push(isAsync);
                if (isAsync) {
                    asyncDepth++;
                }
            }
            case STAR -> {
                // After function*, we're entering a generator function
                if (prevType == TokenType.FUNCTION) {
                    // Update the most recent function to be a generator
                    if (!functionIsGeneratorStack.isEmpty()) {
                        functionIsGeneratorStack.pop();
                        functionIsGeneratorStack.push(true);
                    }
                    generatorDepth++;
                }
                // After *, expressions are allowed (for generator functions or multiplication)
                exprAllowed = TokenTypeProperties.beforeExpr(currentType);
            }
            case COLON -> {
                // Pop function context if present
                if (!contextStack.isEmpty() && contextStack.peek().getToken().equals("function")) {
                    contextStack.pop();
                }
                exprAllowed = true;
            }
            case TEMPLATE_LITERAL -> {
                // Handle template literal context (backtick)
                if (!contextStack.isEmpty() && contextStack.peek() == LexerContext.Q_TMPL) {
                    contextStack.pop();
                } else {
                    contextStack.push(LexerContext.Q_TMPL);
                }
                exprAllowed = false;
            }
            case TEMPLATE_HEAD, TEMPLATE_MIDDLE -> {
                // After ${ in a template, expressions are allowed (including regexes)
                exprAllowed = true;
            }
            case INCREMENT, DECREMENT -> {
                // ++ and -- don't change exprAllowed
            }
            default -> {
                // For all other tokens, set exprAllowed based on beforeExpr property
                if (isKeyword(currentType) && prevType == TokenType.DOT) {
                    // Keywords after dot are property names, not keywords
                    exprAllowed = false;
                } else if (currentType == TokenType.IDENTIFIER && "yield".equals(currentToken.lexeme())) {
                    // Special handling for contextual keyword 'yield'
                    // In generator functions, yield allows expressions after it (like return/throw)
                    // Outside generator functions, yield is just an identifier (no expressions after)
                    exprAllowed = generatorDepth > 0;
                } else if (currentType == TokenType.IDENTIFIER && "await".equals(currentToken.lexeme())) {
                    // Special handling for contextual keyword 'await'
                    // In async functions or module mode (top-level await), await allows expressions after it
                    // Outside these contexts, await is just an identifier (no expressions after)
                    exprAllowed = asyncDepth > 0 || isModule;
                } else {
                    exprAllowed = TokenTypeProperties.beforeExpr(currentType);
                }
            }
        }
    }

    /**
     * Determines if a { should be treated as a block statement or object expression.
     * Based on Acorn's braceIsBlock logic.
     */
    private boolean braceIsBlock(TokenType prevType) {
        if (contextStack.isEmpty()) {
            return true;
        }

        LexerContext parent = contextStack.peek();

        // After function keyword, { is always a block
        if (parent == LexerContext.F_EXPR || parent == LexerContext.F_STAT) {
            return true;
        }

        // After colon in a block or object, check if the parent is expression
        if (prevType == TokenType.COLON && (parent == LexerContext.B_STAT || parent == LexerContext.B_EXPR)) {
            return !parent.isExpr();
        }

        // After return or name (like 'of', 'yield') with line break
        if (prevType == TokenType.RETURN || (prevType == TokenType.IDENTIFIER && exprAllowed)) {
            // In real implementation, would check for line break here
            // For now, assume no line break
            return false;
        }

        // After else, semicolon, EOF, closing paren, arrow
        if (prevType == TokenType.ELSE || prevType == TokenType.SEMICOLON ||
            prevType == null || prevType == TokenType.RPAREN || prevType == TokenType.ARROW) {
            return true;
        }

        // After opening brace, check parent context
        if (prevType == TokenType.LBRACE) {
            return parent == LexerContext.B_STAT;
        }

        // After var, const, name
        if (prevType == TokenType.VAR || prevType == TokenType.CONST || prevType == TokenType.IDENTIFIER) {
            return false;
        }

        // Default: if expressions not allowed, it's a block
        return !exprAllowed;
    }

    /**
     * Check if a TokenType is a keyword.
     */
    private boolean isKeyword(TokenType type) {
        return switch (type) {
            case VAR, LET, CONST, FUNCTION, CLASS, ASYNC, AWAIT, RETURN, IF, ELSE,
                 FOR, WHILE, DO, BREAK, CONTINUE, SWITCH, CASE, DEFAULT, TRY, CATCH,
                 FINALLY, THROW, NEW, TYPEOF, VOID, DELETE, THIS, SUPER, IN, OF,
                 INSTANCEOF, GET, SET, IMPORT, EXPORT, WITH, DEBUGGER -> true;
                 // Note: YIELD removed - now treated as contextual keyword (IDENTIFIER)
            default -> false;
        };
    }

    private Token nextToken() {
        char c = peek();
        int startPos = position;
        int startLine = line;
        int startColumn = column;

        return switch (c) {
            case ';' -> {
                advance();
                yield new Token(TokenType.SEMICOLON, ";", startLine, startColumn, startPos);
            }
            case ',' -> {
                advance();
                yield new Token(TokenType.COMMA, ",", startLine, startColumn, startPos);
            }
            case ':' -> {
                advance();
                yield new Token(TokenType.COLON, ":", startLine, startColumn, startPos);
            }
            case '#' -> {
                advance();
                yield new Token(TokenType.HASH, "#", startLine, startColumn, startPos);
            }
            case '@' -> {
                advance();
                yield new Token(TokenType.AT, "@", startLine, startColumn, startPos);
            }
            case '?' -> {
                advance();
                if (match('?')) {
                    if (match('=')) {
                        yield new Token(TokenType.QUESTION_QUESTION_ASSIGN, "??=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.QUESTION_QUESTION, "??", startLine, startColumn, startPos);
                } else if (match('.')) {
                    // Check if this is optional chaining ?. or not
                    // OptionalChainingPunctuator:: ?.[lookahead ∉ DecimalDigit]
                    // So ?.3 is NOT optional chaining (it's ? and .3), but ?.[, ?.foo, etc. are
                    char nextChar = peek();
                    if (nextChar == '.' || (nextChar >= '0' && nextChar <= '9')) {
                        // Backtrack - it's ? followed by a decimal number like .3
                        // or ? followed by .. (spread/rest)
                        position--;
                        column--;
                        yield new Token(TokenType.QUESTION, "?", startLine, startColumn, startPos);
                    } else {
                        yield new Token(TokenType.QUESTION_DOT, "?.", startLine, startColumn, startPos);
                    }
                } else {
                    yield new Token(TokenType.QUESTION, "?", startLine, startColumn, startPos);
                }
            }
            case '(' -> {
                advance();
                yield new Token(TokenType.LPAREN, "(", startLine, startColumn, startPos);
            }
            case ')' -> {
                advance();
                yield new Token(TokenType.RPAREN, ")", startLine, startColumn, startPos);
            }
            case '{' -> {
                advance();
                yield new Token(TokenType.LBRACE, "{", startLine, startColumn, startPos);
            }
            case '}' -> {
                advance();
                yield new Token(TokenType.RBRACE, "}", startLine, startColumn, startPos);
            }
            case '[' -> {
                advance();
                yield new Token(TokenType.LBRACKET, "[", startLine, startColumn, startPos);
            }
            case ']' -> {
                advance();
                yield new Token(TokenType.RBRACKET, "]", startLine, startColumn, startPos);
            }
            case '.' -> {
                // Check if it's a number like .5 or just a dot
                // Only treat as number if we're not in a member access context
                if (isDigit(peekNext()) && !canPrecedeMemberAccess(lastTokenType)) {
                    // Debug: System.out.println("Calling scanNumber with startColumn=" + startColumn + " startPos=" + startPos);
                    yield scanNumber(startLine, startColumn, startPos);
                }
                advance();
                // Check for ... (spread/rest)
                if (match('.') && match('.')) {
                    yield new Token(TokenType.DOT_DOT_DOT, "...", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.DOT, ".", startLine, startColumn, startPos);
            }
            case '+' -> {
                advance();
                if (match('+')) {
                    yield new Token(TokenType.INCREMENT, "++", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    yield new Token(TokenType.PLUS_ASSIGN, "+=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.PLUS, "+", startLine, startColumn, startPos);
            }
            case '-' -> {
                advance();
                if (match('-')) {
                    // Check for HTML comment closing: --> (only at line start)
                    if (atLineStart && peek() == '>') {
                        advance(); // consume '>'
                        // This is an HTML closing comment, skip to end of line
                        while (!isAtEnd() && !isLineTerminator(peek())) {
                            advance();
                        }
                        yield null; // Skip the comment
                    }
                    yield new Token(TokenType.DECREMENT, "--", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    yield new Token(TokenType.MINUS_ASSIGN, "-=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.MINUS, "-", startLine, startColumn, startPos);
            }
            case '*' -> {
                advance();
                if (match('*')) {
                    if (match('=')) {
                        yield new Token(TokenType.STAR_STAR_ASSIGN, "**=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.STAR_STAR, "**", startLine, startColumn, startPos);
                } else if (match('=')) {
                    yield new Token(TokenType.STAR_ASSIGN, "*=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.STAR, "*", startLine, startColumn, startPos);
            }
            case '/' -> {
                // Peek ahead to check if it's a comment or regex
                char next = peekNext();

                if (next == '/') {
                    // Single-line comment
                    advance(); // consume first /
                    advance(); // consume second /
                    while (!isAtEnd() && !isLineTerminator(peek())) {
                        advance();
                    }
                    yield null; // Skip the comment, return null to signal no token
                } else if (next == '*') {
                    // Multi-line comment
                    advance(); // consume /
                    advance(); // consume *
                    boolean containsLineTerminator = false;
                    while (!isAtEnd()) {
                        if (peek() == '*' && peekNext() == '/') {
                            advance(); // consume *
                            advance(); // consume /
                            break;
                        }
                        char ch = peek();
                        // Handle line terminators: LF, CR, LS, PS
                        if (ch == '\n') {
                            line++;
                            column = 0;
                            containsLineTerminator = true;
                            position++; // consume \n without incrementing column
                        } else if (ch == '\r') {
                            line++;
                            column = 0;
                            containsLineTerminator = true;
                            position++; // consume \r without incrementing column
                            // Handle CRLF as a single line terminator
                            if (peek() == '\n') {
                                position++; // consume the \n too without incrementing column
                            }
                        } else if (ch == '\u2028' || ch == '\u2029') {
                            // Line Separator (LS) and Paragraph Separator (PS)
                            line++;
                            column = 0;
                            containsLineTerminator = true;
                            position++; // consume line separator without incrementing column
                        } else {
                            advance(); // consume regular character in comment
                        }
                    }
                    // If comment contains line terminator, treat it as being at line start
                    if (containsLineTerminator) {
                        atLineStart = true;
                    }
                    yield null; // Skip the comment
                } else if (exprAllowed) {
                    // When expressions are allowed, / starts a regex literal
                    yield scanRegex(startLine, startColumn, startPos);
                } else {
                    advance();
                    if (match('=')) {
                        yield new Token(TokenType.SLASH_ASSIGN, "/=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.SLASH, "/", startLine, startColumn, startPos);
                }
            }
            case '%' -> {
                advance();
                if (match('=')) {
                    yield new Token(TokenType.PERCENT_ASSIGN, "%=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.PERCENT, "%", startLine, startColumn, startPos);
            }
            case '=' -> {
                advance();
                if (match('>')) {
                    yield new Token(TokenType.ARROW, "=>", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    if (match('=')) {
                        yield new Token(TokenType.EQ_STRICT, "===", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.EQ, "==", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.ASSIGN, "=", startLine, startColumn, startPos);
            }
            case '!' -> {
                advance();
                if (match('=')) {
                    if (match('=')) {
                        yield new Token(TokenType.NE_STRICT, "!==", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.NE, "!=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.BANG, "!", startLine, startColumn, startPos);
            }
            case '~' -> {
                advance();
                yield new Token(TokenType.TILDE, "~", startLine, startColumn, startPos);
            }
            case '<' -> {
                advance();
                // Check for HTML comment opening: <!--
                if (peek() == '!' && peekNext() == '-') {
                    advance(); // consume '!'
                    advance(); // consume first '-'
                    if (peek() == '-') {
                        advance(); // consume second '-'
                        // This is an HTML comment, skip to end of line
                        while (!isAtEnd() && !isLineTerminator(peek())) {
                            advance();
                        }
                        yield null; // Skip the comment
                    }
                    // If not followed by another '-', this is an error, but fall through
                }
                if (match('<')) {
                    if (match('=')) {
                        yield new Token(TokenType.LEFT_SHIFT_ASSIGN, "<<=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.LEFT_SHIFT, "<<", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    yield new Token(TokenType.LE, "<=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.LT, "<", startLine, startColumn, startPos);
            }
            case '>' -> {
                advance();
                if (match('>')) {
                    if (match('>')) {
                        if (match('=')) {
                            yield new Token(TokenType.UNSIGNED_RIGHT_SHIFT_ASSIGN, ">>>=", startLine, startColumn, startPos);
                        }
                        yield new Token(TokenType.UNSIGNED_RIGHT_SHIFT, ">>>", startLine, startColumn, startPos);
                    }
                    if (match('=')) {
                        yield new Token(TokenType.RIGHT_SHIFT_ASSIGN, ">>=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.RIGHT_SHIFT, ">>", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    yield new Token(TokenType.GE, ">=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.GT, ">", startLine, startColumn, startPos);
            }
            case '&' -> {
                advance();
                if (match('&')) {
                    if (match('=')) {
                        yield new Token(TokenType.AND_ASSIGN, "&&=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.AND, "&&", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    yield new Token(TokenType.BIT_AND_ASSIGN, "&=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.BIT_AND, "&", startLine, startColumn, startPos);
            }
            case '|' -> {
                advance();
                if (match('|')) {
                    if (match('=')) {
                        yield new Token(TokenType.OR_ASSIGN, "||=", startLine, startColumn, startPos);
                    }
                    yield new Token(TokenType.OR, "||", startLine, startColumn, startPos);
                }
                if (match('=')) {
                    yield new Token(TokenType.BIT_OR_ASSIGN, "|=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.BIT_OR, "|", startLine, startColumn, startPos);
            }
            case '^' -> {
                advance();
                if (match('=')) {
                    yield new Token(TokenType.BIT_XOR_ASSIGN, "^=", startLine, startColumn, startPos);
                }
                yield new Token(TokenType.BIT_XOR, "^", startLine, startColumn, startPos);
            }
            case '"', '\'' -> scanString(startLine, startColumn, startPos);
            case '`' -> scanTemplateLiteral(startLine, startColumn, startPos);
            case '\\' -> {
                // Unicode escape in identifier
                if (peekNext() == 'u') {
                    yield scanIdentifier(startLine, startColumn, startPos);
                }
                throw new RuntimeException(String.format("Unexpected character: %c (U+%04X) at position=%d, line=%d, column=%d", c, (int)c, position, line, column));
            }
            default -> {
                if (isDigit(c)) {
                    yield scanNumber(startLine, startColumn, startPos);
                } else if (isAlpha(c)) {
                    yield scanIdentifier(startLine, startColumn, startPos);
                }
                throw new RuntimeException(String.format("Unexpected character: %c (U+%04X)", c, (int)c));
            }
        };
    }

    private Token scanString(int startLine, int startColumn, int startPos) {
        char quote = advance();
        StringBuilder value = new StringBuilder();

        while (!isAtEnd() && peek() != quote) {
            if (peek() == '\\') {
                advance();
                if (!isAtEnd()) {
                    char escaped = advance();
                    switch (escaped) {
                        case 'n' -> value.append('\n');
                        case 't' -> value.append('\t');
                        case 'r' -> value.append('\r');
                        case 'v' -> value.append('\u000B'); // vertical tab
                        case 'f' -> value.append('\f'); // form feed
                        case 'b' -> value.append('\b'); // backspace
                        case '0', '1', '2', '3', '4', '5', '6', '7' -> {
                            // Legacy octal escape sequences (allowed only in non-strict mode)
                            // EXCEPT: \0 when not followed by a digit IS allowed in strict mode
                            // Grammar:
                            //   OctalDigit [lookahead ∉ OctalDigit]
                            //   ZeroToThree OctalDigit [lookahead ∉ OctalDigit]
                            //   FourToSeven OctalDigit
                            //   ZeroToThree OctalDigit OctalDigit

                            StringBuilder octalBuilder = new StringBuilder();
                            octalBuilder.append(escaped);

                            if (!isAtEnd() && peek() >= '0' && peek() <= '7') {
                                // This is a multi-digit octal escape - NOT allowed in strict mode
                                if (strictMode) {
                                    throw new RuntimeException("Octal escape sequences are not allowed in strict mode");
                                }
                                // Second digit
                                char secondDigit = peek();
                                octalBuilder.append(secondDigit);
                                advance();

                                // Third digit only if first digit is 0-3 and second digit exists
                                if (escaped >= '0' && escaped <= '3' && !isAtEnd() && peek() >= '0' && peek() <= '7') {
                                    char thirdDigit = peek();
                                    // Check if adding third digit would exceed 377 octal (255 decimal)
                                    String tentative = octalBuilder.toString() + thirdDigit;
                                    int tentativeValue = Integer.parseInt(tentative, 8);
                                    if (tentativeValue <= 255) {
                                        octalBuilder.append(thirdDigit);
                                        advance();
                                    }
                                }
                            } else {
                                // Single digit octal - check if it's \1-\7 in strict mode
                                if (strictMode && escaped != '0') {
                                    throw new RuntimeException("Octal escape sequences are not allowed in strict mode");
                                }
                            }

                            int octalValue = Integer.parseInt(octalBuilder.toString(), 8);
                            value.append((char) octalValue);
                        }
                        case '\\' -> value.append('\\');
                        case '"' -> value.append('"');
                        case '\'' -> value.append('\'');
                        case 'u' -> {
                            // Unicode escape: \\uXXXX or \\u{X...X}
                            if (peek() == '{') {
                                // Braced unicode: \\u{...}
                                advance(); // consume {
                                StringBuilder hexBuilder = new StringBuilder();
                                while (!isAtEnd() && peek() != '}') {
                                    hexBuilder.append(advance());
                                }
                                if (peek() == '}') {
                                    advance(); // consume }
                                    try {
                                        int codePoint = Integer.parseInt(hexBuilder.toString(), 16);
                                        value.appendCodePoint(codePoint);
                                    } catch (NumberFormatException e) {
                                        value.append(escaped);
                                    }
                                } else {
                                    value.append(escaped);
                                }
                            } else if (position + 4 <= length) {
                                // Fixed 4-digit unicode: \\uXXXX
                                String hex = new String(buf, position, 4);
                                try {
                                    int codePoint = Integer.parseInt(hex, 16);
                                    value.append((char) codePoint);
                                    // Advance 4 times to skip hex digits and update column
                                    for (int i = 0; i < 4; i++) advance();
                                } catch (NumberFormatException e) {
                                    value.append(escaped); // Invalid unicode escape
                                }
                            } else {
                                value.append(escaped);
                            }
                        }
                        case 'x' -> {
                            // Hex escape: \xXX
                            if (position + 2 <= length) {
                                String hex = new String(buf, position, 2);
                                try {
                                    int codePoint = Integer.parseInt(hex, 16);
                                    value.append((char) codePoint);
                                    // Advance 2 times to skip hex digits and update column
                                    for (int i = 0; i < 2; i++) advance();
                                } catch (NumberFormatException e) {
                                    value.append(escaped);
                                }
                            } else {
                                value.append(escaped);
                            }
                        }
                        case '\n', '\r', '\u2028', '\u2029' -> {
                            // Line continuation: backslash followed by line terminator
                            // This produces an empty string (no characters added to value)
                            // The line terminator was already consumed by escaped = advance()
                            // Update line/column tracking
                            line++;
                            column = 0;
                            // If it's \r followed by \n, consume the \n too (CRLF counts as one line)
                            if (escaped == '\r' && !isAtEnd() && peek() == '\n') {
                                advance(); // consume the \n
                                column = 0; // Reset column after consuming \n (advance incremented it)
                            }
                            // Don't append anything - line continuation produces empty sequence
                        }
                        default -> value.append(escaped);
                    }
                }
            } else {
                char c = advance();
                value.append(c);
                // U+2028 (LINE SEPARATOR) and U+2029 (PARAGRAPH SEPARATOR) are line terminators
                // even when they appear literally (not escaped) in strings
                if (c == '\u2028' || c == '\u2029') {
                    line++;
                    column = 0;
                }
            }
        }

        if (isAtEnd()) {
            throw new RuntimeException("Unterminated string");
        }

        advance(); // closing quote
        String lexeme = source.substring(startPos, position);
        // Use the multi-line token constructor to properly record end line/column for strings with line continuations
        return new Token(TokenType.STRING, lexeme, value.toString(), startLine, startColumn, startPos, position, line, column, null);
    }

    private Token scanNumber(int startLine, int startColumn, int startPos) {
        // Check for hex (0x), octal (0o), or binary (0b) literals
        if (peek() == '0' && (peekNext() == 'x' || peekNext() == 'X')) {
            advance(); // consume 0
            advance(); // consume x
            while (isHexDigit(peek()) || peek() == '_') {
                advance();
            }
            String lexeme = source.substring(startPos, position);
            String hexPart = lexeme.substring(2).replace("_", ""); // Remove "0x" and underscores

            // Check for BigInt suffix
            if (peek() == 'n') {
                advance();
                String bigintLexeme = source.substring(startPos, position);
                return new Token(TokenType.NUMBER, bigintLexeme, "0x" + hexPart + "n", startLine, startColumn, startPos);
            }

            Object literal;
            try {
                literal = Integer.parseInt(hexPart, 16);
            } catch (NumberFormatException e1) {
                try {
                    long longValue = Long.parseLong(hexPart, 16);
                    // JavaScript's Number.MAX_SAFE_INTEGER is 2^53 - 1 = 9007199254740991
                    // Beyond this, IEEE-754 doubles lose precision
                    if (longValue > 9007199254740991L || longValue < -9007199254740991L) {
                        // Convert to double to match JavaScript's precision loss
                        literal = (double) longValue;
                    } else {
                        literal = longValue;
                    }
                } catch (NumberFormatException e2) {
                    // Number too large for Long - use BigInteger then convert to double
                    // to match JavaScript's IEEE-754 behavior
                    java.math.BigInteger bi = new java.math.BigInteger(hexPart, 16);
                    literal = bi.doubleValue();
                }
            }
            return new Token(TokenType.NUMBER, lexeme, literal, startLine, startColumn, startPos);
        } else if (peek() == '0' && (peekNext() == 'o' || peekNext() == 'O')) {
            advance(); // consume 0
            advance(); // consume o
            while ((peek() >= '0' && peek() <= '7') || peek() == '_') {
                advance();
            }
            String lexeme = source.substring(startPos, position);
            String octalPart = lexeme.substring(2).replace("_", ""); // Remove "0o" and underscores

            // Check for BigInt suffix
            if (peek() == 'n') {
                advance();
                String bigintLexeme = source.substring(startPos, position);
                return new Token(TokenType.NUMBER, bigintLexeme, "0o" + octalPart + "n", startLine, startColumn, startPos);
            }

            Object literal;
            try {
                literal = Integer.parseInt(octalPart, 8);
            } catch (NumberFormatException e1) {
                try {
                    long longValue = Long.parseLong(octalPart, 8);
                    // JavaScript's Number.MAX_SAFE_INTEGER is 2^53 - 1
                    if (longValue > 9007199254740991L || longValue < -9007199254740991L) {
                        literal = (double) longValue;
                    } else {
                        literal = longValue;
                    }
                } catch (NumberFormatException e2) {
                    // Number too large for Long
                    java.math.BigInteger bi = new java.math.BigInteger(octalPart, 8);
                    literal = bi.doubleValue();
                }
            }
            return new Token(TokenType.NUMBER, lexeme, literal, startLine, startColumn, startPos);
        } else if (peek() == '0' && peekNext() >= '0' && peekNext() <= '9' && peekNext() != '.') {
            // Could be legacy octal (00-07) or NonOctalDecimalIntegerLiteral (08, 09, etc.)
            advance(); // consume initial 0

            // Scan all digits
            boolean hasNonOctalDigit = false;
            while (isDigit(peek())) {
                char digit = peek();
                if (digit == '8' || digit == '9') {
                    hasNonOctalDigit = true;
                }
                advance();
            }

            String lexeme = source.substring(startPos, position);

            Object literal;
            if (hasNonOctalDigit) {
                // NonOctalDecimalIntegerLiteral - parse as decimal
                try {
                    literal = Integer.parseInt(lexeme);
                } catch (NumberFormatException e1) {
                    try {
                        literal = Long.parseLong(lexeme);
                    } catch (NumberFormatException e2) {
                        literal = Double.parseDouble(lexeme);
                    }
                }
            } else {
                // Legacy octal integer literal (allowed only in non-strict mode)
                try {
                    literal = Integer.parseInt(lexeme, 8);
                } catch (NumberFormatException e1) {
                    try {
                        literal = Long.parseLong(lexeme, 8);
                    } catch (NumberFormatException e2) {
                        literal = (double) Long.parseUnsignedLong(lexeme, 8);
                    }
                }
            }
            return new Token(TokenType.NUMBER, lexeme, literal, startLine, startColumn, startPos);
        } else if (peek() == '0' && (peekNext() == 'b' || peekNext() == 'B')) {
            advance(); // consume 0
            advance(); // consume b
            while (peek() == '0' || peek() == '1' || peek() == '_') {
                advance();
            }
            String lexeme = source.substring(startPos, position);
            String binaryPart = lexeme.substring(2).replace("_", ""); // Remove "0b" and underscores

            // Check for BigInt suffix
            if (peek() == 'n') {
                advance();
                String bigintLexeme = source.substring(startPos, position);
                return new Token(TokenType.NUMBER, bigintLexeme, "0b" + binaryPart + "n", startLine, startColumn, startPos);
            }

            Object literal;
            try {
                literal = Integer.parseInt(binaryPart, 2);
            } catch (NumberFormatException e1) {
                try {
                    long longValue = Long.parseLong(binaryPart, 2);
                    // JavaScript's Number.MAX_SAFE_INTEGER is 2^53 - 1
                    if (longValue > 9007199254740991L || longValue < -9007199254740991L) {
                        literal = (double) longValue;
                    } else {
                        literal = longValue;
                    }
                } catch (NumberFormatException e2) {
                    // Number too large for Long
                    java.math.BigInteger bi = new java.math.BigInteger(binaryPart, 2);
                    literal = bi.doubleValue();
                }
            }
            return new Token(TokenType.NUMBER, lexeme, literal, startLine, startColumn, startPos);
        }

        // Regular decimal number
        boolean isDecimal = false;

        while (isDigit(peek()) || peek() == '_') {
            advance();
        }

        // Handle decimal
        // JavaScript allows numbers ending with '.': 1. is valid and equals 1.0
        // Note: 5..method() is parsed as (5.) . method(), so we DO consume the first dot
        if (peek() == '.') {
            char nextChar = peekNext();
            if (isDigit(nextChar)) {
                // 1.5 - decimal with fractional part
                isDecimal = true;
                advance(); // consume '.'
                while (isDigit(peek()) || peek() == '_') {
                    advance();
                }
            } else {
                // 1. or 1.. - number ending with decimal point
                // Always consume the dot (even for 1..)
                isDecimal = true;
                advance(); // consume '.'
            }
        }

        // Handle exponent (e.g., 1e10, 1.5e-3, 2E+5)
        if (peek() == 'e' || peek() == 'E') {
            isDecimal = true;
            advance(); // consume e/E
            if (peek() == '+' || peek() == '-') {
                advance(); // consume sign
            }
            while (isDigit(peek()) || peek() == '_') {
                advance();
            }
        }

        String lexeme = source.substring(startPos, position);
        // Remove underscores for parsing
        String numberStr = lexeme.replace("_", "");

        // Parse as integer if no decimal point, otherwise as double
        Object literal;
        if (isDecimal) {
            double doubleValue = Double.parseDouble(numberStr);
            // If the double value is a whole number that fits in a long without losing precision
            if (doubleValue == Math.floor(doubleValue) && !Double.isInfinite(doubleValue)) {
                // Try to parse the original string as a long to see if it's in range
                // This avoids floating point precision issues when checking bounds
                try {
                    long longValue = Long.parseLong(numberStr);
                    // Successfully parsed as long, check if it fits in int
                    if (longValue >= Integer.MIN_VALUE && longValue <= Integer.MAX_VALUE) {
                        literal = (int) longValue;
                    } else {
                        literal = longValue;
                    }
                } catch (NumberFormatException e) {
                    // Number is outside long range or has decimal point, keep as double
                    literal = doubleValue;
                }
            } else {
                literal = doubleValue;
            }
        } else {
            // Try to parse as integer first
            try {
                literal = Integer.parseInt(numberStr);
            } catch (NumberFormatException e1) {
                try {
                    long longValue = Long.parseLong(numberStr);
                    // JavaScript's Number.MAX_SAFE_INTEGER is 2^53 - 1 = 9007199254740991
                    // Beyond this, IEEE-754 doubles lose precision
                    if (longValue > 9007199254740991L || longValue < -9007199254740991L) {
                        // Convert to double to match JavaScript's precision loss
                        literal = (double) longValue;
                    } else {
                        literal = longValue;
                    }
                } catch (NumberFormatException e2) {
                    // Number too large for Long - JavaScript uses IEEE-754 doubles
                    // which lose precision beyond 2^53, so we should match that behavior
                    literal = Double.parseDouble(numberStr);
                }
            }
        }

        // Check for BigInt suffix 'n'
        if (peek() == 'n') {
            advance(); // consume 'n'
            String bigintLexeme = source.substring(startPos, position);
            // For BigInt, keep the value as a string (remove underscores but keep the number)
            // The AST can represent it as a string since Java doesn't have native BigInt
            return new Token(TokenType.NUMBER, bigintLexeme, numberStr + "n", startLine, startColumn, startPos);
        }

        return new Token(TokenType.NUMBER, lexeme, literal, startLine, startColumn, startPos);
    }

    private Token scanIdentifier(int startLine, int startColumn, int startPos) {
        // OPTIMIZED: Use fast ASCII scanning first
        // Check if first char is ASCII identifier start (not escape or unicode)
        char firstChar = peek();
        if (firstChar != '\\' && firstChar < 128 && isAsciiIdentifierStart(firstChar)) {
            // Fast path: scan ASCII identifier chars
            int asciiEnd = scanAsciiIdentifierChars(position);

            // Check what stopped us
            if (asciiEnd > position) {
                // Check if we hit something that needs unicode handling
                if (asciiEnd < length) {
                    char stopChar = buf[asciiEnd];
                    if (stopChar == '\\' || stopChar >= 128) {
                        // Need to continue with unicode handling
                        return scanIdentifierWithUnicodeContinuation(startLine, startColumn, startPos, asciiEnd);
                    }
                }

                // Pure ASCII identifier - fast path complete!
                String identifierName = new String(buf, startPos, asciiEnd - startPos);
                int consumed = asciiEnd - position;
                position = asciiEnd;
                column += consumed;

                // Keyword lookup
                TokenType type = lookupKeyword(identifierName);
                Object literal = literalForKeyword(type);

                int endColumn = startColumn + consumed;
                return new Token(type, identifierName, literal, startLine, startColumn, startPos, position, startLine, endColumn, null);
            }
        }

        // Slow path: first char is escape or unicode, use full handling
        return scanIdentifierFull(startLine, startColumn, startPos);
    }

    // Fast ASCII identifier character scanning
    private int scanAsciiIdentifierChars(int start) {
        int pos = start;
        while (pos < length) {
            char c = buf[pos];
            if (c >= 128 || c == '\\') {
                // Non-ASCII or escape - stop here
                return pos;
            }
            if ((c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '_' || c == '$') {
                pos++;
            } else {
                return pos;
            }
        }
        return pos;
    }

    private static boolean isAsciiIdentifierStart(char c) {
        return (c >= 'a' && c <= 'z') ||
               (c >= 'A' && c <= 'Z') ||
               c == '_' || c == '$';
    }

    private TokenType lookupKeyword(String name) {
        return switch (name) {
            case "true" -> TokenType.TRUE;
            case "false" -> TokenType.FALSE;
            case "null" -> TokenType.NULL;
            case "var" -> TokenType.VAR;
            case "let" -> TokenType.LET;
            case "const" -> TokenType.CONST;
            case "function" -> TokenType.FUNCTION;
            case "class" -> TokenType.CLASS;
            case "return" -> TokenType.RETURN;
            case "if" -> TokenType.IF;
            case "else" -> TokenType.ELSE;
            case "for" -> TokenType.FOR;
            case "while" -> TokenType.WHILE;
            case "do" -> TokenType.DO;
            case "break" -> TokenType.BREAK;
            case "continue" -> TokenType.CONTINUE;
            case "switch" -> TokenType.SWITCH;
            case "case" -> TokenType.CASE;
            case "default" -> TokenType.DEFAULT;
            case "try" -> TokenType.TRY;
            case "catch" -> TokenType.CATCH;
            case "finally" -> TokenType.FINALLY;
            case "throw" -> TokenType.THROW;
            case "new" -> TokenType.NEW;
            case "typeof" -> TokenType.TYPEOF;
            case "void" -> TokenType.VOID;
            case "delete" -> TokenType.DELETE;
            case "this" -> TokenType.THIS;
            case "super" -> TokenType.SUPER;
            case "in" -> TokenType.IN;
            case "instanceof" -> TokenType.INSTANCEOF;
            case "import" -> TokenType.IMPORT;
            case "export" -> TokenType.EXPORT;
            case "with" -> TokenType.WITH;
            case "debugger" -> TokenType.DEBUGGER;
            default -> TokenType.IDENTIFIER;
        };
    }

    private Object literalForKeyword(TokenType type) {
        return switch (type) {
            case TRUE -> true;
            case FALSE -> false;
            case NULL -> null;
            default -> null;
        };
    }

    // Continue scanning after ASCII portion hit unicode/escape
    private Token scanIdentifierWithUnicodeContinuation(int startLine, int startColumn, int startPos, int asciiEnd) {
        StringBuilder actualName = new StringBuilder();
        boolean hasEscapes = false;

        // Append the ASCII portion we already scanned
        actualName.append(buf, startPos, asciiEnd - startPos);
        position = asciiEnd;
        column += (asciiEnd - startPos);

        // Continue with unicode handling
        while (!isAtEnd() && (isAlphaNumeric(peek()) || (peek() == '\\' && peekNext() == 'u'))) {
            if (peek() == '\\') {
                hasEscapes = true;
                advance(); // consume \
                advance(); // consume u
                actualName.append(scanUnicodeEscape());
            } else {
                char c = advance();
                actualName.append(c);
                if (Character.isHighSurrogate(c) && !isAtEnd() && Character.isLowSurrogate(peek())) {
                    actualName.append(advance());
                }
            }
        }

        String identifierName = actualName.toString();
        TokenType type = hasEscapes ? TokenType.IDENTIFIER : lookupKeyword(identifierName);
        Object literal = literalForKeyword(type);

        int rawLength = position - startPos;
        int endColumn = startColumn + rawLength;
        return new Token(type, identifierName, literal, startLine, startColumn, startPos, position, startLine, endColumn, null);
    }

    // Original full scanning for when first char is escape/unicode
    private Token scanIdentifierFull(int startLine, int startColumn, int startPos) {
        StringBuilder actualName = new StringBuilder();
        boolean hasEscapes = false;

        // Handle first character (could be unicode escape or regular)
        if (peek() == '\\' && peekNext() == 'u') {
            hasEscapes = true;
            advance(); // consume \
            advance(); // consume u
            actualName.append(scanUnicodeEscape());
        } else {
            char c = advance();
            actualName.append(c);
            if (Character.isHighSurrogate(c) && !isAtEnd() && Character.isLowSurrogate(peek())) {
                actualName.append(advance());
            }
        }

        // Handle remaining characters
        while (!isAtEnd() && (isAlphaNumeric(peek()) || (peek() == '\\' && peekNext() == 'u'))) {
            if (peek() == '\\') {
                hasEscapes = true;
                advance(); // consume \
                advance(); // consume u
                actualName.append(scanUnicodeEscape());
            } else {
                char c = advance();
                actualName.append(c);
                if (Character.isHighSurrogate(c) && !isAtEnd() && Character.isLowSurrogate(peek())) {
                    actualName.append(advance());
                }
            }
        }

        String identifierName = actualName.toString();
        TokenType type = hasEscapes ? TokenType.IDENTIFIER : lookupKeyword(identifierName);
        Object literal = literalForKeyword(type);

        int rawLength = position - startPos;
        int endColumn = startColumn + rawLength;
        return new Token(type, identifierName, literal, startLine, startColumn, startPos, position, startLine, endColumn, null);
    }

    private String scanUnicodeEscape() {
        // Support both \\uXXXX and \\u{X...X} formats
        // Returns a String to support surrogate pairs for code points > 0xFFFF
        if (peek() == '{') {
            // Braced unicode: \\u{...}
            advance(); // consume {
            StringBuilder hex = new StringBuilder();
            while (!isAtEnd() && peek() != '}') {
                if (isHexDigit(peek())) {
                    hex.append(advance());
                } else {
                    throw new RuntimeException("Invalid unicode escape sequence");
                }
            }
            if (peek() != '}') {
                throw new RuntimeException("Invalid unicode escape sequence");
            }
            advance(); // consume }

            if (hex.length() == 0) {
                throw new RuntimeException("Invalid unicode escape sequence");
            }

            int codePoint = Integer.parseInt(hex.toString(), 16);
            // Return the string representation (handles surrogate pairs automatically)
            return Character.toString(codePoint);
        } else {
            // Fixed 4-digit unicode: \\uXXXX
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < 4; i++) {
                if (!isAtEnd() && isHexDigit(peek())) {
                    hex.append(advance());
                } else {
                    throw new RuntimeException("Invalid unicode escape sequence");
                }
            }
            int codePoint = Integer.parseInt(hex.toString(), 16);
            return Character.toString(codePoint);
        }
    }

    private boolean isHexDigit(char c) {
        return (c >= '0' && c <= '9') ||
               (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
    }

    private Token scanRegex(int startLine, int startColumn, int startPos) {
        advance(); // consume '/'

        StringBuilder pattern = new StringBuilder();
        boolean escaped = false;
        boolean inCharClass = false;

        while (!isAtEnd()) {
            char c = peek();

            if (escaped) {
                pattern.append(c);
                advance();
                escaped = false;
                continue;
            }

            if (c == '\\') {
                pattern.append(c);
                advance();
                escaped = true;
                continue;
            }

            if (c == '[') {
                inCharClass = true;
                pattern.append(c);
                advance();
                continue;
            }

            if (c == ']' && inCharClass) {
                inCharClass = false;
                pattern.append(c);
                advance();
                continue;
            }

            if (c == '/' && !inCharClass) {
                advance(); // consume closing /
                break;
            }

            if (c == '\n') {
                throw new RuntimeException("Unterminated regex literal");
            }

            pattern.append(c);
            advance();
        }

        // Scan flags (g, i, m, s, u, y)
        StringBuilder flags = new StringBuilder();
        while (!isAtEnd() && isAlpha(peek())) {
            flags.append(advance());
        }

        String lexeme = source.substring(startPos, position);
        String patternStr = pattern.toString();
        String flagsStr = flags.toString();

        // Validate regex according to ECMAScript specification
        com.jsparser.regex.RegexValidator.validate(
            patternStr, flagsStr,
            startPos, startLine, startColumn
        );

        // Return token with regex info in literal
        return new Token(TokenType.REGEX, lexeme,
            new Literal.RegexInfo(patternStr, flagsStr),
            startLine, startColumn, startPos, position);
    }

    public Token scanRegexAt(int pos) {
        // Save current state
        int savedPosition = position;
        int savedLine = line;
        int savedColumn = column;

        // Move to the specified position
        position = pos;
        // Recalculate line and column for this position
        line = 1;
        column = 0;
        for (int i = 0; i < pos; i++) {
            if (buf[i] == '\n') {
                line++;
                column = 0;
            } else {
                column++;
            }
        }

        int startLine = line;
        int startColumn = column;
        int startPos = position;

        Token token = scanRegex(startLine, startColumn, startPos);

        // Restore state
        position = savedPosition;
        line = savedLine;
        column = savedColumn;

        return token;
    }

    private void skipWhitespace() {
        while (!isAtEnd()) {
            char c = peek();
            // ECMAScript whitespace characters:
            // U+0009 (Tab), U+000B (Vertical Tab), U+000C (Form Feed)
            // U+0020 (Space), U+00A0 (No-break space)
            // U+FEFF (Zero width no-break space / BOM)
            // U+2028 (Line Separator), U+2029 (Paragraph Separator)
            // Plus Unicode category Zs (Space Separator)
            switch (c) {
                case ' ', '\t', '\u000B', '\u000C', '\u00A0', '\uFEFF' -> advance();
                case '\r' -> {
                    // CR is a line terminator
                    line++;
                    column = -1;
                    advance(); // consume \r, column becomes 0
                    atLineStart = true;
                    // Handle CRLF as single line terminator
                    if (!isAtEnd() && peek() == '\n') {
                        position++; // consume \n without incrementing column
                    }
                }
                case '\n', '\u2028', '\u2029' -> {
                    // Line terminators
                    line++;
                    column = -1;  // Will be incremented to 0 by advance()
                    advance();
                    atLineStart = true;
                }
                default -> {
                    // Check for other Unicode space separators (category Zs)
                    if (Character.getType(c) == Character.SPACE_SEPARATOR) {
                        advance();
                    } else {
                        return;
                    }
                }
            }
        }
    }

    private boolean match(char expected) {
        if (isAtEnd() || peek() != expected) {
            return false;
        }
        advance();
        return true;
    }

    private char peek() {
        if (isAtEnd()) return '\0';
        return buf[position];
    }

    private char peekNext() {
        if (position + 1 >= length) return '\0';
        return buf[position + 1];
    }

    private char advance() {
        column++;
        return buf[position++];
    }

    private boolean isAtEnd() {
        return position >= length;
    }

    private boolean isLineTerminator(char c) {
        return c == '\n' || c == '\r' || c == '\u2028' || c == '\u2029';
    }

    private boolean isDigit(char c) {
        return c >= '0' && c <= '9';
    }

    private boolean isAlpha(char c) {
        // Support Unicode identifiers per ECMAScript spec
        // This includes letters from all languages (Greek µ, etc.)
        // Must explicitly include $ and _ as they're valid identifier starts
        // Also accept high surrogates as potential identifier starts
        return Character.isUnicodeIdentifierStart(c) || c == '$' || c == '_' || Character.isHighSurrogate(c);
    }

    private boolean isAlphaNumeric(char c) {
        // Support Unicode identifiers per ECMAScript spec
        // Must explicitly include $ as it's a valid identifier part
        // Also accept high surrogates as potential identifier parts
        return Character.isUnicodeIdentifierPart(c) || c == '$' || Character.isHighSurrogate(c);
    }

    private boolean canPrecedeMemberAccess(TokenType tokenType) {
        // Returns true if the token type can be followed by a member access (.)
        // These are tokens that represent values/expressions that can have properties
        if (tokenType == null) return false;

        return switch (tokenType) {
            case IDENTIFIER,     // obj.prop
                 RPAREN,         // (expr).prop
                 RBRACKET,       // arr[0].prop, [].prop
                 RBRACE,         // {}.prop
                 NUMBER,         // 123..toString()
                 STRING,         // "str".length
                 THIS,           // this.prop
                 SUPER,          // super.prop
                 TRUE, FALSE,    // true.toString()
                 NULL,           // null.toString()
                 TEMPLATE_TAIL,  // `str`.length
                 INCREMENT,      // (x++).prop
                 DECREMENT       // (x--).prop
                -> true;
            default -> false;
        };
    }

    private Token scanTemplateLiteral(int startLine, int startColumn, int startPos) {
        advance(); // consume opening `
        // Push a marker to indicate we're in a template context
        // We'll update this to the brace depth when/if we encounter ${
        templateBraceDepthStack.push(-1); // -1 means "not in interpolation yet"
        return scanTemplateChars(startLine, startColumn, startPos, true);
    }

    private Token scanTemplateContinuation() {
        int startLine = line;
        int startColumn = column;
        int startPos = position;
        advance(); // consume closing }
        return scanTemplateChars(startLine, startColumn, startPos, false);
    }

    private Token scanTemplateChars(int startLine, int startColumn, int startPos, boolean isStart) {
        StringBuilder raw = new StringBuilder();
        StringBuilder cooked = new StringBuilder();
        boolean hasInvalidEscape = false;

        while (!isAtEnd()) {
            char c = peek();

            if (c == '`') {
                // End of template
                advance();
                if (!templateBraceDepthStack.isEmpty()) {
                    templateBraceDepthStack.pop();
                }

                // Determine token type
                TokenType type;
                if (isStart) {
                    // Simple template with no interpolation: `hello`
                    type = TokenType.TEMPLATE_LITERAL;
                } else {
                    // End of template after interpolation: } ... `
                    type = TokenType.TEMPLATE_TAIL;
                }

                // For tagged templates with invalid escapes, cooked should be null
                String cookedValue = hasInvalidEscape ? null : cooked.toString();
                // Lexeme should be the actual source text from startPos to current position
                String lexeme = source.substring(startPos, position);
                // Raw value is the processed string with normalized line continuations
                String rawValue = raw.toString();
                // endPosition should be current position (after the closing `)
                // Use current line/column as endLine/endColumn since we've advanced past the `
                return new Token(type, lexeme, cookedValue, startLine, startColumn, startPos, position, line, column, rawValue);
            } else if (c == '$' && peekNext() == '{') {
                // Start of interpolation
                advance(); // consume $
                advance(); // consume {
                // Update the stack top from -1 (not in interpolation) to 0 (in interpolation, depth 0)
                if (!templateBraceDepthStack.isEmpty()) {
                    templateBraceDepthStack.pop();
                }
                templateBraceDepthStack.push(0);

                // Determine token type
                TokenType type = isStart ? TokenType.TEMPLATE_HEAD : TokenType.TEMPLATE_MIDDLE;

                // For tagged templates with invalid escapes, cooked should be null
                String cookedValue = hasInvalidEscape ? null : cooked.toString();
                // Lexeme should be the actual source text from startPos to current position
                String lexeme = source.substring(startPos, position);
                // Raw value is the processed string with normalized line continuations
                String rawValue = raw.toString();
                // endPosition should be current position (after the ${)
                // Use current line/column as endLine/endColumn since we've advanced past the ${
                return new Token(type, lexeme, cookedValue, startLine, startColumn, startPos, position, line, column, rawValue);
            } else if (c == '\\') {
                // Escape sequence
                raw.append(c);
                advance();
                if (!isAtEnd()) {
                    char escaped = peek();

                    // Handle line continuations specially for raw value normalization
                    boolean isLineContinuation = escaped == '\r' || escaped == '\n' || escaped == '\u2028' || escaped == '\u2029';

                    if (!isLineContinuation) {
                        raw.append(escaped);
                    }
                    advance();

                    // Process escape for cooked value
                    switch (escaped) {
                        case 'n' -> cooked.append('\n');
                        case 't' -> cooked.append('\t');
                        case 'r' -> cooked.append('\r');
                        case 'v' -> cooked.append('\u000B');
                        case 'f' -> cooked.append('\f');
                        case 'b' -> cooked.append('\b');
                        case '0' -> {
                            // Check if followed by a digit (making it an invalid octal escape)
                            if (!isAtEnd() && peek() >= '0' && peek() <= '9') {
                                hasInvalidEscape = true;
                            } else {
                                cooked.append('\0');
                            }
                        }
                        case '\\' -> cooked.append('\\');
                        case '`' -> cooked.append('`');
                        case '"' -> cooked.append('"');
                        case '\'' -> cooked.append('\'');
                        case '$' -> cooked.append('$');
                        case 'u' -> {
                            // Unicode escape in template
                            if (peek() == '{') {
                                // Braced unicode: backslash-u followed by braces
                                raw.append(peek());
                                advance(); // consume {
                                StringBuilder hexBuilder = new StringBuilder();
                                // Collect hex digits until } or invalid char (like `, $, newline)
                                while (!isAtEnd() && peek() != '}' && peek() != '`' && peek() != '$') {
                                    if (isHexDigit(peek())) {
                                        raw.append(peek());
                                        hexBuilder.append(peek());
                                        advance();
                                    } else {
                                        // Invalid character in unicode escape
                                        hasInvalidEscape = true;
                                        break;
                                    }
                                }
                                if (peek() == '}') {
                                    raw.append(peek());
                                    advance(); // consume }
                                    if (!hasInvalidEscape && hexBuilder.length() > 0) {
                                        try {
                                            int codePoint = Integer.parseInt(hexBuilder.toString(), 16);
                                            // Check if it's a valid Unicode code point
                                            if (Character.isValidCodePoint(codePoint)) {
                                                cooked.appendCodePoint(codePoint);
                                            } else {
                                                hasInvalidEscape = true;
                                            }
                                        } catch (IllegalArgumentException e) {
                                            // Catches NumberFormatException and invalid code points
                                            hasInvalidEscape = true;
                                        }
                                    }
                                } else {
                                    // No closing }, invalid escape
                                    hasInvalidEscape = true;
                                }
                            } else if (position + 4 <= length) {
                                // Fixed 4-digit unicode: backslash-u followed by 4 hex digits
                                // Check if all 4 characters are valid hex digits
                                boolean isValidHex = position + 4 <= length &&
                                    isHexDigit(buf[position]) && isHexDigit(buf[position + 1]) &&
                                    isHexDigit(buf[position + 2]) && isHexDigit(buf[position + 3]);
                                String hex = new String(buf, position, 4);

                                if (isValidHex) {
                                    raw.append(hex);
                                    int codePoint = Integer.parseInt(hex, 16);
                                    cooked.append((char) codePoint);
                                    for (int i = 0; i < 4; i++) advance();
                                } else {
                                    // Invalid unicode escape (not all hex digits)
                                    hasInvalidEscape = true;
                                    // Don't consume the invalid characters
                                }
                            } else {
                                // Not enough characters remaining
                                hasInvalidEscape = true;
                            }
                        }
                        case 'x' -> {
                            // Hex escape in template: backslash-x followed by 2 hex digits
                            if (position + 2 <= length) {
                                // Check if both characters are valid hex digits
                                boolean isValidHex = position + 2 <= length &&
                                    isHexDigit(buf[position]) && isHexDigit(buf[position + 1]);
                                String hex = new String(buf, position, 2);

                                if (isValidHex) {
                                    raw.append(hex);
                                    int codePoint = Integer.parseInt(hex, 16);
                                    cooked.append((char) codePoint);
                                    for (int i = 0; i < 2; i++) advance();
                                } else {
                                    // Invalid hex escape like \xg or \x1g
                                    hasInvalidEscape = true;
                                    // Don't consume the invalid characters
                                }
                            } else {
                                // Not enough characters for \xHH
                                hasInvalidEscape = true;
                            }
                        }
                        case '1', '2', '3', '4', '5', '6', '7', '8', '9' -> {
                            // Octal escapes are invalid in templates
                            hasInvalidEscape = true;
                        }
                        case '\r', '\n', '\u2028', '\u2029' -> {
                            // Line continuation in template literal
                            // In raw: backslash + normalized line terminator (CRLF -> \n, \r -> \n, others unchanged)
                            // In cooked: remove entirely (backslash + line terminator = empty string)

                            // Update line tracking
                            line++;
                            column = 0;  // We're at the start of the new line

                            // Handle CRLF as single line terminator
                            if (escaped == '\r' && !isAtEnd() && peek() == '\n') {
                                // CRLF: normalize to \n in raw
                                raw.append('\n');
                                advance(); // consume the \n
                                column = 0;  // Still at start of new line
                            } else if (escaped == '\r') {
                                // Bare \r: normalize to \n in raw
                                raw.append('\n');
                            } else {
                                // \n, U+2028, or U+2029: keep as-is in raw
                                raw.append(escaped);
                            }
                            // Don't append anything to cooked - line continuation is removed
                        }
                        default -> cooked.append(escaped);
                    }
                }
            } else {
                // Literal characters (not escape sequences)
                // Normalize line terminators: CR and CRLF both become LF
                if (c == '\r') {
                    // Check if followed by LF (CRLF sequence)
                    // Don't use advance() for \r since it shouldn't count as a column
                    position++;  // Manually increment position without affecting column
                    if (!isAtEnd() && peek() == '\n') {
                        // CRLF: normalize to LF in both raw and cooked
                        raw.append('\n');
                        cooked.append('\n');
                        advance(); // consume the LF (this also increments column)
                        // After consuming CRLF, set to new line
                        line++;
                        column = 0;  // Reset to start of new line (advance already moved us past the \n)
                    } else {
                        // Just CR: normalize to LF
                        raw.append('\n');
                        cooked.append('\n');
                        line++;
                        column = 0;  // Reset to start of new line (we've moved past the \r)
                    }
                } else if (c == '\n' || c == '\u2028' || c == '\u2029') {
                    // LF, LS, PS: keep as-is
                    raw.append(c);
                    cooked.append(c);
                    advance(); // consume the line terminator (increments column)
                    line++;
                    column = 0;  // Reset to start of new line
                } else {
                    // Regular character
                    raw.append(c);
                    cooked.append(c);
                    advance();
                }
            }
        }

        throw new RuntimeException("Unterminated template literal");
    }
}
