package com.jsparser;

import com.jsparser.ast.Literal;
import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

public class Lexer {
    private final String source;
    private int position = 0;
    private int line = 1;
    private int column = 0;
    private Stack<Integer> templateBraceDepthStack = new Stack<>();
    private TokenType lastTokenType = null;
    private boolean atLineStart = true;

    public Lexer(String source) {
        this.source = source;
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
                lastTokenType = token.type();
                atLineStart = false;
            }
        }

        tokens.add(new Token(TokenType.EOF, "", line, column, position));
        return tokens;
    }

    private boolean shouldParseRegex() {
        // After these tokens, / starts a regex literal, not division
        return lastTokenType == null ||
               lastTokenType == TokenType.ASSIGN ||
               lastTokenType == TokenType.LPAREN ||
               lastTokenType == TokenType.LBRACKET ||
               lastTokenType == TokenType.COMMA ||
               lastTokenType == TokenType.LBRACE ||
               lastTokenType == TokenType.COLON ||
               lastTokenType == TokenType.SEMICOLON ||
               lastTokenType == TokenType.RETURN ||
               lastTokenType == TokenType.IF ||
               lastTokenType == TokenType.WHILE ||
               lastTokenType == TokenType.FOR ||
               lastTokenType == TokenType.BANG ||
               lastTokenType == TokenType.EQ ||
               lastTokenType == TokenType.NE ||
               lastTokenType == TokenType.EQ_STRICT ||
               lastTokenType == TokenType.NE_STRICT ||
               lastTokenType == TokenType.LT ||
               lastTokenType == TokenType.LE ||
               lastTokenType == TokenType.GT ||
               lastTokenType == TokenType.GE ||
               lastTokenType == TokenType.AND ||
               lastTokenType == TokenType.OR ||
               lastTokenType == TokenType.QUESTION ||
               lastTokenType == TokenType.PLUS ||
               lastTokenType == TokenType.MINUS ||
               lastTokenType == TokenType.STAR ||
               lastTokenType == TokenType.PERCENT ||
               lastTokenType == TokenType.BIT_AND ||
               lastTokenType == TokenType.BIT_OR ||
               lastTokenType == TokenType.BIT_XOR ||
               lastTokenType == TokenType.LEFT_SHIFT ||
               lastTokenType == TokenType.RIGHT_SHIFT ||
               lastTokenType == TokenType.UNSIGNED_RIGHT_SHIFT ||
               lastTokenType == TokenType.PLUS_ASSIGN ||
               lastTokenType == TokenType.MINUS_ASSIGN ||
               lastTokenType == TokenType.STAR_ASSIGN ||
               lastTokenType == TokenType.SLASH_ASSIGN ||
               lastTokenType == TokenType.PERCENT_ASSIGN;
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
                        } else if (ch == '\r') {
                            line++;
                            column = 0;
                            containsLineTerminator = true;
                            // Handle CRLF as a single line terminator
                            if (peekNext() == '\n') {
                                advance(); // consume the \r
                                // The \n will be consumed by the next iteration
                            }
                        } else if (ch == '\u2028' || ch == '\u2029') {
                            // Line Separator (LS) and Paragraph Separator (PS)
                            line++;
                            column = 0;
                            containsLineTerminator = true;
                        }
                        advance();
                    }
                    // If comment contains line terminator, treat it as being at line start
                    if (containsLineTerminator) {
                        atLineStart = true;
                    }
                    yield null; // Skip the comment
                } else if (shouldParseRegex()) {
                    // Check if this should be a regex literal
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
                throw new RuntimeException(String.format("Unexpected character: %c (U+%04X)", c, (int)c));
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
                            // Grammar:
                            //   OctalDigit [lookahead ∉ OctalDigit]
                            //   ZeroToThree OctalDigit [lookahead ∉ OctalDigit]
                            //   FourToSeven OctalDigit
                            //   ZeroToThree OctalDigit OctalDigit
                            StringBuilder octalBuilder = new StringBuilder();
                            octalBuilder.append(escaped);

                            if (!isAtEnd() && peek() >= '0' && peek() <= '7') {
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
                            } else if (position + 4 <= source.length()) {
                                // Fixed 4-digit unicode: \\uXXXX
                                String hex = source.substring(position, position + 4);
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
                            if (position + 2 <= source.length()) {
                                String hex = source.substring(position, position + 2);
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
        return new Token(TokenType.STRING, lexeme, value.toString(), startLine, startColumn, startPos);
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
                    literal = Long.parseLong(hexPart, 16);
                } catch (NumberFormatException e2) {
                    // Number too large for Long, convert to double
                    literal = (double) Long.parseUnsignedLong(hexPart, 16);
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
                    literal = Long.parseLong(octalPart, 8);
                } catch (NumberFormatException e2) {
                    // Number too large for Long, convert to double
                    literal = (double) Long.parseUnsignedLong(octalPart, 8);
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
                    literal = Long.parseLong(binaryPart, 2);
                } catch (NumberFormatException e2) {
                    // Number too large for Long, convert to double
                    literal = (double) Long.parseUnsignedLong(binaryPart, 2);
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
                long longValue = (long) doubleValue;
                // Check if converting to long loses precision
                if ((double) longValue == doubleValue) {
                    if (longValue >= Integer.MIN_VALUE && longValue <= Integer.MAX_VALUE) {
                        literal = (int) longValue;
                    } else {
                        literal = longValue;
                    }
                } else {
                    // Conversion loses precision, keep as double
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
        StringBuilder actualName = new StringBuilder();

        // Handle first character (could be unicode escape or regular)
        if (peek() == '\\' && peekNext() == 'u') {
            advance(); // consume \
            advance(); // consume u
            // scanUnicodeEscape now returns String (may be 1 or 2 chars for surrogate pairs)
            actualName.append(scanUnicodeEscape());
        } else {
            char c = advance();
            actualName.append(c);
            // Handle surrogate pairs
            if (Character.isHighSurrogate(c) && !isAtEnd() && Character.isLowSurrogate(peek())) {
                actualName.append(advance());
            }
        }

        // Handle remaining characters
        while (!isAtEnd() && (isAlphaNumeric(peek()) || (peek() == '\\' && peekNext() == 'u'))) {
            if (peek() == '\\') {
                advance(); // consume \
                advance(); // consume u
                // scanUnicodeEscape now returns String (may be 1 or 2 chars for surrogate pairs)
                actualName.append(scanUnicodeEscape());
            } else {
                char c = advance();
                actualName.append(c);
                // Handle surrogate pairs
                if (Character.isHighSurrogate(c) && !isAtEnd() && Character.isLowSurrogate(peek())) {
                    actualName.append(advance());
                }
            }
        }

        String lexeme = source.substring(startPos, position);
        String identifierName = actualName.toString();

        // Keywords with escape sequences should be treated as identifiers
        // Only check for keywords if the lexeme doesn't contain escapes
        boolean hasEscapes = lexeme.contains("\\");

        TokenType type = TokenType.IDENTIFIER;
        if (!hasEscapes) {
            type = switch (identifierName) {
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
                // Note: 'of' is a contextual keyword, treated as identifier by lexer
                // Parser will check for peek().lexeme().equals("of") where needed
                case "instanceof" -> TokenType.INSTANCEOF;
                case "yield" -> TokenType.YIELD;
                case "import" -> TokenType.IMPORT;
                case "export" -> TokenType.EXPORT;
                case "with" -> TokenType.WITH;
                case "debugger" -> TokenType.DEBUGGER;
                default -> TokenType.IDENTIFIER;
            };
        }

        Object literal = switch (type) {
            case TRUE -> true;
            case FALSE -> false;
            case NULL -> null;
            default -> null;
        };

        // Always use the decoded name for identifiers and keywords (not the raw lexeme with escapes)
        // This ensures \u0063onst becomes "const" not "\u0063onst"
        String tokenLexeme = identifierName;
        // endPosition is current position (after consuming all chars including escapes)
        return new Token(type, tokenLexeme, literal, startLine, startColumn, startPos, position);
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

        // Return token with regex info in literal
        return new Token(TokenType.REGEX, lexeme,
            new Literal.RegexInfo(pattern.toString(), flags.toString()),
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
            if (source.charAt(i) == '\n') {
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
        return source.charAt(position);
    }

    private char peekNext() {
        if (position + 1 >= source.length()) return '\0';
        return source.charAt(position + 1);
    }

    private char advance() {
        column++;
        return source.charAt(position++);
    }

    private boolean isAtEnd() {
        return position >= source.length();
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
                // endPosition should be current position (after the closing `)
                return new Token(type, raw.toString(), cookedValue, startLine, startColumn, startPos, position);
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
                // endPosition should be current position (after the ${)
                return new Token(type, raw.toString(), cookedValue, startLine, startColumn, startPos, position);
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
                            } else if (position + 4 <= source.length()) {
                                // Fixed 4-digit unicode: backslash-u followed by 4 hex digits
                                String hex = source.substring(position, position + 4);
                                // Check if all 4 characters are valid hex digits
                                boolean isValidHex = hex.length() == 4 &&
                                    isHexDigit(hex.charAt(0)) && isHexDigit(hex.charAt(1)) &&
                                    isHexDigit(hex.charAt(2)) && isHexDigit(hex.charAt(3));

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
                            if (position + 2 <= source.length()) {
                                String hex = source.substring(position, position + 2);
                                // Check if both characters are valid hex digits
                                boolean isValidHex = hex.length() == 2 &&
                                    isHexDigit(hex.charAt(0)) && isHexDigit(hex.charAt(1));

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
                            column = -1;

                            // Handle CRLF as single line terminator
                            if (escaped == '\r' && !isAtEnd() && peek() == '\n') {
                                // CRLF: normalize to \n in raw
                                raw.append('\n');
                                advance(); // consume the \n
                                column = -1;
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
                    advance();
                    if (!isAtEnd() && peek() == '\n') {
                        // CRLF: normalize to LF in both raw and cooked
                        raw.append('\n');
                        cooked.append('\n');
                        advance(); // consume the LF
                    } else {
                        // Just CR: normalize to LF
                        raw.append('\n');
                        cooked.append('\n');
                    }
                    line++;
                    column = -1;
                } else if (c == '\n' || c == '\u2028' || c == '\u2029') {
                    // LF, LS, PS: keep as-is
                    raw.append(c);
                    cooked.append(c);
                    line++;
                    column = -1;
                    advance();
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
