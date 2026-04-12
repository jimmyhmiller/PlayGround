package jrust;

import java.util.ArrayList;
import java.util.List;

public class Lexer {
    private final String source;
    private int pos;
    private int line;
    private int col;

    public Lexer(String source) {
        this.source = source;
        this.pos = 0;
        this.line = 1;
        this.col = 1;
    }

    public List<Token> tokenize() {
        List<Token> tokens = new ArrayList<>();
        while (pos < source.length()) {
            skipWhitespaceAndComments();
            if (pos >= source.length()) break;

            Token token = nextToken();
            if (token != null) {
                tokens.add(token);
            }
        }
        tokens.add(new Token(Token.Kind.EOF, "", line, col));
        return tokens;
    }

    private void skipWhitespaceAndComments() {
        while (pos < source.length()) {
            char c = source.charAt(pos);
            if (c == ' ' || c == '\t' || c == '\r') {
                advance();
            } else if (c == '\n') {
                advance();
                line++;
                col = 1;
            } else if (c == '/' && pos + 1 < source.length() && source.charAt(pos + 1) == '/') {
                // Line comment
                while (pos < source.length() && source.charAt(pos) != '\n') {
                    advance();
                }
            } else {
                break;
            }
        }
    }

    private char advance() {
        char c = source.charAt(pos);
        pos++;
        col++;
        return c;
    }

    private boolean match(char expected) {
        if (pos < source.length() && source.charAt(pos) == expected) {
            advance();
            return true;
        }
        return false;
    }

    private Token nextToken() {
        int startLine = line;
        int startCol = col;
        char c = advance();

        switch (c) {
            case '(': return new Token(Token.Kind.LPAREN, "(", startLine, startCol);
            case ')': return new Token(Token.Kind.RPAREN, ")", startLine, startCol);
            case '{': return new Token(Token.Kind.LBRACE, "{", startLine, startCol);
            case '}': return new Token(Token.Kind.RBRACE, "}", startLine, startCol);
            case '[': return new Token(Token.Kind.LBRACKET, "[", startLine, startCol);
            case ']': return new Token(Token.Kind.RBRACKET, "]", startLine, startCol);
            case ',': return new Token(Token.Kind.COMMA, ",", startLine, startCol);
            case ';': return new Token(Token.Kind.SEMI, ";", startLine, startCol);
            case '+': return new Token(Token.Kind.PLUS, "+", startLine, startCol);
            case '*': return new Token(Token.Kind.STAR, "*", startLine, startCol);
            case '%': return new Token(Token.Kind.PERCENT, "%", startLine, startCol);
            case '/': return new Token(Token.Kind.SLASH, "/", startLine, startCol);

            case '-':
                if (match('>')) return new Token(Token.Kind.ARROW, "->", startLine, startCol);
                return new Token(Token.Kind.MINUS, "-", startLine, startCol);

            case '=':
                if (match('=')) return new Token(Token.Kind.EQEQ, "==", startLine, startCol);
                if (match('>')) return new Token(Token.Kind.FATARROW, "=>", startLine, startCol);
                return new Token(Token.Kind.EQ, "=", startLine, startCol);

            case '!':
                if (match('=')) return new Token(Token.Kind.BANGEQ, "!=", startLine, startCol);
                return new Token(Token.Kind.BANG, "!", startLine, startCol);

            case '<':
                if (match('=')) return new Token(Token.Kind.LTEQ, "<=", startLine, startCol);
                return new Token(Token.Kind.LT, "<", startLine, startCol);

            case '>':
                if (match('=')) return new Token(Token.Kind.GTEQ, ">=", startLine, startCol);
                return new Token(Token.Kind.GT, ">", startLine, startCol);

            case ':':
                if (match(':')) return new Token(Token.Kind.COLONCOLON, "::", startLine, startCol);
                return new Token(Token.Kind.COLON, ":", startLine, startCol);

            case '.':
                if (match('.')) return new Token(Token.Kind.DOTDOT, "..", startLine, startCol);
                return new Token(Token.Kind.DOT, ".", startLine, startCol);

            case '&':
                if (match('&')) return new Token(Token.Kind.AMPAMP, "&&", startLine, startCol);
                return new Token(Token.Kind.AMP, "&", startLine, startCol);

            case '|':
                if (match('|')) return new Token(Token.Kind.PIPEPIPE, "||", startLine, startCol);
                return new Token(Token.Kind.PIPE, "|", startLine, startCol);

            case '"': return lexString(startLine, startCol);
            case '\'': return lexChar(startLine, startCol);

            default:
                if (c == '_' && (pos >= source.length() || !Character.isLetterOrDigit(source.charAt(pos)) && source.charAt(pos) != '_')) {
                    return new Token(Token.Kind.UNDERSCORE, "_", startLine, startCol);
                }
                if (Character.isDigit(c)) {
                    return lexNumber(c, startLine, startCol);
                }
                if (Character.isLetter(c) || c == '_') {
                    return lexIdentOrKeyword(c, startLine, startCol);
                }
                throw new RuntimeException("Unexpected character '" + c + "' at " + startLine + ":" + startCol);
        }
    }

    private Token lexString(int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        while (pos < source.length() && source.charAt(pos) != '"') {
            char c = source.charAt(pos);
            if (c == '\\') {
                advance();
                if (pos >= source.length()) throw new RuntimeException("Unterminated string at " + startLine + ":" + startCol);
                char escaped = advance();
                switch (escaped) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case '\\': sb.append('\\'); break;
                    case '"': sb.append('"'); break;
                    case '\'': sb.append('\''); break;
                    case '0': sb.append('\0'); break;
                    default: throw new RuntimeException("Unknown escape sequence '\\" + escaped + "' at " + line + ":" + col);
                }
            } else {
                if (c == '\n') {
                    line++;
                    col = 0;
                }
                sb.append(advance());
            }
        }
        if (pos >= source.length()) {
            throw new RuntimeException("Unterminated string at " + startLine + ":" + startCol);
        }
        advance(); // closing "
        return new Token(Token.Kind.STRING_LIT, sb.toString(), startLine, startCol);
    }

    private Token lexChar(int startLine, int startCol) {
        char value;
        if (pos >= source.length()) throw new RuntimeException("Unterminated char literal at " + startLine + ":" + startCol);
        char c = advance();
        if (c == '\\') {
            if (pos >= source.length()) throw new RuntimeException("Unterminated char literal at " + startLine + ":" + startCol);
            char escaped = advance();
            value = switch (escaped) {
                case 'n' -> '\n';
                case 't' -> '\t';
                case 'r' -> '\r';
                case '\\' -> '\\';
                case '\'' -> '\'';
                case '0' -> '\0';
                default -> throw new RuntimeException("Unknown escape in char literal '\\" + escaped + "' at " + line + ":" + col);
            };
        } else {
            value = c;
        }
        if (pos >= source.length() || source.charAt(pos) != '\'') {
            throw new RuntimeException("Unterminated char literal at " + startLine + ":" + startCol);
        }
        advance(); // closing '
        return new Token(Token.Kind.CHAR_LIT, String.valueOf(value), startLine, startCol);
    }

    private Token lexNumber(char first, int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        sb.append(first);
        boolean isFloat = false;
        while (pos < source.length()) {
            char c = source.charAt(pos);
            if (Character.isDigit(c)) {
                sb.append(advance());
            } else if (c == '.' && !isFloat && pos + 1 < source.length() && Character.isDigit(source.charAt(pos + 1))) {
                isFloat = true;
                sb.append(advance()); // .
            } else {
                break;
            }
        }
        return new Token(isFloat ? Token.Kind.FLOAT_LIT : Token.Kind.INT_LIT, sb.toString(), startLine, startCol);
    }

    private Token lexIdentOrKeyword(char first, int startLine, int startCol) {
        StringBuilder sb = new StringBuilder();
        sb.append(first);
        while (pos < source.length()) {
            char c = source.charAt(pos);
            if (Character.isLetterOrDigit(c) || c == '_') {
                sb.append(advance());
            } else {
                break;
            }
        }
        String word = sb.toString();
        Token.Kind kind = switch (word) {
            case "fn" -> Token.Kind.FN;
            case "struct" -> Token.Kind.STRUCT;
            case "impl" -> Token.Kind.IMPL;
            case "let" -> Token.Kind.LET;
            case "mut" -> Token.Kind.MUT;
            case "self" -> Token.Kind.SELF;
            case "import" -> Token.Kind.IMPORT;
            case "return" -> Token.Kind.RETURN;
            case "true" -> Token.Kind.TRUE;
            case "false" -> Token.Kind.FALSE;
            case "if" -> Token.Kind.IF;
            case "else" -> Token.Kind.ELSE;
            case "while" -> Token.Kind.WHILE;
            case "for" -> Token.Kind.FOR;
            case "in" -> Token.Kind.IN;
            case "match" -> Token.Kind.MATCH;
            case "enum" -> Token.Kind.ENUM;
            case "const" -> Token.Kind.CONST;
            case "null" -> Token.Kind.NULL;
            case "break" -> Token.Kind.BREAK;
            case "continue" -> Token.Kind.CONTINUE;
            default -> Token.Kind.IDENT;
        };
        return new Token(kind, word, startLine, startCol);
    }
}
