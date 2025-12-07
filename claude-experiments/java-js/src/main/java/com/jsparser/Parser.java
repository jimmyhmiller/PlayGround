package com.jsparser;

import com.jsparser.ast.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static java.util.Map.entry;

public class Parser {
    /**
     * Complete precedence table for JavaScript operators.
     * Higher precedence values = tighter binding.
     * Based on ECMAScript specification operator precedence.
     *
     * Precedence levels:
     * 1  - Comma (handled separately by parseSequence)
     * 2  - Assignment operators (right-associative)
     * 3  - Conditional/Ternary (handled separately - two-token operator)
     * 4  - Nullish coalescing (??)
     * 5  - Logical OR (||)
     * 6  - Logical AND (&&)
     * 7  - Bitwise OR (|)
     * 8  - Bitwise XOR (^)
     * 9  - Bitwise AND (&)
     * 10 - Equality (==, !=, ===, !==)
     * 11 - Relational (<, <=, >, >=, instanceof, in)
     * 12 - Shift (<<, >>, >>>)
     * 13 - Additive (+, -)
     * 14 - Multiplicative (*, /, %)
     * 15 - Exponentiation (**) (right-associative)
     */
    private static final Map<TokenType, OperatorInfo> OPERATOR_PRECEDENCE = Map.ofEntries(
        // Precedence 2: Assignment operators (right-associative)
        entry(TokenType.ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.PLUS_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.MINUS_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.STAR_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.SLASH_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.PERCENT_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.STAR_STAR_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.LEFT_SHIFT_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.RIGHT_SHIFT_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.UNSIGNED_RIGHT_SHIFT_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.BIT_AND_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.BIT_OR_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.BIT_XOR_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.AND_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.OR_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),
        entry(TokenType.QUESTION_QUESTION_ASSIGN, new OperatorInfo(2, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.ASSIGNMENT)),

        // Precedence 4: Nullish coalescing
        entry(TokenType.QUESTION_QUESTION, new OperatorInfo(4, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.LOGICAL)),

        // Precedence 5: Logical OR
        entry(TokenType.OR, new OperatorInfo(5, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.LOGICAL)),

        // Precedence 6: Logical AND
        entry(TokenType.AND, new OperatorInfo(6, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.LOGICAL)),

        // Precedence 7: Bitwise OR
        entry(TokenType.BIT_OR, new OperatorInfo(7, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 8: Bitwise XOR
        entry(TokenType.BIT_XOR, new OperatorInfo(8, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 9: Bitwise AND
        entry(TokenType.BIT_AND, new OperatorInfo(9, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 10: Equality
        entry(TokenType.EQ, new OperatorInfo(10, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.NE, new OperatorInfo(10, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.EQ_STRICT, new OperatorInfo(10, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.NE_STRICT, new OperatorInfo(10, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 11: Relational
        entry(TokenType.LT, new OperatorInfo(11, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.LE, new OperatorInfo(11, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.GT, new OperatorInfo(11, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.GE, new OperatorInfo(11, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.INSTANCEOF, new OperatorInfo(11, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.IN, new OperatorInfo(11, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 12: Shift
        entry(TokenType.LEFT_SHIFT, new OperatorInfo(12, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.RIGHT_SHIFT, new OperatorInfo(12, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.UNSIGNED_RIGHT_SHIFT, new OperatorInfo(12, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 13: Additive
        entry(TokenType.PLUS, new OperatorInfo(13, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.MINUS, new OperatorInfo(13, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 14: Multiplicative
        entry(TokenType.STAR, new OperatorInfo(14, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.SLASH, new OperatorInfo(14, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),
        entry(TokenType.PERCENT, new OperatorInfo(14, OperatorInfo.Associativity.LEFT, OperatorInfo.ExpressionType.BINARY)),

        // Precedence 15: Exponentiation (right-associative!)
        entry(TokenType.STAR_STAR, new OperatorInfo(15, OperatorInfo.Associativity.RIGHT, OperatorInfo.ExpressionType.BINARY))
    );

    private final List<Token> tokens;
    private final int sourceLength;
    private final Lexer lexer;
    private final String source;
    private final char[] sourceBuf;
    private final int[] lineOffsets; // Starting byte offset of each line
    private int current = 0;
    private boolean allowIn = true;
    private boolean inGenerator = false;
    private boolean inAsyncContext = false;
    private boolean inClassFieldInitializer = false;
    private final boolean forceModuleMode;

    // Strict mode tracking
    private boolean strictMode = false;
    private java.util.Stack<Boolean> strictModeStack = new java.util.Stack<>();

    public Parser(String source) {
        this(source, false, false);
    }

    public Parser(String source, boolean forceModuleMode) {
        this(source, forceModuleMode, false);
    }

    public Parser(String source, boolean forceModuleMode, boolean forceStrictMode) {
        this.source = source;
        this.sourceBuf = source.toCharArray();
        this.sourceLength = sourceBuf.length;

        // Module mode is always strict
        // Force strict mode can also be enabled via flag
        boolean initialStrictMode = forceModuleMode || forceStrictMode;
        this.strictMode = initialStrictMode;

        this.lexer = new Lexer(source, initialStrictMode);
        this.tokens = lexer.tokenize();
        this.forceModuleMode = forceModuleMode;
        this.lineOffsets = buildLineOffsetIndex();
    }

    public Program parse() {
        List<Statement> statements = new ArrayList<>();

        while (!isAtEnd()) {
            statements.add(parseStatement());
        }

        // Process directive prologue for program
        statements = processDirectives(statements);

        // Determine sourceType
        String sourceType;
        if (forceModuleMode) {
            // Explicitly forced to module mode (e.g., from Test262 frontmatter)
            sourceType = "module";
        } else {
            // Auto-detect: if any import/export statements, it's a module
            sourceType = "script";
            for (Statement stmt : statements) {
                if (stmt instanceof ImportDeclaration ||
                    stmt instanceof ExportNamedDeclaration ||
                    stmt instanceof ExportDefaultDeclaration ||
                    stmt instanceof ExportAllDeclaration) {
                    sourceType = "module";
                    // Module mode is always strict
                    strictMode = true;
                    break;
                }
            }
        }

        // Program loc should span entire file (line 1 to last position)
        // Calculate the end position from the actual source length
        SourceLocation.Position endPos = getPositionFromOffset(sourceLength);
        SourceLocation loc = new SourceLocation(
            new SourceLocation.Position(1, 0),
            endPos
        );
        return new Program(0, sourceLength, loc, statements, sourceType);
    }

    private Statement parseStatement() {
        Token token = peek();

        return switch (token.type()) {
            case VAR, CONST -> parseVariableDeclaration();
            case LET -> {
                // Check if 'let' is used as identifier (e.g., let = 5, let[0]) or as declaration keyword
                // If followed by = (not part of destructuring), it's an identifier being assigned
                // If followed by line terminator, ASI applies and 'let' is an identifier
                Token letToken = peek();
                if (checkAhead(1, TokenType.ASSIGN)) {
                    // Parse as expression statement (let as identifier)
                    Expression expr = parseExpression();
                    consumeSemicolon("Expected ';' after expression");
                    Token endToken = previous();
                    yield new ExpressionStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), expr);
                } else if (current + 1 < tokens.size() && letToken.line() != tokens.get(current + 1).line()) {
                    // Line terminator after 'let' - treat as identifier with ASI
                    Expression expr = parseExpression();
                    consumeSemicolon("Expected ';' after expression");
                    Token endToken = previous();
                    yield new ExpressionStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), expr);
                } else {
                    // Parse as variable declaration (includes let x, let [x], let {x}, etc.)
                    yield parseVariableDeclaration();
                }
            }
            case LBRACE -> parseBlockStatement();
            case RETURN -> parseReturnStatement();
            case IF -> parseIfStatement();
            case WHILE -> parseWhileStatement();
            case DO -> parseDoWhileStatement();
            case FOR -> parseForStatement();
            case BREAK -> parseBreakStatement();
            case CONTINUE -> parseContinueStatement();
            case SWITCH -> parseSwitchStatement();
            case THROW -> parseThrowStatement();
            case TRY -> parseTryStatement();
            case WITH -> parseWithStatement();
            case DEBUGGER -> parseDebuggerStatement();
            case SEMICOLON -> parseEmptyStatement();
            case FUNCTION -> parseFunctionDeclaration(false);
            case CLASS -> parseClassDeclaration();
            case IMPORT -> {
                // Check if this is a dynamic import expression: import('./module.js')
                // or import.meta expression
                // vs an import declaration: import { foo } from 'module'
                if (current + 1 < tokens.size()) {
                    TokenType nextType = tokens.get(current + 1).type();
                    if (nextType == TokenType.LPAREN || nextType == TokenType.DOT) {
                        // Dynamic import or import.meta - parse as expression statement
                        Expression expr = parseExpression();
                        consumeSemicolon("Expected ';' after expression");
                        Token endToken = previous();
                        yield new ExpressionStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), expr);
                    } else {
                        // Import declaration
                        yield parseImportDeclaration();
                    }
                } else {
                    // Import declaration
                    yield parseImportDeclaration();
                }
            }
            case EXPORT -> parseExportDeclaration();
            case IDENTIFIER -> {
                // Check for async function declaration
                // No line terminator is allowed between async and function
                if (token.lexeme().equals("async") && current + 1 < tokens.size() &&
                    tokens.get(current + 1).type() == TokenType.FUNCTION &&
                    tokens.get(current).line() == tokens.get(current + 1).line()) {
                    yield parseFunctionDeclaration(true);
                }
                // Otherwise, it's an expression statement
                Expression expr = parseExpression();

                // Check if this is a labeled statement (identifier followed by colon)
                if (expr instanceof Identifier id && check(TokenType.COLON)) {
                    advance(); // consume ':'
                    Statement labeledBody = parseStatement();
                    Token endToken = previous();
                    yield new LabeledStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), id, labeledBody);
                }

                consumeSemicolon("Expected ';' after expression");
                Token endToken = previous();
                yield new ExpressionStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), expr);
            }
            default -> {
                Expression expr = parseExpression();

                // Check if this is a labeled statement (identifier followed by colon)
                if (expr instanceof Identifier id && check(TokenType.COLON)) {
                    advance(); // consume ':'
                    Statement labeledBody = parseStatement();
                    Token endToken = previous();
                    yield new LabeledStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), id, labeledBody);
                }

                consumeSemicolon("Expected ';' after expression");
                Token endToken = previous();
                yield new ExpressionStatement(getStart(token), getEnd(endToken), createLocation(token, endToken), expr);
            }
        };
    }

    private WhileStatement parseWhileStatement() {
        Token startToken = peek();
        advance(); // consume 'while'

        consume(TokenType.LPAREN, "Expected '(' after 'while'");
        Expression test = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after while condition");

        Statement body = parseStatement();

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new WhileStatement(getStart(startToken), getEnd(endToken), loc, test, body);
    }

    private DoWhileStatement parseDoWhileStatement() {
        Token startToken = peek();
        advance(); // consume 'do'

        Statement body = parseStatement();

        consume(TokenType.WHILE, "Expected 'while' after do body");
        consume(TokenType.LPAREN, "Expected '(' after 'while'");
        Expression test = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after do-while condition");

        // Special ASI rule for do-while: semicolon can be inserted even on same line
        // if previous token is ) and it would complete the do-while
        if (check(TokenType.SEMICOLON)) {
            advance();
        }
        // Otherwise ASI applies (line break, }, EOF, or offending token)

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new DoWhileStatement(getStart(startToken), getEnd(endToken), loc, body, test);
    }

    private Statement parseForStatement() {
        Token startToken = peek();
        advance(); // consume 'for'

        // Check for for-await-of: for await (...)
        boolean isAwait = false;
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("await")) {
            advance(); // consume 'await'
            isAwait = true;
        }

        consume(TokenType.LPAREN, "Expected '(' after 'for'");

        // Parse init/left clause - can be var/let/const declaration, expression, or empty
        // Disable 'in' as operator to allow for-in detection
        boolean oldAllowIn = allowIn;
        allowIn = false;

        Node initOrLeft = null;
        Token kindToken = null;
        if (!check(TokenType.SEMICOLON)) {
            // Check for var/let/const
            // BUT: 'let' is treated as identifier (not declaration keyword) when followed by:
            // - 'in' (for-in loop): for (let in obj)
            // - 'of' (for-of loop): for (let of arr)
            // - ';' (expression): for (let; ; )
            // - '=' (assignment): for (let = 3; ; )
            // Note: 'let [' IS a declaration (destructuring), not an identifier
            boolean isDeclaration = check(TokenType.VAR) || check(TokenType.CONST) ||
                (check(TokenType.LET) && !checkAhead(1, TokenType.IN) && !checkAhead(1, TokenType.SEMICOLON) &&
                 !checkAhead(1, TokenType.ASSIGN) &&
                 !(checkAhead(1, TokenType.IDENTIFIER) && tokens.get(current + 1).lexeme().equals("of")));

            if (isDeclaration && match(TokenType.VAR, TokenType.LET, TokenType.CONST)) {
                // Variable declaration - support destructuring patterns
                kindToken = previous();
                String kind = kindToken.lexeme();
                List<VariableDeclarator> declarators = new ArrayList<>();

                do {
                    Token patternStart = peek();
                    // Use parsePatternBase to support destructuring
                    Pattern pattern = parsePatternBase();

                    Expression initExpr = null;
                    if (match(TokenType.ASSIGN)) {
                        initExpr = parseAssignment();
                    }

                    Token declaratorEnd = previous();

                    int declaratorStart = getStart(patternStart);
                    int declaratorEndPos;
                    SourceLocation declaratorLoc;

                    // Use the end of the last token (which includes closing parens)
                    declaratorEndPos = getEnd(declaratorEnd);
                    declaratorLoc = createLocation(patternStart, declaratorEnd);

                    declarators.add(new VariableDeclarator(declaratorStart, declaratorEndPos, declaratorLoc, pattern, initExpr));

                } while (match(TokenType.COMMA));

                Token endToken = previous();
                SourceLocation declLoc = createLocation(kindToken, endToken);
                int declStart = getStart(kindToken);
                int declEnd = getEnd(endToken);
                initOrLeft = new VariableDeclaration(declStart, declEnd, declLoc, declarators, kind);
            } else {
                initOrLeft = parseExpression();
            }
        }

        // Restore allowIn flag
        allowIn = oldAllowIn;

        // Check for for-in or for-of
        // Note: 'of' is a contextual keyword (IDENTIFIER token with lexeme "of")
        boolean isOfKeyword = check(TokenType.IDENTIFIER) && peek().lexeme().equals("of");

        if (check(TokenType.IN)) {
            advance(); // consume 'in'

            // Validate for-in initializers
            // 1. Assignment expressions are never allowed: for (a = 0 in obj)
            // 2. Variable declaration initializers are only allowed in sloppy mode (Annex B): for (var a = 0 in obj)
            if (initOrLeft instanceof VariableDeclaration) {
                VariableDeclaration varDecl = (VariableDeclaration) initOrLeft;

                // Check if any declarator has an initializer
                for (VariableDeclarator declarator : varDecl.declarations()) {
                    if (declarator.init() != null) {
                        // Destructuring patterns (ObjectPattern/ArrayPattern) with initializers are NEVER allowed
                        Pattern pattern = declarator.id();
                        if (pattern instanceof ObjectPattern || pattern instanceof ArrayPattern) {
                            throw new ParseException("SyntaxError", peek(), null, "for-in statement",
                                "for-in loop variable declaration with destructuring may not have an initializer");
                        }

                        // In strict mode, initializers are never allowed
                        if (strictMode) {
                            throw new ParseException("SyntaxError", peek(), null, "for-in statement",
                                "for-in loop variable declaration may not have an initializer in strict mode");
                        }
                        // In sloppy mode, only single 'var' declarations with initializers are allowed (Annex B)
                        // let/const are never allowed, even in sloppy mode
                        if (!varDecl.kind().equals("var")) {
                            throw new ParseException("SyntaxError", peek(), null, "for-in statement",
                                "for-in loop " + varDecl.kind() + " declaration may not have an initializer");
                        }
                        if (varDecl.declarations().size() > 1) {
                            throw new ParseException("SyntaxError", peek(), null, "for-in statement",
                                "for-in loop variable declaration may not have an initializer");
                        }
                    }
                }
            } else if (initOrLeft instanceof Expression) {
                // Check if it's an assignment expression
                Expression expr = (Expression) initOrLeft;
                if (expr instanceof AssignmentExpression) {
                    throw new ParseException("SyntaxError", peek(), null, "for-in statement",
                        "Invalid left-hand side in for-in loop: assignment expression not allowed");
                }

                // Convert left to pattern if it's an expression (for destructuring)
                initOrLeft = convertToPatternIfNeeded(initOrLeft);
            }

            Expression right = parseExpression();
            consume(TokenType.RPAREN, "Expected ')' after for-in");
            Statement body = parseStatement();
            Token endToken = previous();
            SourceLocation loc = createLocation(startToken, endToken);
            return new ForInStatement(getStart(startToken), getEnd(endToken), loc, initOrLeft, right, body);
        } else if (isOfKeyword) {
            advance(); // consume 'of'
            // Convert left to pattern if it's an expression (for destructuring)
            if (initOrLeft instanceof Expression) {
                initOrLeft = convertToPatternIfNeeded(initOrLeft);
            }
            Expression right = parseExpression();
            consume(TokenType.RPAREN, "Expected ')' after for-of");
            Statement body = parseStatement();
            Token endToken = previous();
            SourceLocation loc = createLocation(startToken, endToken);
            return new ForOfStatement(getStart(startToken), getEnd(endToken), loc, isAwait, initOrLeft, right, body);
        }

        // Regular for loop
        consume(TokenType.SEMICOLON, "Expected ';' after for loop initializer");

        // Parse test clause - can be expression or empty
        Expression test = null;
        if (!check(TokenType.SEMICOLON)) {
            test = parseExpression();
        }
        consume(TokenType.SEMICOLON, "Expected ';' after for loop condition");

        // Parse update clause - can be expression or empty
        Expression update = null;
        if (!check(TokenType.RPAREN)) {
            update = parseExpression();
        }
        consume(TokenType.RPAREN, "Expected ')' after for clauses");

        Statement body = parseStatement();

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ForStatement(getStart(startToken), getEnd(endToken), loc, initOrLeft, test, update, body);
    }

    private IfStatement parseIfStatement() {
        Token startToken = peek();
        advance(); // consume 'if'

        consume(TokenType.LPAREN, "Expected '(' after 'if'");
        Expression test = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after if condition");

        Statement consequent = parseStatement();

        Statement alternate = null;
        if (match(TokenType.ELSE)) {
            alternate = parseStatement();
        }

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new IfStatement(getStart(startToken), getEnd(endToken), loc, test, consequent, alternate);
    }

    private ReturnStatement parseReturnStatement() {
        Token startToken = peek();
        Token returnToken = startToken;
        advance(); // consume 'return'

        Expression argument = null;

        // [no LineTerminator here] restriction
        // If there's a line break after 'return', treat it as return with no argument
        Token nextToken = peek();
        boolean hasLineBreak = returnToken.line() < nextToken.line();

        // Check if there's an argument
        if (!check(TokenType.SEMICOLON) && !check(TokenType.RBRACE) && !hasLineBreak && !isAtEnd()) {
            argument = parseExpression();
        }

        consumeSemicolon("Expected ';' after return statement");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ReturnStatement(getStart(startToken), getEnd(endToken), loc, argument);
    }

    private BreakStatement parseBreakStatement() {
        Token startToken = peek();
        advance(); // consume 'break'
        Token breakToken = previous(); // the 'break' token we just consumed

        // Optional label for break statement
        // No line terminator is allowed between break and its label
        Identifier label = null;
        if (check(TokenType.IDENTIFIER) && !check(TokenType.SEMICOLON) &&
            breakToken.line() == peek().line()) {
            Token labelToken = peek();
            advance();
            label = new Identifier(getStart(labelToken), getEnd(labelToken), createLocation(labelToken, labelToken), labelToken.lexeme());
        }

        consumeSemicolon("Expected ';' after break statement");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new BreakStatement(getStart(startToken), getEnd(endToken), loc, label);
    }

    private ContinueStatement parseContinueStatement() {
        Token startToken = peek();
        advance(); // consume 'continue'
        Token continueToken = previous(); // the 'continue' token we just consumed

        // Optional label for continue statement
        // No line terminator is allowed between continue and its label
        Identifier label = null;
        if (check(TokenType.IDENTIFIER) && !check(TokenType.SEMICOLON) &&
            continueToken.line() == peek().line()) {
            Token labelToken = peek();
            advance();
            label = new Identifier(getStart(labelToken), getEnd(labelToken), createLocation(labelToken, labelToken), labelToken.lexeme());
        }

        consumeSemicolon("Expected ';' after continue statement");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ContinueStatement(getStart(startToken), getEnd(endToken), loc, label);
    }

    private SwitchStatement parseSwitchStatement() {
        Token startToken = peek();
        advance(); // consume 'switch'

        consume(TokenType.LPAREN, "Expected '(' after 'switch'");
        Expression discriminant = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after switch discriminant");

        consume(TokenType.LBRACE, "Expected '{' before switch body");

        List<SwitchCase> cases = new ArrayList<>();

        while (!check(TokenType.RBRACE) && !isAtEnd()) {
            Token caseStart = peek();

            if (match(TokenType.CASE)) {
                // case test: consequent
                Expression test = parseExpression();
                consume(TokenType.COLON, "Expected ':' after case test");

                // Parse consequent statements until we hit another case/default or closing brace
                List<Statement> consequent = new ArrayList<>();
                while (!check(TokenType.CASE) && !check(TokenType.DEFAULT) && !check(TokenType.RBRACE) && !isAtEnd()) {
                    consequent.add(parseStatement());
                }

                Token caseEnd = previous();
                SourceLocation caseLoc = createLocation(caseStart, caseEnd);
                cases.add(new SwitchCase(getStart(caseStart), getEnd(caseEnd), caseLoc, test, consequent));

            } else if (match(TokenType.DEFAULT)) {
                // default: consequent
                consume(TokenType.COLON, "Expected ':' after 'default'");

                // Parse consequent statements
                List<Statement> consequent = new ArrayList<>();
                while (!check(TokenType.CASE) && !check(TokenType.DEFAULT) && !check(TokenType.RBRACE) && !isAtEnd()) {
                    consequent.add(parseStatement());
                }

                Token caseEnd = previous();
                SourceLocation caseLoc = createLocation(caseStart, caseEnd);
                cases.add(new SwitchCase(getStart(caseStart), getEnd(caseEnd), caseLoc, null, consequent));

            } else {
                throw new ExpectedTokenException("'case' or 'default' in switch body", peek());
            }
        }

        consume(TokenType.RBRACE, "Expected '}' after switch body");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new SwitchStatement(getStart(startToken), getEnd(endToken), loc, discriminant, cases);
    }

    private ThrowStatement parseThrowStatement() {
        Token startToken = peek();
        advance(); // consume 'throw'

        // ECMAScript spec: No line terminator is allowed between 'throw' and its expression
        Token prevToken = previous();
        Token nextToken = peek();
        if (prevToken.line() < nextToken.line()) {
            throw new ParseException("ValidationError", peek(), null, "throw statement", "Line break is not allowed between 'throw' and its expression");
        }

        // Parse the expression to throw
        Expression argument = parseExpression();
        consumeSemicolon("Expected ';' after throw statement");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ThrowStatement(getStart(startToken), getEnd(endToken), loc, argument);
    }

    private TryStatement parseTryStatement() {
        Token startToken = peek();
        advance(); // consume 'try'

        // Parse the try block
        BlockStatement block = parseBlockStatement();

        // Parse optional catch clause
        CatchClause handler = null;
        if (check(TokenType.CATCH)) {
            Token catchStart = peek();
            advance(); // consume 'catch'

            // Parse optional catch parameter (ES2019+ allows catch without parameter)
            Pattern param = null;
            if (match(TokenType.LPAREN)) {
                param = parsePatternBase();
                consume(TokenType.RPAREN, "Expected ')' after catch parameter");
            }

            // Parse catch body
            BlockStatement body = parseBlockStatement();
            Token catchEnd = previous();
            SourceLocation catchLoc = createLocation(catchStart, catchEnd);
            handler = new CatchClause(getStart(catchStart), getEnd(catchEnd), catchLoc, param, body);
        }

        // Parse optional finally clause
        BlockStatement finalizer = null;
        if (match(TokenType.FINALLY)) {
            finalizer = parseBlockStatement();
        }

        // Must have either catch or finally (or both)
        if (handler == null && finalizer == null) {
            throw new ParseException("ValidationError", peek(), null, "try statement", "Missing catch or finally after try");
        }

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new TryStatement(getStart(startToken), getEnd(endToken), loc, block, handler, finalizer);
    }

    private WithStatement parseWithStatement() {
        Token startToken = peek();
        advance(); // consume 'with'

        // Strict mode validation: with statements are not allowed in strict mode
        if (strictMode) {
            throw new ExpectedTokenException("'with' statement is not allowed in strict mode", startToken);
        }

        consume(TokenType.LPAREN, "Expected '(' after 'with'");
        Expression object = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after with object");

        Statement body = parseStatement();

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new WithStatement(getStart(startToken), getEnd(endToken), loc, object, body);
    }

    private DebuggerStatement parseDebuggerStatement() {
        Token startToken = peek();
        advance(); // consume 'debugger'
        consumeSemicolon("Expected ';' after debugger statement");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new DebuggerStatement(getStart(startToken), getEnd(endToken), loc);
    }

    private EmptyStatement parseEmptyStatement() {
        Token startToken = peek();
        advance(); // consume ';'
        SourceLocation loc = createLocation(startToken, startToken);
        return new EmptyStatement(getStart(startToken), getEnd(startToken), loc);
    }

    private FunctionDeclaration parseFunctionDeclaration(boolean isAsync) {
        return parseFunctionDeclaration(isAsync, false);
    }

    private FunctionDeclaration parseFunctionDeclaration(boolean isAsync, boolean allowAnonymous) {
        Token startToken = peek();

        if (isAsync) {
            advance(); // consume 'async'
        }

        advance(); // consume 'function'

        // Check for generator
        boolean isGenerator = match(TokenType.STAR);

        // Parse function name (allow yield, of, let as function names in appropriate contexts)
        Identifier id = null;
        if (check(TokenType.IDENTIFIER) || check(TokenType.OF) || check(TokenType.LET)) {
            Token nameToken = peek();
            advance();
            id = new Identifier(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.lexeme());
        } else if (!allowAnonymous) {
            throw new ExpectedTokenException("function name", peek());
        }

        // Set generator/async context before parsing parameters
        // (parameters can have default values that need correct context)
        boolean savedInGenerator = inGenerator;
        boolean savedInAsyncContext = inAsyncContext;
        boolean savedStrictMode = strictMode;
        boolean savedInClassFieldInitializer = inClassFieldInitializer;
        inGenerator = isGenerator;
        inAsyncContext = isAsync;
        inClassFieldInitializer = false; // Function bodies are never class field initializers

        // Parse parameters
        consume(TokenType.LPAREN, "Expected '(' after function name");
        List<Pattern> params = new ArrayList<>();

        if (!check(TokenType.RPAREN)) {
            do {
                // Check for trailing comma: function foo(a, b,) {}
                if (check(TokenType.RPAREN)) {
                    break;
                }
                // Check for rest parameter: ...param
                if (match(TokenType.DOT_DOT_DOT)) {
                    Token restStart = previous();
                    Pattern argument = parsePatternBase();
                    Token restEnd = previous();
                    SourceLocation restLoc = createLocation(restStart, restEnd);
                    params.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                    // Rest parameter must be last
                    if (match(TokenType.COMMA)) {
                        throw new ParseException("ValidationError", peek(), null, "parameter list", "Rest parameter must be last");
                    }
                    break;
                } else {
                    params.add(parsePattern());
                }
            } while (match(TokenType.COMMA));
        }

        consume(TokenType.RPAREN, "Expected ')' after parameters");

        // Reset strict mode for function body (unless in module mode)
        // Functions can have their own "use strict" directive
        if (!forceModuleMode) {
            strictMode = false;
        }

        // Parse body (context already set above)
        // The block statement will call processDirectives which may set strictMode
        BlockStatement body = parseBlockStatement(true); // Function body

        // Check for duplicate parameters if in strict mode
        // This must be done AFTER parsing the body (which might contain "use strict")
        // but BEFORE restoring the saved strict mode
        validateNoDuplicateParameters(params, startToken);

        // Restore context
        inGenerator = savedInGenerator;
        inAsyncContext = savedInAsyncContext;
        strictMode = savedStrictMode;
        inClassFieldInitializer = savedInClassFieldInitializer;

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new FunctionDeclaration(getStart(startToken), getEnd(endToken), loc, id, false, isGenerator, isAsync, params, body);
    }

    private ClassDeclaration parseClassDeclaration() {
        return parseClassDeclaration(false);
    }

    private ClassDeclaration parseClassDeclaration(boolean allowAnonymous) {
        Token startToken = peek();
        advance(); // consume 'class'

        // Parse class name (but not if it's 'extends' which starts the extends clause)
        Identifier id = null;
        if (check(TokenType.IDENTIFIER) && !peek().lexeme().equals("extends")) {
            Token nameToken = peek();
            advance();
            id = new Identifier(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.lexeme());
        } else if (!allowAnonymous && !(check(TokenType.IDENTIFIER) && peek().lexeme().equals("extends"))) {
            throw new ExpectedTokenException("class name", peek());
        }

        // Check for extends
        Expression superClass = null;
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("extends")) {
            advance(); // consume 'extends'
            superClass = parseConditional(); // Parse the superclass expression (can be any expression except assignment)
        }

        // Parse class body
        ClassBody body = parseClassBody();

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ClassDeclaration(getStart(startToken), getEnd(endToken), loc, id, superClass, body);
    }

    private ClassBody parseClassBody() {
        Token startToken = peek();
        consume(TokenType.LBRACE, "Expected '{' before class body");

        List<Node> bodyElements = new ArrayList<>();

        while (!check(TokenType.RBRACE) && !isAtEnd()) {
            // Skip empty statements (semicolons) in class body
            if (match(TokenType.SEMICOLON)) {
                continue;
            }

            // Track the start of the member (before any modifiers)
            Token memberStart = peek();

            boolean isStatic = false;

            // Check for 'static' keyword (but not if it's a method named "static" or field named "static")
            if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("static")) {
                // Look ahead to see if this is "static()" (method name), "static;" or "static =" (field name),
                // or "static something" (modifier)
                TokenType nextType = current + 1 < tokens.size() ? tokens.get(current + 1).type() : null;
                // static is a modifier unless followed by ( ; = or nothing
                if (nextType != null && nextType != TokenType.LPAREN &&
                    nextType != TokenType.SEMICOLON && nextType != TokenType.ASSIGN) {
                    advance();
                    isStatic = true;

                    // Check for static initialization block: static { ... }
                    if (check(TokenType.LBRACE)) {
                        Token blockStart = memberStart;
                        Token braceStart = peek();
                        advance(); // consume '{'

                        List<Statement> blockBody = new ArrayList<>();
                        while (!check(TokenType.RBRACE) && !isAtEnd()) {
                            blockBody.add(parseStatement());
                        }

                        Token blockEnd = peek();
                        consume(TokenType.RBRACE, "Expected '}' after static block body");

                        SourceLocation blockLoc = createLocation(blockStart, blockEnd);
                        bodyElements.add(new StaticBlock(getStart(blockStart), getEnd(blockEnd), blockLoc, blockBody));
                        continue;
                    }
                }
            }

            // Check for 'async' keyword (but not if it's a method named "async")
            boolean isAsync = false;
            if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
                // Look ahead to see if this is "async()" (method name) or "async something" (modifier)
                if (current + 1 < tokens.size() && tokens.get(current + 1).type() != TokenType.LPAREN) {
                    advance();
                    isAsync = true;
                }
            }

            // Check for getter/setter
            String kind = "method";
            if (check(TokenType.IDENTIFIER) && (peek().lexeme().equals("get") || peek().lexeme().equals("set"))) {
                // Look ahead to see if this is "get()" / "set()" (method names) or "get something" / "set something" (accessor)
                if (current + 1 < tokens.size()) {
                    Token currentToken = peek();
                    Token nextToken = tokens.get(current + 1);
                    TokenType nextType = nextToken.type();

                    // If next token is LPAREN, this is a method named "get" or "set"
                    // If next token is ASSIGN or SEMICOLON, this is a field named "get" or "set"
                    // Otherwise, check for ASI and potentially treat as accessor
                    if (nextType != TokenType.LPAREN && nextType != TokenType.ASSIGN && nextType != TokenType.SEMICOLON) {
                        // Check for ASI: if there's a line break between "get"/"set" and the next token,
                        // it's a field, not an accessor
                        boolean hasLineBreak = nextToken.line() > currentToken.line();

                        if (!hasLineBreak) {
                            kind = peek().lexeme(); // "get" or "set"
                            advance(); // consume get/set keyword
                        }
                    }
                }
            }

            // Check for generator method (*)
            boolean isGenerator = false;
            if (match(TokenType.STAR)) {
                isGenerator = true;
            }

            // Check for private field/method
            boolean isPrivate = false;
            Token hashToken = null;

            if (match(TokenType.HASH)) {
                isPrivate = true;
                hashToken = previous();
            }

            // Parse property key
            Token keyToken = peek();
            Expression key;
            boolean computed = false;

            if (isPrivate) {
                // Private identifier - allow keywords as private names
                if (!check(TokenType.IDENTIFIER) && !isKeyword(keyToken)) {
                    throw new ExpectedTokenException("identifier after '#'", peek());
                }
                advance();
                // PrivateIdentifier starts at #, but method/property start is memberStart (may include static)
                key = new PrivateIdentifier(getStart(hashToken), getEnd(keyToken), createLocation(hashToken, keyToken), keyToken.lexeme());
            } else if (match(TokenType.LBRACKET)) {
                // Computed property name - allow 'in' operator inside
                computed = true;
                boolean savedAllowIn = allowIn;
                allowIn = true;
                key = parseExpression();
                allowIn = savedAllowIn;
                consume(TokenType.RBRACKET, "Expected ']' after computed property name");
            } else if (check(TokenType.STRING) || check(TokenType.NUMBER)) {
                // Literal property name (string or number)
                advance();
                SourceLocation keyLoc = createLocation(keyToken, keyToken);
                String keyLexeme = keyToken.lexeme();

                // Check if this is a BigInt literal (ends with 'n')
                if (keyLexeme.endsWith("n")) {
                    // BigInt literal: value is null, bigint field has the numeric part
                    String bigintValue = keyLexeme.substring(0, keyLexeme.length() - 1).replace("_", "");

                    // Convert hex/octal/binary to decimal for the bigint field
                    if (bigintValue.startsWith("0x") || bigintValue.startsWith("0X")) {
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 16);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    } else if (bigintValue.startsWith("0o") || bigintValue.startsWith("0O")) {
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 8);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    } else if (bigintValue.startsWith("0b") || bigintValue.startsWith("0B")) {
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 2);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    }

                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyLoc, null, keyLexeme, null, bigintValue);
                } else {
                    Object literalValue = keyToken.literal();
                    if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                        literalValue = null;
                    }
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyLoc, literalValue, keyLexeme);
                }
            } else if (check(TokenType.DOT) && current + 1 < tokens.size() && tokens.get(current + 1).type() == TokenType.NUMBER) {
                // Handle .1 as a numeric literal (0.1)
                Token dotToken = peek();
                advance(); // consume DOT
                Token numToken = peek();
                advance(); // consume NUMBER
                String lexeme = "." + numToken.lexeme();
                double value = Double.parseDouble(lexeme);
                SourceLocation keyLoc = createLocation(dotToken, numToken);
                key = new Literal(getStart(dotToken), getEnd(numToken), keyLoc, value, lexeme);
            } else if (check(TokenType.IDENTIFIER) || check(TokenType.GET) || check(TokenType.SET) ||
                       check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL) ||
                       isKeyword(peek())) {
                // Regular identifier, get/set, keyword, or boolean/null literal as property name
                advance();
                SourceLocation keyLoc = createLocation(keyToken, keyToken);
                key = new Identifier(getStart(keyToken), getEnd(keyToken), keyLoc, keyToken.lexeme());
            } else {
                throw new ExpectedTokenException("property name in class body", peek());
            }

            // Check if it's a method or a field
            if (match(TokenType.LPAREN)) {
                // It's a method - parse as function expression
                Token fnStart = previous(); // The opening paren
                List<Pattern> params = new ArrayList<>();

                while (!check(TokenType.RPAREN)) {
                    if (match(TokenType.DOT_DOT_DOT)) {
                        Token restStart = previous();
                        Pattern argument = parsePatternBase();
                        Token restEnd = previous();
                        SourceLocation restLoc = createLocation(restStart, restEnd);
                        params.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                        if (match(TokenType.COMMA)) {
                            throw new ParseException("ValidationError", peek(), null, "parameter list", "Rest parameter must be last");
                        }
                        break;
                    } else {
                        params.add(parsePattern());
                    }

                    if (!match(TokenType.COMMA)) {
                        break;
                    }
                }

                consume(TokenType.RPAREN, "Expected ')' after parameters");

                // Parse body with proper generator/async context
                boolean savedInGenerator = inGenerator;
                boolean savedInAsyncContext = inAsyncContext;
                boolean savedStrictMode = strictMode;
                inGenerator = isGenerator;
                inAsyncContext = isAsync;

                // Reset strict mode for method body (unless in module mode)
                if (!forceModuleMode) {
                    strictMode = false;
                }

                BlockStatement body = parseBlockStatement(true); // Method body

                // Check for duplicate parameters if in strict mode
                validateNoDuplicateParameters(params, memberStart);

                inGenerator = savedInGenerator;
                inAsyncContext = savedInAsyncContext;
                strictMode = savedStrictMode;

                Token methodEnd = previous();
                SourceLocation methodLoc = createLocation(memberStart, methodEnd);
                SourceLocation fnLoc = createLocation(fnStart, methodEnd);
                FunctionExpression fnExpr = new FunctionExpression(
                    getStart(fnStart),
                    getEnd(methodEnd),
                    fnLoc,
                    null,  // No id for method
                    false, // expression (methods are not expression context)
                    isGenerator, // generator
                    isAsync, // async
                    params,
                    body
                );

                // Determine method kind (or use the one already set for get/set)
                // Only non-static, non-computed methods can be constructors
                if (kind.equals("method") && !isStatic && !computed) {
                    if (key instanceof Identifier id && id.name().equals("constructor")) {
                        kind = "constructor";
                    } else if (key instanceof Literal lit && "constructor".equals(lit.value())) {
                        kind = "constructor";
                    }
                }

                MethodDefinition method = new MethodDefinition(
                    getStart(memberStart),
                    getEnd(methodEnd),
                    methodLoc,
                    key,
                    fnExpr,
                    kind,
                    computed,
                    isStatic
                );
                bodyElements.add(method);

                // Consume optional semicolon after method
                match(TokenType.SEMICOLON);
            } else {
                // It's a property field
                Expression value = null;
                if (match(TokenType.ASSIGN)) {
                    // Class field initializers are not in async context
                    boolean oldInClassFieldInitializer = inClassFieldInitializer;
                    boolean oldInAsyncContext = inAsyncContext;
                    inClassFieldInitializer = true;
                    inAsyncContext = false;  // Reset async context for class field initializers
                    value = parseAssignment();
                    inClassFieldInitializer = oldInClassFieldInitializer;
                    inAsyncContext = oldInAsyncContext;
                }

                // Consume optional semicolon
                match(TokenType.SEMICOLON);

                Token propertyEnd = previous();
                SourceLocation propertyLoc = createLocation(memberStart, propertyEnd);
                PropertyDefinition property = new PropertyDefinition(
                    getStart(memberStart),
                    getEnd(propertyEnd),
                    propertyLoc,
                    key,
                    value,
                    computed,
                    isStatic
                );
                bodyElements.add(property);
            }
        }

        consume(TokenType.RBRACE, "Expected '}' after class body");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ClassBody(getStart(startToken), getEnd(endToken), loc, bodyElements);
    }

    private ImportDeclaration parseImportDeclaration() {
        Token startToken = peek();
        advance(); // consume 'import'

        List<Node> specifiers = new ArrayList<>();

        // Check for import 'module' (side-effect import)
        if (check(TokenType.STRING)) {
            Token sourceToken = advance();
            Literal source = new Literal(getStart(sourceToken), getEnd(sourceToken), createLocation(sourceToken, sourceToken), sourceToken.literal(), sourceToken.lexeme());
            List<ImportAttribute> attributes = parseImportAttributes();
            consumeSemicolon("Expected ';' after import");
            Token endToken = previous();
            return new ImportDeclaration(getStart(startToken), getEnd(endToken), createLocation(startToken, endToken), specifiers, source, attributes);
        }

        // Parse import specifiers
        // import defaultName from 'module'
        // import * as name from 'module'
        // import { name1, name2 } from 'module'
        // import defaultName, { name1 } from 'module'
        // import defaultName, * as name from 'module'

        // Check for default import
        // Note: 'from' can be used as an identifier: import from from 'module'
        // We distinguish by checking if the next token after the identifier is 'from' (keyword) or comma
        if (check(TokenType.IDENTIFIER)) {
            Token localToken = peek();
            // Check if this is actually a default import binding or the 'from' keyword
            // If it's 'from' and the next token is STRING, then this 'from' is the keyword, not a binding
            if (localToken.lexeme().equals("from") && checkAhead(1, TokenType.STRING)) {
                // This is the 'from' keyword, not a binding - don't consume it
            } else {
                // This is a binding name (could be 'from' if followed by 'from' keyword)
                advance();
                Identifier local = new Identifier(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), localToken.lexeme());
                specifiers.add(new ImportDefaultSpecifier(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), local));

                // Check for comma (means there are more specifiers)
                if (match(TokenType.COMMA)) {
                    // Continue parsing
                }
            }
        }

        // Check for namespace import: * as name
        if (match(TokenType.STAR)) {
            Token starToken = previous();
            consume(TokenType.IDENTIFIER, "Expected 'as' after '*'");
            Token asToken = previous();
            if (!asToken.lexeme().equals("as")) {
                throw new ExpectedTokenException("'as' after '*'", peek());
            }
            Token localToken = peek();
            if (!check(TokenType.IDENTIFIER)) {
                throw new ExpectedTokenException("identifier after 'as'", peek());
            }
            advance();
            Identifier local = new Identifier(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), localToken.lexeme());
            specifiers.add(new ImportNamespaceSpecifier(getStart(starToken), getEnd(localToken), createLocation(starToken, localToken), local));
        } else if (match(TokenType.LBRACE)) {
            // Named imports: { name1, name2 as alias }
            // Handle empty specifiers: { }
            if (!check(TokenType.RBRACE)) {
                do {
                    Token importedToken = peek();
                    Node imported;
                    Identifier local;
                    boolean isStringImport = check(TokenType.STRING);

                    if (isStringImport) {
                        advance();
                        imported = new Literal(getStart(importedToken), getEnd(importedToken), createLocation(importedToken, importedToken), importedToken.literal(), importedToken.lexeme());
                        // String imports MUST have 'as' with local binding
                        if (!check(TokenType.IDENTIFIER) || !peek().lexeme().equals("as")) {
                            throw new ExpectedTokenException("'as' after string import specifier", peek());
                        }
                        advance(); // consume 'as'
                        Token localToken = peek();
                        if (!check(TokenType.IDENTIFIER) && !isKeyword(localToken)) {
                            throw new ExpectedTokenException("identifier after 'as'", peek());
                        }
                        advance();
                        local = new Identifier(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), localToken.lexeme());
                    } else if (check(TokenType.IDENTIFIER) || isKeyword(importedToken)) {
                        advance();
                        Identifier importedId = new Identifier(getStart(importedToken), getEnd(importedToken), createLocation(importedToken, importedToken), importedToken.lexeme());
                        imported = importedId;
                        local = importedId;
                        // Check for 'as'
                        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("as")) {
                            advance(); // consume 'as'
                            Token localToken = peek();
                            if (!check(TokenType.IDENTIFIER) && !isKeyword(localToken)) {
                                throw new ExpectedTokenException("identifier after 'as'", peek());
                            }
                            advance();
                            local = new Identifier(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), localToken.lexeme());
                        }
                    } else {
                        throw new ExpectedTokenException("identifier or string in import specifier", peek());
                    }

                    specifiers.add(new ImportSpecifier(getStart(importedToken), getEnd(previous()), createLocation(importedToken, previous()), imported, local));

                    // Handle trailing comma: { a, }
                    if (match(TokenType.COMMA)) {
                        if (check(TokenType.RBRACE)) {
                            break;
                        }
                    } else {
                        break;
                    }
                } while (true);
            }

            consume(TokenType.RBRACE, "Expected '}' after import specifiers");
        }

        // Parse 'from' clause
        Token fromToken = peek();
        if (!check(TokenType.IDENTIFIER) || !fromToken.lexeme().equals("from")) {
            throw new ExpectedTokenException("'from' after import specifiers", peek());
        }
        advance(); // consume 'from'

        // Parse module source
        Token sourceToken = peek();
        if (!check(TokenType.STRING)) {
            throw new ExpectedTokenException("string literal after 'from'", peek());
        }
        advance();
        Literal source = new Literal(getStart(sourceToken), getEnd(sourceToken), createLocation(sourceToken, sourceToken), sourceToken.literal(), sourceToken.lexeme());

        // Parse import attributes: with { type: 'json' }
        List<ImportAttribute> attributes = parseImportAttributes();

        consumeSemicolon("Expected ';' after import");
        Token endToken = previous();
        return new ImportDeclaration(getStart(startToken), getEnd(endToken), createLocation(startToken, endToken), specifiers, source, attributes);
    }

    private List<ImportAttribute> parseImportAttributes() {
        List<ImportAttribute> attributes = new ArrayList<>();

        // Check for 'with' keyword (can be IDENTIFIER or WITH token type)
        if ((check(TokenType.IDENTIFIER) && peek().lexeme().equals("with")) || check(TokenType.WITH)) {
            advance(); // consume 'with'
            consume(TokenType.LBRACE, "Expected '{' after 'with'");

            while (!check(TokenType.RBRACE) && !isAtEnd()) {
                Token keyToken = peek();
                Node key;

                if (check(TokenType.STRING)) {
                    advance();
                    key = new Literal(getStart(keyToken), getEnd(keyToken), createLocation(keyToken, keyToken), keyToken.literal(), keyToken.lexeme());
                } else if (check(TokenType.IDENTIFIER) || isKeyword(keyToken)) {
                    advance();
                    key = new Identifier(getStart(keyToken), getEnd(keyToken), createLocation(keyToken, keyToken), keyToken.lexeme());
                } else {
                    throw new ExpectedTokenException("identifier or string in import attribute", peek());
                }

                consume(TokenType.COLON, "Expected ':' after import attribute key");

                Token valueToken = peek();
                if (!check(TokenType.STRING)) {
                    throw new ExpectedTokenException("string value in import attribute", peek());
                }
                advance();
                Literal value = new Literal(getStart(valueToken), getEnd(valueToken), createLocation(valueToken, valueToken), valueToken.literal(), valueToken.lexeme());

                Token attrEnd = previous();
                attributes.add(new ImportAttribute(getStart(keyToken), getEnd(attrEnd), createLocation(keyToken, attrEnd), key, value));

                if (!match(TokenType.COMMA)) {
                    break;
                }
            }

            consume(TokenType.RBRACE, "Expected '}' after import attributes");
        }

        return attributes;
    }

    private Statement parseExportDeclaration() {
        Token startToken = peek();
        advance(); // consume 'export'

        // export default ...
        if (match(TokenType.DEFAULT)) {

            // Parse the default export value
            Node declaration;
            if (check(TokenType.FUNCTION)) {
                // Both named and anonymous export default functions are FunctionDeclarations
                // Anonymous just has id: null
                declaration = parseFunctionDeclaration(false, true);
            } else if (check(TokenType.CLASS)) {
                // Both named and anonymous export default classes are ClassDeclarations
                // Anonymous just has id: null
                declaration = parseClassDeclaration(true);
            } else if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
                // Check for async function declaration
                int savedCurrent = current;
                advance(); // consume 'async'
                boolean isFunction = check(TokenType.FUNCTION);
                current = savedCurrent; // restore position

                if (isFunction) {
                    // Both named and anonymous async functions are FunctionDeclarations
                    declaration = parseFunctionDeclaration(true, true);
                } else {
                    declaration = parseAssignment();
                    consumeSemicolon("Expected ';' after export default");
                }
            } else {
                // Expression
                declaration = parseAssignment();
                consumeSemicolon("Expected ';' after export default");
            }

            Token endToken = previous();
            return new ExportDefaultDeclaration(getStart(startToken), getEnd(endToken), createLocation(startToken, endToken), declaration);
        }

        // export * from 'module' or export * as name from 'module'
        if (match(TokenType.STAR)) {
            Node exported = null;

            // Check for 'as'
            if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("as")) {
                advance(); // consume 'as'
                Token nameToken = peek();
                // Allow keywords as identifiers or strings after 'as'
                if (check(TokenType.STRING)) {
                    advance();
                    exported = new Literal(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.literal(), nameToken.lexeme());
                } else if (check(TokenType.IDENTIFIER) || isKeyword(nameToken) ||
                           check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL)) {
                    advance();
                    exported = new Identifier(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.lexeme());
                } else {
                    throw new ExpectedTokenException("identifier or string after 'as'", peek());
                }
            }

            // Parse 'from'
            Token fromToken = peek();
            if (!check(TokenType.IDENTIFIER) || !fromToken.lexeme().equals("from")) {
                throw new ExpectedTokenException("'from' after export *", peek());
            }
            advance(); // consume 'from'

            // Parse source
            Token sourceToken = peek();
            if (!check(TokenType.STRING)) {
                throw new ExpectedTokenException("string literal after 'from'", peek());
            }
            advance();
            Literal source = new Literal(getStart(sourceToken), getEnd(sourceToken), createLocation(sourceToken, sourceToken), sourceToken.literal(), sourceToken.lexeme());

            List<ImportAttribute> attributes = parseImportAttributes();
            consumeSemicolon("Expected ';' after export");
            Token endToken = previous();
            return new ExportAllDeclaration(getStart(startToken), getEnd(endToken), createLocation(startToken, endToken), source, exported, attributes);
        }

        // export { name1, name2 } or export { name1 as alias } from 'module'
        if (match(TokenType.LBRACE)) {
            List<Node> specifiers = new ArrayList<>();

            // Handle empty specifiers: { }
            if (!check(TokenType.RBRACE)) {
                do {
                    Token localToken = peek();
                    Node local;
                    if (check(TokenType.STRING)) {
                        advance();
                        local = new Literal(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), localToken.literal(), localToken.lexeme());
                    } else if (check(TokenType.IDENTIFIER) || isKeyword(localToken)) {
                        advance();
                        local = new Identifier(getStart(localToken), getEnd(localToken), createLocation(localToken, localToken), localToken.lexeme());
                    } else {
                        throw new ExpectedTokenException("identifier or string in export specifier", peek());
                    }

                    Node exported = local;
                    // Check for 'as'
                    if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("as")) {
                        advance(); // consume 'as'
                        Token exportedToken = peek();
                        if (check(TokenType.STRING)) {
                            advance();
                            exported = new Literal(getStart(exportedToken), getEnd(exportedToken), createLocation(exportedToken, exportedToken), exportedToken.literal(), exportedToken.lexeme());
                        } else if (check(TokenType.IDENTIFIER) || isKeyword(exportedToken) ||
                                   check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL)) {
                            advance();
                            exported = new Identifier(getStart(exportedToken), getEnd(exportedToken), createLocation(exportedToken, exportedToken), exportedToken.lexeme());
                        } else {
                            throw new ExpectedTokenException("identifier or string after 'as'", peek());
                        }
                    }

                    specifiers.add(new ExportSpecifier(getStart(localToken), getEnd(previous()), createLocation(localToken, previous()), local, exported));

                    // Handle trailing comma: { a, }
                    if (match(TokenType.COMMA)) {
                        if (check(TokenType.RBRACE)) {
                            break;
                        }
                    } else {
                        break;
                    }
                } while (true);
            }

            consume(TokenType.RBRACE, "Expected '}' after export specifiers");

            // Check for 'from' (re-export)
            Literal source = null;
            List<ImportAttribute> attributes = new ArrayList<>();
            if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("from")) {
                advance(); // consume 'from'
                Token sourceToken = peek();
                if (!check(TokenType.STRING)) {
                    throw new ExpectedTokenException("string literal after 'from'", peek());
                }
                advance();
                source = new Literal(getStart(sourceToken), getEnd(sourceToken), createLocation(sourceToken, sourceToken), sourceToken.literal(), sourceToken.lexeme());
                attributes = parseImportAttributes();
            }

            consumeSemicolon("Expected ';' after export");
            Token endToken = previous();
            return new ExportNamedDeclaration(getStart(startToken), getEnd(endToken), createLocation(startToken, endToken), null, specifiers, source, attributes);
        }

        // export var/let/const/function/class declaration
        Statement declaration = null;
        if (check(TokenType.VAR) || check(TokenType.LET) || check(TokenType.CONST)) {
            declaration = parseVariableDeclaration();
        } else if (check(TokenType.FUNCTION)) {
            declaration = parseFunctionDeclaration(false);
        } else if (check(TokenType.CLASS)) {
            declaration = parseClassDeclaration();
        } else if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
            // export async function
            // Don't consume 'async' - let parseFunctionDeclaration handle it
            declaration = parseFunctionDeclaration(true); // pass true for async
        } else {
            throw new UnexpectedTokenException(peek(), "export keyword", "export statement");
        }

        Token endToken = previous();
        return new ExportNamedDeclaration(getStart(startToken), getEnd(endToken), createLocation(startToken, endToken), declaration, new ArrayList<>(), null, new ArrayList<>());
    }

    private BlockStatement parseBlockStatement() {
        return parseBlockStatement(false);
    }

    private BlockStatement parseBlockStatement(boolean isFunctionBody) {
        Token startToken = peek();
        consume(TokenType.LBRACE, "Expected '{'");

        // Temporarily allow 'in' inside blocks
        boolean oldAllowIn = allowIn;
        allowIn = true;

        List<Statement> statements = new ArrayList<>();
        while (!check(TokenType.RBRACE) && !isAtEnd()) {
            statements.add(parseStatement());
        }

        // Process directive prologue only for function bodies
        if (isFunctionBody) {
            statements = processDirectives(statements);
        }

        consume(TokenType.RBRACE, "Expected '}'");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);

        // Restore allowIn
        allowIn = oldAllowIn;

        return new BlockStatement(getStart(startToken), getEnd(endToken), loc, statements);
    }

    // Process directive prologue: add directive property to string literal expression statements at the start
    private List<Statement> processDirectives(List<Statement> statements) {
        List<Statement> processed = new ArrayList<>();
        boolean inPrologue = true;

        for (Statement stmt : statements) {
            if (inPrologue && stmt instanceof ExpressionStatement exprStmt) {
                if (exprStmt.expression() instanceof Literal lit && lit.value() instanceof String) {
                    // This is a directive
                    // Directive value is the raw string content (without quotes) to preserve escape sequences
                    String directiveValue = lit.raw().substring(1, lit.raw().length() - 1);

                    // Detect "use strict" directive
                    if (directiveValue.equals("use strict")) {
                        strictMode = true;
                    }

                    processed.add(new ExpressionStatement(
                        exprStmt.start(),
                        exprStmt.end(),
                        exprStmt.loc(),
                        exprStmt.expression(),
                        directiveValue
                    ));
                    continue;
                } else {
                    // Non-string-literal expression ends the prologue
                    inPrologue = false;
                }
            } else {
                // Non-expression statement ends the prologue
                inPrologue = false;
            }
            processed.add(stmt);
        }

        return processed;
    }

    private VariableDeclaration parseVariableDeclaration() {
        Token startToken = peek();
        Token kindToken = advance(); // var, let, or const
        String kind = kindToken.lexeme();

        List<VariableDeclarator> declarators = new ArrayList<>();

        do {
            Token patternStart = peek();
            // Don't parse default values at top level - those are initializers, not defaults
            Pattern pattern = parsePatternBase();

            Expression init = null;
            if (match(TokenType.ASSIGN)) {
                init = parseAssignment();
            }

            Token declaratorEnd = previous();

            int declaratorStart = getStart(patternStart);
            int declaratorEndPos;
            SourceLocation declaratorLoc;

            // Use the end of the last token (which includes closing parens)
            declaratorEndPos = getEnd(declaratorEnd);
            declaratorLoc = createLocation(patternStart, declaratorEnd);

            declarators.add(new VariableDeclarator(declaratorStart, declaratorEndPos, declaratorLoc, pattern, init));

        } while (match(TokenType.COMMA));

        consumeSemicolon("Expected ';' after variable declaration");

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new VariableDeclaration(getStart(startToken), getEnd(endToken), loc, declarators, kind);
    }

    private Pattern parsePattern() {
        return parsePatternWithDefault();
    }

    private Pattern parsePatternWithDefault() {
        Token startToken = peek();
        Pattern pattern = parsePatternBase();

        // Check for default value: pattern = defaultValue
        if (match(TokenType.ASSIGN)) {
            Expression defaultValue = parseAssignment();
            Token endToken = previous();
            SourceLocation loc = createLocation(startToken, endToken);
            return new AssignmentPattern(getStart(startToken), getEnd(endToken), loc, pattern, defaultValue);
        }

        return pattern;
    }

    private Pattern parsePatternBase() {
        Token startToken = peek();

        if (match(TokenType.LBRACE)) {
            // Object pattern: { x, y, z }
            return parseObjectPattern(startToken);
        } else if (match(TokenType.LBRACKET)) {
            // Array pattern: [ a, b, c ]
            return parseArrayPattern(startToken);
        } else if (check(TokenType.IDENTIFIER) || isKeyword(peek())) {
            // Simple identifier pattern (keywords allowed as identifiers in patterns)
            Token idToken = advance();
            return new Identifier(getStart(idToken), getEnd(idToken), createLocation(idToken, idToken), idToken.lexeme());
        } else {
            throw new ExpectedTokenException("identifier in variable declaration", peek());
        }
    }

    private ObjectPattern parseObjectPattern(Token startToken) {
        List<Node> properties = new ArrayList<>();

        while (!check(TokenType.RBRACE) && !isAtEnd()) {
            // Check for rest element in object pattern: {...rest}
            if (match(TokenType.DOT_DOT_DOT)) {
                Token restStart = previous();
                Pattern argument = parsePatternBase();
                Token restEnd = previous();
                SourceLocation restLoc = createLocation(restStart, restEnd);
                properties.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                // Rest element must be last
                if (match(TokenType.COMMA)) {
                    throw new ParseException("ValidationError", peek(), null, "object pattern", "Rest element must be last in object pattern");
                }
                break;
            }

            Token propStart = peek();

            // Parse the key (identifier or computed)
            Node key;
            boolean computed = false;
            boolean shorthand = false;

            if (match(TokenType.LBRACKET)) {
                // Computed property: [expr]
                computed = true;
                key = parseAssignment();
                consume(TokenType.RBRACKET, "Expected ']' after computed property");
            } else if (check(TokenType.STRING) || check(TokenType.NUMBER)) {
                // Literal key (string or numeric)
                Token keyToken = advance();
                SourceLocation keyLoc = createLocation(keyToken, keyToken);
                String keyLexeme = keyToken.lexeme();

                // Check if this is a BigInt literal (ends with 'n')
                if (keyLexeme.endsWith("n")) {
                    // BigInt literal: value is null, bigint field has the numeric part
                    String bigintValue = keyLexeme.substring(0, keyLexeme.length() - 1).replace("_", "");

                    // Convert hex/octal/binary to decimal for the bigint field
                    if (bigintValue.startsWith("0x") || bigintValue.startsWith("0X")) {
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 16);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    } else if (bigintValue.startsWith("0o") || bigintValue.startsWith("0O")) {
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 8);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    } else if (bigintValue.startsWith("0b") || bigintValue.startsWith("0B")) {
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 2);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    }

                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyLoc, null, keyLexeme, null, bigintValue);
                } else {
                    Object literalValue = keyToken.literal();
                    if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                        literalValue = null;
                    }
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyLoc, literalValue, keyLexeme);
                }
            } else {
                // Allow identifiers, keywords, and boolean/null literals as property names
                if (!check(TokenType.IDENTIFIER) && !isKeyword(peek()) &&
                    !check(TokenType.TRUE) && !check(TokenType.FALSE) && !check(TokenType.NULL)) {
                    throw new ExpectedTokenException("property name", peek());
                }
                Token keyToken = advance();
                key = new Identifier(getStart(keyToken), getEnd(keyToken), createLocation(keyToken, keyToken), keyToken.lexeme());
            }

            // Parse the value (pattern)
            Pattern value;
            if (match(TokenType.COLON)) {
                // Full form: { x: y }
                value = parsePattern();
            } else {
                // Shorthand form: { x } or { x = defaultValue }
                shorthand = true;
                if (key instanceof Identifier id) {
                    value = id;
                    // Check for default value in shorthand: { x = 1 }
                    if (match(TokenType.ASSIGN)) {
                        Token assignStart = previous();
                        Expression defaultValue = parseAssignment();
                        Token assignEnd = previous();
                        SourceLocation assignLoc = createLocation(propStart, assignEnd);
                        value = new AssignmentPattern(getStart(propStart), getEnd(assignEnd), assignLoc, id, defaultValue);
                    }
                } else {
                    throw new ParseException("ValidationError", peek(), null, "object pattern property", "Shorthand property must have identifier key");
                }
            }

            Token propEnd = previous();
            SourceLocation propLoc = createLocation(propStart, propEnd);
            properties.add(new Property(
                getStart(propStart), getEnd(propEnd), propLoc,
                false, shorthand, computed, key, value, "init"
            ));

            if (!match(TokenType.COMMA)) {
                break;
            }
        }

        consume(TokenType.RBRACE, "Expected '}' after object pattern");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ObjectPattern(getStart(startToken), getEnd(endToken), loc, properties);
    }

    private ArrayPattern parseArrayPattern(Token startToken) {
        List<Pattern> elements = new ArrayList<>();

        while (!check(TokenType.RBRACKET) && !isAtEnd()) {
            if (match(TokenType.DOT_DOT_DOT)) {
                // Rest element: ...rest (no default value allowed)
                Token restStart = previous();
                Pattern argument = parsePatternBase();
                Token restEnd = previous();
                SourceLocation restLoc = createLocation(restStart, restEnd);
                elements.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                // Rest element must be last
                if (match(TokenType.COMMA)) {
                    throw new ParseException("ValidationError", peek(), null, "array pattern", "Rest element must be last in array pattern");
                }
                break;
            } else if (check(TokenType.COMMA)) {
                // Hole in array pattern
                elements.add(null);
            } else {
                // Regular pattern element
                elements.add(parsePattern());
            }

            if (!match(TokenType.COMMA)) {
                break;
            }
        }

        consume(TokenType.RBRACKET, "Expected ']' after array pattern");
        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new ArrayPattern(getStart(startToken), getEnd(endToken), loc, elements);
    }

    // Expression parsing with precedence climbing
    // expression -> sequence
    // sequence -> assignment ( "," assignment )*
    // assignment -> conditional ( "=" assignment )?
    private Expression parseExpression() {
        return parseSequence();
    }

    private Expression parseSequence() {
        Token startToken = peek();
        Expression expr = parseAssignment();

        if (!check(TokenType.COMMA)) {
            return expr;
        }

        List<Expression> expressions = new ArrayList<>();
        expressions.add(expr);

        while (match(TokenType.COMMA)) {
            expressions.add(parseAssignment());
        }

        Token endToken = previous();
        SourceLocation loc = createLocation(startToken, endToken);
        return new SequenceExpression(getStart(startToken), getEnd(endToken), loc, expressions);
    }

    private Expression parseAssignment() {
        Token startToken = peek();

        // Yield expressions - only in generator context
        // Yield has assignment-level precedence, so check it here
        // Note: we don't check for COLON here because parseAssignment is only called in expression contexts,
        // so "yield:" as a label will be handled at the statement level, not here
        if (inGenerator && check(TokenType.IDENTIFIER) && peek().lexeme().equals("yield") &&
            !checkAhead(1, TokenType.ASSIGN) && !checkAhead(1, TokenType.PLUS_ASSIGN) &&
            !checkAhead(1, TokenType.MINUS_ASSIGN) && !checkAhead(1, TokenType.STAR_ASSIGN) &&
            !checkAhead(1, TokenType.SLASH_ASSIGN) && !checkAhead(1, TokenType.PERCENT_ASSIGN)) {
            advance(); // consume 'yield'
            Token yieldToken = previous();
            boolean delegate = false;
            Expression argument = null;

            // Check for yield* (delegate)
            if (match(TokenType.STAR)) {
                delegate = true;
            }

            // Check if there's an argument (not standalone yield)
            // Per ECMAScript spec: [no LineTerminator here] before the argument (but only for non-delegate yield)
            // For yield*, the * "counts as starting the RHS", so a newline after * is allowed
            // Don't consume the argument if it's a semicolon, closing delimiter, etc., or on a new line (for non-delegate)
            boolean hasLineTerminator = !delegate && !isAtEnd() && peek().line() > yieldToken.line();
            if (!hasLineTerminator &&
                !check(TokenType.SEMICOLON) && !check(TokenType.RBRACE) && !check(TokenType.EOF) &&
                !check(TokenType.RPAREN) && !check(TokenType.COMMA) && !check(TokenType.RBRACKET) &&
                !check(TokenType.TEMPLATE_MIDDLE) && !check(TokenType.TEMPLATE_TAIL) &&
                !check(TokenType.COLON)) {
                argument = parseAssignment();
            }

            Token endToken = previous();
            SourceLocation loc = createLocation(yieldToken, endToken);
            return new YieldExpression(getStart(yieldToken), getEnd(endToken), loc, delegate, argument);
        }

        // Check for async arrow function: async identifier => expr or async (params) => expr
        boolean isAsync = false;
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
            // Look ahead to see if this is async arrow function
            if (current + 1 < tokens.size()) {
                Token asyncToken = peek();
                Token nextToken = tokens.get(current + 1);

                // Check for line terminator between async and next token
                // Grammar: async [no LineTerminator here] AsyncArrowBindingIdentifier
                if (asyncToken.line() != nextToken.line()) {
                    // Line terminator present - not an async arrow function
                    // Fall through to parse 'async' as identifier
                } else if (nextToken.type() == TokenType.IDENTIFIER || nextToken.type() == TokenType.LPAREN) {
                    // Could be async arrow function
                    if (nextToken.type() == TokenType.IDENTIFIER) {
                        // async id => ...
                        if (current + 2 < tokens.size() && tokens.get(current + 2).type() == TokenType.ARROW) {
                            startToken = peek(); // Keep async as start
                            advance(); // consume 'async'
                            isAsync = true;
                        }
                    } else {
                        // async (...) => ...
                        // Need to check if there's actually an => after the (...)
                        int savedCurrent = current;
                        advance(); // consume 'async'
                        advance(); // consume '('

                        // Use lookahead to check if this is an arrow function
                        boolean isArrow = isArrowFunctionParameters();

                        // Restore position
                        current = savedCurrent;

                        if (isArrow) {
                            startToken = peek(); // Keep async as start
                            advance(); // consume 'async'
                            isAsync = true;
                        }
                    }
                }
            }
        }

        // Check for arrow function: identifier => expr (allow of, let as parameter names)
        if ((check(TokenType.IDENTIFIER) || check(TokenType.OF) || check(TokenType.LET)) && !isAsync) {
            Token idToken = peek();
            if (current + 1 < tokens.size() && tokens.get(current + 1).type() == TokenType.ARROW) {
                advance(); // consume identifier/yield/of/let
                List<Pattern> params = new ArrayList<>();
                params.add(new Identifier(getStart(idToken), getEnd(idToken), createLocation(idToken, idToken), idToken.lexeme()));

                consume(TokenType.ARROW, "Expected '=>'");
                return parseArrowFunctionBody(startToken, params, isAsync);
            }
        } else if ((check(TokenType.IDENTIFIER) || check(TokenType.OF) || check(TokenType.LET)) && isAsync) {
            // async identifier => expr
            Token idToken = peek();
            advance(); // consume identifier/yield/of/let
            List<Pattern> params = new ArrayList<>();
            params.add(new Identifier(getStart(idToken), getEnd(idToken), createLocation(idToken, idToken), idToken.lexeme()));

            consume(TokenType.ARROW, "Expected '=>'");
            return parseArrowFunctionBody(startToken, params, isAsync);
        }

        // Check for arrow function: (params) => expr
        if (check(TokenType.LPAREN)) {
            int savedCurrent = current;
            Token lparenToken = peek();
            advance(); // consume (

            // Use lookahead to check if this is an arrow function
            // We need to scan ahead to find the matching ) and check for =>
            boolean isArrow = isArrowFunctionParameters();

            if (isArrow) {
                // Parse parameters as patterns (supports destructuring)
                List<Pattern> params = new ArrayList<>();

                if (!check(TokenType.RPAREN)) {
                    do {
                        // Check for trailing comma: (a, b,) =>
                        if (check(TokenType.RPAREN)) {
                            break;
                        }
                        // Check for rest parameter: ...param
                        if (match(TokenType.DOT_DOT_DOT)) {
                            Token restStart = previous();
                            Pattern argument = parsePatternBase();
                            Token restEnd = previous();
                            SourceLocation restLoc = createLocation(restStart, restEnd);
                            params.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                            // Rest parameter must be last
                            if (match(TokenType.COMMA)) {
                                throw new ParseException("ValidationError", peek(), null, "parameter list", "Rest parameter must be last");
                            }
                            break;
                        } else {
                            // Parse as pattern to support destructuring
                            params.add(parsePattern());
                        }
                    } while (match(TokenType.COMMA));
                }

                consume(TokenType.RPAREN, "Expected ')' after parameters");
                consume(TokenType.ARROW, "Expected '=>'");
                return parseArrowFunctionBody(startToken, params, isAsync);
            } else {
                // Not an arrow function, backtrack and parse as expression
                current = savedCurrent;
            }
        }

        Expression left = parseConditional();

        if (match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                  TokenType.STAR_ASSIGN, TokenType.STAR_STAR_ASSIGN, TokenType.SLASH_ASSIGN, TokenType.PERCENT_ASSIGN,
                  TokenType.LEFT_SHIFT_ASSIGN, TokenType.RIGHT_SHIFT_ASSIGN, TokenType.UNSIGNED_RIGHT_SHIFT_ASSIGN,
                  TokenType.BIT_AND_ASSIGN, TokenType.BIT_OR_ASSIGN, TokenType.BIT_XOR_ASSIGN,
                  TokenType.AND_ASSIGN, TokenType.OR_ASSIGN, TokenType.QUESTION_QUESTION_ASSIGN)) {
            Token operator = previous();
            Expression right = parseAssignment();
            Token endToken = previous();

            // Convert left side to pattern if it's a destructuring target
            Node leftNode = convertToPatternIfNeeded(left);

            // Use endToken for accurate end position (handles parenthesized expressions correctly)
            int assignEnd = getEnd(endToken);
            SourceLocation loc = createLocation(startToken, endToken);

            return new AssignmentExpression(getStart(startToken), assignEnd, loc, operator.lexeme(), leftNode, right);
        }

        return left;
    }

    // Convert Expression to Pattern for destructuring assignments
    private Node convertToPatternIfNeeded(Node node) {
        // If it's not an expression, return as-is
        if (!(node instanceof Expression)) {
            return node;
        }
        Expression expr = (Expression) node;
        if (expr instanceof ArrayExpression arrayExpr) {
            // Convert ArrayExpression to ArrayPattern
            List<Pattern> patternElements = new ArrayList<>();
            for (Expression element : arrayExpr.elements()) {
                if (element instanceof Identifier id) {
                    patternElements.add(id);
                } else if (element instanceof ArrayExpression || element instanceof ObjectExpression) {
                    // Recursively convert nested destructuring
                    patternElements.add((Pattern) convertToPatternIfNeeded(element));
                } else if (element instanceof AssignmentExpression assignExpr) {
                    // Convert AssignmentExpression to AssignmentPattern (for default values)
                    Node leftNode = convertToPatternIfNeeded(assignExpr.left());
                    Pattern left = (Pattern) leftNode;
                    patternElements.add(new AssignmentPattern(assignExpr.start(), assignExpr.end(), assignExpr.loc(), left, assignExpr.right()));
                } else if (element instanceof SpreadElement spreadElem) {
                    // Convert SpreadElement to RestElement
                    Node argNode = convertToPatternIfNeeded(spreadElem.argument());
                    Pattern argument = (Pattern) argNode;
                    patternElements.add(new RestElement(spreadElem.start(), spreadElem.end(), spreadElem.loc(), argument));
                } else {
                    // For other expressions, keep as is (this might be an error in real code)
                    patternElements.add((Pattern) element);
                }
            }
            return new ArrayPattern(arrayExpr.start(), arrayExpr.end(), arrayExpr.loc(), patternElements);
        } else if (expr instanceof ObjectExpression objExpr) {
            // Convert ObjectExpression to ObjectPattern
            // Need to convert properties: AssignmentExpression -> AssignmentPattern, SpreadElement -> RestElement
            List<Node> convertedProperties = new ArrayList<>();
            for (Node prop : objExpr.properties()) {
                if (prop instanceof Property property) {
                    Node value = property.value();
                    // Convert the value if needed
                    if (value instanceof AssignmentExpression assignExpr) {
                        // Convert AssignmentExpression to AssignmentPattern
                        Node leftNode = convertToPatternIfNeeded(assignExpr.left());
                        Pattern left = (Pattern) leftNode;
                        AssignmentPattern pattern = new AssignmentPattern(assignExpr.start(), assignExpr.end(), assignExpr.loc(), left, assignExpr.right());
                        convertedProperties.add(new Property(property.start(), property.end(), property.loc(),
                            property.method(), property.shorthand(), property.computed(),
                            property.key(), pattern, property.kind()));
                    } else if (value instanceof ObjectExpression || value instanceof ArrayExpression) {
                        // Recursively convert nested destructuring
                        Node convertedValue = convertToPatternIfNeeded(value);
                        convertedProperties.add(new Property(property.start(), property.end(), property.loc(),
                            property.method(), property.shorthand(), property.computed(),
                            property.key(), convertedValue, property.kind()));
                    } else {
                        convertedProperties.add(property);
                    }
                } else if (prop instanceof SpreadElement spreadElem) {
                    // Convert SpreadElement to RestElement in object patterns
                    Node argNode = convertToPatternIfNeeded(spreadElem.argument());
                    Pattern argument = (Pattern) argNode;
                    convertedProperties.add(new RestElement(spreadElem.start(), spreadElem.end(), spreadElem.loc(), argument));
                } else {
                    convertedProperties.add(prop);
                }
            }
            return new ObjectPattern(objExpr.start(), objExpr.end(), objExpr.loc(), convertedProperties);
        } else if (expr instanceof AssignmentExpression assignExpr) {
            // Convert AssignmentExpression to AssignmentPattern (for default values)
            Node leftNode = convertToPatternIfNeeded(assignExpr.left());
            Pattern left = (Pattern) leftNode;
            return new AssignmentPattern(assignExpr.start(), assignExpr.end(), assignExpr.loc(), left, assignExpr.right());
        } else if (expr instanceof Identifier) {
            // Identifier is both Expression and Pattern, return as-is
            return expr;
        } else if (expr instanceof MemberExpression) {
            // MemberExpression is valid as an assignment target, return as-is
            return expr;
        }

        // For other expressions, return as-is
        return expr;
    }

    // Helper method to check if the current position (after opening paren) looks like arrow function parameters
    // This scans ahead to find the matching ) and checks if it's followed by =>
    private boolean isArrowFunctionParameters() {
        int depth = 1; // We've already consumed the opening (
        int checkCurrent = current;

        // Scan ahead to find the matching )
        while (checkCurrent < tokens.size() && depth > 0) {
            TokenType type = tokens.get(checkCurrent).type();

            if (type == TokenType.LPAREN || type == TokenType.LBRACKET || type == TokenType.LBRACE) {
                depth++;
            } else if (type == TokenType.RPAREN || type == TokenType.RBRACKET || type == TokenType.RBRACE) {
                depth--;
                if (depth == 0) {
                    // Found the matching ), check if followed by =>
                    if (checkCurrent + 1 < tokens.size() &&
                        tokens.get(checkCurrent + 1).type() == TokenType.ARROW) {
                        return true;
                    }
                    return false;
                }
            }
            checkCurrent++;
        }

        return false;
    }

    private Expression parseArrowFunctionBody(Token startToken, List<Pattern> params, boolean isAsync) {
        // Save and set async context for arrow function body
        boolean savedInAsyncContext = inAsyncContext;
        boolean savedInClassFieldInitializer = inClassFieldInitializer;
        inAsyncContext = isAsync;
        inClassFieldInitializer = false; // Function bodies are never class field initializers

        try {
            // Arrow function body can be an expression or block statement
            if (check(TokenType.LBRACE)) {
                // Block body: () => { statements }
                BlockStatement body = parseBlockStatement(true); // Arrow function body
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                return new ArrowFunctionExpression(getStart(startToken), getEnd(endToken), loc, null, false, false, isAsync, params, body);
            } else {
                // Expression body: () => expr
                Expression body = parseAssignment();
            Token endToken = previous();

            // Use body.end() and accurate location for template literals and complex expressions
            int arrowEnd;
            SourceLocation loc;
            if (body instanceof TemplateLiteral) {
                arrowEnd = body.end();
                // Create location using getPositionFromOffset for accurate line/column
                SourceLocation.Position startPos = new SourceLocation.Position(startToken.line(), startToken.column());
                SourceLocation.Position endPos = getPositionFromOffset(arrowEnd);
                loc = new SourceLocation(startPos, endPos);
            } else {
                arrowEnd = getEnd(endToken);
                loc = createLocation(startToken, endToken);
            }

            return new ArrowFunctionExpression(getStart(startToken), arrowEnd, loc, null, true, false, isAsync, params, body);
            }
        } finally {
            inAsyncContext = savedInAsyncContext;
            inClassFieldInitializer = savedInClassFieldInitializer;
        }
    }

    private Expression parseTemplateLiteral() {
        Token startToken = peek();
        TokenType startType = startToken.type();

        if (startType == TokenType.TEMPLATE_LITERAL) {
            // Simple template with no interpolation: `hello`
            advance();
            List<Expression> expressions = new ArrayList<>();
            List<TemplateElement> quasis = new ArrayList<>();

            // Create the single quasi
            // Use the raw value from the token (already processed by lexer)
            String raw = startToken.raw();
            String cooked = (String) startToken.literal();
            int elemStart = getStart(startToken) + 1; // +1 to skip opening `
            int elemEnd = getEnd(startToken) - 1; // -1 to exclude closing `
            SourceLocation.Position elemStartPos = getPositionFromOffset(elemStart);
            SourceLocation.Position elemEndPos = getPositionFromOffset(elemEnd);
            SourceLocation elemLoc = new SourceLocation(elemStartPos, elemEndPos);
            quasis.add(new TemplateElement(
                elemStart,
                elemEnd,
                elemLoc,
                new TemplateElement.TemplateElementValue(raw, cooked),
                true // tail
            ));

            SourceLocation.Position templateStartPos = getPositionFromOffset(getStart(startToken));
            SourceLocation.Position templateEndPos = getPositionFromOffset(getEnd(startToken));
            SourceLocation loc = new SourceLocation(templateStartPos, templateEndPos);
            return new TemplateLiteral(getStart(startToken), getEnd(startToken), loc, expressions, quasis);
        } else if (startType == TokenType.TEMPLATE_HEAD) {
            // Template with interpolations
            int templateStart = getStart(startToken);
            advance(); // consume TEMPLATE_HEAD

            List<Expression> expressions = new ArrayList<>();
            List<TemplateElement> quasis = new ArrayList<>();

            // Add the head quasi
            // Use the raw value from the token (already processed by lexer)
            String raw = startToken.raw();
            String cooked = (String) startToken.literal();
            int elemStart = templateStart + 1; // +1 for opening `
            // Token endPosition includes the ${ delimiter, so we need to subtract 2
            // This works correctly for both LF and CRLF files since we use actual token positions
            int elemEnd = getEnd(startToken) - 2;
            SourceLocation.Position elemStartPos = getPositionFromOffset(elemStart);
            SourceLocation.Position elemEndPos = getPositionFromOffset(elemEnd);
            SourceLocation elemLoc = new SourceLocation(elemStartPos, elemEndPos);
            quasis.add(new TemplateElement(
                elemStart,
                elemEnd,
                elemLoc,
                new TemplateElement.TemplateElementValue(raw, cooked),
                false // not tail
            ));

            // Parse expressions and middle/tail quasis
            while (true) {
                // Parse the expression
                Expression expr = parseExpression();
                expressions.add(expr);

                // Next token should be TEMPLATE_MIDDLE or TEMPLATE_TAIL
                Token quasiToken = peek();
                TokenType quasiType = quasiToken.type();

                if (quasiType == TokenType.TEMPLATE_MIDDLE) {
                    advance();
                    // Use the raw value from the token (already processed by lexer)
                    String quasiRaw = quasiToken.raw();
                    String quasiCooked = (String) quasiToken.literal();
                    int quasiStart = getStart(quasiToken) + 1; // +1 to skip }
                    // Token endPosition includes the ${ delimiter, so we need to subtract 2
                    // This works correctly for both LF and CRLF files since we use actual token positions
                    int quasiEnd = getEnd(quasiToken) - 2;
                    SourceLocation.Position quasiStartPos = getPositionFromOffset(quasiStart);
                    SourceLocation.Position quasiEndPos = getPositionFromOffset(quasiEnd);
                    SourceLocation quasiLoc = new SourceLocation(quasiStartPos, quasiEndPos);
                    quasis.add(new TemplateElement(
                        quasiStart,
                        quasiEnd,
                        quasiLoc,
                        new TemplateElement.TemplateElementValue(quasiRaw, quasiCooked),
                        false // not tail
                    ));
                } else if (quasiType == TokenType.TEMPLATE_TAIL) {
                    Token endToken = quasiToken;
                    advance();
                    // Use the raw value from the token (already processed by lexer)
                    String quasiRaw = quasiToken.raw();
                    String quasiCooked = (String) quasiToken.literal();
                    int quasiStart = getStart(quasiToken) + 1; // +1 to skip }
                    // Use token's actual end position - 1 to exclude closing `
                    // This is important for files with CRLF line endings where raw is normalized
                    int quasiEnd = getEnd(quasiToken) - 1;
                    SourceLocation.Position quasiStartPos = getPositionFromOffset(quasiStart);
                    SourceLocation.Position quasiEndPos = getPositionFromOffset(quasiEnd);
                    SourceLocation quasiLoc = new SourceLocation(quasiStartPos, quasiEndPos);
                    quasis.add(new TemplateElement(
                        quasiStart,
                        quasiEnd,
                        quasiLoc,
                        new TemplateElement.TemplateElementValue(quasiRaw, quasiCooked),
                        true // tail
                    ));

                    // Calculate template end position (token already includes closing `)
                    int templateEnd = getEnd(endToken);
                    SourceLocation.Position templateStartPos = getPositionFromOffset(templateStart);
                    SourceLocation.Position templateEndPos = getPositionFromOffset(templateEnd);
                    SourceLocation loc = new SourceLocation(templateStartPos, templateEndPos);
                    return new TemplateLiteral(templateStart, templateEnd, loc, expressions, quasis);
                } else {
                    throw new ExpectedTokenException("TEMPLATE_MIDDLE or TEMPLATE_TAIL after expression in template literal", peek());
                }
            }
        } else {
            throw new ExpectedTokenException("template literal token", peek());
        }
    }

    /**
     * Parse binary expressions using table-driven precedence climbing.
     * This replaces the previous 12-method precedence hierarchy with a single unified method.
     *
     * Algorithm: Precedence Climbing (also known as Pratt parsing)
     * - Start with minimum precedence
     * - Parse left-hand side (unary expression)
     * - While current operator has precedence >= minimum:
     *   - Consume operator
     *   - Calculate next precedence based on associativity
     *   - Recursively parse right-hand side with next precedence
     *   - Create binary/logical expression node
     *   - Continue with result as new left-hand side
     *
     * This approach reduces stack depth from 17+ method calls to 1-3 calls
     * and makes operator precedence explicit via the OPERATOR_PRECEDENCE table.
     *
     * @param minPrecedence Minimum operator precedence to parse (higher values = tighter binding)
     * @return Parsed expression
     */
    private Expression parseBinaryExpression(int minPrecedence) {
        Token startToken = peek();
        Expression left = parseUnary();

        while (current < tokens.size()) {
            Token operatorToken = peek();
            OperatorInfo opInfo = OPERATOR_PRECEDENCE.get(operatorToken.type());

            // Stop parsing if:
            // 1. Current token is not a binary operator
            // 2. Operator precedence is lower than minimum (lower precedence = looser binding)
            // 3. Context check fails (e.g., 'in' operator when allowIn=false)
            if (opInfo == null || opInfo.precedence() < minPrecedence) {
                break;
            }

            // Special case: 'in' operator respects allowIn context flag
            // Example: for (let x in obj) - 'in' should NOT be parsed as binary operator
            // Example: (x in obj) - 'in' SHOULD be parsed as binary operator
            if (operatorToken.type() == TokenType.IN && !allowIn) {
                break;
            }

            // Consume the operator token
            advance();
            Token operator = previous();

            // Calculate next precedence level based on associativity:
            // - Left-associative: next precedence = current + 1
            //   Example: 1 + 2 + 3 = (1 + 2) + 3
            //   First + parses with precedence 13, second + needs precedence 14 to stop
            // - Right-associative: next precedence = current
            //   Example: 2 ** 3 ** 4 = 2 ** (3 ** 4)
            //   Both ** parse with precedence 15, allowing right recursion
            int nextPrecedence = opInfo.isLeftAssociative()
                ? opInfo.precedence() + 1
                : opInfo.precedence();

            // Recursively parse right-hand side with calculated precedence
            Expression right = parseBinaryExpression(nextPrecedence);

            // Create appropriate AST node based on operator type
            Token endToken = previous();
            SourceLocation loc = createLocation(startToken, endToken);

            if (opInfo.isLogicalExpression()) {
                // LogicalExpression for ||, &&, ?? (short-circuiting operators)
                left = new LogicalExpression(
                    getStart(startToken),
                    getEnd(endToken),
                    loc,
                    operator.lexeme(),
                    left,
                    right
                );
            } else {
                // BinaryExpression for all other binary operators
                // (arithmetic, bitwise, comparison, shift, etc.)
                left = new BinaryExpression(
                    getStart(startToken),
                    getEnd(endToken),
                    loc,
                    left,
                    operator.lexeme(),
                    right
                );
            }
        }

        return left;
    }

    // conditional -> nullishCoalescing ( "?" assignment ":" assignment )?
    private Expression parseConditional() {
        Token startToken = peek();
        // Parse up to (and including) nullish coalescing level (precedence 4)
        // This handles: ??, ||, &&, |, ^, &, ==, !=, ===, !==, <, <=, >, >=,
        // instanceof, in, <<, >>, >>>, +, -, *, /, %, **
        Expression test = parseBinaryExpression(4);

        if (match(TokenType.QUESTION)) {
            // Allow 'in' operator in ternary branches
            boolean oldAllowIn = allowIn;
            allowIn = true;
            Expression consequent = parseAssignment();
            consume(TokenType.COLON, "Expected ':' in ternary expression");
            Expression alternate = parseAssignment();
            allowIn = oldAllowIn;
            Token endToken = previous();

            // Always use endToken to include any closing parens
            int conditionalEnd = getEnd(endToken);
            SourceLocation loc = createLocation(startToken, endToken);

            return new ConditionalExpression(getStart(startToken), conditionalEnd, loc, test, consequent, alternate);
        }

        return test;
    }

    // unary -> ( "!" | "-" | "+" | "~" | "typeof" | "void" | "delete" | "++" | "--" ) unary | postfix
    private Expression parseUnary() {
        Token token = peek();

        // Prefix update operators (++x, --x)
        if (match(TokenType.INCREMENT, TokenType.DECREMENT)) {
            Token operator = previous();
            Expression argument = parseUnary();  // Right-associative
            Token endToken = previous();
            SourceLocation loc = createLocation(token, endToken);
            return new UpdateExpression(getStart(token), getEnd(endToken), loc, operator.lexeme(), true, argument);
        }

        // Await expressions (contextual keyword)
        // In async context, await is a keyword (use minimal lookahead for top-level await in modules)
        // In script mode (non-async), use lookahead to distinguish from identifier usage
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("await")) {
            boolean shouldParseAsAwait = false;

            if (inAsyncContext && !inClassFieldInitializer) {
                // Async context: await is ALWAYS a keyword, parse as AwaitExpression
                // No lookahead needed - await is unambiguous in async contexts
                shouldParseAsAwait = true;
            } else if (!inAsyncContext && forceModuleMode && !inClassFieldInitializer) {
                // Module mode (top-level await): await is a keyword, but use minimal lookahead
                // to avoid parsing await labels/assignments
                // Only exclude the cases that are syntactically impossible as AwaitExpression
                shouldParseAsAwait = !checkAhead(1, TokenType.COLON) &&
                                   !checkAhead(1, TokenType.ASSIGN) &&
                                   !checkAhead(1, TokenType.PLUS_ASSIGN) &&
                                   !checkAhead(1, TokenType.MINUS_ASSIGN);
            } else if (!inAsyncContext && !forceModuleMode) {
                // Script mode (non-async, non-module): await is ALWAYS a regular identifier
                // It is NEVER an AwaitExpression in script mode
                shouldParseAsAwait = false;
            }

            // Class field initializer validation: reject AwaitExpression patterns
            // Even though we won't parse it as AwaitExpression, we need to detect and reject the pattern
            // NOTE: Class field initializers cannot use await even in module mode
            if (inClassFieldInitializer && !shouldParseAsAwait) {
                // Check if 'await' is followed by what looks like an AwaitExpression argument
                // Reject patterns like: await foo, await x.y, await (expr), await func(), etc.
                // Allow patterns like: await; await = x, await: label
                boolean looksLikeAwaitExpression = checkAhead(1, TokenType.IDENTIFIER) ||
                                                   checkAhead(1, TokenType.LPAREN) ||
                                                   checkAhead(1, TokenType.LBRACKET) ||
                                                   checkAhead(1, TokenType.THIS) ||
                                                   checkAhead(1, TokenType.SUPER) ||
                                                   checkAhead(1, TokenType.NEW) ||
                                                   checkAhead(1, TokenType.CLASS) ||
                                                   checkAhead(1, TokenType.FUNCTION) ||
                                                   checkAhead(1, TokenType.ASYNC) ||
                                                   checkAhead(1, TokenType.STRING) ||
                                                   checkAhead(1, TokenType.NUMBER) ||
                                                   checkAhead(1, TokenType.TRUE) ||
                                                   checkAhead(1, TokenType.FALSE) ||
                                                   checkAhead(1, TokenType.NULL);

                if (looksLikeAwaitExpression) {
                    Token awaitToken = peek();
                    throw new ParseException("SyntaxError", awaitToken, null, null,
                        "Cannot use keyword 'await' outside an async function");
                }
            }

            if (shouldParseAsAwait) {
                Token awaitToken = advance();

                // Check if there's an argument (not standalone await)
                // Don't consume the argument if it's a semicolon, closing delimiter, binary operator, etc.
                Expression argument = null;
                if (!check(TokenType.SEMICOLON) && !check(TokenType.RBRACE) && !check(TokenType.EOF) &&
                    !check(TokenType.RPAREN) && !check(TokenType.COMMA) && !check(TokenType.RBRACKET) &&
                    !check(TokenType.INSTANCEOF) && !check(TokenType.IN) &&
                    !check(TokenType.QUESTION) && !check(TokenType.COLON)) {
                    argument = parseUnary();  // Right-associative
                }

                Token endToken = previous();
                SourceLocation loc = createLocation(awaitToken, endToken);
                return new AwaitExpression(getStart(awaitToken), getEnd(endToken), loc, argument);
            }
        }

        // Unary operators
        if (match(TokenType.BANG, TokenType.MINUS, TokenType.PLUS, TokenType.TILDE,
                  TokenType.TYPEOF, TokenType.VOID, TokenType.DELETE)) {
            Token operator = previous();
            Expression argument = parseUnary();  // Right-associative
            Token endToken = previous();

            // Strict mode validation: delete on identifiers is not allowed
            if (strictMode && operator.type() == TokenType.DELETE && argument instanceof Identifier) {
                throw new ExpectedTokenException("Delete of an unqualified identifier is not allowed in strict mode", operator);
            }

            SourceLocation loc = createLocation(token, endToken);
            return new UnaryExpression(getStart(token), getEnd(endToken), loc, operator.lexeme(), true, argument);
        }

        return parsePostfix();
    }

    // Handle member access (obj.prop, obj[prop]) and function calls (func())
    private Expression parsePostfix() {
        Token startToken = peek();
        Expression expr = parsePrimary();

        // Track if we need to wrap in ChainExpression
        boolean hasOptionalChaining = false;
        Token chainStartToken = startToken;

        while (true) {
            if (match(TokenType.QUESTION_DOT)) {
                // Mark that we've started an optional chain
                if (!hasOptionalChaining) {
                    hasOptionalChaining = true;
                    chainStartToken = startToken;
                }
                // Optional chaining: obj?.prop, obj?.[expr], or obj?.(args)
                if (check(TokenType.LPAREN)) {
                    // Optional call: obj?.(args)
                    advance(); // consume (
                    List<Expression> args = new ArrayList<>();
                    if (!check(TokenType.RPAREN)) {
                        do {
                            // Check for trailing comma: foo?.(1, 2,)
                            if (check(TokenType.RPAREN)) {
                                break;
                            }
                            // Check for spread element: foo?.(...arr)
                            if (match(TokenType.DOT_DOT_DOT)) {
                                Token spreadStart = previous();
                                Expression argument = parseAssignment();
                                Token spreadEnd = previous();
                                SourceLocation spreadLoc = createLocation(spreadStart, spreadEnd);
                                args.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadLoc, argument));
                            } else {
                                args.add(parseAssignment());
                            }
                        } while (match(TokenType.COMMA));
                    }
                    consume(TokenType.RPAREN, "Expected ')' after arguments");
                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    expr = new CallExpression(getStart(startToken), getEnd(endToken), loc, expr, args, true);
                } else if (check(TokenType.LBRACKET)) {
                    // Optional computed member: obj?.[expr]
                    advance(); // consume [
                    Expression property = parseExpression();
                    consume(TokenType.RBRACKET, "Expected ']' after computed property");
                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    expr = new MemberExpression(getStart(startToken), getEnd(endToken), loc, expr, property, true, true);
                } else if (match(TokenType.HASH)) {
                    // Optional private field: obj?.#x - allow keywords as private names
                    Token hashToken = previous();
                    Token propertyToken = peek();
                    if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken)) {
                        throw new ExpectedTokenException("identifier after '#'", peek());
                    }
                    advance();
                    Expression property = new PrivateIdentifier(getStart(hashToken), getEnd(propertyToken), createLocation(hashToken, propertyToken), propertyToken.lexeme());
                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    expr = new MemberExpression(getStart(startToken), getEnd(endToken), loc, expr, property, false, true);
                } else {
                    // Optional property: obj?.x (allows keywords, numbers, strings, booleans, null)
                    Token propertyToken = peek();
                    if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken) &&
                        !check(TokenType.NUMBER) && !check(TokenType.STRING) &&
                        !check(TokenType.TRUE) && !check(TokenType.FALSE) && !check(TokenType.NULL)) {
                        throw new ExpectedTokenException("property name after '?.'", peek());
                    }
                    advance();
                    Expression property = new Identifier(getStart(propertyToken), getEnd(propertyToken), createLocation(propertyToken, propertyToken), propertyToken.lexeme());
                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    expr = new MemberExpression(getStart(startToken), getEnd(endToken), loc, expr, property, false, true);
                }
            } else if (match(TokenType.DOT)) {
                // Member expression: obj.property or obj.#privateProperty
                if (match(TokenType.HASH)) {
                    // Private field: obj.#x - allow keywords as private names
                    Token hashToken = previous();
                    Token propertyToken = peek();
                    if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken)) {
                        throw new ExpectedTokenException("identifier after '#'", peek());
                    }
                    advance();
                    Expression property = new PrivateIdentifier(getStart(hashToken), getEnd(propertyToken), createLocation(hashToken, propertyToken), propertyToken.lexeme());
                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    expr = new MemberExpression(getStart(startToken), getEnd(endToken), loc, expr, property, false, false);
                } else {
                    // Regular property: obj.x (allows keywords, numbers, strings, and boolean/null literals)
                    Token propertyToken = peek();
                    if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken) &&
                        !check(TokenType.NUMBER) && !check(TokenType.STRING) &&
                        !check(TokenType.TRUE) && !check(TokenType.FALSE) && !check(TokenType.NULL)) {
                        throw new ExpectedTokenException("property name after '.'", peek());
                    }
                    advance();
                    Expression property = new Identifier(getStart(propertyToken), getEnd(propertyToken), createLocation(propertyToken, propertyToken), propertyToken.lexeme());
                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    expr = new MemberExpression(getStart(startToken), getEnd(endToken), loc, expr, property, false, false);
                }
            } else if (match(TokenType.LBRACKET)) {
                // Computed member expression: obj[property]
                Expression property = parseExpression();
                consume(TokenType.RBRACKET, "Expected ']' after computed property");
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                expr = new MemberExpression(getStart(startToken), getEnd(endToken), loc, expr, property, true, false);
            } else if (match(TokenType.LPAREN)) {
                // Call expression: func(args)
                List<Expression> args = new ArrayList<>();
                if (!check(TokenType.RPAREN)) {
                    do {
                        // Check for trailing comma: foo(1, 2,)
                        if (check(TokenType.RPAREN)) {
                            break;
                        }
                        // Check for spread element: foo(...arr)
                        if (match(TokenType.DOT_DOT_DOT)) {
                            Token spreadStart = previous();
                            Expression argument = parseAssignment();
                            Token spreadEnd = previous();
                            SourceLocation spreadLoc = createLocation(spreadStart, spreadEnd);
                            args.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadLoc, argument));
                        } else {
                            args.add(parseAssignment()); // Allow assignments in arguments
                        }
                    } while (match(TokenType.COMMA));
                }
                consume(TokenType.RPAREN, "Expected ')' after arguments");
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                expr = new CallExpression(getStart(startToken), getEnd(endToken), loc, expr, args, false);
            } else if (check(TokenType.INCREMENT) || check(TokenType.DECREMENT)) {
                // Postfix update operators: x++, x--
                // [no LineTerminator here] restriction: line break not allowed before ++ or --
                Token prevToken = previous();
                Token nextToken = peek();
                if (prevToken.line() < nextToken.line()) {
                    // Line break before postfix operator - cannot apply postfix
                    break;
                }
                advance(); // consume ++ or --
                Token operator = previous();
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                expr = new UpdateExpression(getStart(startToken), getEnd(endToken), loc, operator.lexeme(), false, expr);
            } else if (check(TokenType.TEMPLATE_LITERAL) || check(TokenType.TEMPLATE_HEAD)) {
                // Tagged template literal: tag`template`
                Expression template = parseTemplateLiteral();
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                expr = new TaggedTemplateExpression(getStart(startToken), template.end(), loc, expr, (TemplateLiteral) template);
            } else {
                break;
            }
        }

        // Wrap in ChainExpression if we used optional chaining
        if (hasOptionalChaining) {
            Token endToken = previous();
            SourceLocation loc = createLocation(chainStartToken, endToken);
            return new ChainExpression(getStart(chainStartToken), getEnd(endToken), loc, expr);
        }

        return expr;
    }

    // primary -> NUMBER | STRING | "true" | "false" | "null" | IDENTIFIER | "(" expression ")" | "[" elements "]"
    private Expression parsePrimary() {
        Token token = peek();

        return switch (token.type()) {
            case HASH -> {
                // Private identifier for `#field in obj` expressions - allow keywords as private names
                advance(); // consume #
                Token nameToken = peek();
                if (!check(TokenType.IDENTIFIER) && !isKeyword(nameToken)) {
                    throw new ExpectedTokenException("identifier after '#'", peek());
                }
                advance();
                SourceLocation loc = createLocation(token, nameToken);
                yield new PrivateIdentifier(getStart(token), getEnd(nameToken), loc, nameToken.lexeme());
            }
            case NUMBER -> {
                advance();
                SourceLocation loc = createLocation(token, token);

                // Check if this is a BigInt literal (ends with 'n')
                String lexeme = token.lexeme();
                if (lexeme.endsWith("n")) {
                    // BigInt literal: value is null, bigint field has the numeric part
                    String bigintValue = lexeme.substring(0, lexeme.length() - 1).replace("_", "");

                    // Convert hex/octal/binary to decimal for the bigint field
                    if (bigintValue.startsWith("0x") || bigintValue.startsWith("0X")) {
                        // Hex BigInt
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 16);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    } else if (bigintValue.startsWith("0o") || bigintValue.startsWith("0O")) {
                        // Octal BigInt
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 8);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    } else if (bigintValue.startsWith("0b") || bigintValue.startsWith("0B")) {
                        // Binary BigInt
                        try {
                            java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 2);
                            bigintValue = bi.toString();
                        } catch (NumberFormatException e) {
                            // Keep original if conversion fails
                        }
                    }

                    yield new Literal(getStart(token), getEnd(token), loc, null, lexeme, null, bigintValue);
                }

                // For Infinity/-Infinity/NaN, the value should be null per ESTree spec
                Object literalValue = token.literal();
                if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                    literalValue = null;
                }
                yield new Literal(getStart(token), getEnd(token), loc, literalValue, token.lexeme());
            }
            case STRING -> {
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Literal(getStart(token), getEnd(token), loc, token.literal(), token.lexeme());
            }
            case TEMPLATE_LITERAL, TEMPLATE_HEAD -> parseTemplateLiteral();
            case TRUE -> {
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Literal(getStart(token), getEnd(token), loc, true, "true");
            }
            case FALSE -> {
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Literal(getStart(token), getEnd(token), loc, false, "false");
            }
            case NULL -> {
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Literal(getStart(token), getEnd(token), loc, null, "null");
            }
            case IDENTIFIER -> {
                // Check for async function expression
                // No line terminator is allowed between async and function
                if (token.lexeme().equals("async") && current + 1 < tokens.size() &&
                    tokens.get(current + 1).type() == TokenType.FUNCTION &&
                    tokens.get(current).line() == tokens.get(current + 1).line()) {
                    Token startToken = token;
                    advance(); // consume 'async'
                    advance(); // consume 'function'

                    // Check for generator
                    boolean isGenerator = match(TokenType.STAR);

                    // Optional function name (can be null for anonymous)
                    Identifier id = null;
                    if (check(TokenType.IDENTIFIER)) {
                        Token nameToken = peek();
                        advance();
                        id = new Identifier(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.lexeme());
                    }

                    // Parse parameters
                    consume(TokenType.LPAREN, "Expected '(' after function");
                    List<Pattern> params = new ArrayList<>();

                    if (!check(TokenType.RPAREN)) {
                        do {
                            // Check for trailing comma: async function(a, b,) {}
                            if (check(TokenType.RPAREN)) {
                                break;
                            }
                            if (match(TokenType.DOT_DOT_DOT)) {
                                Token restStart = previous();
                                Pattern argument = parsePatternBase();
                                Token restEnd = previous();
                                SourceLocation restLoc = createLocation(restStart, restEnd);
                                params.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                                if (match(TokenType.COMMA)) {
                                    throw new ParseException("ValidationError", peek(), null, "parameter list", "Rest parameter must be last");
                                }
                                break;
                            } else {
                                params.add(parsePattern());
                            }
                        } while (match(TokenType.COMMA));
                    }

                    consume(TokenType.RPAREN, "Expected ')' after parameters");

                    // Parse body with proper generator/async context
                    boolean savedInGenerator = inGenerator;
                    boolean savedInAsyncContext = inAsyncContext;
                    boolean savedStrictMode = strictMode;
                    boolean savedInClassFieldInitializer = inClassFieldInitializer;
                    inGenerator = isGenerator;
                    inAsyncContext = true; // async function expression
                    inClassFieldInitializer = false; // Function bodies are never class field initializers

                    // Reset strict mode for function body (unless in module mode)
                    if (!forceModuleMode) {
                        strictMode = false;
                    }

                    BlockStatement body = parseBlockStatement(true); // Function expression body

                    // Check for duplicate parameters if in strict mode
                    validateNoDuplicateParameters(params, startToken);

                    inGenerator = savedInGenerator;
                    inAsyncContext = savedInAsyncContext;
                    strictMode = savedStrictMode;
                    inClassFieldInitializer = savedInClassFieldInitializer;

                    Token endToken = previous();
                    SourceLocation loc = createLocation(startToken, endToken);
                    yield new FunctionExpression(getStart(startToken), getEnd(endToken), loc, id, false, isGenerator, true, params, body);
                }

                advance();

                // In module/async mode, 'await' is a reserved keyword
                // If we reach here, it means the AwaitExpression check didn't catch it
                // This can only happen in error cases (e.g., trying to assign to await)
                // Exception: class field initializers reset the async context, so 'await' can be an identifier there
                if ((forceModuleMode || inAsyncContext) && !inClassFieldInitializer && token.lexeme().equals("await")) {
                    String context = forceModuleMode ? "module code" : "async function";
                    throw new ParseException("SyntaxError", token, null, null,
                        "Unexpected use of 'await' as identifier in " + context);
                }

                SourceLocation loc = createLocation(token, token);
                yield new Identifier(getStart(token), getEnd(token), loc, token.lexeme());
            }
            case THIS -> {
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new ThisExpression(getStart(token), getEnd(token), loc);
            }
            case SUPER -> {
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Super(getStart(token), getEnd(token), loc);
            }
            case IMPORT -> {
                // Handle import.meta or dynamic import()
                Token importToken = token;
                advance(); // consume 'import'

                // Check for import.meta
                if (match(TokenType.DOT)) {
                    Token propertyToken = peek();
                    if (check(TokenType.IDENTIFIER) && propertyToken.lexeme().equals("meta")) {
                        advance(); // consume 'meta'
                        SourceLocation metaLoc = createLocation(importToken, importToken);
                        SourceLocation propLoc = createLocation(propertyToken, propertyToken);
                        Identifier meta = new Identifier(getStart(importToken), getEnd(importToken), metaLoc, "import");
                        Identifier property = new Identifier(getStart(propertyToken), getEnd(propertyToken), propLoc, "meta");
                        SourceLocation loc = createLocation(importToken, propertyToken);
                        yield new MetaProperty(getStart(importToken), getEnd(propertyToken), loc, meta, property);
                    } else {
                        throw new ExpectedTokenException("'meta'", peek());
                    }
                } else if (match(TokenType.LPAREN)) {
                    // Dynamic import: import(specifier) or import(specifier, options)
                    // Per spec, arguments use AssignmentExpression[+In, ...]
                    boolean savedAllowIn = allowIn;
                    allowIn = true;

                    Expression source = parseAssignment();
                    Expression options = null;

                    // Check for optional second argument (import attributes)
                    if (match(TokenType.COMMA)) {
                        // Allow trailing comma: import(source,)
                        if (!check(TokenType.RPAREN)) {
                            options = parseAssignment();
                            // Allow trailing comma after options: import(source, options,)
                            match(TokenType.COMMA);
                        }
                    }

                    allowIn = savedAllowIn;

                    consume(TokenType.RPAREN, "Expected ')' after import source");
                    Token endToken = previous();
                    SourceLocation loc = createLocation(importToken, endToken);
                    yield new ImportExpression(getStart(importToken), getEnd(endToken), loc, source, options);
                } else {
                    throw new UnexpectedTokenException(peek(), null, "expression context");
                }
            }
            case REGEX -> {
                advance();
                Literal.RegexInfo regexInfo = (Literal.RegexInfo) token.literal();
                SourceLocation loc = createLocation(token, token);
                // value is {} representing a RegExp object (JSON can't serialize actual RegExp)
                yield new Literal(getStart(token), getEnd(token), loc,
                    new java.util.HashMap<>(), token.lexeme(), regexInfo);
            }
            case SLASH -> {
                // If we encounter SLASH in primary position, it might be division in wrong context
                // or we missed a regex. Try rescanning as regex as fallback.
                Token regexToken = lexer.scanRegexAt(token.position());
                // Update current to skip the regex token
                while (current < tokens.size() && tokens.get(current).position() < regexToken.endPosition()) {
                    current++;
                }
                Literal.RegexInfo regexInfo = (Literal.RegexInfo) regexToken.literal();
                SourceLocation loc = createLocation(regexToken, regexToken);
                // value is {} representing a RegExp object
                yield new Literal(getStart(regexToken), getEnd(regexToken), loc,
                    new java.util.HashMap<>(), regexToken.lexeme(), regexInfo);
            }
            case LPAREN -> {
                advance(); // consume '('
                Token startAfterParen = peek();  // Save the position after the '('

                // Temporarily allow 'in' inside parentheses
                boolean oldAllowIn = allowIn;
                allowIn = true;

                Expression expr = parseAssignment();

                // Check for sequence expression (comma operator)
                if (check(TokenType.COMMA)) {
                    List<Expression> expressions = new ArrayList<>();
                    expressions.add(expr);

                    while (match(TokenType.COMMA)) {
                        expressions.add(parseAssignment());
                    }

                    // Determine the end position for the SequenceExpression
                    // Acorn includes any RPAREN tokens between the last expression and the outer RPAREN
                    Expression lastExpr = expressions.get(expressions.size() - 1);
                    int seqEnd = lastExpr.end();
                    SourceLocation.Position seqEndPos = lastExpr.loc().end();

                    // Look backwards from current position to find RPARENs that were consumed
                    // while parsing the last expression. These should be included in the sequence.
                    // We stop at the last expression's start position.
                    int checkPos = current - 1;
                    Token lastRparen = null;
                    while (checkPos >= 0) {
                        Token t = tokens.get(checkPos);
                        if (getStart(t) < lastExpr.start()) {
                            // Went too far back
                            break;
                        }
                        if (t.type() == TokenType.RPAREN && getStart(t) >= lastExpr.end()) {
                            // This RPAREN comes after the last expression ended
                            lastRparen = t;
                        }
                        checkPos--;
                    }

                    // If we found RPARENs after the last expression, include the last one
                    if (lastRparen != null) {
                        seqEnd = getEnd(lastRparen);
                        seqEndPos = new SourceLocation.Position(lastRparen.line(), lastRparen.column() + 1);
                    }

                    consume(TokenType.RPAREN, "Expected ')' after expression");

                    SourceLocation loc = new SourceLocation(
                        new SourceLocation.Position(startAfterParen.line(), startAfterParen.column()),
                        seqEndPos
                    );
                    allowIn = oldAllowIn; // Restore allowIn
                    yield new SequenceExpression(getStart(startAfterParen), seqEnd, loc, expressions);
                }

                consume(TokenType.RPAREN, "Expected ')' after expression");
                allowIn = oldAllowIn; // Restore allowIn
                yield expr;
            }
            case LBRACKET -> {
                Token startToken = token;
                advance();
                // Allow 'in' operator inside array literals
                boolean savedAllowIn = allowIn;
                allowIn = true;
                List<Expression> elements = new ArrayList<>();
                if (!check(TokenType.RBRACKET)) {
                    do {
                        // Check for elision (hole): [,] or [1,,3]
                        if (check(TokenType.COMMA)) {
                            elements.add(null);
                        }
                        // Check for spread element: [...expr]
                        else if (match(TokenType.DOT_DOT_DOT)) {
                            Token spreadStart = previous();
                            Expression argument = parseAssignment();
                            Token spreadEnd = previous();
                            SourceLocation spreadLoc = createLocation(spreadStart, spreadEnd);
                            elements.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadLoc, argument));
                        } else {
                            elements.add(parseAssignment());
                        }
                    } while (match(TokenType.COMMA) && !check(TokenType.RBRACKET));
                }
                allowIn = savedAllowIn;
                consume(TokenType.RBRACKET, "Expected ']' after array elements");
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                yield new ArrayExpression(getStart(startToken), getEnd(endToken), loc, elements);
            }
            case LBRACE -> {
                // Object literal
                Token startToken = token;
                advance();
                // Allow 'in' operator inside object literals
                boolean savedAllowInObj = allowIn;
                allowIn = true;
                List<Node> properties = new ArrayList<>();

                while (!check(TokenType.RBRACE) && !isAtEnd()) {
                    // Check for spread property: {...expr}
                    if (match(TokenType.DOT_DOT_DOT)) {
                        Token spreadStart = previous();
                        Expression argument = parseAssignment();
                        Token spreadEnd = previous();
                        SourceLocation spreadLoc = createLocation(spreadStart, spreadEnd);
                        properties.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadLoc, argument));

                        if (!match(TokenType.COMMA)) {
                            break;
                        }
                        continue;
                    }

                    Token propStartToken = peek();  // Save start for Property position
                    Token keyToken = peek();
                    Expression key;
                    boolean computed = false;
                    String kind = "init";

                    // Check for generator method: *foo() {}
                    boolean isGenerator = false;
                    boolean isAsync = false;

                    if (match(TokenType.STAR)) {
                        isGenerator = true;
                        keyToken = peek();
                    } else if (check(TokenType.IDENTIFIER)) {
                        String ident = peek().lexeme();

                        // Check for async method
                        if (ident.equals("async")) {
                            Token nextToken = tokens.get(current + 1);
                            if (nextToken.type() == TokenType.IDENTIFIER ||
                                nextToken.type() == TokenType.STRING ||
                                nextToken.type() == TokenType.NUMBER ||
                                nextToken.type() == TokenType.LBRACKET ||
                                nextToken.type() == TokenType.STAR ||
                                nextToken.type() == TokenType.TRUE ||
                                nextToken.type() == TokenType.FALSE ||
                                nextToken.type() == TokenType.NULL ||
                                isKeyword(nextToken)) {
                                advance(); // consume 'async'
                                isAsync = true;
                                if (match(TokenType.STAR)) {
                                    isGenerator = true;
                                }
                                keyToken = peek();
                            }
                        }
                        // Check for getter/setter (contextual keywords)
                        else if ((ident.equals("get") || ident.equals("set"))) {
                            Token nextToken = tokens.get(current + 1);
                            // Only treat as getter/setter if followed by property key (including keywords)
                            if (nextToken.type() == TokenType.IDENTIFIER ||
                                nextToken.type() == TokenType.STRING ||
                                nextToken.type() == TokenType.NUMBER ||
                                nextToken.type() == TokenType.LBRACKET ||
                                nextToken.type() == TokenType.TRUE ||
                                nextToken.type() == TokenType.FALSE ||
                                nextToken.type() == TokenType.NULL ||
                                isKeyword(nextToken)) {
                                advance();  // consume 'get' or 'set'
                                kind = ident;
                                keyToken = peek();  // now points to property name
                                // propStartToken still points to 'get'/'set'
                            }
                        }
                    }

                    if (match(TokenType.LBRACKET)) {
                        // Computed property: [expression]: value
                        // Allow 'in' operator inside computed property
                        boolean savedAllowInComputed = allowIn;
                        allowIn = true;
                        key = parseAssignment();
                        allowIn = savedAllowInComputed;
                        consume(TokenType.RBRACKET, "Expected ']' after computed property");
                        computed = true;
                    } else if (check(TokenType.STRING) || check(TokenType.NUMBER)) {
                        // Literal key
                        advance();
                        SourceLocation keyLoc = createLocation(keyToken, keyToken);
                        String keyLexeme = keyToken.lexeme();

                        // Check if this is a BigInt literal (ends with 'n')
                        if (keyLexeme.endsWith("n")) {
                            // BigInt literal: value is null, bigint field has the numeric part
                            String bigintValue = keyLexeme.substring(0, keyLexeme.length() - 1).replace("_", "");

                            // Convert hex/octal/binary to decimal for the bigint field
                            if (bigintValue.startsWith("0x") || bigintValue.startsWith("0X")) {
                                try {
                                    java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 16);
                                    bigintValue = bi.toString();
                                } catch (NumberFormatException e) {
                                    // Keep original if conversion fails
                                }
                            } else if (bigintValue.startsWith("0o") || bigintValue.startsWith("0O")) {
                                try {
                                    java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 8);
                                    bigintValue = bi.toString();
                                } catch (NumberFormatException e) {
                                    // Keep original if conversion fails
                                }
                            } else if (bigintValue.startsWith("0b") || bigintValue.startsWith("0B")) {
                                try {
                                    java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 2);
                                    bigintValue = bi.toString();
                                } catch (NumberFormatException e) {
                                    // Keep original if conversion fails
                                }
                            }

                            key = new Literal(getStart(keyToken), getEnd(keyToken), keyLoc, null, keyLexeme, null, bigintValue);
                        } else {
                            // For Infinity/-Infinity, the value should be null per ESTree spec
                            Object literalValue = keyToken.literal();
                            if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                                literalValue = null;
                            }
                            key = new Literal(getStart(keyToken), getEnd(keyToken), keyLoc, literalValue, keyLexeme);
                        }
                    } else if (check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL)) {
                        // Boolean and null literals as keys: { true: 1, false: 2, null: 3 }
                        advance();
                        SourceLocation keyLoc = createLocation(keyToken, keyToken);
                        key = new Identifier(getStart(keyToken), getEnd(keyToken), keyLoc, keyToken.lexeme());
                    } else if (check(TokenType.IDENTIFIER) || isKeyword(peek())) {
                        // Identifier or keyword as key - could be shorthand {x}, method {x(){}}, or regular {x: value}
                        // Keywords are allowed as unquoted property names in ES5+
                        advance();
                        SourceLocation keyLoc = createLocation(keyToken, keyToken);
                        key = new Identifier(getStart(keyToken), getEnd(keyToken), keyLoc, keyToken.lexeme());
                    } else {
                        throw new ExpectedTokenException("property key", peek());
                    }

                    // Check for method shorthand {foo() {}} or property shorthand {x} or regular {x: value}
                    Node value;
                    boolean shorthand = false;
                    boolean isMethod = false;

                    if (check(TokenType.LPAREN)) {
                        // Method shorthand: {foo() {}} or {[computed]() {}} or getter/setter
                        // Getters and setters have method=false but kind="get"/"set"
                        isMethod = !kind.equals("get") && !kind.equals("set");
                        // FunctionExpression starts at '(' for all methods
                        Token methodStart = peek();

                        consume(TokenType.LPAREN, "Expected '(' for method");
                        List<Pattern> params = new ArrayList<>();

                        if (!check(TokenType.RPAREN)) {
                            do {
                                // Check for trailing comma: {foo(a, b,) {}}
                                if (check(TokenType.RPAREN)) {
                                    break;
                                }
                                if (match(TokenType.DOT_DOT_DOT)) {
                                    Token restStart = previous();
                                    Pattern argument = parsePatternBase();
                                    Token restEnd = previous();
                                    SourceLocation restLoc = createLocation(restStart, restEnd);
                                    params.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                                    if (match(TokenType.COMMA)) {
                                        throw new ParseException("ValidationError", peek(), null, "parameter list", "Rest parameter must be last");
                                    }
                                    break;
                                } else {
                                    params.add(parsePattern());
                                }
                            } while (match(TokenType.COMMA));
                        }

                        consume(TokenType.RPAREN, "Expected ')' after parameters");

                        // Parse body with proper generator and async context
                        boolean savedInGenerator = inGenerator;
                        boolean savedInAsyncContext = inAsyncContext;
                        boolean savedStrictMode = strictMode;
                        inGenerator = isGenerator;
                        inAsyncContext = isAsync;

                        // Reset strict mode for method body (unless in module mode)
                        if (!forceModuleMode) {
                            strictMode = false;
                        }

                        BlockStatement body = parseBlockStatement(true); // Object method body

                        // Check for duplicate parameters if in strict mode
                        validateNoDuplicateParameters(params, methodStart);

                        inGenerator = savedInGenerator;
                        inAsyncContext = savedInAsyncContext;
                        strictMode = savedStrictMode;

                        Token funcEnd = previous();
                        SourceLocation funcLoc = createLocation(methodStart, funcEnd);
                        value = new FunctionExpression(getStart(methodStart), getEnd(funcEnd), funcLoc, null, false, isGenerator, isAsync, params, body);
                    } else if (check(TokenType.COLON)) {
                        consume(TokenType.COLON, "Expected ':' after property key");
                        value = parseAssignment();
                    } else if (!computed && key instanceof Identifier && (check(TokenType.ASSIGN) || check(TokenType.COMMA) || check(TokenType.RBRACE))) {
                        // Shorthand property: {x} is equivalent to {x: x}
                        // Or shorthand with default: {x = defaultValue} for destructuring
                        shorthand = true;

                        if (match(TokenType.ASSIGN)) {
                            // Destructuring with default value: {x = 5}
                            Expression defaultValue = parseAssignment();
                            Token assignEnd = previous();
                            SourceLocation assignLoc = createLocation(keyToken, assignEnd);
                            // The value is an AssignmentPattern wrapping the identifier
                            value = new AssignmentPattern(getStart(keyToken), getEnd(assignEnd), assignLoc, (Identifier) key, defaultValue);
                        } else {
                            // Regular shorthand: {x}
                            value = key;
                        }
                    } else {
                        throw new ExpectedTokenException("':'", peek());
                    }

                    // For shorthand properties without default, the end is at the key token
                    Token propEnd = (shorthand && !(value instanceof AssignmentPattern)) ? keyToken : previous();
                    SourceLocation propLoc = createLocation(propStartToken, propEnd);
                    properties.add(new Property(getStart(propStartToken), getEnd(propEnd), propLoc, isMethod, shorthand, computed, key, value, kind));

                    if (!match(TokenType.COMMA)) {
                        break;
                    }
                    // Allow trailing comma: {a: 1, b: 2,}
                    if (check(TokenType.RBRACE)) {
                        break;
                    }
                }

                allowIn = savedAllowInObj;
                consume(TokenType.RBRACE, "Expected '}' after object properties");
                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                yield new ObjectExpression(getStart(startToken), getEnd(endToken), loc, properties);
            }
            case NEW -> {
                Token newToken = token;  // Save 'new' token
                advance(); // consume 'new'

                // Check for new.target
                if (match(TokenType.DOT)) {
                    Token propertyToken = peek();
                    if (check(TokenType.IDENTIFIER) && propertyToken.lexeme().equals("target")) {
                        advance(); // consume 'target'
                        SourceLocation metaLoc = createLocation(newToken, newToken);
                        SourceLocation propLoc = createLocation(propertyToken, propertyToken);
                        Identifier meta = new Identifier(getStart(newToken), getEnd(newToken), metaLoc, "new");
                        Identifier property = new Identifier(getStart(propertyToken), getEnd(propertyToken), propLoc, "target");
                        SourceLocation loc = createLocation(newToken, propertyToken);
                        yield new MetaProperty(getStart(newToken), getEnd(propertyToken), loc, meta, property);
                    } else {
                        throw new ExpectedTokenException("'target'", peek());
                    }
                }

                // Parse constructor (member expression is allowed, but not call)
                Token calleeStartToken = peek();  // Save callee start for MemberExpression
                Expression callee = parsePrimary();

                // Handle member/subscript access on constructor
                // Loop to handle chains like: new A.B[C].D[E].F
                while (true) {
                    if (match(TokenType.DOT)) {
                        if (match(TokenType.HASH)) {
                            // Private field: new obj.#method()
                            Token hashToken = previous();
                            Token propertyToken = peek();
                            if (!check(TokenType.IDENTIFIER)) {
                                throw new ExpectedTokenException("identifier", peek());
                            }
                            advance();
                            Expression property = new PrivateIdentifier(getStart(hashToken), getEnd(propertyToken), createLocation(hashToken, propertyToken), propertyToken.lexeme());
                            Token memberEnd = previous();
                            SourceLocation memberLoc = createLocation(calleeStartToken, memberEnd);
                            callee = new MemberExpression(getStart(calleeStartToken), getEnd(memberEnd), memberLoc, callee, property, false, false);
                        } else {
                            Token propertyToken = peek();
                            if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken) &&
                                !check(TokenType.TRUE) && !check(TokenType.FALSE) && !check(TokenType.NULL)) {
                                throw new ExpectedTokenException("property name", peek());
                            }
                            advance();
                            Expression property = new Identifier(getStart(propertyToken), getEnd(propertyToken), createLocation(propertyToken, propertyToken), propertyToken.lexeme());
                            Token memberEnd = previous();
                            SourceLocation memberLoc = createLocation(calleeStartToken, memberEnd);
                            callee = new MemberExpression(getStart(calleeStartToken), getEnd(memberEnd), memberLoc, callee, property, false, false);
                        }
                    } else if (match(TokenType.LBRACKET)) {
                        // Handle computed member expression
                        Expression property = parseExpression();
                        consume(TokenType.RBRACKET, "Expected ']' after computed property");
                        Token memberEnd = previous();
                        SourceLocation memberLoc = createLocation(calleeStartToken, memberEnd);
                        callee = new MemberExpression(getStart(calleeStartToken), getEnd(memberEnd), memberLoc, callee, property, true, false);
                    } else {
                        // No more member access
                        break;
                    }
                }

                // Handle tagged template: new tag`template`
                // Tagged templates are MemberExpressions, so they bind before 'new'
                if (check(TokenType.TEMPLATE_LITERAL) || check(TokenType.TEMPLATE_HEAD)) {
                    Expression template = parseTemplateLiteral();
                    Token templateEnd = previous();
                    SourceLocation taggedLoc = createLocation(calleeStartToken, templateEnd);
                    callee = new TaggedTemplateExpression(getStart(calleeStartToken), getEnd(templateEnd), taggedLoc, callee, (TemplateLiteral) template);
                }

                // Arguments are required for new expressions (even if empty)
                List<Expression> args = new ArrayList<>();
                if (check(TokenType.LPAREN)) {
                    advance();
                    if (!check(TokenType.RPAREN)) {
                        do {
                            // Check for trailing comma: new Foo(1, 2,)
                            if (check(TokenType.RPAREN)) {
                                break;
                            }
                            // Check for spread element: new Foo(...arr)
                            if (match(TokenType.DOT_DOT_DOT)) {
                                Token spreadStart = previous();
                                Expression argument = parseAssignment();
                                Token spreadEnd = previous();
                                SourceLocation spreadLoc = createLocation(spreadStart, spreadEnd);
                                args.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadLoc, argument));
                            } else {
                                args.add(parseAssignment());
                            }
                        } while (match(TokenType.COMMA));
                    }
                    consume(TokenType.RPAREN, "Expected ')' after arguments");
                }

                Token endToken = previous();
                SourceLocation loc = createLocation(newToken, endToken);
                yield new NewExpression(getStart(newToken), getEnd(endToken), loc, callee, args);
            }
            case FUNCTION -> {
                Token startToken = token;
                advance(); // consume 'function'

                // Check for generator
                boolean isGenerator = match(TokenType.STAR);

                // Optional function name (can be null for anonymous, allow of/let as names)
                Identifier id = null;
                if (check(TokenType.IDENTIFIER) || check(TokenType.OF) || check(TokenType.LET)) {
                    Token nameToken = peek();
                    advance();
                    id = new Identifier(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.lexeme());
                }

                // Set generator context before parsing parameters
                // (parameters can have default values that need correct context)
                boolean savedInGenerator = inGenerator;
                boolean savedStrictMode = strictMode;
                inGenerator = isGenerator;

                // Parse parameters
                consume(TokenType.LPAREN, "Expected '(' after function");
                List<Pattern> params = new ArrayList<>();

                if (!check(TokenType.RPAREN)) {
                    do {
                        // Check for trailing comma: function(a, b,) {}
                        if (check(TokenType.RPAREN)) {
                            break;
                        }
                        if (match(TokenType.DOT_DOT_DOT)) {
                            Token restStart = previous();
                            Pattern argument = parsePatternBase();
                            Token restEnd = previous();
                            SourceLocation restLoc = createLocation(restStart, restEnd);
                            params.add(new RestElement(getStart(restStart), getEnd(restEnd), restLoc, argument));
                            if (match(TokenType.COMMA)) {
                                throw new ParseException("ValidationError", peek(), null, "parameter list", "Rest parameter must be last");
                            }
                            break;
                        } else {
                            params.add(parsePattern());
                        }
                    } while (match(TokenType.COMMA));
                }

                consume(TokenType.RPAREN, "Expected ')' after parameters");

                // Reset strict mode for function body (unless in module mode)
                // Functions can have their own "use strict" directive
                if (!forceModuleMode) {
                    strictMode = false;
                }

                // Parse body (context already set above)
                BlockStatement body = parseBlockStatement(true); // Function expression body

                // Check for duplicate parameters if in strict mode
                validateNoDuplicateParameters(params, startToken);

                // Restore context
                inGenerator = savedInGenerator;
                strictMode = savedStrictMode;

                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                yield new FunctionExpression(getStart(startToken), getEnd(endToken), loc, id, false, isGenerator, false, params, body);
            }
            case CLASS -> {
                Token startToken = token;
                advance(); // consume 'class'

                // Optional class name (can be null for anonymous)
                Identifier id = null;
                if (check(TokenType.IDENTIFIER) && !peek().lexeme().equals("extends")) {
                    Token nameToken = peek();
                    advance();
                    id = new Identifier(getStart(nameToken), getEnd(nameToken), createLocation(nameToken, nameToken), nameToken.lexeme());
                }

                // Check for extends
                Expression superClass = null;
                if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("extends")) {
                    advance(); // consume 'extends'
                    superClass = parseConditional(); // Parse the superclass expression (can be any expression except assignment)
                }

                // Parse class body
                ClassBody body = parseClassBody();

                Token endToken = previous();
                SourceLocation loc = createLocation(startToken, endToken);
                yield new ClassExpression(getStart(startToken), getEnd(endToken), loc, id, superClass, body);
            }
            // Note: 'yield' is now tokenized as IDENTIFIER and handled in the IDENTIFIER case
            case OF -> {
                // 'of' can be used as an identifier outside of for-of loops
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Identifier(getStart(token), getEnd(token), loc, "of");
            }
            case LET -> {
                // 'let' can be used as an identifier in non-strict mode
                advance();
                SourceLocation loc = createLocation(token, token);
                yield new Identifier(getStart(token), getEnd(token), loc, "let");
            }
            default -> throw new UnexpectedTokenException(token);
        };
    }

    // Helper method to create SourceLocation from tokens
    private SourceLocation createLocation(Token start, Token end) {
        SourceLocation.Position startPos = new SourceLocation.Position(start.line(), start.column());
        // Use getPositionFromOffset for accurate end position (handles unicode escapes, etc.)
        SourceLocation.Position endPos = getPositionFromOffset(end.endPosition());
        return new SourceLocation(startPos, endPos);
    }

    // Helper method to get start byte position from token
    private int getStart(Token token) {
        return token.position();
    }

    // Helper method to get end byte position from token
    private int getEnd(Token token) {
        // Use endPosition if available (correct for tokens with escapes)
        return token.endPosition();
    }

    // Build line offset index once during construction (O(n) operation)
    private int[] buildLineOffsetIndex() {
        List<Integer> offsets = new ArrayList<>();
        offsets.add(0); // Line 1 starts at offset 0

        for (int i = 0; i < sourceLength; i++) {
            char ch = sourceBuf[i];
            // Handle all line terminators: LF, CR, CRLF, LS, PS
            if (ch == '\n') {
                offsets.add(i + 1);
            } else if (ch == '\r') {
                // Check for CRLF (skip the LF if present)
                if (i + 1 < sourceLength && sourceBuf[i + 1] == '\n') {
                    i++; // Skip the LF
                }
                offsets.add(i + 1);
            } else if (ch == '\u2028' || ch == '\u2029') {
                // Line Separator (LS) and Paragraph Separator (PS)
                offsets.add(i + 1);
            }
        }

        return offsets.stream().mapToInt(Integer::intValue).toArray();
    }

    // Helper method to compute line and column from a position in source (O(log n) operation)
    private SourceLocation.Position getPositionFromOffset(int offset) {
        // Clamp offset to valid range
        offset = Math.max(0, Math.min(offset, sourceLength));

        // Binary search to find the line
        int low = 0;
        int high = lineOffsets.length - 1;
        int line = 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            if (lineOffsets[mid] <= offset) {
                line = mid + 1; // Lines are 1-indexed
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        // Calculate column as offset from start of line
        int lineStartOffset = lineOffsets[line - 1];
        int column = offset - lineStartOffset;

        return new SourceLocation.Position(line, column);
    }

    // Helper methods

    private boolean match(TokenType... types) {
        for (TokenType type : types) {
            if (check(type)) {
                advance();
                return true;
            }
        }
        return false;
    }

    private boolean check(TokenType type) {
        if (isAtEnd()) return false;
        return peek().type() == type;
    }

    private boolean checkAhead(int offset, TokenType type) {
        int pos = current + offset;
        if (pos >= tokens.size()) return false;
        return tokens.get(pos).type() == type;
    }

    private boolean isKeyword(Token token) {
        TokenType type = token.type();
        return type == TokenType.VAR || type == TokenType.LET || type == TokenType.CONST ||
               type == TokenType.FUNCTION || type == TokenType.CLASS ||
               type == TokenType.RETURN || type == TokenType.IF || type == TokenType.ELSE ||
               type == TokenType.FOR || type == TokenType.WHILE || type == TokenType.DO ||
               type == TokenType.BREAK || type == TokenType.CONTINUE || type == TokenType.SWITCH ||
               type == TokenType.CASE || type == TokenType.DEFAULT || type == TokenType.TRY ||
               type == TokenType.CATCH || type == TokenType.FINALLY || type == TokenType.THROW ||
               type == TokenType.NEW || type == TokenType.TYPEOF || type == TokenType.VOID ||
               type == TokenType.DELETE || type == TokenType.THIS || type == TokenType.SUPER ||
               type == TokenType.IN || type == TokenType.OF || type == TokenType.INSTANCEOF ||
               type == TokenType.GET || type == TokenType.SET ||
               type == TokenType.IMPORT || type == TokenType.EXPORT || type == TokenType.WITH ||
               type == TokenType.DEBUGGER;
    }

    private Token advance() {
        if (!isAtEnd()) current++;
        return previous();
    }

    private boolean isAtEnd() {
        // Fast path: direct position check instead of peek() + type() + enum comparison
        return current >= tokens.size() - 1;
    }

    private Token peek() {
        return tokens.get(current);
    }

    private Token previous() {
        return tokens.get(current - 1);
    }

    private void consume(TokenType type, String message) {
        if (check(type)) {
            advance();
            return;
        }
        throw new ExpectedTokenException(message, peek());
    }

    /**
     * Validate that an identifier name is allowed in the current context.
     * Throws an exception if the identifier is a strict mode reserved word or
     * otherwise invalid in the current parsing context.
     */
    private void validateIdentifier(String name, Token token) {
        // Check for strict mode reserved words
        if (strictMode) {
            // Future reserved words in strict mode (ECMAScript spec section 12.1.1)
            if (name.equals("implements") || name.equals("interface") ||
                name.equals("package") || name.equals("private") ||
                name.equals("protected") || name.equals("public") ||
                name.equals("static")) {
                throw new ExpectedTokenException("'" + name + "' is a reserved identifier in strict mode", token);
            }

            // 'yield' is reserved in strict mode (outside generators)
            if (name.equals("yield") && !inGenerator) {
                throw new ExpectedTokenException("'yield' is a reserved identifier in strict mode", token);
            }
        }

        // 'yield' is always reserved inside generators (even in non-strict mode)
        if (inGenerator && name.equals("yield")) {
            throw new ExpectedTokenException("'yield' is a reserved identifier in generators", token);
        }
    }

    /**
     * Validate that an identifier is not eval or arguments when used as an assignment target.
     * In strict mode, eval and arguments cannot be assigned to.
     */
    private void validateAssignmentTarget(String name, Token token) {
        if (strictMode && (name.equals("eval") || name.equals("arguments"))) {
            throw new ExpectedTokenException("Cannot assign to '" + name + "' in strict mode", token);
        }
    }

    /**
     * Check for duplicate parameter names in a function parameter list.
     * In strict mode, duplicate parameters are not allowed.
     */
    private void validateNoDuplicateParameters(List<Pattern> params, Token functionToken) {
        if (!strictMode) {
            return; // Duplicates are only forbidden in strict mode
        }

        java.util.Set<String> paramNames = new java.util.HashSet<>();
        for (Pattern param : params) {
            collectParameterNames(param, paramNames, functionToken);
        }
    }

    /**
     * Recursively collect all parameter names from a pattern, checking for duplicates.
     */
    private void collectParameterNames(Pattern pattern, java.util.Set<String> names, Token functionToken) {
        if (pattern instanceof Identifier id) {
            String name = id.name();
            if (names.contains(name)) {
                throw new ExpectedTokenException("Duplicate parameter name '" + name + "' not allowed in strict mode", functionToken);
            }
            names.add(name);
        } else if (pattern instanceof AssignmentPattern ap) {
            collectParameterNames(ap.left(), names, functionToken);
        } else if (pattern instanceof ArrayPattern ap) {
            for (Pattern element : ap.elements()) {
                if (element != null) {
                    collectParameterNames(element, names, functionToken);
                }
            }
        } else if (pattern instanceof ObjectPattern op) {
            for (Node node : op.properties()) {
                if (node instanceof Property prop && prop.value() instanceof Pattern p) {
                    collectParameterNames(p, names, functionToken);
                } else if (node instanceof RestElement re) {
                    collectParameterNames(re.argument(), names, functionToken);
                }
            }
        } else if (pattern instanceof RestElement re) {
            collectParameterNames(re.argument(), names, functionToken);
        }
    }

    // ASI-aware semicolon consumption
    // According to ECMAScript spec, semicolons can be automatically inserted when:
    // 1. The next token is }
    // 2. The next token is EOF
    // 3. There's a line terminator between the previous token and the current token
    // 4. The next token would cause a grammatical error (e.g., seeing 'import' after var declaration)
    private void consumeSemicolon(String message) {
        if (check(TokenType.SEMICOLON)) {
            advance();
            return;
        }

        // ASI: Allow missing semicolon if next token is } or EOF
        if (check(TokenType.RBRACE) || isAtEnd()) {
            return;
        }

        // ASI: Allow missing semicolon if there's a line break before the next token
        Token prev = previous();
        Token next = peek();
        if (prev.line() < next.line()) {
            return;
        }

        // ASI: Allow missing semicolon if the next token would start a new statement
        // that cannot be part of the current statement (restricted production)
        TokenType nextType = peek().type();
        if (nextType == TokenType.IMPORT || nextType == TokenType.EXPORT ||
            nextType == TokenType.FUNCTION || nextType == TokenType.CLASS ||
            nextType == TokenType.CONST || nextType == TokenType.LET || nextType == TokenType.VAR) {
            return;
        }

        throw new ExpectedTokenException(message, peek());
    }

    public static Program parse(String source) {
        return new Parser(source).parse();
    }

    public static Program parse(String source, boolean forceModuleMode) {
        return new Parser(source, forceModuleMode).parse();
    }

    public static Program parse(String source, boolean forceModuleMode, boolean forceStrictMode) {
        return new Parser(source, forceModuleMode, forceStrictMode).parse();
    }

    /**
     * Check if source has 'module' flag in Test262 frontmatter
     */
    public static boolean hasModuleFlag(String source) {
        // Parse Test262 YAML frontmatter for module flag
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
            "/\\*---\\n([\\s\\S]*?)\\n---\\*/");
        java.util.regex.Matcher matcher = pattern.matcher(source);

        if (!matcher.find()) return false;

        String yaml = matcher.group(1);

        // Check inline format: flags: [module, async]
        // Need (?m) for ^ to match start of line
        // Use [^\]]* to match only within the brackets, not past them
        if (yaml.matches("(?sm).*^flags:\\s*\\[[^\\]]*\\bmodule\\b[^\\]]*\\].*")) {
            return true;
        }

        // Check multiline format:
        //   flags:
        //     - module
        if (yaml.matches("(?sm).*^flags:\\s*\\n(?:\\s+-\\s+\\w+\\s*\\n)*?\\s+-\\s+module\\s*(?:\\n|$).*")) {
            return true;
        }

        return false;
    }

    public static boolean isNegativeParseTest(String source) {
        // Parse Test262 YAML frontmatter for negative parse test
        // Format:
        //   negative:
        //     phase: parse
        //     type: SyntaxError
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
            "/\\*---\\n([\\s\\S]*?)\\n---\\*/");
        java.util.regex.Matcher matcher = pattern.matcher(source);

        if (!matcher.find()) return false;

        String yaml = matcher.group(1);

        // Check if negative section exists with phase: parse
        // Match "negative:" followed by any content, then "phase: parse"
        return yaml.matches("(?sm).*^negative:\\s*\\n.*?^\\s*phase:\\s*parse.*");
    }

    public static boolean hasOnlyStrictFlag(String source) {
        // Parse Test262 YAML frontmatter for onlyStrict flag
        java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
            "/\\*---\\n([\\s\\S]*?)\\n---\\*/");
        java.util.regex.Matcher matcher = pattern.matcher(source);

        if (!matcher.find()) return false;

        String yaml = matcher.group(1);

        // Check inline format: flags: [onlyStrict]
        if (yaml.matches("(?sm).*^flags:\\s*\\[[^\\]]*\\bonlyStrict\\b[^\\]]*\\].*")) {
            return true;
        }

        // Check multiline format:
        //   flags:
        //     - onlyStrict
        if (yaml.matches("(?sm).*^flags:\\s*\\n(?:\\s+-\\s+\\w+\\s*\\n)*?\\s+-\\s+onlyStrict\\s*(?:\\n|$).*")) {
            return true;
        }

        return false;
    }
}
