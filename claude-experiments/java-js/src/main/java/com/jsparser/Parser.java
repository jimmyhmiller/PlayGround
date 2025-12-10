package com.jsparser;

import com.jsparser.ast.*;

import java.util.ArrayList;
import java.util.List;

public class Parser {
    // ========================================================================
    // Binding Power Constants for Pratt Parser
    // ========================================================================
    // Higher binding power = tighter binding (higher precedence)
    // These values correspond to JavaScript operator precedence levels
    private static final int BP_NONE = 0;           // Lowest - used as minimum for top-level
    private static final int BP_COMMA = 1;          // Comma/Sequence operator
    private static final int BP_ASSIGNMENT = 2;     // Assignment (=, +=, etc.) - right-associative
    private static final int BP_TERNARY = 3;        // Conditional (? :)
    private static final int BP_NULLISH = 4;        // Nullish coalescing (??)
    private static final int BP_OR = 5;             // Logical OR (||)
    private static final int BP_AND = 6;            // Logical AND (&&)
    private static final int BP_BIT_OR = 7;         // Bitwise OR (|)
    private static final int BP_BIT_XOR = 8;        // Bitwise XOR (^)
    private static final int BP_BIT_AND = 9;        // Bitwise AND (&)
    private static final int BP_EQUALITY = 10;      // Equality (==, !=, ===, !==)
    private static final int BP_RELATIONAL = 11;    // Relational (<, <=, >, >=, instanceof, in)
    private static final int BP_SHIFT = 12;         // Shift (<<, >>, >>>)
    private static final int BP_ADDITIVE = 13;      // Additive (+, -)
    private static final int BP_MULTIPLICATIVE = 14;// Multiplicative (*, /, %)
    private static final int BP_EXPONENT = 15;      // Exponentiation (**) - right-associative
    private static final int BP_UNARY = 16;         // Prefix unary (!, -, +, ~, typeof, void, delete, ++, --)
    private static final int BP_POSTFIX = 17;       // Postfix (x++, x--, call, member access, optional chaining)

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

    // Pratt parser context - tracks outer expression start for proper location
    private int exprStartPos = 0;
    private SourceLocation.Position exprStartLoc = null;

    // Track whether current expression came from a parenthesized context (for directive detection)
    private boolean lastExpressionWasParenthesized = false;

    // Track whether we're parsing statements in a context where directives are valid
    // (Program body or function body - not loop bodies, if consequents, etc.)
    private boolean inDirectiveContext = false;

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

        this.lexer = new Lexer(source, initialStrictMode, forceModuleMode);
        this.tokens = lexer.tokenize();
        this.forceModuleMode = forceModuleMode;
        this.lineOffsets = buildLineOffsetIndex();
    }

    public Program parse() {
        List<Statement> statements = new ArrayList<>();

        // Enable directive context for program body
        inDirectiveContext = true;
        while (!isAtEnd()) {
            statements.add(parseStatement());
        }
        inDirectiveContext = false;

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
        return new Program(0, sourceLength, 1, 0, endPos.line(), endPos.column(), statements, sourceType);
    }

    /**
     * Parse a statement using switch dispatch.
     *
     * Most statement types are handled by inline switch.
     * Special cases requiring lookahead are handled explicitly:
     * - LET: Could be identifier or declaration keyword
     * - IMPORT: Could be dynamic import expression or import declaration
     * - IDENTIFIER: Could be async function, labeled statement, or expression
     */
    private Statement parseStatement() {
        Token token = peek();
        TokenType type = token.type();

        // Inline switch dispatch for performance (avoids lambda indirection)
        return switch (type) {
            // Variable declarations
            case VAR, CONST -> parseVariableDeclaration();

            // Block statement
            case LBRACE -> parseBlockStatement();

            // Control flow statements
            case IF -> parseIfStatement();
            case WHILE -> parseWhileStatement();
            case DO -> parseDoWhileStatement();
            case FOR -> parseForStatement();
            case SWITCH -> parseSwitchStatement();

            // Jump statements
            case RETURN -> parseReturnStatement();
            case BREAK -> parseBreakStatement();
            case CONTINUE -> parseContinueStatement();
            case THROW -> parseThrowStatement();

            // Exception handling
            case TRY -> parseTryStatement();

            // Other statements
            case WITH -> parseWithStatement();
            case DEBUGGER -> parseDebuggerStatement();
            case SEMICOLON -> parseEmptyStatement();

            // Declarations
            case FUNCTION -> parseFunctionDeclaration(false);
            case CLASS -> parseClassDeclaration();

            // Module declarations
            case EXPORT -> parseExportDeclaration();

            // Special cases requiring lookahead
            case LET -> parseLetStatementOrExpression(token);
            case IMPORT -> parseImportStatementOrExpression(token);
            case IDENTIFIER -> parseIdentifierStatement(token);

            // Default: Parse as expression statement
            default -> parseExpressionStatement(token);
        };
    }

    /**
     * Parse a nested statement (e.g., for-loop body, if consequent).
     * Clears directive context since directives are only valid at the top level of Program/function body.
     */
    private Statement parseNestedStatement() {
        boolean savedDirectiveContext = inDirectiveContext;
        inDirectiveContext = false;
        try {
            return parseStatement();
        } finally {
            inDirectiveContext = savedDirectiveContext;
        }
    }

    /**
     * Handle LET which can be either a declaration keyword or an identifier.
     * - let x = 1     → VariableDeclaration
     * - let = 5       → ExpressionStatement (let as identifier)
     * - let[0] = 1    → ExpressionStatement (let as identifier)
     * - let\n x       → ExpressionStatement with ASI
     */
    private Statement parseLetStatementOrExpression(Token token) {
        Token letToken = peek();

        // If followed by = (not part of destructuring), it's an identifier being assigned
        if (checkAhead(1, TokenType.ASSIGN)) {
            return parseExpressionStatement(token);
        }

        // If followed by line terminator, ASI applies and 'let' is an identifier
        if (current + 1 < tokens.size() && letToken.line() != tokens.get(current + 1).line()) {
            return parseExpressionStatement(token);
        }

        // Parse as variable declaration (includes let x, let [x], let {x}, etc.)
        return parseVariableDeclaration();
    }

    /**
     * Handle IMPORT which can be either a declaration or an expression.
     * - import { foo } from 'module'  → ImportDeclaration
     * - import('./module.js')         → ExpressionStatement (dynamic import)
     * - import.meta                   → ExpressionStatement (import.meta)
     */
    private Statement parseImportStatementOrExpression(Token token) {
        if (current + 1 < tokens.size()) {
            TokenType nextType = tokens.get(current + 1).type();
            if (nextType == TokenType.LPAREN || nextType == TokenType.DOT) {
                // Dynamic import or import.meta - parse as expression statement
                return parseExpressionStatement(token);
            }
        }
        // Import declaration
        return parseImportDeclaration();
    }

    /**
     * Handle IDENTIFIER which can be async function, labeled statement, or expression.
     * - async function foo() {}  → FunctionDeclaration
     * - label: statement         → LabeledStatement
     * - expression;              → ExpressionStatement
     */
    private Statement parseIdentifierStatement(Token token) {
        // Check for async function declaration
        // No line terminator is allowed between async and function
        if (token.lexeme().equals("async") && current + 1 < tokens.size() &&
            tokens.get(current + 1).type() == TokenType.FUNCTION &&
            tokens.get(current).line() == tokens.get(current + 1).line()) {
            return parseFunctionDeclaration(true);
        }

        // Parse as expression first
        Expression expr = parseExpression();

        // Check if this is a labeled statement (identifier followed by colon)
        if (expr instanceof Identifier id && check(TokenType.COLON)) {
            advance(); // consume ':'
            Statement labeledBody = parseNestedStatement();
            Token endToken = previous();
            return new LabeledStatement(getStart(token), getEnd(endToken), token.line(), token.column(), endToken.endLine(), endToken.endColumn(), id, labeledBody);
        }

        // Regular expression statement
        consumeSemicolon("Expected ';' after expression");
        Token endToken = previous();
        return new ExpressionStatement(getStart(token), getEnd(endToken), token.line(), token.column(), endToken.endLine(), endToken.endColumn(), expr);
    }

    /**
     * Parse an expression statement (default case for unknown statement types).
     * Also handles labeled statements if the expression is an identifier followed by colon.
     */
    private Statement parseExpressionStatement(Token token) {
        // Reset parenthesized flag before parsing expression
        lastExpressionWasParenthesized = false;
        Expression expr = parseExpression();
        // Capture whether the expression was parenthesized (for directive detection)
        boolean wasParenthesized = lastExpressionWasParenthesized;

        // Check if this is a labeled statement (identifier followed by colon)
        if (expr instanceof Identifier id && check(TokenType.COLON)) {
            advance(); // consume ':'
            Statement labeledBody = parseNestedStatement();
            Token endToken = previous();
            return new LabeledStatement(getStart(token), getEnd(endToken), token.line(), token.column(), endToken.endLine(), endToken.endColumn(), id, labeledBody);
        }

        consumeSemicolon("Expected ';' after expression");
        Token endToken = previous();

        // If expression was parenthesized and we're in a directive context, mark it so it won't be treated as directive
        // Only use the empty directive marker in directive contexts (Program/function body)
        String directive = (inDirectiveContext && wasParenthesized) ? "" : null;
        return new ExpressionStatement(getStart(token), getEnd(endToken), token.line(), token.column(), endToken.endLine(), endToken.endColumn(), expr, directive);
    }

    private WhileStatement parseWhileStatement() {
        Token startToken = peek();
        advance(); // consume 'while'

        consume(TokenType.LPAREN, "Expected '(' after 'while'");
        Expression test = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after while condition");

        Statement body = parseNestedStatement();

        Token endToken = previous();
        return new WhileStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), test, body);
    }

    private DoWhileStatement parseDoWhileStatement() {
        Token startToken = peek();
        advance(); // consume 'do'

        Statement body = parseNestedStatement();

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
        return new DoWhileStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), body, test);
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
                        initExpr = parseExpr(BP_ASSIGNMENT);
                    }

                    Token declaratorEnd = previous();

                    int declaratorStart = getStart(patternStart);
                    int declaratorEndPos = getEnd(declaratorEnd);

                    declarators.add(new VariableDeclarator(declaratorStart, declaratorEndPos, patternStart.line(), patternStart.column(), declaratorEnd.endLine(), declaratorEnd.endColumn(), pattern, initExpr));

                } while (match(TokenType.COMMA));

                Token endToken = previous();
                int declStart = getStart(kindToken);
                int declEnd = getEnd(endToken);
                initOrLeft = new VariableDeclaration(declStart, declEnd, kindToken.line(), kindToken.column(), endToken.endLine(), endToken.endColumn(), declarators, kind);
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
            Statement body = parseNestedStatement();
            Token endToken = previous();
            return new ForInStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), initOrLeft, right, body);
        } else if (isOfKeyword) {
            advance(); // consume 'of'
            // Convert left to pattern if it's an expression (for destructuring)
            if (initOrLeft instanceof Expression) {
                initOrLeft = convertToPatternIfNeeded(initOrLeft);
            }
            Expression right = parseExpression();
            consume(TokenType.RPAREN, "Expected ')' after for-of");
            Statement body = parseNestedStatement();
            Token endToken = previous();
            return new ForOfStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), isAwait, initOrLeft, right, body);
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

        Statement body = parseNestedStatement();

        Token endToken = previous();
        return new ForStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), initOrLeft, test, update, body);
    }

    private IfStatement parseIfStatement() {
        Token startToken = peek();
        advance(); // consume 'if'

        consume(TokenType.LPAREN, "Expected '(' after 'if'");
        Expression test = parseExpression();
        consume(TokenType.RPAREN, "Expected ')' after if condition");

        Statement consequent = parseNestedStatement();

        Statement alternate = null;
        if (match(TokenType.ELSE)) {
            alternate = parseNestedStatement();
        }

        Token endToken = previous();
        return new IfStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), test, consequent, alternate);
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
        return new ReturnStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), argument);
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
            label = new Identifier(getStart(labelToken), getEnd(labelToken), labelToken.line(), labelToken.column(), labelToken.endLine(), labelToken.endColumn(), labelToken.lexeme());
        }

        consumeSemicolon("Expected ';' after break statement");
        Token endToken = previous();
        return new BreakStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), label);
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
            label = new Identifier(getStart(labelToken), getEnd(labelToken), labelToken.line(), labelToken.column(), labelToken.endLine(), labelToken.endColumn(), labelToken.lexeme());
        }

        consumeSemicolon("Expected ';' after continue statement");
        Token endToken = previous();
        return new ContinueStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), label);
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
                    consequent.add(parseNestedStatement());
                }

                Token caseEnd = previous();
                cases.add(new SwitchCase(getStart(caseStart), getEnd(caseEnd), caseStart.line(), caseStart.column(), caseEnd.endLine(), caseEnd.endColumn(), test, consequent));

            } else if (match(TokenType.DEFAULT)) {
                // default: consequent
                consume(TokenType.COLON, "Expected ':' after 'default'");

                // Parse consequent statements
                List<Statement> consequent = new ArrayList<>();
                while (!check(TokenType.CASE) && !check(TokenType.DEFAULT) && !check(TokenType.RBRACE) && !isAtEnd()) {
                    consequent.add(parseNestedStatement());
                }

                Token caseEnd = previous();
                cases.add(new SwitchCase(getStart(caseStart), getEnd(caseEnd), caseStart.line(), caseStart.column(), caseEnd.endLine(), caseEnd.endColumn(), null, consequent));

            } else {
                throw new ExpectedTokenException("'case' or 'default' in switch body", peek());
            }
        }

        consume(TokenType.RBRACE, "Expected '}' after switch body");
        Token endToken = previous();
        return new SwitchStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), discriminant, cases);
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
        return new ThrowStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), argument);
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
            handler = new CatchClause(getStart(catchStart), getEnd(catchEnd), catchStart.line(), catchStart.column(), catchEnd.endLine(), catchEnd.endColumn(), param, body);
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
        return new TryStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), block, handler, finalizer);
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

        Statement body = parseNestedStatement();

        Token endToken = previous();
        return new WithStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), object, body);
    }

    private DebuggerStatement parseDebuggerStatement() {
        Token startToken = peek();
        advance(); // consume 'debugger'
        consumeSemicolon("Expected ';' after debugger statement");
        Token endToken = previous();
        return new DebuggerStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn());
    }

    private EmptyStatement parseEmptyStatement() {
        Token startToken = peek();
        advance(); // consume ';'
        return new EmptyStatement(getStart(startToken), getEnd(startToken), startToken.line(), startToken.column(), startToken.endLine(), startToken.endColumn());
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
            id = new Identifier(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
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
                    params.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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
        return new FunctionDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), id, false, isGenerator, isAsync, params, body);
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
            id = new Identifier(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
        } else if (!allowAnonymous && !(check(TokenType.IDENTIFIER) && peek().lexeme().equals("extends"))) {
            throw new ExpectedTokenException("class name", peek());
        }

        // Check for extends
        Expression superClass = null;
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("extends")) {
            advance(); // consume 'extends'
            superClass = parseExpr(BP_TERNARY + 1); // Parse the superclass expression (can be any expression except assignment or ternary)
        }

        // Parse class body
        ClassBody body = parseClassBody();

        Token endToken = previous();
        return new ClassDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), id, superClass, body);
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
                            blockBody.add(parseNestedStatement());
                        }

                        Token blockEnd = peek();
                        consume(TokenType.RBRACE, "Expected '}' after static block body");

                        bodyElements.add(new StaticBlock(getStart(blockStart), getEnd(blockEnd), blockStart.line(), blockStart.column(), blockEnd.endLine(), blockEnd.endColumn(), blockBody));
                        continue;
                    }
                }
            }

            // Check for 'async' keyword (but not if it's a method named "async" or field named "async")
            boolean isAsync = false;
            if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
                // Look ahead to see if this is "async()" (method name), "async;" or "async =" (field),
                // or "async something" (modifier)
                if (current + 1 < tokens.size()) {
                    Token currentToken = peek();
                    Token nextToken = tokens.get(current + 1);
                    TokenType nextType = nextToken.type();

                    // async is NOT a modifier if followed by ( ; = or if there's a line break (ASI)
                    if (nextType != TokenType.LPAREN && nextType != TokenType.SEMICOLON &&
                        nextType != TokenType.ASSIGN) {
                        // Check for ASI: if there's a line break after "async", it's a field, not a modifier
                        boolean hasLineBreak = nextToken.line() > currentToken.line();
                        if (!hasLineBreak) {
                            advance();
                            isAsync = true;
                        }
                    }
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
                key = new PrivateIdentifier(getStart(hashToken), getEnd(keyToken), hashToken.line(), hashToken.column(), keyToken.endLine(), keyToken.endColumn(), keyToken.lexeme());
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
                    // value should be the decimal bigint WITHOUT 'n' to match Acorn's JSON serialization
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), bigintValue, keyLexeme, null, bigintValue);
                } else {
                    Object literalValue = keyToken.literal();
                    if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                        literalValue = null;
                    }
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), literalValue, keyLexeme);
                }
            } else if (check(TokenType.DOT) && current + 1 < tokens.size() && tokens.get(current + 1).type() == TokenType.NUMBER) {
                // Handle .1 as a numeric literal (0.1)
                Token dotToken = peek();
                advance(); // consume DOT
                Token numToken = peek();
                advance(); // consume NUMBER
                String lexeme = "." + numToken.lexeme();
                double value = Double.parseDouble(lexeme);
                key = new Literal(getStart(dotToken), getEnd(numToken), dotToken.line(), dotToken.column(), numToken.endLine(), numToken.endColumn(), value, lexeme);
            } else if (check(TokenType.IDENTIFIER) || check(TokenType.GET) || check(TokenType.SET) ||
                       check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL) ||
                       isKeyword(peek())) {
                // Regular identifier, get/set, keyword, or boolean/null literal as property name
                advance();
                key = new Identifier(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), keyToken.lexeme());
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
                        params.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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
                FunctionExpression fnExpr = new FunctionExpression(
                    getStart(fnStart),
                    getEnd(methodEnd),
                    fnStart.line(), fnStart.column(), methodEnd.endLine(), methodEnd.endColumn(),
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
                    memberStart.line(), memberStart.column(), methodEnd.endLine(), methodEnd.endColumn(),
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
                    value = parseExpr(BP_ASSIGNMENT);
                    inClassFieldInitializer = oldInClassFieldInitializer;
                    inAsyncContext = oldInAsyncContext;
                }

                // Consume optional semicolon
                match(TokenType.SEMICOLON);

                Token propertyEnd = previous();
                PropertyDefinition property = new PropertyDefinition(
                    getStart(memberStart),
                    getEnd(propertyEnd),
                    memberStart.line(), memberStart.column(), propertyEnd.endLine(), propertyEnd.endColumn(),
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
        return new ClassBody(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), bodyElements);
    }

    private ImportDeclaration parseImportDeclaration() {
        Token startToken = peek();
        advance(); // consume 'import'

        List<Node> specifiers = new ArrayList<>();

        // Check for import 'module' (side-effect import)
        if (check(TokenType.STRING)) {
            Token sourceToken = advance();
            Literal source = new Literal(getStart(sourceToken), getEnd(sourceToken), sourceToken.line(), sourceToken.column(), sourceToken.endLine(), sourceToken.endColumn(), sourceToken.literal(), sourceToken.lexeme());
            List<ImportAttribute> attributes = parseImportAttributes();
            consumeSemicolon("Expected ';' after import");
            Token endToken = previous();
            return new ImportDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), specifiers, source, attributes);
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
                Identifier local = new Identifier(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), localToken.lexeme());
                specifiers.add(new ImportDefaultSpecifier(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), local));

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
            Identifier local = new Identifier(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), localToken.lexeme());
            specifiers.add(new ImportNamespaceSpecifier(getStart(starToken), getEnd(localToken), starToken.line(), starToken.column(), localToken.endLine(), localToken.endColumn(), local));
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
                        imported = new Literal(getStart(importedToken), getEnd(importedToken), importedToken.line(), importedToken.column(), importedToken.endLine(), importedToken.endColumn(), importedToken.literal(), importedToken.lexeme());
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
                        local = new Identifier(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), localToken.lexeme());
                    } else if (check(TokenType.IDENTIFIER) || isKeyword(importedToken)) {
                        advance();
                        Identifier importedId = new Identifier(getStart(importedToken), getEnd(importedToken), importedToken.line(), importedToken.column(), importedToken.endLine(), importedToken.endColumn(), importedToken.lexeme());
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
                            local = new Identifier(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), localToken.lexeme());
                        }
                    } else {
                        throw new ExpectedTokenException("identifier or string in import specifier", peek());
                    }

                    Token prevToken = previous();
                    specifiers.add(new ImportSpecifier(getStart(importedToken), getEnd(prevToken), importedToken.line(), importedToken.column(), prevToken.endLine(), prevToken.endColumn(), imported, local));

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
        Literal source = new Literal(getStart(sourceToken), getEnd(sourceToken), sourceToken.line(), sourceToken.column(), sourceToken.endLine(), sourceToken.endColumn(), sourceToken.literal(), sourceToken.lexeme());

        // Parse import attributes: with { type: 'json' }
        List<ImportAttribute> attributes = parseImportAttributes();

        consumeSemicolon("Expected ';' after import");
        Token endToken = previous();
        return new ImportDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), specifiers, source, attributes);
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
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), keyToken.literal(), keyToken.lexeme());
                } else if (check(TokenType.IDENTIFIER) || isKeyword(keyToken)) {
                    advance();
                    key = new Identifier(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), keyToken.lexeme());
                } else {
                    throw new ExpectedTokenException("identifier or string in import attribute", peek());
                }

                consume(TokenType.COLON, "Expected ':' after import attribute key");

                Token valueToken = peek();
                if (!check(TokenType.STRING)) {
                    throw new ExpectedTokenException("string value in import attribute", peek());
                }
                advance();
                Literal value = new Literal(getStart(valueToken), getEnd(valueToken), valueToken.line(), valueToken.column(), valueToken.endLine(), valueToken.endColumn(), valueToken.literal(), valueToken.lexeme());

                Token attrEnd = previous();
                attributes.add(new ImportAttribute(getStart(keyToken), getEnd(attrEnd), keyToken.line(), keyToken.column(), attrEnd.endLine(), attrEnd.endColumn(), key, value));

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
                    declaration = parseExpr(BP_ASSIGNMENT);
                    consumeSemicolon("Expected ';' after export default");
                }
            } else {
                // Expression
                declaration = parseExpr(BP_ASSIGNMENT);
                consumeSemicolon("Expected ';' after export default");
            }

            Token endToken = previous();
            return new ExportDefaultDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), declaration);
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
                    exported = new Literal(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.literal(), nameToken.lexeme());
                } else if (check(TokenType.IDENTIFIER) || isKeyword(nameToken) ||
                           check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL)) {
                    advance();
                    exported = new Identifier(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
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
            Literal source = new Literal(getStart(sourceToken), getEnd(sourceToken), sourceToken.line(), sourceToken.column(), sourceToken.endLine(), sourceToken.endColumn(), sourceToken.literal(), sourceToken.lexeme());

            List<ImportAttribute> attributes = parseImportAttributes();
            consumeSemicolon("Expected ';' after export");
            Token endToken = previous();
            return new ExportAllDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), source, exported, attributes);
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
                        local = new Literal(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), localToken.literal(), localToken.lexeme());
                    } else if (check(TokenType.IDENTIFIER) || isKeyword(localToken)) {
                        advance();
                        local = new Identifier(getStart(localToken), getEnd(localToken), localToken.line(), localToken.column(), localToken.endLine(), localToken.endColumn(), localToken.lexeme());
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
                            exported = new Literal(getStart(exportedToken), getEnd(exportedToken), exportedToken.line(), exportedToken.column(), exportedToken.endLine(), exportedToken.endColumn(), exportedToken.literal(), exportedToken.lexeme());
                        } else if (check(TokenType.IDENTIFIER) || isKeyword(exportedToken) ||
                                   check(TokenType.TRUE) || check(TokenType.FALSE) || check(TokenType.NULL)) {
                            advance();
                            exported = new Identifier(getStart(exportedToken), getEnd(exportedToken), exportedToken.line(), exportedToken.column(), exportedToken.endLine(), exportedToken.endColumn(), exportedToken.lexeme());
                        } else {
                            throw new ExpectedTokenException("identifier or string after 'as'", peek());
                        }
                    }

                    Token prevToken = previous();
                    specifiers.add(new ExportSpecifier(getStart(localToken), getEnd(prevToken), localToken.line(), localToken.column(), prevToken.endLine(), prevToken.endColumn(), local, exported));

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
                source = new Literal(getStart(sourceToken), getEnd(sourceToken), sourceToken.line(), sourceToken.column(), sourceToken.endLine(), sourceToken.endColumn(), sourceToken.literal(), sourceToken.lexeme());
                attributes = parseImportAttributes();
            }

            consumeSemicolon("Expected ';' after export");
            Token endToken = previous();
            return new ExportNamedDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), null, specifiers, source, attributes);
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
        return new ExportNamedDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), declaration, new ArrayList<>(), null, new ArrayList<>());
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

        // Enable directive context for function bodies
        boolean oldDirectiveContext = inDirectiveContext;
        if (isFunctionBody) {
            inDirectiveContext = true;
        } else {
            inDirectiveContext = false;
        }

        List<Statement> statements = new ArrayList<>();
        while (!check(TokenType.RBRACE) && !isAtEnd()) {
            statements.add(parseStatement());
        }

        // Restore directive context
        inDirectiveContext = oldDirectiveContext;

        // Process directive prologue only for function bodies
        if (isFunctionBody) {
            statements = processDirectives(statements);
        }

        consume(TokenType.RBRACE, "Expected '}'");
        Token endToken = previous();

        // Restore allowIn
        allowIn = oldAllowIn;

        return new BlockStatement(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), statements);
    }

    // Process directive prologue: add directive property to string literal expression statements at the start
    private List<Statement> processDirectives(List<Statement> statements) {
        List<Statement> processed = new ArrayList<>();
        boolean inPrologue = true;

        for (Statement stmt : statements) {
            // First, handle parenthesized expressions (marked with empty directive)
            // These need their empty directive cleared, regardless of prologue state
            if (stmt instanceof ExpressionStatement exprStmt && exprStmt.directive() != null && exprStmt.directive().isEmpty()) {
                // Clear the empty directive marker and add as regular statement
                processed.add(new ExpressionStatement(
                    exprStmt.start(),
                    exprStmt.end(),
                    exprStmt.startLine(),
                    exprStmt.startCol(),
                    exprStmt.endLine(),
                    exprStmt.endCol(),
                    exprStmt.expression(),
                    null
                ));
                // Parenthesized expression ends the prologue
                inPrologue = false;
                continue;
            }

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
                        exprStmt.startLine(),
                        exprStmt.startCol(),
                        exprStmt.endLine(),
                        exprStmt.endCol(),
                        exprStmt.expression(),
                        directiveValue
                    ));
                    continue;
                } else {
                    // Non-string-literal expression ends the prologue
                    inPrologue = false;
                }
            } else if (inPrologue) {
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
                init = parseExpr(BP_ASSIGNMENT);
            }

            Token declaratorEnd = previous();

            int declaratorStart = getStart(patternStart);
            int declaratorEndPos = getEnd(declaratorEnd);

            declarators.add(new VariableDeclarator(declaratorStart, declaratorEndPos, patternStart.line(), patternStart.column(), declaratorEnd.endLine(), declaratorEnd.endColumn(), pattern, init));

        } while (match(TokenType.COMMA));

        consumeSemicolon("Expected ';' after variable declaration");

        Token endToken = previous();
        return new VariableDeclaration(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), declarators, kind);
    }

    private Pattern parsePattern() {
        return parsePatternWithDefault();
    }

    private Pattern parsePatternWithDefault() {
        Token startToken = peek();
        Pattern pattern = parsePatternBase();

        // Check for default value: pattern = defaultValue
        if (match(TokenType.ASSIGN)) {
            // Inside destructuring patterns, 'in' is always the operator, not for-in keyword
            // For example: for (let [x = 'a' in {}] = []; ...) - the 'in' is an operator
            boolean savedAllowIn = allowIn;
            allowIn = true;
            Expression defaultValue = parseExpr(BP_ASSIGNMENT);
            allowIn = savedAllowIn;
            Token endToken = previous();
            return new AssignmentPattern(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), pattern, defaultValue);
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
            return new Identifier(getStart(idToken), getEnd(idToken), idToken.line(), idToken.column(), idToken.endLine(), idToken.endColumn(), idToken.lexeme());
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
                properties.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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
                key = parseExpr(BP_ASSIGNMENT);
                consume(TokenType.RBRACKET, "Expected ']' after computed property");
            } else if (check(TokenType.STRING) || check(TokenType.NUMBER)) {
                // Literal key (string or numeric)
                Token keyToken = advance();
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
                    // value should be the decimal bigint WITHOUT 'n' to match Acorn's JSON serialization
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), bigintValue, keyLexeme, null, bigintValue);
                } else {
                    Object literalValue = keyToken.literal();
                    if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                        literalValue = null;
                    }
                    key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), literalValue, keyLexeme);
                }
            } else {
                // Allow identifiers, keywords, and boolean/null literals as property names
                if (!check(TokenType.IDENTIFIER) && !isKeyword(peek()) &&
                    !check(TokenType.TRUE) && !check(TokenType.FALSE) && !check(TokenType.NULL)) {
                    throw new ExpectedTokenException("property name", peek());
                }
                Token keyToken = advance();
                key = new Identifier(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), keyToken.lexeme());
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
                        // Inside destructuring patterns, 'in' is always the operator, not for-in keyword
                        boolean savedAllowIn = allowIn;
                        allowIn = true;
                        Expression defaultValue = parseExpr(BP_ASSIGNMENT);
                        allowIn = savedAllowIn;
                        Token assignEnd = previous();
                        value = new AssignmentPattern(getStart(propStart), getEnd(assignEnd), propStart.line(), propStart.column(), assignEnd.endLine(), assignEnd.endColumn(), id, defaultValue);
                    }
                } else {
                    throw new ParseException("ValidationError", peek(), null, "object pattern property", "Shorthand property must have identifier key");
                }
            }

            Token propEnd = previous();
            properties.add(new Property(
                getStart(propStart), getEnd(propEnd), propStart.line(), propStart.column(), propEnd.endLine(), propEnd.endColumn(),
                false, shorthand, computed, key, value, "init"
            ));

            if (!match(TokenType.COMMA)) {
                break;
            }
        }

        consume(TokenType.RBRACE, "Expected '}' after object pattern");
        Token endToken = previous();
        return new ObjectPattern(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), properties);
    }

    private ArrayPattern parseArrayPattern(Token startToken) {
        List<Pattern> elements = new ArrayList<>();

        while (!check(TokenType.RBRACKET) && !isAtEnd()) {
            if (match(TokenType.DOT_DOT_DOT)) {
                // Rest element: ...rest (no default value allowed)
                Token restStart = previous();
                Pattern argument = parsePatternBase();
                Token restEnd = previous();
                elements.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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
        return new ArrayPattern(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), elements);
    }

    // Expression parsing with precedence climbing
    // expression -> sequence
    // sequence -> assignment ( "," assignment )*
    // assignment -> conditional ( "=" assignment )?
    private Expression parseExpression() {
        return parseExpr(BP_COMMA);
    }

    // ========================================================================
    // Unified Pratt Parser - parseExpr(int minBp)
    // ========================================================================
    // This is the core of the Pratt parsing algorithm.
    // It parses expressions with binding power >= minBp.
    //
    // The algorithm:
    // 1. Parse a prefix expression (NUD)
    // 2. While the next token has binding power >= minBp, parse infix (LED)
    //
    // This collapses the 8-method chain:
    //   parseExpression → parseSequence → parseAssignment → parseConditional
    //   → parseBinaryExpression → parseUnary → parsePostfix → parsePrimary
    // into a single unified method with flat call stack.

    private Expression parseExpr(int minBp) {
        Token startToken = peek();
        TokenType startType = startToken.type();
        Expression left = null;

        // Handle contextual keywords: yield and await
        // These have assignment-level precedence and need special handling
        if (startType == TokenType.IDENTIFIER) {
            String lexeme = startToken.lexeme();

            // Yield expression
            if (inGenerator && lexeme.equals("yield") &&
                !checkAhead(1, TokenType.ASSIGN) && !checkAhead(1, TokenType.PLUS_ASSIGN) &&
                !checkAhead(1, TokenType.MINUS_ASSIGN) && !checkAhead(1, TokenType.STAR_ASSIGN) &&
                !checkAhead(1, TokenType.SLASH_ASSIGN) && !checkAhead(1, TokenType.PERCENT_ASSIGN)) {
                left = parseYieldExpr();
            }

            // Await expression (complex logic in shouldParseAwait)
            if (left == null && lexeme.equals("await") && shouldParseAwait()) {
                left = parseAwaitExpr();
            }

            // Quick check for arrow function: id => or async ...
            if (left == null && current + 1 < tokens.size()) {
                TokenType nextType = tokens.get(current + 1).type();
                if (nextType == TokenType.ARROW) {
                    // Simple arrow: id =>
                    left = tryParseArrowFunction(startToken);
                } else if (lexeme.equals("async") && startToken.line() == tokens.get(current + 1).line()) {
                    // Potential async arrow
                    left = tryParseArrowFunction(startToken);
                }
            }
        } else if (startType == TokenType.OF || startType == TokenType.LET) {
            // of => or let => (rare but valid)
            if (current + 1 < tokens.size() && tokens.get(current + 1).type() == TokenType.ARROW) {
                left = tryParseArrowFunction(startToken);
            }
        } else if (startType == TokenType.LPAREN) {
            // Check for arrow: (params) =>
            // Only call tryParseArrowFunction if it looks like it could be arrow params
            int savedCurrent = current;
            advance(); // consume (
            boolean isArrow = isArrowFunctionParameters();
            current = savedCurrent;

            if (isArrow) {
                left = tryParseArrowFunction(startToken);
            }
        }

        // Prefix handling (NUD - Null Denotation) - only if no special case handled above
        // Inlined switch instead of map lookup + lambda for better performance
        if (left == null) {
            Token token = peek();
            advance();
            Token prevToken = previous();
            left = switch (token.type()) {
                // Literals
                case NUMBER -> prefixNumber(this, prevToken);
                case STRING -> prefixString(this, prevToken);
                case TRUE -> prefixTrue(this, prevToken);
                case FALSE -> prefixFalse(this, prevToken);
                case NULL -> prefixNull(this, prevToken);
                case REGEX -> prefixRegex(this, prevToken);

                // Identifiers and keywords
                case IDENTIFIER -> prefixIdentifier(this, prevToken);
                case LET, OF -> prefixIdentifier(this, prevToken); // let and of are valid identifiers in non-strict mode
                case THIS -> prefixThis(this, prevToken);
                case SUPER -> prefixSuper(this, prevToken);

                // Grouping and collections
                case LPAREN -> prefixGroupedOrArrow(this, prevToken);
                case LBRACKET -> prefixArray(this, prevToken);
                case LBRACE -> prefixObject(this, prevToken);

                // Function/class expressions
                case FUNCTION -> prefixFunction(this, prevToken);
                case CLASS -> prefixClass(this, prevToken);
                case NEW -> prefixNew(this, prevToken);

                // Unary operators
                case BANG, MINUS, PLUS, TILDE, TYPEOF, VOID, DELETE -> prefixUnary(this, prevToken);
                case INCREMENT, DECREMENT -> prefixUpdate(this, prevToken);

                // Templates
                case TEMPLATE_LITERAL, TEMPLATE_HEAD -> prefixTemplate(this, prevToken);

                // Special
                case IMPORT -> prefixImport(this, prevToken);
                case HASH -> prefixPrivateIdentifier(this, prevToken);

                default -> throw new UnexpectedTokenException(token, "expression");
            };
        }

        // ========================================================================
        // Infix loop (LED - Left Denotation) - inlined from continueInfix
        // ========================================================================

        // Store the outer expression start - we need these local vars because handlers may recursively call parseExpr
        int outerStartPos = getStart(startToken);
        SourceLocation.Position outerStartLoc = new SourceLocation.Position(startToken.line(), startToken.column());

        // Track optional chaining for ChainExpression wrapping
        boolean hasOptionalChaining = false;
        Token chainEndToken = null; // Token where the chain portion ends

        // Infix/Postfix loop
        while (true) {
            Token token = peek();

            // Special case: postfix ++/-- with line terminator restriction
            if ((token.type() == TokenType.INCREMENT || token.type() == TokenType.DECREMENT)) {
                Token prevToken = previous();
                if (prevToken.line() < token.line()) {
                    // Line terminator before postfix operator - stop
                    break;
                }
                // Handle as postfix update
                if (BP_POSTFIX >= minBp) {
                    // If we have optional chaining and this is a non-chain operator, wrap first
                    if (hasOptionalChaining) {
                        chainEndToken = previous();
                        left = new ChainExpression(outerStartPos, getEnd(chainEndToken), startToken.line(), startToken.column(), chainEndToken.endLine(), chainEndToken.endColumn(), left);
                        hasOptionalChaining = false;
                        // Update start positions for the outer expression
                        outerStartPos = getStart(startToken);
                        outerStartLoc = new SourceLocation.Position(startToken.line(), startToken.column());
                    }
                    advance();
                    Token endToken = previous();
                    left = new UpdateExpression(outerStartPos, getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), token.lexeme(), false, left);
                    continue;
                }
                break;
            }

            // Get binding power for this token - inlined for performance
            TokenType tt = token.type();
            int lbp = switch (tt) {
                case COMMA -> BP_COMMA;
                case ASSIGN, PLUS_ASSIGN, MINUS_ASSIGN, STAR_ASSIGN, SLASH_ASSIGN, PERCENT_ASSIGN,
                     STAR_STAR_ASSIGN, LEFT_SHIFT_ASSIGN, RIGHT_SHIFT_ASSIGN, UNSIGNED_RIGHT_SHIFT_ASSIGN,
                     BIT_AND_ASSIGN, BIT_OR_ASSIGN, BIT_XOR_ASSIGN, AND_ASSIGN, OR_ASSIGN, QUESTION_QUESTION_ASSIGN -> BP_ASSIGNMENT;
                case QUESTION -> BP_TERNARY;
                case QUESTION_QUESTION -> BP_NULLISH;
                case OR -> BP_OR;
                case AND -> BP_AND;
                case BIT_OR -> BP_BIT_OR;
                case BIT_XOR -> BP_BIT_XOR;
                case BIT_AND -> BP_BIT_AND;
                case EQ, NE, EQ_STRICT, NE_STRICT -> BP_EQUALITY;
                case LT, LE, GT, GE, INSTANCEOF, IN -> BP_RELATIONAL;
                case LEFT_SHIFT, RIGHT_SHIFT, UNSIGNED_RIGHT_SHIFT -> BP_SHIFT;
                case PLUS, MINUS -> BP_ADDITIVE;
                case STAR, SLASH, PERCENT -> BP_MULTIPLICATIVE;
                case STAR_STAR -> BP_EXPONENT;
                case DOT, QUESTION_DOT, LBRACKET, LPAREN, TEMPLATE_LITERAL, TEMPLATE_HEAD -> BP_POSTFIX;
                default -> -1; // Not an infix operator
            };

            if (lbp < 0 || lbp < minBp) {
                break;
            }

            // Special case: 'in' operator respects allowIn flag
            if (tt == TokenType.IN && !allowIn) {
                break;
            }

            // ASI rules for ( and [:
            // - If previous token is }, ASI applies before ( (but not [)
            // - In class field context, ASI applies before [
            if (previous().line() < peek().line()) {
                Token prevToken = previous();
                if (tt == TokenType.LPAREN && prevToken.type() == TokenType.RBRACE) {
                    // ASI before ( when previous is } (e.g., arrow function body)
                    break;
                }
                if (tt == TokenType.LBRACKET && inClassFieldInitializer) {
                    // ASI before [ in class field context
                    break;
                }
            }

            // Track optional chaining - only ?. and subsequent chain operations
            boolean isChainOperator = tt == TokenType.QUESTION_DOT ||
                (hasOptionalChaining && (
                    tt == TokenType.DOT ||
                    tt == TokenType.LBRACKET ||
                    tt == TokenType.LPAREN ||
                    tt == TokenType.TEMPLATE_LITERAL ||
                    tt == TokenType.TEMPLATE_HEAD
                ));

            if (tt == TokenType.QUESTION_DOT) {
                hasOptionalChaining = true;
            } else if (hasOptionalChaining && !isChainOperator) {
                // We're leaving the chain portion - wrap in ChainExpression first
                chainEndToken = previous();
                left = new ChainExpression(outerStartPos, getEnd(chainEndToken), startToken.line(), startToken.column(), chainEndToken.endLine(), chainEndToken.endColumn(), left);
                hasOptionalChaining = false;
                // Don't update start positions - the ChainExpression is now part of the larger expression
            }

            advance();
            Token opToken = previous();
            // Set instance vars for handler to use, then call handler - inlined for performance
            exprStartPos = outerStartPos;
            exprStartLoc = outerStartLoc;
            left = switch (tt) {
                case COMMA -> infixComma(this, left, opToken);
                case ASSIGN, PLUS_ASSIGN, MINUS_ASSIGN, STAR_ASSIGN, SLASH_ASSIGN, PERCENT_ASSIGN,
                     STAR_STAR_ASSIGN, LEFT_SHIFT_ASSIGN, RIGHT_SHIFT_ASSIGN, UNSIGNED_RIGHT_SHIFT_ASSIGN,
                     BIT_AND_ASSIGN, BIT_OR_ASSIGN, BIT_XOR_ASSIGN, AND_ASSIGN, OR_ASSIGN, QUESTION_QUESTION_ASSIGN -> infixAssignment(this, left, opToken);
                case QUESTION -> infixTernary(this, left, opToken);
                case QUESTION_QUESTION, OR, AND -> infixLogical(this, left, opToken);
                case BIT_OR, BIT_XOR, BIT_AND, EQ, NE, EQ_STRICT, NE_STRICT,
                     LT, LE, GT, GE, INSTANCEOF, IN,
                     LEFT_SHIFT, RIGHT_SHIFT, UNSIGNED_RIGHT_SHIFT,
                     PLUS, MINUS, STAR, SLASH, PERCENT, STAR_STAR -> infixBinary(this, left, opToken);
                case DOT -> infixMember(this, left, opToken);
                case QUESTION_DOT -> infixOptionalChain(this, left, opToken);
                case LBRACKET -> infixComputed(this, left, opToken);
                case LPAREN -> infixCall(this, left, opToken);
                case TEMPLATE_LITERAL, TEMPLATE_HEAD -> infixTaggedTemplate(this, left, opToken);
                default -> left; // unreachable due to lbp check above
            };
            // Note: handler may have overwritten exprStartPos/exprStartLoc via recursive parseExpr calls,
            // but we have our local outerStartPos/outerStartLoc preserved
        }

        // Wrap in ChainExpression if we still have optional chaining at the end
        if (hasOptionalChaining) {
            Token endToken = previous();
            left = new ChainExpression(outerStartPos, getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), left);
        }

        return left;
    }

    // ========================================================================
    // Prefix Handlers (NUD)
    // ========================================================================

    private static Expression prefixNumber(Parser p, Token token) {
        String lexeme = token.lexeme();

        // Check for BigInt literal (ends with 'n')
        if (lexeme.endsWith("n")) {
            String bigintValue = lexeme.substring(0, lexeme.length() - 1).replace("_", "");
            // Convert hex/octal/binary to decimal
            if (bigintValue.startsWith("0x") || bigintValue.startsWith("0X")) {
                try {
                    java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 16);
                    bigintValue = bi.toString();
                } catch (NumberFormatException e) { /* keep original */ }
            } else if (bigintValue.startsWith("0o") || bigintValue.startsWith("0O")) {
                try {
                    java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 8);
                    bigintValue = bi.toString();
                } catch (NumberFormatException e) { /* keep original */ }
            } else if (bigintValue.startsWith("0b") || bigintValue.startsWith("0B")) {
                try {
                    java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 2);
                    bigintValue = bi.toString();
                } catch (NumberFormatException e) { /* keep original */ }
            }
            // value should be the decimal bigint WITHOUT 'n' to match Acorn's JSON serialization of BigInt
            // (BigInt.toString() outputs just the number, no 'n' suffix)
            return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), bigintValue, lexeme, null, bigintValue);
        }

        // Handle Infinity/-Infinity/NaN - value should be null per ESTree spec
        Object literalValue = token.literal();
        if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
            literalValue = null;
        }
        return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), literalValue, token.lexeme());
    }

    private static Expression prefixString(Parser p, Token token) {
        return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), token.literal(), token.lexeme());
    }

    private static Expression prefixTrue(Parser p, Token token) {
        return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), true, "true");
    }

    private static Expression prefixFalse(Parser p, Token token) {
        return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), false, "false");
    }

    private static Expression prefixNull(Parser p, Token token) {
        return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), null, "null");
    }

    private static Expression prefixRegex(Parser p, Token token) {
        // Value is {} (empty object), the actual regex info goes in the 'regex' field
        Literal.RegexInfo regexInfo = (Literal.RegexInfo) token.literal();
        return new Literal(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(),
            java.util.Collections.emptyMap(), token.lexeme(), regexInfo);
    }

    private static Expression prefixIdentifier(Parser p, Token token) {
        // Check for async function expression
        if (token.lexeme().equals("async") && p.current < p.tokens.size() &&
            p.tokens.get(p.current).type() == TokenType.FUNCTION &&
            token.line() == p.tokens.get(p.current).line()) {
            return p.parseAsyncFunctionExpressionFromIdentifier(token);
        }

        // In module/async mode, 'await' is a reserved keyword
        if ((p.forceModuleMode || p.inAsyncContext) && !p.inClassFieldInitializer && token.lexeme().equals("await")) {
            String context = p.forceModuleMode ? "module code" : "async function";
            throw new ParseException("SyntaxError", token, null, null,
                "Unexpected use of 'await' as identifier in " + context);
        }

        return new Identifier(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn(), token.lexeme());
    }

    private static Expression prefixThis(Parser p, Token token) {
        return new ThisExpression(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn());
    }

    private static Expression prefixSuper(Parser p, Token token) {
        return new Super(p.getStart(token), p.getEnd(token), token.line(), token.column(), token.endLine(), token.endColumn());
    }

    private static Expression prefixGroupedOrArrow(Parser p, Token token) {
        // This is called when we see '(' and it's NOT an arrow function
        // (arrow functions are handled in tryParseArrowFunction before prefix dispatch)
        // So this is a grouped/parenthesized expression

        // Save and enable allowIn - inside parentheses, 'in' is always allowed as an operator
        // This is important for cases like: for (var x = (a in b) ? 1 : 2; ...)
        boolean oldAllowIn = p.allowIn;
        p.allowIn = true;
        Expression expr = p.parseExpr(BP_COMMA);
        p.allowIn = oldAllowIn;

        p.consume(TokenType.RPAREN, "Expected ')' after expression");

        // Mark that this expression was parenthesized (for directive detection)
        p.lastExpressionWasParenthesized = true;

        return expr;
    }

    private static Expression prefixArray(Parser p, Token token) {
        return p.parseArrayLiteral(token);
    }

    private static Expression prefixObject(Parser p, Token token) {
        return p.parseObjectLiteral(token);
    }

    private static Expression prefixFunction(Parser p, Token token) {
        return p.parseFunctionExpression(token, false);
    }

    private static Expression prefixClass(Parser p, Token token) {
        return p.parseClassExpression(token);
    }

    private static Expression prefixNew(Parser p, Token token) {
        return p.parseNewExpression(token);
    }

    private static Expression prefixUnary(Parser p, Token token) {
        Expression argument = p.parseExpr(BP_UNARY);
        Token endToken = p.previous();

        // Strict mode validation: delete on identifiers is not allowed
        if (p.strictMode && token.type() == TokenType.DELETE && argument instanceof Identifier) {
            throw new ExpectedTokenException("Delete of an unqualified identifier is not allowed in strict mode", token);
        }

        return new UnaryExpression(p.getStart(token), p.getEnd(endToken), token.line(), token.column(), endToken.endLine(), endToken.endColumn(), token.lexeme(), true, argument);
    }

    private static Expression prefixUpdate(Parser p, Token token) {
        Expression argument = p.parseExpr(BP_UNARY);
        Token endToken = p.previous();
        return new UpdateExpression(p.getStart(token), p.getEnd(endToken), token.line(), token.column(), endToken.endLine(), endToken.endColumn(), token.lexeme(), true, argument);
    }

    private static Expression prefixTemplate(Parser p, Token token) {
        // Back up one token since parseTemplateLiteral expects to start at the template token
        p.current--;
        return p.parseTemplateLiteral();
    }

    private static Expression prefixImport(Parser p, Token token) {
        return p.parseImportExpression(token);
    }

    private static Expression prefixPrivateIdentifier(Parser p, Token token) {
        Token nameToken = p.peek();
        if (!p.check(TokenType.IDENTIFIER) && !p.isKeyword(nameToken)) {
            throw new ExpectedTokenException("identifier after '#'", p.peek());
        }
        p.advance();
        return new PrivateIdentifier(p.getStart(token), p.getEnd(nameToken), token.line(), token.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
    }

    // ========================================================================
    // Infix Handlers (LED)
    // ========================================================================

    private static Expression infixComma(Parser p, Expression left, Token op) {
        // Save outer expression start before recursive calls
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        List<Expression> expressions = new ArrayList<>();
        expressions.add(left);

        // First comma already consumed, parse rest
        expressions.add(p.parseExpr(BP_COMMA + 1)); // Don't allow further commas at same level

        // Continue parsing more comma-separated expressions
        while (p.match(TokenType.COMMA)) {
            expressions.add(p.parseExpr(BP_COMMA + 1));
        }

        Token endToken = p.previous();
        int endPos = p.getEnd(endToken);
        return new SequenceExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), expressions);
    }

    private static Expression infixAssignment(Parser p, Expression left, Token op) {
        // Save outer expression start before recursive calls
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        Expression right = p.parseExpr(BP_ASSIGNMENT); // Right-associative
        Token endToken = p.previous();

        // Convert left side to pattern if it's a destructuring target
        Node leftNode = p.convertToPatternIfNeeded(left);

        int endPos = p.getEnd(endToken);
        return new AssignmentExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), op.lexeme(), leftNode, right);
    }

    private static Expression infixTernary(Parser p, Expression test, Token question) {
        // Save outer expression start before recursive calls
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        boolean oldAllowIn = p.allowIn;
        p.allowIn = true;
        Expression consequent = p.parseExpr(BP_ASSIGNMENT);
        p.consume(TokenType.COLON, "Expected ':'");
        Expression alternate = p.parseExpr(BP_ASSIGNMENT);
        p.allowIn = oldAllowIn;

        Token endToken = p.previous();
        int endPos = p.getEnd(endToken);
        return new ConditionalExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), test, consequent, alternate);
    }

    private static Expression infixLogical(Parser p, Expression left, Token op) {
        // Save outer expression start before recursive calls
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        // Logical operators are left-associative: RBP = LBP + 1
        int rbp = switch (op.type()) {
            case QUESTION_QUESTION -> BP_NULLISH + 1;
            case OR -> BP_OR + 1;
            case AND -> BP_AND + 1;
            default -> BP_AND + 1; // should not happen
        };
        Expression right = p.parseExpr(rbp);
        Token endToken = p.previous();
        int endPos = p.getEnd(endToken);
        return new LogicalExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), left, op.lexeme(), right);
    }

    private static Expression infixBinary(Parser p, Expression left, Token op) {
        // Save outer expression start before recursive calls
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        // Get RBP based on operator - most are left-associative (RBP = LBP + 1)
        // Only ** is right-associative (RBP = LBP)
        int rbp = switch (op.type()) {
            case BIT_OR -> BP_BIT_OR + 1;
            case BIT_XOR -> BP_BIT_XOR + 1;
            case BIT_AND -> BP_BIT_AND + 1;
            case EQ, NE, EQ_STRICT, NE_STRICT -> BP_EQUALITY + 1;
            case LT, LE, GT, GE, INSTANCEOF, IN -> BP_RELATIONAL + 1;
            case LEFT_SHIFT, RIGHT_SHIFT, UNSIGNED_RIGHT_SHIFT -> BP_SHIFT + 1;
            case PLUS, MINUS -> BP_ADDITIVE + 1;
            case STAR, SLASH, PERCENT -> BP_MULTIPLICATIVE + 1;
            case STAR_STAR -> BP_EXPONENT; // right-associative
            default -> BP_ADDITIVE + 1; // should not happen
        };
        Expression right = p.parseExpr(rbp);
        Token endToken = p.previous();
        int endPos = p.getEnd(endToken);
        return new BinaryExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), left, op.lexeme(), right);
    }

    private static Expression infixMember(Parser p, Expression object, Token dot) {
        if (p.match(TokenType.HASH)) {
            // Private field: obj.#x
            Token hashToken = p.previous();
            Token propertyToken = p.peek();
            if (!p.check(TokenType.IDENTIFIER) && !p.isKeyword(propertyToken)) {
                throw new ExpectedTokenException("identifier after '#'", p.peek());
            }
            p.advance();
            Expression property = new PrivateIdentifier(p.getStart(hashToken), p.getEnd(propertyToken),
                hashToken.line(), hashToken.column(), propertyToken.endLine(), propertyToken.endColumn(), propertyToken.lexeme());
            Token endToken = p.previous();
            int endPos = p.getEnd(endToken);
            return new MemberExpression(p.exprStartPos, endPos, p.exprStartLoc.line(), p.exprStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, property, false, false);
        } else {
            // Regular property: obj.x
            Token propertyToken = p.peek();
            if (!p.check(TokenType.IDENTIFIER) && !p.isKeyword(propertyToken) &&
                !p.check(TokenType.NUMBER) && !p.check(TokenType.STRING) &&
                !p.check(TokenType.TRUE) && !p.check(TokenType.FALSE) && !p.check(TokenType.NULL)) {
                throw new ExpectedTokenException("property name after '.'", p.peek());
            }
            p.advance();
            Expression property = new Identifier(p.getStart(propertyToken), p.getEnd(propertyToken),
                propertyToken.line(), propertyToken.column(), propertyToken.endLine(), propertyToken.endColumn(), propertyToken.lexeme());
            Token endToken = p.previous();
            int endPos = p.getEnd(endToken);
            return new MemberExpression(p.exprStartPos, endPos, p.exprStartLoc.line(), p.exprStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, property, false, false);
        }
    }

    private static Expression infixOptionalChain(Parser p, Expression object, Token questionDot) {
        // Save outer expression start before any recursive calls
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        if (p.check(TokenType.LPAREN)) {
            // Optional call: obj?.(args)
            p.advance(); // consume (
            List<Expression> args = p.parseArgumentList();
            p.consume(TokenType.RPAREN, "Expected ')' after arguments");
            Token endToken = p.previous();
            int endPos = p.getEnd(endToken);
            return new CallExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, args, true);
        } else if (p.check(TokenType.LBRACKET)) {
            // Optional computed: obj?.[expr]
            p.advance(); // consume [
            Expression property = p.parseExpr(BP_COMMA);
            p.consume(TokenType.RBRACKET, "Expected ']' after computed property");
            Token endToken = p.previous();
            int endPos = p.getEnd(endToken);
            return new MemberExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, property, true, true);
        } else if (p.match(TokenType.HASH)) {
            // Optional private: obj?.#x
            Token hashToken = p.previous();
            Token propertyToken = p.peek();
            if (!p.check(TokenType.IDENTIFIER) && !p.isKeyword(propertyToken)) {
                throw new ExpectedTokenException("identifier after '#'", p.peek());
            }
            p.advance();
            Expression property = new PrivateIdentifier(p.getStart(hashToken), p.getEnd(propertyToken),
                hashToken.line(), hashToken.column(), propertyToken.endLine(), propertyToken.endColumn(), propertyToken.lexeme());
            Token endToken = p.previous();
            int endPos = p.getEnd(endToken);
            return new MemberExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, property, false, true);
        } else {
            // Optional property: obj?.x
            Token propertyToken = p.peek();
            if (!p.check(TokenType.IDENTIFIER) && !p.isKeyword(propertyToken) &&
                !p.check(TokenType.NUMBER) && !p.check(TokenType.STRING) &&
                !p.check(TokenType.TRUE) && !p.check(TokenType.FALSE) && !p.check(TokenType.NULL)) {
                throw new ExpectedTokenException("property name after '?.'", p.peek());
            }
            p.advance();
            Expression property = new Identifier(p.getStart(propertyToken), p.getEnd(propertyToken),
                propertyToken.line(), propertyToken.column(), propertyToken.endLine(), propertyToken.endColumn(), propertyToken.lexeme());
            Token endToken = p.previous();
            int endPos = p.getEnd(endToken);
            return new MemberExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, property, false, true);
        }
    }

    private static Expression infixComputed(Parser p, Expression object, Token lbracket) {
        // Save outer expression start before recursive parseExpr call
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        Expression property = p.parseExpr(BP_COMMA);
        p.consume(TokenType.RBRACKET, "Expected ']' after computed property");
        Token endToken = p.previous();
        int endPos = p.getEnd(endToken);
        return new MemberExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), object, property, true, false);
    }

    private static Expression infixCall(Parser p, Expression callee, Token lparen) {
        // Save outer expression start before recursive parseArgumentList calls parseExpr
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        List<Expression> args = p.parseArgumentList();
        p.consume(TokenType.RPAREN, "Expected ')' after arguments");
        Token endToken = p.previous();
        int endPos = p.getEnd(endToken);
        return new CallExpression(savedStartPos, endPos, savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), callee, args, false);
    }

    private static Expression infixTaggedTemplate(Parser p, Expression tag, Token templateStart) {
        // Save outer expression start before parseTemplateLiteral which may call parseExpr for interpolations
        int savedStartPos = p.exprStartPos;
        SourceLocation.Position savedStartLoc = p.exprStartLoc;

        // Back up one token since parseTemplateLiteral expects to start at the template token
        p.current--;
        Expression template = p.parseTemplateLiteral();
        Token endToken = p.previous();
        return new TaggedTemplateExpression(savedStartPos, template.end(), savedStartLoc.line(), savedStartLoc.column(), endToken.endLine(), endToken.endColumn(), tag, (TemplateLiteral) template);
    }

    // ========================================================================
    // Helper Methods for Pratt Parser
    // ========================================================================

    private boolean shouldParseAwait() {
        if (!check(TokenType.IDENTIFIER) || !peek().lexeme().equals("await")) {
            return false;
        }

        if (inAsyncContext && !inClassFieldInitializer) {
            return true;
        }

        if (!inAsyncContext && forceModuleMode && !inClassFieldInitializer) {
            return !checkAhead(1, TokenType.COLON) &&
                   !checkAhead(1, TokenType.ASSIGN) &&
                   !checkAhead(1, TokenType.PLUS_ASSIGN) &&
                   !checkAhead(1, TokenType.MINUS_ASSIGN);
        }

        // Class field initializer validation
        if (inClassFieldInitializer) {
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
                throw new ParseException("SyntaxError", peek(), null, null,
                    "Cannot use keyword 'await' outside an async function");
            }
        }

        return false;
    }

    private Expression parseYieldExpr() {
        advance(); // consume 'yield'
        Token yieldToken = previous();
        boolean delegate = false;
        Expression argument = null;

        if (match(TokenType.STAR)) {
            delegate = true;
        }

        boolean hasLineTerminator = !delegate && !isAtEnd() && peek().line() > yieldToken.line();
        if (!hasLineTerminator &&
            !check(TokenType.SEMICOLON) && !check(TokenType.RBRACE) && !check(TokenType.EOF) &&
            !check(TokenType.RPAREN) && !check(TokenType.COMMA) && !check(TokenType.RBRACKET) &&
            !check(TokenType.TEMPLATE_MIDDLE) && !check(TokenType.TEMPLATE_TAIL) &&
            !check(TokenType.COLON)) {
            argument = parseExpr(BP_ASSIGNMENT);
        }

        Token endToken = previous();
        return new YieldExpression(getStart(yieldToken), getEnd(endToken), yieldToken.line(), yieldToken.column(), endToken.endLine(), endToken.endColumn(), delegate, argument);
    }

    private Expression parseAwaitExpr() {
        Token awaitToken = advance();

        Expression argument = null;
        if (!check(TokenType.SEMICOLON) && !check(TokenType.RBRACE) && !check(TokenType.EOF) &&
            !check(TokenType.RPAREN) && !check(TokenType.COMMA) && !check(TokenType.RBRACKET) &&
            !check(TokenType.INSTANCEOF) && !check(TokenType.IN) &&
            !check(TokenType.QUESTION) && !check(TokenType.COLON)) {
            argument = parseExpr(BP_UNARY);
        }

        Token endToken = previous();
        return new AwaitExpression(getStart(awaitToken), getEnd(endToken), awaitToken.line(), awaitToken.column(), endToken.endLine(), endToken.endColumn(), argument);
    }

    private Expression tryParseArrowFunction(Token startToken) {
        // Check for async arrow function: async identifier => or async (params) =>
        boolean isAsync = false;
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
            if (current + 1 < tokens.size()) {
                Token asyncToken = peek();
                Token nextToken = tokens.get(current + 1);

                if (asyncToken.line() != nextToken.line()) {
                    // Line terminator - not async arrow
                    return null;
                }

                if (nextToken.type() == TokenType.IDENTIFIER) {
                    if (current + 2 < tokens.size() && tokens.get(current + 2).type() == TokenType.ARROW) {
                        advance(); // consume 'async'
                        isAsync = true;
                    }
                } else if (nextToken.type() == TokenType.LPAREN) {
                    int savedCurrent = current;
                    advance(); // consume 'async'
                    advance(); // consume '('
                    boolean isArrow = isArrowFunctionParameters();
                    current = savedCurrent;

                    if (isArrow) {
                        advance(); // consume 'async'
                        isAsync = true;
                    }
                }
            }
        }

        // Check for simple arrow: identifier => or (params) =>
        if ((check(TokenType.IDENTIFIER) || check(TokenType.OF) || check(TokenType.LET))) {
            Token idToken = peek();
            if (current + 1 < tokens.size() && tokens.get(current + 1).type() == TokenType.ARROW) {
                advance(); // consume identifier
                List<Pattern> params = new ArrayList<>();
                params.add(new Identifier(getStart(idToken), getEnd(idToken), idToken.line(), idToken.column(), idToken.endLine(), idToken.endColumn(), idToken.lexeme()));
                consume(TokenType.ARROW, "Expected '=>'");
                return parseArrowFunctionBody(startToken, params, isAsync);
            }
        }

        // Check for parenthesized arrow: (params) =>
        if (check(TokenType.LPAREN)) {
            int savedCurrent = current;
            advance(); // consume (

            boolean isArrow = isArrowFunctionParameters();

            if (isArrow) {
                List<Pattern> params = new ArrayList<>();
                if (!check(TokenType.RPAREN)) {
                    do {
                        if (check(TokenType.RPAREN)) break;
                        if (match(TokenType.DOT_DOT_DOT)) {
                            Token restStart = previous();
                            Pattern argument = parsePatternBase();
                            Token restEnd = previous();
                            params.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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
                consume(TokenType.ARROW, "Expected '=>'");
                return parseArrowFunctionBody(startToken, params, isAsync);
            } else {
                current = savedCurrent;
            }
        }

        return null;
    }

    private List<Expression> parseArgumentList() {
        List<Expression> args = new ArrayList<>();
        // Arguments to function calls should always allow 'in' operator
        boolean savedAllowIn = allowIn;
        allowIn = true;
        if (!check(TokenType.RPAREN)) {
            do {
                if (check(TokenType.RPAREN)) break; // trailing comma
                if (match(TokenType.DOT_DOT_DOT)) {
                    Token spreadStart = previous();
                    Expression argument = parseExpr(BP_ASSIGNMENT);
                    Token spreadEnd = previous();
                    args.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadStart.line(), spreadStart.column(), spreadEnd.endLine(), spreadEnd.endColumn(), argument));
                } else {
                    args.add(parseExpr(BP_ASSIGNMENT));
                }
            } while (match(TokenType.COMMA));
        }
        allowIn = savedAllowIn;
        return args;
    }

    private SourceLocation createLocationFromPositions(int start, int end, SourceLocation.Position startPos, Token endToken) {
        // Use endLine/endColumn from token for accurate multi-line token support
        SourceLocation.Position endPos = new SourceLocation.Position(endToken.endLine(), endToken.endColumn());
        return new SourceLocation(startPos, endPos);
    }

    private Expression parseAsyncFunctionExpressionFromIdentifier(Token asyncToken) {
        advance(); // consume 'function'

        boolean isGenerator = match(TokenType.STAR);

        Identifier id = null;
        if (check(TokenType.IDENTIFIER)) {
            Token nameToken = peek();
            advance();
            id = new Identifier(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
        }

        consume(TokenType.LPAREN, "Expected '(' after function");
        List<Pattern> params = new ArrayList<>();

        if (!check(TokenType.RPAREN)) {
            do {
                if (check(TokenType.RPAREN)) break;
                if (match(TokenType.DOT_DOT_DOT)) {
                    Token restStart = previous();
                    Pattern argument = parsePatternBase();
                    Token restEnd = previous();
                    params.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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

        boolean savedInGenerator = inGenerator;
        boolean savedInAsyncContext = inAsyncContext;
        boolean savedStrictMode = strictMode;
        boolean savedInClassFieldInitializer = inClassFieldInitializer;
        inGenerator = isGenerator;
        inAsyncContext = true;
        inClassFieldInitializer = false;

        if (!forceModuleMode) {
            strictMode = false;
        }

        BlockStatement body = parseBlockStatement(true);
        validateNoDuplicateParameters(params, asyncToken);

        inGenerator = savedInGenerator;
        inAsyncContext = savedInAsyncContext;
        strictMode = savedStrictMode;
        inClassFieldInitializer = savedInClassFieldInitializer;

        Token endToken = previous();
        return new FunctionExpression(getStart(asyncToken), getEnd(endToken), asyncToken.line(), asyncToken.column(), endToken.endLine(), endToken.endColumn(), id, false, isGenerator, true, params, body);
    }

    private Expression parseImportExpression(Token importToken) {
        if (match(TokenType.DOT)) {
            Token propertyToken = peek();
            if (check(TokenType.IDENTIFIER) && propertyToken.lexeme().equals("meta")) {
                advance();
                Identifier meta = new Identifier(getStart(importToken), getEnd(importToken), importToken.line(), importToken.column(), importToken.endLine(), importToken.endColumn(), "import");
                Identifier property = new Identifier(getStart(propertyToken), getEnd(propertyToken), propertyToken.line(), propertyToken.column(), propertyToken.endLine(), propertyToken.endColumn(), "meta");
                Token endToken = previous();
                return new MetaProperty(getStart(importToken), getEnd(endToken), importToken.line(), importToken.column(), endToken.endLine(), endToken.endColumn(), meta, property);
            }
            throw new ExpectedTokenException("meta", peek());
        }

        // Dynamic import: import(source)
        consume(TokenType.LPAREN, "Expected '(' after import");
        Expression source = parseExpr(BP_ASSIGNMENT);

        // Check for options argument (import attributes)
        Expression options = null;
        if (match(TokenType.COMMA)) {
            if (!check(TokenType.RPAREN)) {
                options = parseExpr(BP_ASSIGNMENT);
            }
        }

        consume(TokenType.RPAREN, "Expected ')' after import source");
        Token endToken = previous();
        return new ImportExpression(getStart(importToken), getEnd(endToken), importToken.line(), importToken.column(), endToken.endLine(), endToken.endColumn(), source, options);
    }

    private Expression parseNewExpression(Token newToken) {
        // Handle new.target
        if (match(TokenType.DOT)) {
            Token targetToken = peek();
            if (check(TokenType.IDENTIFIER) && targetToken.lexeme().equals("target")) {
                advance();
                Identifier meta = new Identifier(getStart(newToken), getEnd(newToken), newToken.line(), newToken.column(), newToken.endLine(), newToken.endColumn(), "new");
                Identifier property = new Identifier(getStart(targetToken), getEnd(targetToken), targetToken.line(), targetToken.column(), targetToken.endLine(), targetToken.endColumn(), "target");
                return new MetaProperty(getStart(newToken), getEnd(targetToken), newToken.line(), newToken.column(), targetToken.endLine(), targetToken.endColumn(), meta, property);
            }
            throw new ExpectedTokenException("target", peek());
        }

        // Parse callee - need to handle member expressions without calls
        Expression callee = parseNewCallee();

        // Optional arguments
        List<Expression> args = new ArrayList<>();
        if (match(TokenType.LPAREN)) {
            args = parseArgumentList();
            consume(TokenType.RPAREN, "Expected ')' after arguments");
        }

        Token endToken = previous();
        return new NewExpression(getStart(newToken), getEnd(endToken), newToken.line(), newToken.column(), endToken.endLine(), endToken.endColumn(), callee, args);
    }

    private Expression parseNewCallee() {
        // Parse primary expression for new callee
        Token token = peek();
        Expression callee;

        if (check(TokenType.NEW)) {
            // Nested new: new new Foo()
            advance();
            callee = parseNewExpression(previous());
        } else {
            // Inline prefix handler dispatch for performance
            advance();
            Token prevToken = previous();
            callee = switch (token.type()) {
                case NUMBER -> prefixNumber(this, prevToken);
                case STRING -> prefixString(this, prevToken);
                case TRUE -> prefixTrue(this, prevToken);
                case FALSE -> prefixFalse(this, prevToken);
                case NULL -> prefixNull(this, prevToken);
                case REGEX -> prefixRegex(this, prevToken);
                case IDENTIFIER -> prefixIdentifier(this, prevToken);
                case THIS -> prefixThis(this, prevToken);
                case SUPER -> prefixSuper(this, prevToken);
                case LPAREN -> prefixGroupedOrArrow(this, prevToken);
                case LBRACKET -> prefixArray(this, prevToken);
                case LBRACE -> prefixObject(this, prevToken);
                case FUNCTION -> prefixFunction(this, prevToken);
                case CLASS -> prefixClass(this, prevToken);
                case TEMPLATE_LITERAL, TEMPLATE_HEAD -> prefixTemplate(this, prevToken);
                case IMPORT -> prefixImport(this, prevToken);
                default -> throw new UnexpectedTokenException(token, "expression");
            };
        }

        // Parse member access only (no calls for new callee)
        Token startToken = token;
        while (true) {
            if (match(TokenType.DOT)) {
                if (match(TokenType.HASH)) {
                    Token hashToken = previous();
                    Token propertyToken = peek();
                    if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken)) {
                        throw new ExpectedTokenException("identifier after '#'", peek());
                    }
                    advance();
                    Expression property = new PrivateIdentifier(getStart(hashToken), getEnd(propertyToken),
                        hashToken.line(), hashToken.column(), propertyToken.endLine(), propertyToken.endColumn(), propertyToken.lexeme());
                    Token endToken = previous();
                    callee = new MemberExpression(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), callee, property, false, false);
                } else {
                    Token propertyToken = peek();
                    if (!check(TokenType.IDENTIFIER) && !isKeyword(propertyToken)) {
                        throw new ExpectedTokenException("property name after '.'", peek());
                    }
                    advance();
                    Expression property = new Identifier(getStart(propertyToken), getEnd(propertyToken),
                        propertyToken.line(), propertyToken.column(), propertyToken.endLine(), propertyToken.endColumn(), propertyToken.lexeme());
                    Token endToken = previous();
                    callee = new MemberExpression(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), callee, property, false, false);
                }
            } else if (match(TokenType.LBRACKET)) {
                Expression property = parseExpr(BP_COMMA);
                consume(TokenType.RBRACKET, "Expected ']' after computed property");
                Token endToken = previous();
                callee = new MemberExpression(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), callee, property, true, false);
            } else {
                break;
            }
        }

        return callee;
    }

    private Expression parseFunctionExpression(Token functionToken, boolean isGenerator) {
        if (!isGenerator) {
            isGenerator = match(TokenType.STAR);
        }

        Identifier id = null;
        if (check(TokenType.IDENTIFIER)) {
            Token nameToken = peek();
            advance();
            id = new Identifier(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
        }

        consume(TokenType.LPAREN, "Expected '(' after function");
        List<Pattern> params = new ArrayList<>();

        if (!check(TokenType.RPAREN)) {
            do {
                if (check(TokenType.RPAREN)) break;
                if (match(TokenType.DOT_DOT_DOT)) {
                    Token restStart = previous();
                    Pattern argument = parsePatternBase();
                    Token restEnd = previous();
                    params.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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

        boolean savedInGenerator = inGenerator;
        boolean savedInAsyncContext = inAsyncContext;
        boolean savedStrictMode = strictMode;
        boolean savedInClassFieldInitializer = inClassFieldInitializer;
        inGenerator = isGenerator;
        inAsyncContext = false;
        inClassFieldInitializer = false;

        if (!forceModuleMode) {
            strictMode = false;
        }

        BlockStatement body = parseBlockStatement(true);
        validateNoDuplicateParameters(params, functionToken);

        inGenerator = savedInGenerator;
        inAsyncContext = savedInAsyncContext;
        strictMode = savedStrictMode;
        inClassFieldInitializer = savedInClassFieldInitializer;

        Token endToken = previous();
        return new FunctionExpression(getStart(functionToken), getEnd(endToken), functionToken.line(), functionToken.column(), endToken.endLine(), endToken.endColumn(), id, false, isGenerator, false, params, body);
    }

    private Expression parseClassExpression(Token classToken) {
        // Optional class name - but NOT if the next token is 'extends' (contextual keyword)
        Identifier id = null;
        if (check(TokenType.IDENTIFIER) && !peek().lexeme().equals("extends")) {
            Token nameToken = peek();
            advance();
            id = new Identifier(getStart(nameToken), getEnd(nameToken), nameToken.line(), nameToken.column(), nameToken.endLine(), nameToken.endColumn(), nameToken.lexeme());
        }

        // Optional extends
        Expression superClass = null;
        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("extends")) {
            advance(); // consume 'extends'
            // Parse superclass - use BP_COMMA to allow member access like `orig.Minimatch`
            // but stop at comma, etc. Using parseLeftHandSideExpression would be ideal,
            // but BP_COMMA achieves similar result for practical cases
            superClass = parseExpr(BP_COMMA);
        }

        // Class body
        ClassBody body = parseClassBody();

        Token endToken = previous();
        return new ClassExpression(getStart(classToken), getEnd(endToken), classToken.line(), classToken.column(), endToken.endLine(), endToken.endColumn(), id, superClass, body);
    }

    private Expression parseArrayLiteral(Token lbracket) {
        List<Expression> elements = new ArrayList<>();

        while (!check(TokenType.RBRACKET) && !isAtEnd()) {
            if (check(TokenType.COMMA)) {
                // Elision (hole in array)
                elements.add(null);
                advance();
            } else if (match(TokenType.DOT_DOT_DOT)) {
                // Spread element
                Token spreadStart = previous();
                Expression argument = parseExpr(BP_ASSIGNMENT);
                Token spreadEnd = previous();
                elements.add(new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadStart.line(), spreadStart.column(), spreadEnd.endLine(), spreadEnd.endColumn(), argument));
                if (!check(TokenType.RBRACKET)) {
                    consume(TokenType.COMMA, "Expected ',' after spread element");
                }
            } else {
                elements.add(parseExpr(BP_ASSIGNMENT));
                if (!check(TokenType.RBRACKET)) {
                    consume(TokenType.COMMA, "Expected ',' or ']'");
                }
            }
        }

        consume(TokenType.RBRACKET, "Expected ']' after array elements");
        Token endToken = previous();
        return new ArrayExpression(getStart(lbracket), getEnd(endToken), lbracket.line(), lbracket.column(), endToken.endLine(), endToken.endColumn(), elements);
    }

    private Expression parseObjectLiteral(Token lbrace) {
        List<Node> properties = new ArrayList<>();

        while (!check(TokenType.RBRACE) && !isAtEnd()) {
            properties.add(parseObjectPropertyNode());
            if (!check(TokenType.RBRACE)) {
                consume(TokenType.COMMA, "Expected ',' or '}'");
            }
        }

        consume(TokenType.RBRACE, "Expected '}' after object properties");
        Token endToken = previous();
        return new ObjectExpression(getStart(lbrace), getEnd(endToken), lbrace.line(), lbrace.column(), endToken.endLine(), endToken.endColumn(), properties);
    }

    private Node parseObjectPropertyNode() {
        Token startToken = peek();

        // Check for spread property - return SpreadElement directly
        if (match(TokenType.DOT_DOT_DOT)) {
            Token spreadStart = previous();
            Expression argument = parseExpr(BP_ASSIGNMENT);
            Token spreadEnd = previous();
            return new SpreadElement(getStart(spreadStart), getEnd(spreadEnd), spreadStart.line(), spreadStart.column(), spreadEnd.endLine(), spreadEnd.endColumn(), argument);
        }

        // Check for method shorthand: get/set/async/generator
        boolean isAsync = false;
        boolean isGenerator = false;
        String kind = "init";

        if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("async")) {
            if (current + 1 < tokens.size()) {
                Token nextToken = tokens.get(current + 1);
                // async without line terminator followed by property name is async method
                if (peek().line() == nextToken.line() &&
                    nextToken.type() != TokenType.COLON &&
                    nextToken.type() != TokenType.COMMA &&
                    nextToken.type() != TokenType.RBRACE &&
                    nextToken.type() != TokenType.LPAREN) {
                    advance(); // consume 'async'
                    isAsync = true;
                }
            }
        }

        if (match(TokenType.STAR)) {
            isGenerator = true;
        }

        if (!isAsync && !isGenerator && check(TokenType.IDENTIFIER)) {
            String lexeme = peek().lexeme();
            if (lexeme.equals("get") || lexeme.equals("set")) {
                if (current + 1 < tokens.size()) {
                    Token nextToken = tokens.get(current + 1);
                    if (nextToken.type() != TokenType.COLON &&
                        nextToken.type() != TokenType.COMMA &&
                        nextToken.type() != TokenType.RBRACE &&
                        nextToken.type() != TokenType.LPAREN) {
                        kind = lexeme;
                        advance(); // consume 'get' or 'set'
                    }
                }
            }
        }

        // Parse property key
        boolean computed = false;
        Expression key;

        if (match(TokenType.LBRACKET)) {
            computed = true;
            key = parseExpr(BP_ASSIGNMENT);
            consume(TokenType.RBRACKET, "Expected ']' after computed property name");
        } else if (check(TokenType.STRING) || check(TokenType.NUMBER)) {
            Token keyToken = advance();
            String keyLexeme = keyToken.lexeme();

            // Check if this is a BigInt literal (ends with 'n')
            if (keyToken.type() == TokenType.NUMBER && keyLexeme.endsWith("n")) {
                // BigInt literal: value is null, bigint field has the numeric part
                String bigintValue = keyLexeme.substring(0, keyLexeme.length() - 1).replace("_", "");
                // Convert hex/octal/binary BigInt to decimal string
                if (bigintValue.startsWith("0x") || bigintValue.startsWith("0X")) {
                    try {
                        java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 16);
                        bigintValue = bi.toString();
                    } catch (NumberFormatException e) { /* keep original */ }
                } else if (bigintValue.startsWith("0o") || bigintValue.startsWith("0O")) {
                    try {
                        java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 8);
                        bigintValue = bi.toString();
                    } catch (NumberFormatException e) { /* keep original */ }
                } else if (bigintValue.startsWith("0b") || bigintValue.startsWith("0B")) {
                    try {
                        java.math.BigInteger bi = new java.math.BigInteger(bigintValue.substring(2), 2);
                        bigintValue = bi.toString();
                    } catch (NumberFormatException e) { /* keep original */ }
                }
                // value should be the decimal bigint WITHOUT 'n' to match Acorn's JSON serialization
                key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), bigintValue, keyLexeme, null, bigintValue);
            } else {
                // Handle Infinity/-Infinity/NaN - value should be null per ESTree spec
                Object literalValue = keyToken.literal();
                if (literalValue instanceof Double d && (d.isInfinite() || d.isNaN())) {
                    literalValue = null;
                }
                key = new Literal(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), literalValue, keyLexeme);
            }
        } else {
            Token keyToken = peek();
            if (!check(TokenType.IDENTIFIER) && !isKeyword(keyToken)) {
                throw new ExpectedTokenException("property name", keyToken);
            }
            advance();
            key = new Identifier(getStart(keyToken), getEnd(keyToken), keyToken.line(), keyToken.column(), keyToken.endLine(), keyToken.endColumn(), keyToken.lexeme());
        }

        // Check for method or shorthand
        if (check(TokenType.LPAREN) || isGenerator || isAsync || !kind.equals("init")) {
            // Method - FunctionExpression starts at '(' not at the method name
            Token funcStartToken = peek(); // Save the '(' token for FunctionExpression start
            advance(); // consume (
            List<Pattern> params = new ArrayList<>();

            if (!check(TokenType.RPAREN)) {
                do {
                    if (check(TokenType.RPAREN)) break;
                    if (match(TokenType.DOT_DOT_DOT)) {
                        Token restStart = previous();
                        Pattern argument = parsePatternBase();
                        Token restEnd = previous();
                        params.add(new RestElement(getStart(restStart), getEnd(restEnd), restStart.line(), restStart.column(), restEnd.endLine(), restEnd.endColumn(), argument));
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

            boolean savedInGenerator = inGenerator;
            boolean savedInAsyncContext = inAsyncContext;
            boolean savedStrictMode = strictMode;
            boolean savedInClassFieldInitializer = inClassFieldInitializer;
            inGenerator = isGenerator;
            inAsyncContext = isAsync;
            inClassFieldInitializer = false;

            BlockStatement body = parseBlockStatement(true);

            inGenerator = savedInGenerator;
            inAsyncContext = savedInAsyncContext;
            strictMode = savedStrictMode;
            inClassFieldInitializer = savedInClassFieldInitializer;

            Token endToken = previous();
            // FunctionExpression starts at '(' per ESTree spec for method definitions
            FunctionExpression value = new FunctionExpression(getStart(funcStartToken), getEnd(endToken), funcStartToken.line(), funcStartToken.column(), endToken.endLine(), endToken.endColumn(), null, false, isGenerator, isAsync, params, body);

            // Getters and setters have method=false, only actual methods have method=true
            boolean isMethod = kind.equals("init");
            // Property: start, end, loc, method, shorthand, computed, key, value, kind
            return new Property(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), isMethod, false, computed, key, value, kind);
        } else if (match(TokenType.COLON)) {
            // Regular property
            Expression value = parseExpr(BP_ASSIGNMENT);
            Token endToken = previous();
            // Property: start, end, loc, method, shorthand, computed, key, value, kind
            return new Property(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), false, false, computed, key, value, "init");
        } else if (match(TokenType.ASSIGN)) {
            // Shorthand property with default value: { x = value }
            // This is only valid in destructuring patterns but we parse it as an object expression
            // and convert it later. The value is an AssignmentExpression.
            if (!(key instanceof Identifier)) {
                throw new ExpectedTokenException("identifier", peek());
            }
            Identifier id = (Identifier) key;
            // Inside object literals with default values (destructuring), 'in' is always the operator
            boolean savedAllowIn = allowIn;
            allowIn = true;
            Expression defaultValue = parseExpr(BP_ASSIGNMENT);
            allowIn = savedAllowIn;
            Token endToken = previous();
            // Create an AssignmentExpression as the value, which will be converted to AssignmentPattern later
            AssignmentExpression assignExpr = new AssignmentExpression(
                getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(),
                "=", id, defaultValue);
            // Property: start, end, loc, method, shorthand, computed, key, value, kind
            return new Property(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), false, true, computed, key, assignExpr, "init");
        } else {
            // Shorthand property: { x } means { x: x }
            if (!(key instanceof Identifier)) {
                throw new ExpectedTokenException("':' after property name", peek());
            }
            Token endToken = previous();
            // Property: start, end, loc, method, shorthand, computed, key, value, kind
            return new Property(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), false, true, computed, key, key, "init");
        }
    }

    // End of Pratt parser section
    // ========================================================================

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
                    patternElements.add(new AssignmentPattern(assignExpr.start(), assignExpr.end(), assignExpr.startLine(), assignExpr.startCol(), assignExpr.endLine(), assignExpr.endCol(), left, assignExpr.right()));
                } else if (element instanceof SpreadElement spreadElem) {
                    // Convert SpreadElement to RestElement
                    Node argNode = convertToPatternIfNeeded(spreadElem.argument());
                    Pattern argument = (Pattern) argNode;
                    patternElements.add(new RestElement(spreadElem.start(), spreadElem.end(), spreadElem.startLine(), spreadElem.startCol(), spreadElem.endLine(), spreadElem.endCol(), argument));
                } else {
                    // For other expressions, keep as is (this might be an error in real code)
                    patternElements.add((Pattern) element);
                }
            }
            return new ArrayPattern(arrayExpr.start(), arrayExpr.end(), arrayExpr.startLine(), arrayExpr.startCol(), arrayExpr.endLine(), arrayExpr.endCol(), patternElements);
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
                        AssignmentPattern pattern = new AssignmentPattern(assignExpr.start(), assignExpr.end(), assignExpr.startLine(), assignExpr.startCol(), assignExpr.endLine(), assignExpr.endCol(), left, assignExpr.right());
                        convertedProperties.add(new Property(property.start(), property.end(), property.startLine(), property.startCol(), property.endLine(), property.endCol(),
                            property.method(), property.shorthand(), property.computed(),
                            property.key(), pattern, property.kind()));
                    } else if (value instanceof ObjectExpression || value instanceof ArrayExpression) {
                        // Recursively convert nested destructuring
                        Node convertedValue = convertToPatternIfNeeded(value);
                        convertedProperties.add(new Property(property.start(), property.end(), property.startLine(), property.startCol(), property.endLine(), property.endCol(),
                            property.method(), property.shorthand(), property.computed(),
                            property.key(), convertedValue, property.kind()));
                    } else {
                        convertedProperties.add(property);
                    }
                } else if (prop instanceof SpreadElement spreadElem) {
                    // Convert SpreadElement to RestElement in object patterns
                    Node argNode = convertToPatternIfNeeded(spreadElem.argument());
                    Pattern argument = (Pattern) argNode;
                    convertedProperties.add(new RestElement(spreadElem.start(), spreadElem.end(), spreadElem.startLine(), spreadElem.startCol(), spreadElem.endLine(), spreadElem.endCol(), argument));
                } else {
                    convertedProperties.add(prop);
                }
            }
            return new ObjectPattern(objExpr.start(), objExpr.end(), objExpr.startLine(), objExpr.startCol(), objExpr.endLine(), objExpr.endCol(), convertedProperties);
        } else if (expr instanceof AssignmentExpression assignExpr) {
            // Convert AssignmentExpression to AssignmentPattern (for default values)
            Node leftNode = convertToPatternIfNeeded(assignExpr.left());
            Pattern left = (Pattern) leftNode;
            return new AssignmentPattern(assignExpr.start(), assignExpr.end(), assignExpr.startLine(), assignExpr.startCol(), assignExpr.endLine(), assignExpr.endCol(), left, assignExpr.right());
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
                return new ArrowFunctionExpression(getStart(startToken), getEnd(endToken), startToken.line(), startToken.column(), endToken.endLine(), endToken.endColumn(), null, false, false, isAsync, params, body);
            } else {
                // Expression body: () => expr
                Expression body = parseExpr(BP_ASSIGNMENT);
                Token endToken = previous();

                // Always use endToken for arrow end position - this correctly handles
                // parenthesized expressions like () => (expr) where the ) is consumed
                int arrowEnd = getEnd(endToken);
                int endLine = endToken.endLine();
                int endCol = endToken.endColumn();

                return new ArrowFunctionExpression(getStart(startToken), arrowEnd, startToken.line(), startToken.column(), endLine, endCol, null, true, false, isAsync, params, body);
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
            quasis.add(new TemplateElement(
                elemStart,
                elemEnd,
                elemStartPos.line(), elemStartPos.column(), elemEndPos.line(), elemEndPos.column(),
                new TemplateElement.TemplateElementValue(raw, cooked),
                true // tail
            ));

            SourceLocation.Position templateStartPos = getPositionFromOffset(getStart(startToken));
            SourceLocation.Position templateEndPos = getPositionFromOffset(getEnd(startToken));
            return new TemplateLiteral(getStart(startToken), getEnd(startToken), templateStartPos.line(), templateStartPos.column(), templateEndPos.line(), templateEndPos.column(), expressions, quasis);
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
            quasis.add(new TemplateElement(
                elemStart,
                elemEnd,
                elemStartPos.line(), elemStartPos.column(), elemEndPos.line(), elemEndPos.column(),
                new TemplateElement.TemplateElementValue(raw, cooked),
                false // not tail
            ));

            // Parse expressions and middle/tail quasis
            while (true) {
                // Parse the expression - allow 'in' operator in template interpolations
                boolean savedAllowIn = allowIn;
                allowIn = true;
                Expression expr = parseExpression();
                allowIn = savedAllowIn;
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
                    quasis.add(new TemplateElement(
                        quasiStart,
                        quasiEnd,
                        quasiStartPos.line(), quasiStartPos.column(), quasiEndPos.line(), quasiEndPos.column(),
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
                    quasis.add(new TemplateElement(
                        quasiStart,
                        quasiEnd,
                        quasiStartPos.line(), quasiStartPos.column(), quasiEndPos.line(), quasiEndPos.column(),
                        new TemplateElement.TemplateElementValue(quasiRaw, quasiCooked),
                        true // tail
                    ));

                    // Calculate template end position (token already includes closing `)
                    int templateEnd = getEnd(endToken);
                    SourceLocation.Position templateStartPos = getPositionFromOffset(templateStart);
                    SourceLocation.Position templateEndPos = getPositionFromOffset(templateEnd);
                    return new TemplateLiteral(templateStart, templateEnd, templateStartPos.line(), templateStartPos.column(), templateEndPos.line(), templateEndPos.column(), expressions, quasis);
                } else {
                    throw new ExpectedTokenException("TEMPLATE_MIDDLE or TEMPLATE_TAIL after expression in template literal", peek());
                }
            }
        } else {
            throw new ExpectedTokenException("template literal token", peek());
        }
    }


    // Helper method to create SourceLocation from tokens
    private SourceLocation createLocation(Token start, Token end) {
        SourceLocation.Position startPos = new SourceLocation.Position(start.line(), start.column());
        // Use token's endLine/endColumn for accurate multi-line token support
        SourceLocation.Position endPos = new SourceLocation.Position(end.endLine(), end.endColumn());
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
               type == TokenType.DEBUGGER || type == TokenType.ASYNC || type == TokenType.AWAIT ||
               type == TokenType.TRUE || type == TokenType.FALSE || type == TokenType.NULL;
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
