package jrust;

import jrust.ast.*;

import java.util.ArrayList;
import java.util.List;

public class Parser {
    private final List<Token> tokens;
    private int pos;

    public Parser(List<Token> tokens) {
        this.tokens = tokens;
        this.pos = 0;
    }

    public Program parse() {
        List<Item> items = new ArrayList<>();
        while (!atEnd()) {
            items.add(parseItem());
        }
        return new Program(items);
    }

    // --- Item parsing ---

    private Item parseItem() {
        if (check(Token.Kind.FN)) return parseFnDef();
        if (check(Token.Kind.STRUCT)) return parseStructDef();
        if (check(Token.Kind.IMPL)) return parseImplDef();
        if (check(Token.Kind.IMPORT)) return parseImport();
        if (check(Token.Kind.ENUM)) return parseEnumDef();
        if (check(Token.Kind.CONST)) return parseConstDef();
        throw error("Expected item (fn, struct, impl, import, enum, or const), got " + current().kind);
    }

    private Item.FnDef parseFnDef() {
        expect(Token.Kind.FN);
        String name = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.LPAREN);
        List<Item.Param> params = parseParams();
        expect(Token.Kind.RPAREN);

        Type returnType = new Type.Void();
        if (matchToken(Token.Kind.ARROW)) {
            returnType = parseType();
        }

        List<Stmt> body = parseBlock();
        return new Item.FnDef(name, params, returnType, body);
    }

    private List<Item.Param> parseParams() {
        List<Item.Param> params = new ArrayList<>();
        if (check(Token.Kind.RPAREN)) return params;

        params.add(parseParam());
        while (matchToken(Token.Kind.COMMA)) {
            if (check(Token.Kind.RPAREN)) break;
            params.add(parseParam());
        }
        return params;
    }

    private Item.Param parseParam() {
        // Handle self / mut self / &self / &mut self
        if (check(Token.Kind.SELF)) {
            advance();
            return new Item.Param("self", null, false, true);
        }
        if (check(Token.Kind.MUT) && peekKind(1) == Token.Kind.SELF) {
            advance(); // mut
            advance(); // self
            return new Item.Param("self", null, true, true);
        }
        if (check(Token.Kind.AMP)) {
            advance(); // &
            boolean mutable = false;
            if (matchToken(Token.Kind.MUT)) {
                mutable = true;
            }
            expect(Token.Kind.SELF);
            return new Item.Param("self", null, mutable, true);
        }

        // Normal param: [mut] name: type
        boolean mutable = matchToken(Token.Kind.MUT);
        String name = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.COLON);
        Type type = parseType();
        return new Item.Param(name, type, mutable, false);
    }

    private Item.StructDef parseStructDef() {
        expect(Token.Kind.STRUCT);
        String name = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.LBRACE);
        List<Item.Field> fields = new ArrayList<>();
        while (!check(Token.Kind.RBRACE)) {
            String fieldName = expect(Token.Kind.IDENT).text;
            expect(Token.Kind.COLON);
            Type type = parseType();
            fields.add(new Item.Field(fieldName, type));
            matchToken(Token.Kind.COMMA);
        }
        expect(Token.Kind.RBRACE);
        return new Item.StructDef(name, fields);
    }

    private Item.ImplDef parseImplDef() {
        expect(Token.Kind.IMPL);
        String typeName = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.LBRACE);
        List<Item.FnDef> methods = new ArrayList<>();
        while (!check(Token.Kind.RBRACE)) {
            methods.add(parseFnDef());
        }
        expect(Token.Kind.RBRACE);
        return new Item.ImplDef(typeName, methods);
    }

    private Item.Import parseImport() {
        expect(Token.Kind.IMPORT);
        StringBuilder path = new StringBuilder();
        path.append(expect(Token.Kind.IDENT).text);
        while (matchToken(Token.Kind.DOT)) {
            path.append(".");
            // Handle wildcard import: import java.util.*
            if (check(Token.Kind.STAR)) {
                path.append(advance().text);
                break;
            }
            path.append(expect(Token.Kind.IDENT).text);
        }
        expect(Token.Kind.SEMI);
        return new Item.Import(path.toString());
    }

    private Item.EnumDef parseEnumDef() {
        expect(Token.Kind.ENUM);
        String name = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.LBRACE);
        List<Item.EnumVariant> variants = new ArrayList<>();
        while (!check(Token.Kind.RBRACE)) {
            String variantName = expect(Token.Kind.IDENT).text;
            List<Item.Field> fields = new ArrayList<>();
            if (matchToken(Token.Kind.LBRACE)) {
                while (!check(Token.Kind.RBRACE)) {
                    String fieldName = expect(Token.Kind.IDENT).text;
                    expect(Token.Kind.COLON);
                    Type type = parseType();
                    fields.add(new Item.Field(fieldName, type));
                    matchToken(Token.Kind.COMMA);
                }
                expect(Token.Kind.RBRACE);
            }
            variants.add(new Item.EnumVariant(variantName, fields));
            matchToken(Token.Kind.COMMA);
        }
        expect(Token.Kind.RBRACE);
        return new Item.EnumDef(name, variants);
    }

    private Item.ConstDef parseConstDef() {
        expect(Token.Kind.CONST);
        String name = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.COLON);
        Type type = parseType();
        expect(Token.Kind.EQ);
        Expr value = parseExpr();
        expect(Token.Kind.SEMI);
        return new Item.ConstDef(name, type, value);
    }

    // --- Type parsing ---

    private Type parseType() {
        String name = expect(Token.Kind.IDENT).text;
        // Check for generic parameters: Vec<T>, Map<K, V>
        if (matchToken(Token.Kind.LT)) {
            List<Type> args = new ArrayList<>();
            args.add(parseType());
            while (matchToken(Token.Kind.COMMA)) {
                args.add(parseType());
            }
            expect(Token.Kind.GT);
            return new Type.Generic(name, args);
        }
        return new Type.Simple(name);
    }

    // --- Statement parsing ---

    private List<Stmt> parseBlock() {
        expect(Token.Kind.LBRACE);
        List<Stmt> stmts = new ArrayList<>();
        while (!check(Token.Kind.RBRACE)) {
            stmts.add(parseStmt());
        }
        expect(Token.Kind.RBRACE);
        return stmts;
    }

    private Stmt parseStmt() {
        if (check(Token.Kind.LET)) return parseLetStmt();
        if (check(Token.Kind.BREAK)) { advance(); expect(Token.Kind.SEMI); return new Stmt.Break(); }
        if (check(Token.Kind.CONTINUE)) { advance(); expect(Token.Kind.SEMI); return new Stmt.Continue(); }
        if (check(Token.Kind.RETURN)) return parseReturnStmt();
        return parseExprStmt();
    }

    private Stmt.Let parseLetStmt() {
        expect(Token.Kind.LET);
        boolean mutable = matchToken(Token.Kind.MUT);
        String name = expect(Token.Kind.IDENT).text;

        Type type = null;
        if (matchToken(Token.Kind.COLON)) {
            type = parseType();
        }

        Expr init = null;
        if (matchToken(Token.Kind.EQ)) {
            init = parseExpr();
        }

        expect(Token.Kind.SEMI);
        return new Stmt.Let(name, mutable, type, init);
    }

    private Stmt.Return parseReturnStmt() {
        expect(Token.Kind.RETURN);
        Expr value = null;
        if (!check(Token.Kind.SEMI)) {
            value = parseExpr();
        }
        expect(Token.Kind.SEMI);
        return new Stmt.Return(value);
    }

    private Stmt.ExprStmt parseExprStmt() {
        Expr expr = parseExpr();
        // Block-like expressions don't need trailing semicolons
        if (!(expr instanceof Expr.If) && !(expr instanceof Expr.While)
                && !(expr instanceof Expr.Block) && !(expr instanceof Expr.ForRange)
                && !(expr instanceof Expr.ForEach) && !(expr instanceof Expr.Match)) {
            expect(Token.Kind.SEMI);
        } else {
            matchToken(Token.Kind.SEMI); // optional semicolon after block expressions
        }
        return new Stmt.ExprStmt(expr);
    }

    // --- Expression parsing (precedence climbing) ---

    private Expr parseExpr() {
        return parseAssignment();
    }

    private Expr parseAssignment() {
        Expr left = parseOr();
        if (matchToken(Token.Kind.EQ)) {
            Expr right = parseAssignment();
            return new Expr.Assign(left, right);
        }
        return left;
    }

    private Expr parseOr() {
        Expr left = parseAnd();
        while (matchToken(Token.Kind.PIPEPIPE)) {
            Expr right = parseAnd();
            left = new Expr.Binary(left, "||", right);
        }
        return left;
    }

    private Expr parseAnd() {
        Expr left = parseBitwiseOr();
        while (matchToken(Token.Kind.AMPAMP)) {
            Expr right = parseBitwiseOr();
            left = new Expr.Binary(left, "&&", right);
        }
        return left;
    }

    private Expr parseBitwiseOr() {
        Expr left = parseEquality();
        while (matchToken(Token.Kind.PIPE)) {
            Expr right = parseEquality();
            left = new Expr.Binary(left, "|", right);
        }
        return left;
    }

    private Expr parseEquality() {
        Expr left = parseComparison();
        while (check(Token.Kind.EQEQ) || check(Token.Kind.BANGEQ)) {
            String op = advance().text;
            Expr right = parseComparison();
            left = new Expr.Binary(left, op, right);
        }
        return left;
    }

    private Expr parseComparison() {
        Expr left = parseAddition();
        while (check(Token.Kind.LT) || check(Token.Kind.GT) || check(Token.Kind.LTEQ) || check(Token.Kind.GTEQ)) {
            String op = advance().text;
            Expr right = parseAddition();
            left = new Expr.Binary(left, op, right);
        }
        return left;
    }

    private Expr parseAddition() {
        Expr left = parseMultiplication();
        while (check(Token.Kind.PLUS) || check(Token.Kind.MINUS)) {
            String op = advance().text;
            Expr right = parseMultiplication();
            left = new Expr.Binary(left, op, right);
        }
        return left;
    }

    private Expr parseMultiplication() {
        Expr left = parseUnary();
        while (check(Token.Kind.STAR) || check(Token.Kind.SLASH) || check(Token.Kind.PERCENT)) {
            String op = advance().text;
            Expr right = parseUnary();
            left = new Expr.Binary(left, op, right);
        }
        return left;
    }

    private Expr parseUnary() {
        if (check(Token.Kind.MINUS) || check(Token.Kind.BANG)) {
            String op = advance().text;
            Expr operand = parseUnary();
            return new Expr.Unary(op, operand);
        }
        return parsePostfix();
    }

    private Expr parsePostfix() {
        Expr expr = parsePrimary();
        while (true) {
            if (matchToken(Token.Kind.DOT)) {
                String name = expect(Token.Kind.IDENT).text;
                if (matchToken(Token.Kind.LPAREN)) {
                    List<Expr> args = parseArgList();
                    expect(Token.Kind.RPAREN);
                    expr = new Expr.MethodCall(expr, name, args);
                } else {
                    expr = new Expr.FieldAccess(expr, name);
                }
            } else if (matchToken(Token.Kind.LBRACKET)) {
                Expr index = parseExpr();
                expect(Token.Kind.RBRACKET);
                expr = new Expr.Index(expr, index);
            } else {
                break;
            }
        }
        return expr;
    }

    private Expr parsePrimary() {
        // Int / Float literals
        if (check(Token.Kind.INT_LIT)) {
            Token t = advance();
            return new Expr.IntLit(Long.parseLong(t.text));
        }
        if (check(Token.Kind.FLOAT_LIT)) {
            Token t = advance();
            return new Expr.FloatLit(Double.parseDouble(t.text));
        }
        if (check(Token.Kind.STRING_LIT)) {
            Token t = advance();
            return new Expr.StringLit(t.text);
        }
        if (check(Token.Kind.CHAR_LIT)) {
            Token t = advance();
            return new Expr.CharLit(t.text.charAt(0));
        }
        if (check(Token.Kind.TRUE)) {
            advance();
            return new Expr.BoolLit(true);
        }
        if (check(Token.Kind.FALSE)) {
            advance();
            return new Expr.BoolLit(false);
        }
        if (check(Token.Kind.NULL)) {
            advance();
            return new Expr.NullLit();
        }
        if (check(Token.Kind.SELF)) {
            advance();
            return new Expr.SelfExpr();
        }

        // If expression
        if (check(Token.Kind.IF)) {
            return parseIfExpr();
        }

        // While expression
        if (check(Token.Kind.WHILE)) {
            return parseWhileExpr();
        }

        // For expression
        if (check(Token.Kind.FOR)) {
            return parseForExpr();
        }

        // Match expression
        if (check(Token.Kind.MATCH)) {
            return parseMatchExpr();
        }

        // Block expression
        if (check(Token.Kind.LBRACE)) {
            List<Stmt> stmts = parseBlock();
            return new Expr.Block(stmts);
        }

        // Array literal [a, b, c]
        if (check(Token.Kind.LBRACKET)) {
            return parseArrayLit();
        }

        // Identifier or call or static call or struct init or enum init
        if (check(Token.Kind.IDENT)) {
            Token ident = advance();

            // Static call / enum init: Type::method(args) or Type::Variant { ... } or Type::Variant
            if (matchToken(Token.Kind.COLONCOLON)) {
                String member = expect(Token.Kind.IDENT).text;

                // Enum variant with fields: Type::Variant { field: val, ... }
                if (check(Token.Kind.LBRACE) && isEnumInit()) {
                    advance(); // {
                    List<Expr.FieldValue> fields = new ArrayList<>();
                    while (!check(Token.Kind.RBRACE)) {
                        String fieldName = expect(Token.Kind.IDENT).text;
                        expect(Token.Kind.COLON);
                        Expr value = parseExpr();
                        fields.add(new Expr.FieldValue(fieldName, value));
                        matchToken(Token.Kind.COMMA);
                    }
                    expect(Token.Kind.RBRACE);
                    return new Expr.EnumInit(ident.text, member, fields);
                }

                // Static method call: Type::method(args)
                if (matchToken(Token.Kind.LPAREN)) {
                    List<Expr> args = parseArgList();
                    expect(Token.Kind.RPAREN);

                    // Check for subclass expression: Type::new(args) with { fn ... }
                    if (check(Token.Kind.IDENT) && current().text.equals("with")) {
                        advance(); // consume 'with'
                        expect(Token.Kind.LBRACE);
                        List<Item.FnDef> methods = new ArrayList<>();
                        while (check(Token.Kind.FN)) {
                            methods.add(parseFnDef());
                        }
                        expect(Token.Kind.RBRACE);
                        return new Expr.Subclass(ident.text, args, methods);
                    }

                    return new Expr.StaticCall(ident.text, member, args);
                }

                // Fieldless enum variant: Type::Variant (no parens, no braces)
                return new Expr.EnumInit(ident.text, member, List.of());
            }

            // Function call: name(args)
            if (matchToken(Token.Kind.LPAREN)) {
                List<Expr> args = parseArgList();
                expect(Token.Kind.RPAREN);
                return new Expr.Call(ident.text, args);
            }

            // Struct init: Name { field: val, ... }
            if (check(Token.Kind.LBRACE) && isStructInit(ident.text)) {
                advance(); // {
                List<Expr.FieldValue> fields = new ArrayList<>();
                while (!check(Token.Kind.RBRACE)) {
                    String fieldName = expect(Token.Kind.IDENT).text;
                    expect(Token.Kind.COLON);
                    Expr value = parseExpr();
                    fields.add(new Expr.FieldValue(fieldName, value));
                    matchToken(Token.Kind.COMMA);
                }
                expect(Token.Kind.RBRACE);
                return new Expr.StructInit(ident.text, fields);
            }

            return new Expr.Ident(ident.text);
        }

        // Parenthesized expression
        if (matchToken(Token.Kind.LPAREN)) {
            Expr expr = parseExpr();
            expect(Token.Kind.RPAREN);
            return expr;
        }

        throw error("Expected expression, got " + current().kind + " '" + current().text + "'");
    }

    // Heuristic: struct init if name starts with uppercase and next is { ident :
    private boolean isStructInit(String name) {
        if (!Character.isUpperCase(name.charAt(0))) return false;
        if (peekKind(0) != Token.Kind.LBRACE) return false;
        if (peekKind(1) == Token.Kind.IDENT && peekKind(2) == Token.Kind.COLON) return true;
        if (peekKind(1) == Token.Kind.RBRACE) return true; // empty struct
        return false;
    }

    // Heuristic: enum init with fields if { ident : follows
    private boolean isEnumInit() {
        if (peekKind(0) != Token.Kind.LBRACE) return false;
        if (peekKind(1) == Token.Kind.IDENT && peekKind(2) == Token.Kind.COLON) return true;
        if (peekKind(1) == Token.Kind.RBRACE) return true;
        return false;
    }

    private Expr.If parseIfExpr() {
        expect(Token.Kind.IF);
        Expr condition = parseExpr();
        List<Stmt> thenBlock = parseBlock();
        List<Stmt> elseBlock = null;
        if (matchToken(Token.Kind.ELSE)) {
            if (check(Token.Kind.IF)) {
                Expr.If elseIf = parseIfExpr();
                elseBlock = List.of(new Stmt.ExprStmt(elseIf));
            } else {
                elseBlock = parseBlock();
            }
        }
        return new Expr.If(condition, thenBlock, elseBlock);
    }

    private Expr.While parseWhileExpr() {
        expect(Token.Kind.WHILE);
        Expr condition = parseExpr();
        List<Stmt> body = parseBlock();
        return new Expr.While(condition, body);
    }

    private Expr parseForExpr() {
        expect(Token.Kind.FOR);
        String var = expect(Token.Kind.IDENT).text;
        expect(Token.Kind.IN);

        Expr start = parseExpr();

        // Check for range: start..end
        if (matchToken(Token.Kind.DOTDOT)) {
            Expr end = parseExpr();
            List<Stmt> body = parseBlock();
            return new Expr.ForRange(var, start, end, body);
        }

        // Otherwise for-each: for item in iterable { ... }
        List<Stmt> body = parseBlock();
        return new Expr.ForEach(var, start, body);
    }

    private Expr parseMatchExpr() {
        expect(Token.Kind.MATCH);
        Expr subject = parseExpr();
        expect(Token.Kind.LBRACE);
        List<Expr.MatchArm> arms = new ArrayList<>();
        while (!check(Token.Kind.RBRACE)) {
            Pattern pattern = parsePattern();
            expect(Token.Kind.FATARROW);
            List<Stmt> body;
            if (check(Token.Kind.LBRACE)) {
                body = parseBlock();
            } else {
                // Single expression arm
                Expr expr = parseExpr();
                body = List.of(new Stmt.ExprStmt(expr));
            }
            matchToken(Token.Kind.COMMA);
            arms.add(new Expr.MatchArm(pattern, body));
        }
        expect(Token.Kind.RBRACE);
        return new Expr.Match(subject, arms);
    }

    private Pattern parsePattern() {
        // Wildcard: _
        if (check(Token.Kind.UNDERSCORE)) {
            advance();
            return new Pattern.Wildcard();
        }

        // Literal patterns: numbers, strings, bools, chars, null
        if (check(Token.Kind.INT_LIT)) {
            Token t = advance();
            return new Pattern.Literal(new Expr.IntLit(Long.parseLong(t.text)));
        }
        if (check(Token.Kind.FLOAT_LIT)) {
            Token t = advance();
            return new Pattern.Literal(new Expr.FloatLit(Double.parseDouble(t.text)));
        }
        if (check(Token.Kind.STRING_LIT)) {
            Token t = advance();
            return new Pattern.Literal(new Expr.StringLit(t.text));
        }
        if (check(Token.Kind.CHAR_LIT)) {
            Token t = advance();
            return new Pattern.Literal(new Expr.CharLit(t.text.charAt(0)));
        }
        if (check(Token.Kind.TRUE)) {
            advance();
            return new Pattern.Literal(new Expr.BoolLit(true));
        }
        if (check(Token.Kind.FALSE)) {
            advance();
            return new Pattern.Literal(new Expr.BoolLit(false));
        }
        if (check(Token.Kind.NULL)) {
            advance();
            return new Pattern.Literal(new Expr.NullLit());
        }

        // Negative number literals
        if (check(Token.Kind.MINUS) && peekKind(1) == Token.Kind.INT_LIT) {
            advance(); // -
            Token t = advance();
            return new Pattern.Literal(new Expr.IntLit(-Long.parseLong(t.text)));
        }

        // Enum variant pattern: Type::Variant or Type::Variant { field1, field2 }
        if (check(Token.Kind.IDENT) && peekKind(1) == Token.Kind.COLONCOLON) {
            String enumName = advance().text;
            expect(Token.Kind.COLONCOLON);
            String variant = expect(Token.Kind.IDENT).text;
            List<String> bindings = new ArrayList<>();
            if (matchToken(Token.Kind.LBRACE)) {
                while (!check(Token.Kind.RBRACE)) {
                    bindings.add(expect(Token.Kind.IDENT).text);
                    matchToken(Token.Kind.COMMA);
                }
                expect(Token.Kind.RBRACE);
            }
            return new Pattern.EnumVariant(enumName, variant, bindings);
        }

        throw error("Expected pattern, got " + current().kind + " '" + current().text + "'");
    }

    private Expr parseArrayLit() {
        expect(Token.Kind.LBRACKET);
        List<Expr> elements = new ArrayList<>();
        if (!check(Token.Kind.RBRACKET)) {
            elements.add(parseExpr());
            while (matchToken(Token.Kind.COMMA)) {
                if (check(Token.Kind.RBRACKET)) break;
                elements.add(parseExpr());
            }
        }
        expect(Token.Kind.RBRACKET);
        return new Expr.ArrayLit(elements);
    }

    private List<Expr> parseArgList() {
        List<Expr> args = new ArrayList<>();
        if (check(Token.Kind.RPAREN)) return args;
        args.add(parseExpr());
        while (matchToken(Token.Kind.COMMA)) {
            if (check(Token.Kind.RPAREN)) break;
            args.add(parseExpr());
        }
        return args;
    }

    // --- Helpers ---

    private boolean atEnd() {
        return current().kind == Token.Kind.EOF;
    }

    private Token current() {
        return tokens.get(pos);
    }

    private Token.Kind peekKind(int offset) {
        int idx = pos + offset;
        if (idx < tokens.size()) return tokens.get(idx).kind;
        return Token.Kind.EOF;
    }

    private boolean check(Token.Kind kind) {
        return current().kind == kind;
    }

    private boolean matchToken(Token.Kind kind) {
        if (check(kind)) {
            advance();
            return true;
        }
        return false;
    }

    private Token advance() {
        Token t = current();
        pos++;
        return t;
    }

    private Token expect(Token.Kind kind) {
        if (!check(kind)) {
            throw error("Expected " + kind + ", got " + current().kind + " '" + current().text + "'");
        }
        return advance();
    }

    private RuntimeException error(String msg) {
        Token t = current();
        return new RuntimeException("Parse error at " + t.line + ":" + t.col + ": " + msg);
    }
}
