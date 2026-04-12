package jrust.ast;

import java.util.List;

public sealed interface Expr {
    record IntLit(long value) implements Expr {}
    record FloatLit(double value) implements Expr {}
    record StringLit(String value) implements Expr {}
    record BoolLit(boolean value) implements Expr {}
    record CharLit(char value) implements Expr {}
    record NullLit() implements Expr {}
    record Ident(String name) implements Expr {}
    record SelfExpr() implements Expr {}
    record Binary(Expr left, String op, Expr right) implements Expr {}
    record Unary(String op, Expr operand) implements Expr {}
    record Call(String name, List<Expr> args) implements Expr {}
    record MethodCall(Expr receiver, String method, List<Expr> args) implements Expr {}
    record FieldAccess(Expr receiver, String field) implements Expr {}
    record StructInit(String name, List<FieldValue> fields) implements Expr {}
    record StaticCall(String typeName, String method, List<Expr> args) implements Expr {}
    record Assign(Expr target, Expr value) implements Expr {}
    record If(Expr condition, List<Stmt> thenBlock, List<Stmt> elseBlock) implements Expr {}
    record While(Expr condition, List<Stmt> body) implements Expr {}
    record Block(List<Stmt> stmts) implements Expr {}
    record ForRange(String var, Expr start, Expr end, List<Stmt> body) implements Expr {}
    record ForEach(String var, Expr iterable, List<Stmt> body) implements Expr {}
    record Match(Expr subject, List<MatchArm> arms) implements Expr {}
    record Index(Expr receiver, Expr index) implements Expr {}
    record EnumInit(String enumName, String variant, List<FieldValue> fields) implements Expr {}
    record Throw(Expr message) implements Expr {}
    record ArrayLit(List<Expr> elements) implements Expr {}
    record Cast(Expr value, Type type) implements Expr {}
    record Subclass(String typeName, List<Expr> args, List<Item.FnDef> methods) implements Expr {}

    record FieldValue(String name, Expr value) {}
    record MatchArm(Pattern pattern, List<Stmt> body) {}
}
