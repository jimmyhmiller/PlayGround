package jrust.ast;

public sealed interface Stmt {
    record Let(String name, boolean mutable, Type type, Expr init) implements Stmt {}
    record Return(Expr value) implements Stmt {}
    record Break() implements Stmt {}
    record Continue() implements Stmt {}
    record ExprStmt(Expr expr) implements Stmt {}
}
