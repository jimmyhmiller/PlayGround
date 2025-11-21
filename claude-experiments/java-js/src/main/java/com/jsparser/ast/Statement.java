package com.jsparser.ast;

public sealed interface Statement extends Node permits
    ExpressionStatement,
    VariableDeclaration,
    BlockStatement,
    ReturnStatement,
    IfStatement,
    WhileStatement,
    DoWhileStatement,
    ForStatement,
    ForInStatement,
    ForOfStatement,
    BreakStatement,
    ContinueStatement,
    FunctionDeclaration,
    ClassDeclaration,
    ImportDeclaration,
    ExportNamedDeclaration,
    ExportDefaultDeclaration,
    ExportAllDeclaration,
    ThrowStatement,
    TryStatement,
    WithStatement,
    DebuggerStatement,
    EmptyStatement,
    LabeledStatement,
    SwitchStatement {
}
