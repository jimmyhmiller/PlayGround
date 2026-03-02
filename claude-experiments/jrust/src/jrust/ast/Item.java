package jrust.ast;

import java.util.List;

public sealed interface Item {
    record FnDef(String name, List<Param> params, Type returnType, List<Stmt> body) implements Item {}
    record StructDef(String name, List<Field> fields) implements Item {}
    record ImplDef(String typeName, List<FnDef> methods) implements Item {}
    record Import(String path) implements Item {}
    record EnumDef(String name, List<EnumVariant> variants) implements Item {}
    record ConstDef(String name, Type type, Expr value) implements Item {}

    record Param(String name, Type type, boolean mutable, boolean isSelf) {}
    record Field(String name, Type type) {}
    record EnumVariant(String name, List<Field> fields) {}
}
