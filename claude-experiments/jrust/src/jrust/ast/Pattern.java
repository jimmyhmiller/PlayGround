package jrust.ast;

import java.util.List;

public sealed interface Pattern {
    record Wildcard() implements Pattern {}
    record Literal(Expr expr) implements Pattern {}
    record EnumVariant(String enumName, String variant, List<String> bindings) implements Pattern {}
}
