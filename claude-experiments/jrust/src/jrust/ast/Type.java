package jrust.ast;

import java.util.List;

public sealed interface Type {
    record Simple(String name) implements Type {}
    record Generic(String name, List<Type> args) implements Type {}
    record Array(Type element) implements Type {}
    record Void() implements Type {}
}
