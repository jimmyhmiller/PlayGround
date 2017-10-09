package io.github.jimmyhmiller;

public interface Set<T> {
    Boolean isEmpty();
    Boolean contains(T t);
    Set<T> insert(T t);
    Set<T> union(Set<T> t);
}
