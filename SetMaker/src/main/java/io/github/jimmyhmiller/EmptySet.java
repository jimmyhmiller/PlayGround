package io.github.jimmyhmiller;

public class EmptySet<T> implements Set<T> {
    @Override
    public Boolean isEmpty() {
        return true;
    }

    @Override
    public Boolean contains(T t) {
        return false;
    }

    @Override
    public Set<T> insert(T t) {
        return new InsertSet<>(this, t);
    }

    @Override
    public Set<T> union(Set<T> t) {
        return new UnionSet<>(this, t);
    }
}
