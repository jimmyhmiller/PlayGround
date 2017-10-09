package io.github.jimmyhmiller;

public class InsertSet<T> implements Set<T> {


    private final Set<T> other;
    private final T t;

    public InsertSet(Set<T> other, T t) {
        this.other = other;
        this.t = t;
    }

    @Override
    public Boolean isEmpty() {
        return false;
    }

    @Override
    public Boolean contains(T t) {
        return t == this.t || other.contains(t);
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
