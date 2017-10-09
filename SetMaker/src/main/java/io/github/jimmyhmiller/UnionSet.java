package io.github.jimmyhmiller;

public class UnionSet<T> implements Set<T> {


    private final Set<T> set1;
    private final Set<T> set2;

    public UnionSet(Set<T> set1, Set<T> set2) {
        this.set1 = set1;
        this.set2 = set2;
    }

    @Override
    public Boolean isEmpty() {
        return set1.isEmpty() && set2.isEmpty();
    }

    @Override
    public Boolean contains(T t) {
        return set1.contains(t) || set2.contains(t);
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
