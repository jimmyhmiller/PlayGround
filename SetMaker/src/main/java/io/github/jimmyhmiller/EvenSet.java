package io.github.jimmyhmiller;

public class EvenSet implements Set<Integer> {
    @Override
    public Boolean isEmpty() {
        return false;
    }

    @Override
    public Boolean contains(Integer i) {
        return i % 2 == 0;
    }

    @Override
    public Set<Integer> insert(Integer t) {
        return new InsertSet<>(this, t);
    }

    @Override
    public Set<Integer> union(Set<Integer> t) {
        return new UnionSet<>(this, t);
    }
}
