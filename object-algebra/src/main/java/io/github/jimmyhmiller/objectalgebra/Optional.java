package io.github.jimmyhmiller.objectalgebra;


public interface Optional<T, A> {
    A nothing();
    A just(T t);
}
