package io.github.jimmyhmiller.objectalgebra;


import java.util.function.Function;

public interface Functor<A> {

    static <T> Monad<T> pure(T t) {
        throw new RuntimeException("must be implemented");
    }

    <B> Functor<B> map(Function<A, B> f);

}
