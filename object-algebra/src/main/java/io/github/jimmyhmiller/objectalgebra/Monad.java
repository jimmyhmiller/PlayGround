package io.github.jimmyhmiller.objectalgebra;


import java.util.function.Function;

public interface Monad<A> extends Functor<A> {
    <B> Monad<B> flatMap(Function<A, Monad<B>> f);
}
