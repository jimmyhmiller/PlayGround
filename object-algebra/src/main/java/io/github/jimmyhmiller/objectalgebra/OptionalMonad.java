package io.github.jimmyhmiller.objectalgebra;


import java.util.function.Function;

public class OptionalMonad<T> implements Optional<T, Monad<T>> {

    public static <U> Monad<U> pure(U u) {
        if (u == null) {
            return new OptionalMonad<U>().nothing();
        }
        return new OptionalMonad<U>().just(u);
    }


    @Override
    public Monad<T> nothing() {
        return new Monad<T>() {
            @Override
            public <B> Functor<B> map(Function<T, B> f) {
                return new OptionalMonad<B>().nothing();
            }
            @Override
            public <B> Monad<B> flatMap(Function<T, Monad<B>> f) {
                return new OptionalMonad<B>().nothing();
            }
            public String toString() {
                return "Nothing";
            }
        };
    }

    @Override
    public Monad<T> just(T t) {
        return new Monad<T>() {
            @Override
            public <B> Functor<B> map(Function<T, B> f) {
                return OptionalMonad.pure(f.apply(t));
            }
            @Override
            public <B> Monad<B> flatMap(Function<T, Monad<B>> f) {
                return f.apply(t);
            }
            public String toString() {
                return "Just " + t.toString();
            }
        };
    }

}
