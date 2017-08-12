package io.github.jimmyhmiller;


import javafx.util.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

public interface Sequence<T> {
    T first();
    Sequence<T> rest();
    Sequence<T> cons(T elem);
    Sequence<T> empty();
    Boolean isEmpty();

    default T second() {
        return this.rest().first();
    }

    default T nth(Integer n) {
        return this.drop(n).first();
    }

    default T last() {
        if (this.isEmpty()) {
            return null;
        } else if (this.rest().isEmpty()) {
            return this.first();
        } else {
            return this.rest().last();
        }
    }

    default Sequence<T> drop(Integer n) {
        if (n <= 0 || this.isEmpty()) {
            return this;
        } else {
            return this.rest().drop(n-1);
        }
    }


    default <U> Sequence<U> map(Function<T, U> f) {
        return this.reduce((Sequence<U>) this.empty(), (uSequence, t) -> uSequence.cons(f.apply(t)));
    }

    default <U> Sequence<U> mapWithIndex(BiFunction<T, Integer, U> f) {
        return this.reduce(new Pair<>(0, (Sequence<U>) this.empty()),
                (pair, t) -> new Pair<>(pair.getKey() + 1, pair.getValue().cons(f.apply(t, pair.getKey())))).getValue();
    }

    default Sequence<T> filter(Predicate<T> pred) {
        return this.reduce(this.empty(), (Sequence<T> tSequence, T t) -> {
            if (pred.test(t)) {
               return tSequence.cons(t);
            } else {
                return tSequence;
            }

        });
    }

    default <U> U reduce(U init, BiFunction<U, T, U> f) {
        if (this.isEmpty()) {
            return init;
        } else {
            return this.rest().reduce(f.apply(init, this.first()), f);
        }
    }

    default List<T> toList() {
        return reduce(new ArrayList<>(), ((ts, t) -> {
            ts.add(t);
            return ts;
        }));
    }

    default Integer count() {
        return this.reduce(0, (n, e) -> n + 1);
    }

    default Sequence<T> concat(Sequence<T> seq) {
        if (this.isEmpty()) {
            return seq;
        } else {
            return this.rest().concat(seq).cons(this.first());
        }
    }
}

