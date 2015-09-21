package io.github.jimmyhmiller.objectalgebra;


import de.scravy.pair.Pair;
import de.scravy.pair.Pairs;

import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MapCollection<T, U> implements Collection<Converter<Map<T,U>>> {


    private Collection<Converter<U>> converter;


    public MapCollection(Collection<Converter<U>> converter) {
        this.converter = converter;
    }

    public static <A> Builder<A> c(Object... args) {
        return (Collection<Converter<A>> c) -> c.create(args);
    }

    public static <A> A build(Builder<A> builder, Collection<Converter<A>> c) {
        return builder.apply(c).convert();
    }

    private U convertType(Object o) {
        System.out.println(Builder.class.isInstance(o));
        if (Builder.class.isInstance(o)) {
            return build((Builder<U>) o, this.converter);
        }
        return (U) o;
    }

    @Override
    public Converter<Map<T, U>> create(Object... args) {
        return () -> IntStream.range(1, args.length)
                .filter(i -> i % 2 != 0)
                .mapToObj(i -> Pairs.from((T) args[i - 1], convertType(args[i])))
                .collect(Collectors.toMap(Pair::getFirst, Pair::getSecond));
    }
}
