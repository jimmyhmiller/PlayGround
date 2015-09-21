package io.github.jimmyhmiller.objectalgebra;

import java.util.function.Function;


public interface Builder<A> extends Function<Collection<Converter<A>>, Converter<A>> {}
