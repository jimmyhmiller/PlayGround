package io.github.jimmyhmiller.objectalgebra;


import java.util.Arrays;
import java.util.List;

public class ListCollection<T> implements Collection<Converter<List<T>>> {

    @Override
    public Converter<List<T>> create(Object... args) {
        return () -> Arrays.asList((T[]) args);
    }

}
