package io.github.jimmyhmiller.objectalgebra;


public interface JsonObject<E> {
    JsonObject<E> add(String key, E value);
    E build();
}
