package io.github.jimmyhmiller.objectalgebra;


public interface JsonStruct<E> {
    E String(String s);
    JsonObject<E> Object();
}
