package io.github.jimmyhmiller.injector;

import java.lang.reflect.Constructor;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

public class Injector {

    public Map<Class, Supplier<Object>> factories = new HashMap<>();
    public Map<Class, Object> singletons = new HashMap<>();

    public <T> Injector addFactory(Class<T> clazz, Supplier<T> f) {
        factories.put(clazz, (Supplier<Object>)f);
        return this;
    }
    public <T> Injector addSingleton(Class<T> clazz, T o) {
        singletons.put(clazz, (Object)o);
        return this;
    }

    public <T> Constructor<T> chooseConstructor(Class<T> clazz) {
        return (Constructor<T>)clazz.getConstructors()[0];
    }

    public <T> Object[] getConstructorArgs(Constructor<T> constructor) {
        return Arrays.stream(constructor.getParameterTypes())
                .map(this::getInstance)
                .toArray();
    }

    public <T> T getInstance(Class<T> clazz) {
        if (singletons.containsKey(clazz)) {
            return (T)singletons.get(clazz);
        }
        if (factories.containsKey(clazz)) {
            return (T)factories.get(clazz).get();
        }
        try {
            Constructor<T> constructor = chooseConstructor(clazz);
            return constructor.newInstance(getConstructorArgs(constructor));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
