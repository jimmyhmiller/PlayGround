package io.github.jimmyhmiller.objectalgebra;

import javax.json.Json;
import javax.json.JsonArrayBuilder;
import javax.json.JsonValue;
import java.util.Arrays;

public class JsonValueCollection implements Collection<Converter<JsonValue>> {
    @Override
    public Converter<JsonValue> create(Object... args) {
        return () -> {
            JsonArrayBuilder b = Json.createArrayBuilder();
            Arrays.asList(args).forEach( o -> b.add((String) o));
            return b.build();
        };
    }
}
