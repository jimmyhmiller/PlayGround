package io.github.jimmyhmiller.objectalgebra;

public class JsonStringToJsonValue implements JsonString<ToJsonValue> {
    @Override
    public ToJsonValue lit(String s) {
        return () -> javax.json.Json.createObjectBuilder().add("test", s).build().getJsonString("test");
    }
}
