package io.github.jimmyhmiller.objectalgebra;


public class JsonToJsonStructValue implements JsonStruct<ToJsonValue> {
    @Override
    public ToJsonValue String(String s) {
        return new JsonStringToJsonValue().lit(s);
    }

    @Override
    public JsonObject<ToJsonValue> Object() {
        return new JsonObjectToJsonValue();
    }
}
