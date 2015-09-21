package io.github.jimmyhmiller.objectalgebra;

import javax.json.*;


public class JsonObjectToJsonValue implements JsonObject<ToJsonValue> {

    JsonObjectBuilder builder = javax.json.Json.createObjectBuilder();

    @Override
    public JsonObject<ToJsonValue> add(String key, ToJsonValue value) {
        builder.add(key, value.convert());
        return this;
    }

    @Override
    public ToJsonValue build() {
        return builder::build;
    }
}
