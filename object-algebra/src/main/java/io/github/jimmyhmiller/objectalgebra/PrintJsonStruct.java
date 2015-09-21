package io.github.jimmyhmiller.objectalgebra;

/**
 * Created by jimmyhmiller on 9/9/15.
 */
public class PrintJsonStruct extends PrintJsonObject implements JsonStruct<Print> {
    @Override
    public Print String(String s) {
        return new PrintJsonString().lit(s);
    }

    @Override
    public JsonObject<Print> Object() {
        return new PrintJsonObject();
    }
}
