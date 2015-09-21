package io.github.jimmyhmiller.objectalgebra;

class PrintJsonString implements JsonString<Print> {

    @Override
    public Print lit(String s) {
        return level -> "\"" + s + "\"";
    }
}