package io.github.jimmyhmiller.objectalgebra;


import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class PrintJsonObject extends PrintJsonString implements JsonObject<Print> {

    public PrintJsonObject() {}

    private PrintJsonObject(Map<String, Print> map) {
        this.map = map;
    }

    private static String stringOfNLength(Integer n, String s) {
        StringBuilder builder = new StringBuilder(n);
        for (int i = 0; i < n; i++){
            builder.append(s);
        }
        return builder.toString();
    }

    private Map<String, Print> map = new HashMap<>();


    @Override
    public JsonObject<Print> add(String key, Print value) {
        map.put(key, value);
        return new PrintJsonObject(map.entrySet().stream().collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
    }

    @Override
    public Print build() {
        return level -> {
            StringBuilder s = new StringBuilder();
            s.append("{\n");

            Set<Map.Entry<String,Print>> set = map.entrySet();
            Integer i = 0;
            for (Map.Entry<String, Print> e : set) {
                s.append(stringOfNLength((level + 1) * 4, " "));
                s.append(e.getKey());
                s.append(": ");
                s.append(e.getValue().print(level + 1));
                if (i < set.size() -1) {
                    s.append(",");
                }
                s.append("\n");
                i += 1;
            }

            map.forEach((k, v) -> {

            });
            s.append(stringOfNLength(level * 4, " "));
            s.append("}");
            return s.toString();
        };
    }
}
