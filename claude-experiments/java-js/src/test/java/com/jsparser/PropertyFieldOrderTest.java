package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.Map;

public class PropertyFieldOrderTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testFieldOrderDoesntMatterInStructuralComparison() throws Exception {
        // Create two JSONs with different field order but same values
        String json1 = "{\"kind\": \"init\", \"value\": 42}";
        String json2 = "{\"value\": 42, \"kind\": \"init\"}";

        Object obj1 = mapper.readValue(json1, Object.class);
        Object obj2 = mapper.readValue(json2, Object.class);

        System.out.println("obj1: " + obj1);
        System.out.println("obj2: " + obj2);
        System.out.println("Are they equal? " + java.util.Objects.deepEquals(obj1, obj2));
        System.out.println("obj1 class: " + obj1.getClass());
        System.out.println("obj2 class: " + obj2.getClass());

        if (obj1 instanceof Map && obj2 instanceof Map) {
            Map<?, ?> map1 = (Map<?, ?>) obj1;
            Map<?, ?> map2 = (Map<?, ?>) obj2;
            System.out.println("map1.equals(map2): " + map1.equals(map2));
        }
    }
}
