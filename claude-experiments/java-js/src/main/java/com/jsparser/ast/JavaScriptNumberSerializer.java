package com.jsparser.ast;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import java.io.IOException;

/**
 * Custom serializer for JavaScript numbers that formats doubles as integers
 * when they represent whole numbers, matching JavaScript's JSON.stringify behavior.
 */
public class JavaScriptNumberSerializer extends JsonSerializer<Object> {
    @Override
    public void serialize(Object value, JsonGenerator gen, SerializerProvider serializers) throws IOException {
        if (value == null) {
            gen.writeNull();
        } else if (value instanceof Double) {
            double d = (Double) value;
            // Check if the double represents a whole number
            if (d == Math.floor(d) && !Double.isInfinite(d) && !Double.isNaN(d)) {
                // JavaScript uses scientific notation for numbers >= 1e21
                // For smaller whole numbers, format as integer without scientific notation
                if (Math.abs(d) < 1e21) {
                    // Use BigDecimal to get the exact decimal representation
                    java.math.BigDecimal bd = java.math.BigDecimal.valueOf(d);
                    gen.writeNumber(bd.toBigInteger());
                } else {
                    // Large numbers: use scientific notation like JavaScript
                    gen.writeNumber(d);
                }
            } else {
                gen.writeNumber(d);
            }
        } else if (value instanceof Float) {
            float f = (Float) value;
            if (f == Math.floor(f) && !Float.isInfinite(f) && !Float.isNaN(f)) {
                if (Math.abs(f) < 1e21) {
                    java.math.BigDecimal bd = java.math.BigDecimal.valueOf(f);
                    gen.writeNumber(bd.toBigInteger());
                } else {
                    gen.writeNumber(f);
                }
            } else {
                gen.writeNumber(f);
            }
        } else if (value instanceof Integer) {
            gen.writeNumber((Integer) value);
        } else if (value instanceof Long) {
            gen.writeNumber((Long) value);
        } else if (value instanceof Boolean) {
            gen.writeBoolean((Boolean) value);
        } else if (value instanceof String) {
            gen.writeString((String) value);
        } else {
            // Fallback for other types
            gen.writeObject(value);
        }
    }
}
