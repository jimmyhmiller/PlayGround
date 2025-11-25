import com.jsparser.*;
import com.jsparser.ast.*;
import com.fasterxml.jackson.databind.ObjectMapper;

public class TestParseBig {
    public static void main(String[] args) throws Exception {
        String code = "var x = 9223372036854776000;";
        Program ast = Parser.parse(code, false);
        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(ast);
        
        // Find the value field
        if (json.contains("9223372036854776000")) {
            System.out.println("✓ Contains exact number: 9223372036854776000");
        } else if (json.contains("9223372036854775807")) {
            System.out.println("✗ Truncated to Long.MAX_VALUE: 9223372036854775807");
        } else if (json.contains("9.223372036854776E18")) {
            System.out.println("✓ Stored as double: 9.223372036854776E18");
        } else {
            System.out.println("Number representation: " + json.substring(json.indexOf("value") + 7, json.indexOf("value") + 30));
        }
    }
}
