import com.jsparser.Parser;
import com.jsparser.ast.Program;

public class test_parser {
    public static void main(String[] args) {
        String[] testCases = {
            "throw new Error('test');",
            "try { x(); } catch(e) { }",
            "function f() { var x = [1, 2, 3]; }",
            "var x = { ...y };",
            "function f(...args) { }",
            "[...arr]"
        };

        for (String code : testCases) {
            System.out.println("\nTesting: " + code);
            try {
                Program ast = Parser.parse(code);
                System.out.println("  ✓ Parsed successfully");
            } catch (Exception e) {
                System.out.println("  ✗ " + e.getMessage());
            }
        }
    }
}
