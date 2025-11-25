import com.jsparser.*;
import com.jsparser.ast.*;

public class TestParseLarge {
    public static void main(String[] args) throws Exception {
        String code = "var x = 10000000000000000; var y = {value: 10000000000000000};";
        try {
            Program ast = Parser.parse(code, false);
            System.out.println("✓ Successfully parsed code with large number!");
            System.out.println("AST has " + ast.body().size() + " statements");
        } catch (Exception e) {
            System.out.println("✗ Failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
