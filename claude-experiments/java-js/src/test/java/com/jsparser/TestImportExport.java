package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestImportExport {
    @Test
    void testImportDefault() throws Exception {
        try {
            String source = "import foo from 'module';";
            Parser.parse(source);
            System.out.println("✓ import default works");
        } catch (Exception e) {
            System.out.println("✗ import default failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testImportNamed() throws Exception {
        try {
            String source = "import { foo, bar } from 'module';";
            Parser.parse(source);
            System.out.println("✓ import named works");
        } catch (Exception e) {
            System.out.println("✗ import named failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testImportNamespace() throws Exception {
        try {
            String source = "import * as foo from 'module';";
            Parser.parse(source);
            System.out.println("✓ import namespace works");
        } catch (Exception e) {
            System.out.println("✗ import namespace failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testImportSideEffect() throws Exception {
        try {
            String source = "import 'module';";
            Parser.parse(source);
            System.out.println("✓ import side-effect works");
        } catch (Exception e) {
            System.out.println("✗ import side-effect failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testExportNamed() throws Exception {
        try {
            String source = "export const x = 1;";
            Parser.parse(source);
            System.out.println("✓ export named works");
        } catch (Exception e) {
            System.out.println("✗ export named failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testExportDefault() throws Exception {
        try {
            String source = "export default function() { }";
            Parser.parse(source);
            System.out.println("✓ export default works");
        } catch (Exception e) {
            System.out.println("✗ export default failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testExportList() throws Exception {
        try {
            String source = "export { foo, bar };";
            Parser.parse(source);
            System.out.println("✓ export list works");
        } catch (Exception e) {
            System.out.println("✗ export list failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testExportAll() throws Exception {
        try {
            String source = "export * from 'module';";
            Parser.parse(source);
            System.out.println("✓ export all works");
        } catch (Exception e) {
            System.out.println("✗ export all failed: " + e.getMessage());
            throw e;
        }
    }
}
