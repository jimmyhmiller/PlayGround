package jrust;

import org.objectweb.asm.ClassWriter;

import java.io.*;
import java.nio.file.*;

/**
 * Runtime utilities for JRust programs.
 * Provides file I/O, process management, argument handling, and ClassWriter creation.
 */
public class JRustRuntime {

    // === ClassWriter creation (needs subclass override) ===

    public static ClassWriter create_class_writer(int flags) {
        return new ClassWriter(flags) {
            @Override
            protected String getCommonSuperClass(String type1, String type2) {
                try {
                    return super.getCommonSuperClass(type1, type2);
                } catch (RuntimeException e) {
                    if (type1.contains("$") && type2.contains("$")) {
                        String base1 = type1.substring(0, type1.indexOf('$'));
                        String base2 = type2.substring(0, type2.indexOf('$'));
                        if (base1.equals(base2)) {
                            return base1;
                        }
                    }
                    if (type1.contains("$")) {
                        String base1 = type1.substring(0, type1.indexOf('$'));
                        if (base1.equals(type2)) return type2;
                    }
                    if (type2.contains("$")) {
                        String base2 = type2.substring(0, type2.indexOf('$'));
                        if (base2.equals(type1)) return type1;
                    }
                    return "java/lang/Object";
                }
            }
        };
    }

    // === Write class file ===

    public static void write_class(ClassWriter cw, String dir, String name) {
        byte[] bytes = cw.toByteArray();
        try {
            File d = new File(dir);
            d.mkdirs();
            File f = new File(d, name + ".class");
            try (FileOutputStream fos = new FileOutputStream(f)) {
                fos.write(bytes);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to write class file: " + e.getMessage());
        }
    }

    // === File I/O ===

    public static String read_file(String path) {
        try {
            return Files.readString(Path.of(path));
        } catch (IOException e) {
            throw new RuntimeException("Failed to read file: " + path + ": " + e.getMessage());
        }
    }

    public static void write_file(String path, String content) {
        try {
            Files.writeString(Path.of(path), content);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write file: " + path + ": " + e.getMessage());
        }
    }

    public static void mkdir(String path) {
        new File(path).mkdirs();
    }

    // === Process ===

    public static int run_command(String command) {
        try {
            ProcessBuilder pb = new ProcessBuilder("sh", "-c", command);
            pb.inheritIO();
            Process proc = pb.start();
            return proc.waitFor();
        } catch (Exception e) {
            throw new RuntimeException("Failed to run command: " + command + ": " + e.getMessage());
        }
    }

    // === Number parsing ===

    public static int parse_int(String s) { return Integer.parseInt(s); }
    public static long parse_long(String s) { return Long.parseLong(s); }
    public static double parse_double(String s) { return Double.parseDouble(s); }

    // === Args ===

    private static String[] programArgs;

    public static void set_args(String[] args) {
        programArgs = args;
    }

    public static int args_len() {
        return programArgs != null ? programArgs.length : 0;
    }

    public static String args_get(int index) {
        return programArgs[index];
    }
}
