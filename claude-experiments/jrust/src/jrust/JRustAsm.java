package jrust;

import org.objectweb.asm.*;

import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;

import static org.objectweb.asm.Opcodes.*;

/**
 * Handle-based wrapper around ASM for use from JRust programs.
 * All methods use simple types (int, long, double, String) so JRust can call them.
 * Objects are stored internally and referenced by integer handles.
 */
public class JRustAsm {
    private static final ArrayList<Object> objects = new ArrayList<>();

    private static int store(Object obj) {
        objects.add(obj);
        return objects.size() - 1;
    }

    @SuppressWarnings("unchecked")
    private static <T> T get(int handle) {
        return (T) objects.get(handle);
    }

    // === ClassWriter ===

    public static int cw_new(int flags) {
        return store(new ClassWriter(flags) {
            @Override
            protected String getCommonSuperClass(String type1, String type2) {
                try {
                    return super.getCommonSuperClass(type1, type2);
                } catch (RuntimeException e) {
                    // For enum variants like "Expr$FieldAccessE" and "Expr$MethodCallE",
                    // the common superclass is the base enum type "Expr"
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
        });
    }

    public static void cw_visit(int h, int version, int access, String name, String superName) {
        ClassWriter cw = get(h);
        cw.visit(version, access, name, null, superName, null);
    }

    public static int cw_visit_method(int h, int access, String name, String desc) {
        ClassWriter cw = get(h);
        MethodVisitor mv = cw.visitMethod(access, name, desc, null, null);
        return store(mv);
    }

    public static int cw_visit_field(int h, int access, String name, String desc) {
        ClassWriter cw = get(h);
        FieldVisitor fv = cw.visitField(access, name, desc, null, null);
        return store(fv);
    }

    public static int cw_visit_field_int(int h, int access, String name, String desc, int val) {
        ClassWriter cw = get(h);
        FieldVisitor fv = cw.visitField(access, name, desc, null, val);
        return store(fv);
    }

    public static int cw_visit_field_str(int h, int access, String name, String desc, String val) {
        ClassWriter cw = get(h);
        FieldVisitor fv = cw.visitField(access, name, desc, null, val);
        return store(fv);
    }

    public static void cw_end(int h) {
        ClassWriter cw = get(h);
        cw.visitEnd();
    }

    public static void cw_write(int h, String dir, String name) {
        ClassWriter cw = get(h);
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

    // === MethodVisitor ===

    public static void mv_code(int h) {
        MethodVisitor mv = get(h);
        mv.visitCode();
    }

    public static void mv_insn(int h, int opcode) {
        MethodVisitor mv = get(h);
        mv.visitInsn(opcode);
    }

    public static void mv_int_insn(int h, int opcode, int operand) {
        MethodVisitor mv = get(h);
        mv.visitIntInsn(opcode, operand);
    }

    public static void mv_var_insn(int h, int opcode, int slot) {
        MethodVisitor mv = get(h);
        mv.visitVarInsn(opcode, slot);
    }

    public static void mv_field_insn(int h, int opcode, String owner, String name, String desc) {
        MethodVisitor mv = get(h);
        mv.visitFieldInsn(opcode, owner, name, desc);
    }

    public static void mv_method_insn(int h, int opcode, String owner, String name, String desc, int isInterface) {
        MethodVisitor mv = get(h);
        mv.visitMethodInsn(opcode, owner, name, desc, isInterface != 0);
    }

    public static void mv_jump_insn(int h, int opcode, int label) {
        MethodVisitor mv = get(h);
        Label l = get(label);
        mv.visitJumpInsn(opcode, l);
    }

    public static void mv_label(int h, int label) {
        MethodVisitor mv = get(h);
        Label l = get(label);
        mv.visitLabel(l);
    }

    public static void mv_ldc_str(int h, String value) {
        MethodVisitor mv = get(h);
        mv.visitLdcInsn(value);
    }

    public static void mv_ldc_int(int h, int value) {
        MethodVisitor mv = get(h);
        mv.visitLdcInsn(value);
    }

    public static void mv_ldc_long(int h, long value) {
        MethodVisitor mv = get(h);
        mv.visitLdcInsn(value);
    }

    public static void mv_ldc_double(int h, double value) {
        MethodVisitor mv = get(h);
        mv.visitLdcInsn(value);
    }

    public static void mv_type_insn(int h, int opcode, String type) {
        MethodVisitor mv = get(h);
        mv.visitTypeInsn(opcode, type);
    }

    public static void mv_iinc(int h, int slot, int increment) {
        MethodVisitor mv = get(h);
        mv.visitIincInsn(slot, increment);
    }

    public static void mv_maxs(int h, int maxStack, int maxLocals) {
        MethodVisitor mv = get(h);
        mv.visitMaxs(maxStack, maxLocals);
    }

    public static void mv_end(int h) {
        MethodVisitor mv = get(h);
        mv.visitEnd();
    }

    // === FieldVisitor ===

    public static void fv_end(int h) {
        FieldVisitor fv = get(h);
        fv.visitEnd();
    }

    // === Label ===

    public static int label_new() {
        return store(new Label());
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
