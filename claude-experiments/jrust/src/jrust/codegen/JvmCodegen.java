package jrust.codegen;

import jrust.ast.Expr;
import jrust.ast.Item;
import jrust.ast.Pattern;
import jrust.ast.Program;
import jrust.ast.Stmt;
import jrust.ast.Type;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;

import static org.objectweb.asm.Opcodes.*;

public class JvmCodegen {
    private final Program program;
    private final String outputDir;
    private final Map<String, Item.StructDef> structs = new HashMap<>();
    private final Map<String, Item.EnumDef> enums = new HashMap<>();
    private final Map<String, List<Item.FnDef>> implMethods = new HashMap<>();
    private final List<Item.FnDef> topLevelFns = new ArrayList<>();
    private final List<Item.ConstDef> constants = new ArrayList<>();
    private final Map<String, String> imports = new HashMap<>(); // simple name -> jvm class name

    // Per-method state
    private MethodVisitor mv;
    private final Map<String, Integer> locals = new HashMap<>();
    private final Map<String, Type> localTypes = new HashMap<>();
    private int nextLocal;
    private String currentClass;
    private Type expectedType; // set by let binding for imported class return types
    private Label breakLabel; // target for break statements
    private Label continueLabel; // target for continue statements

    public JvmCodegen(Program program, String outputDir) {
        this.program = program;
        this.outputDir = outputDir;
    }

    private static ClassWriter createClassWriter() {
        return new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS) {
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
        };
    }

    public void generate() throws IOException {
        // First pass: collect structs, enums, impls, top-level fns, constants, imports
        for (Item item : program.items()) {
            switch (item) {
                case Item.StructDef sd -> structs.put(sd.name(), sd);
                case Item.EnumDef ed -> enums.put(ed.name(), ed);
                case Item.ImplDef id -> {
                    implMethods.computeIfAbsent(id.typeName(), k -> new ArrayList<>()).addAll(id.methods());
                }
                case Item.FnDef fd -> topLevelFns.add(fd);
                case Item.Import imp -> registerImport(imp.path());
                case Item.ConstDef cd -> constants.add(cd);
            }
        }

        // Generate struct classes
        for (Item.StructDef sd : structs.values()) {
            generateStruct(sd);
        }

        // Generate enum classes
        for (Item.EnumDef ed : enums.values()) {
            generateEnum(ed);
        }

        // Generate Main class with top-level fns and constants
        generateMainClass();
    }

    private void registerImport(String path) {
        // e.g., "java.util.ArrayList" -> "ArrayList" maps to "java/util/ArrayList"
        String jvmName = path.replace('.', '/');
        String simpleName = path.substring(path.lastIndexOf('.') + 1);
        imports.put(simpleName, jvmName);
    }

    // --- Struct class generation ---

    private void generateStruct(Item.StructDef sd) throws IOException {
        ClassWriter cw = createClassWriter();
        cw.visit(V21, ACC_PUBLIC | ACC_SUPER, sd.name(), null, "java/lang/Object", null);

        // Fields
        for (Item.Field field : sd.fields()) {
            cw.visitField(ACC_PUBLIC, field.name(), typeDescriptor(field.type()), null, null).visitEnd();
        }

        // Default constructor
        MethodVisitor init = cw.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
        init.visitCode();
        init.visitVarInsn(ALOAD, 0);
        init.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        init.visitInsn(RETURN);
        init.visitMaxs(0, 0);
        init.visitEnd();

        // Impl methods
        List<Item.FnDef> methods = implMethods.getOrDefault(sd.name(), List.of());
        for (Item.FnDef method : methods) {
            generateMethod(cw, sd.name(), method);
        }

        cw.visitEnd();
        writeClass(sd.name(), cw.toByteArray());
    }

    // --- Enum class generation ---

    private void generateEnum(Item.EnumDef ed) throws IOException {
        // Base class
        ClassWriter baseCw = createClassWriter();
        baseCw.visit(V21, ACC_PUBLIC | ACC_SUPER, ed.name(), null, "java/lang/Object", null);

        // Default constructor
        MethodVisitor baseInit = baseCw.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
        baseInit.visitCode();
        baseInit.visitVarInsn(ALOAD, 0);
        baseInit.visitMethodInsn(INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        baseInit.visitInsn(RETURN);
        baseInit.visitMaxs(0, 0);
        baseInit.visitEnd();

        // Impl methods on the enum base class
        List<Item.FnDef> methods = implMethods.getOrDefault(ed.name(), List.of());
        for (Item.FnDef method : methods) {
            generateMethod(baseCw, ed.name(), method);
        }

        baseCw.visitEnd();
        writeClass(ed.name(), baseCw.toByteArray());

        // Variant subclasses
        for (Item.EnumVariant variant : ed.variants()) {
            generateEnumVariant(ed.name(), variant);
        }
    }

    private void generateEnumVariant(String enumName, Item.EnumVariant variant) throws IOException {
        String variantClass = enumName + "$" + variant.name();
        ClassWriter cw = createClassWriter();
        cw.visit(V21, ACC_PUBLIC | ACC_SUPER, variantClass, null, enumName, null);

        // Fields
        for (Item.Field field : variant.fields()) {
            cw.visitField(ACC_PUBLIC, field.name(), typeDescriptor(field.type()), null, null).visitEnd();
        }

        // Constructor
        MethodVisitor init = cw.visitMethod(ACC_PUBLIC, "<init>", "()V", null, null);
        init.visitCode();
        init.visitVarInsn(ALOAD, 0);
        init.visitMethodInsn(INVOKESPECIAL, enumName, "<init>", "()V", false);
        init.visitInsn(RETURN);
        init.visitMaxs(0, 0);
        init.visitEnd();

        cw.visitEnd();
        writeClass(variantClass, cw.toByteArray());
    }

    // --- Main class generation ---

    private void generateMainClass() throws IOException {
        ClassWriter cw = createClassWriter();
        cw.visit(V21, ACC_PUBLIC | ACC_SUPER, "Main", null, "java/lang/Object", null);

        // Constants as static fields
        for (Item.ConstDef cd : constants) {
            cw.visitField(ACC_PUBLIC | ACC_STATIC | ACC_FINAL, cd.name(),
                    typeDescriptor(cd.type()), null, getConstantValue(cd)).visitEnd();
        }

        // Static initializer for non-trivial constants
        if (!constants.isEmpty()) {
            MethodVisitor clinit = cw.visitMethod(ACC_STATIC, "<clinit>", "()V", null, null);
            clinit.visitCode();
            for (Item.ConstDef cd : constants) {
                if (getConstantValue(cd) == null) {
                    // Non-trivial constant: need to evaluate
                    locals.clear();
                    localTypes.clear();
                    nextLocal = 0;
                    currentClass = "Main";
                    mv = clinit;
                    generateExpr(cd.value());
                    clinit.visitFieldInsn(PUTSTATIC, "Main", cd.name(), typeDescriptor(cd.type()));
                }
            }
            clinit.visitInsn(RETURN);
            clinit.visitMaxs(0, 0);
            clinit.visitEnd();
        }

        for (Item.FnDef fn : topLevelFns) {
            generateMethod(cw, "Main", fn);
        }

        cw.visitEnd();
        writeClass("Main", cw.toByteArray());
    }

    private Object getConstantValue(Item.ConstDef cd) {
        // Return compile-time constant for simple literals
        if (cd.value() instanceof Expr.IntLit il) {
            if (cd.type() instanceof Type.Simple s && s.name().equals("i64")) return il.value();
            return (int) il.value();
        }
        if (cd.value() instanceof Expr.FloatLit fl) return fl.value();
        if (cd.value() instanceof Expr.StringLit sl) return sl.value();
        if (cd.value() instanceof Expr.BoolLit bl) return bl.value() ? 1 : 0;
        return null; // need runtime initialization
    }

    // --- Method generation ---

    private void generateMethod(ClassWriter cw, String className, Item.FnDef fn) {
        locals.clear();
        localTypes.clear();
        nextLocal = 0;
        currentClass = className;

        boolean isMain = fn.name().equals("main") && className.equals("Main");
        boolean hasSelfParam = !fn.params().isEmpty() && fn.params().get(0).isSelf();
        boolean isStatic = !hasSelfParam;

        String descriptor = methodDescriptor(fn, isStatic, className);
        int access = ACC_PUBLIC | (isStatic ? ACC_STATIC : 0);
        String methodName = isMain ? "main" : fn.name();

        mv = cw.visitMethod(access, methodName, descriptor, null, null);
        mv.visitCode();

        // Assign local slots for parameters
        if (isMain && fn.name().equals("main")) {
            locals.put("__args__", nextLocal);
            // Pass args to JRustAsm so args_len/args_get work
            mv.visitVarInsn(ALOAD, nextLocal);
            mv.visitMethodInsn(INVOKESTATIC, "jrust/JRustAsm", "set_args",
                    "([Ljava/lang/String;)V", false);
            nextLocal++;
        }

        for (Item.Param param : fn.params()) {
            if (param.isSelf()) {
                locals.put("self", nextLocal);
                localTypes.put("self", new Type.Simple(className));
                nextLocal++;
            } else {
                locals.put(param.name(), nextLocal);
                localTypes.put(param.name(), param.type());
                nextLocal += localSize(param.type());
            }
        }

        // Generate body
        for (Stmt stmt : fn.body()) {
            generateStmt(stmt);
        }

        // Add return if void and no explicit return
        if (fn.returnType() instanceof Type.Void) {
            mv.visitInsn(RETURN);
        }

        mv.visitMaxs(0, 0);
        mv.visitEnd();
    }

    private String methodDescriptor(Item.FnDef fn, boolean isStatic, String className) {
        StringBuilder desc = new StringBuilder("(");

        if (fn.name().equals("main") && className.equals("Main")) {
            desc.append("[Ljava/lang/String;");
        } else {
            for (Item.Param param : fn.params()) {
                if (param.isSelf()) continue;
                desc.append(typeDescriptor(param.type()));
            }
        }
        desc.append(")");
        desc.append(typeDescriptor(fn.returnType()));
        return desc.toString();
    }

    // --- Statement codegen ---

    private void generateStmt(Stmt stmt) {
        switch (stmt) {
            case Stmt.Let let -> generateLet(let);
            case Stmt.Return ret -> generateReturn(ret);
            case Stmt.Break b -> { if (breakLabel != null) mv.visitJumpInsn(GOTO, breakLabel); }
            case Stmt.Continue c -> { if (continueLabel != null) mv.visitJumpInsn(GOTO, continueLabel); }
            case Stmt.ExprStmt es -> {
                Type t = generateExpr(es.expr());
                // Pop value if non-void
                if (t != null && !(t instanceof Type.Void)) {
                    if (isWide(t)) {
                        mv.visitInsn(POP2);
                    } else {
                        mv.visitInsn(POP);
                    }
                }
            }
        }
    }

    private void generateLet(Stmt.Let let) {
        Type type = let.type();
        if (type == null && let.init() != null) {
            type = inferType(let.init());
        }
        if (type == null) {
            throw new RuntimeException("Cannot infer type for let binding '" + let.name() + "'");
        }

        int slot = nextLocal;
        locals.put(let.name(), slot);
        localTypes.put(let.name(), type);
        nextLocal += localSize(type);

        if (let.init() != null) {
            Type prevExpected = expectedType;
            expectedType = type;
            generateExpr(let.init());
            expectedType = prevExpected;
            storeLocal(slot, type);
        }
    }

    private void generateReturn(Stmt.Return ret) {
        if (ret.value() == null) {
            mv.visitInsn(RETURN);
        } else {
            Type t = generateExpr(ret.value());
            generateReturnInsn(t);
        }
    }

    private void generateReturnInsn(Type t) {
        if (t instanceof Type.Void) {
            mv.visitInsn(RETURN);
        } else if (t instanceof Type.Simple s) {
            switch (s.name()) {
                case "i32", "bool", "char" -> mv.visitInsn(IRETURN);
                case "i64" -> mv.visitInsn(LRETURN);
                case "f64" -> mv.visitInsn(DRETURN);
                default -> mv.visitInsn(ARETURN);
            }
        } else {
            mv.visitInsn(ARETURN);
        }
    }

    // --- Expression codegen ---

    private Type generateExpr(Expr expr) {
        return switch (expr) {
            case Expr.IntLit il -> {
                long val = il.value();
                if (val >= Integer.MIN_VALUE && val <= Integer.MAX_VALUE) {
                    pushInt((int) val);
                    yield new Type.Simple("i32");
                } else {
                    mv.visitLdcInsn(val);
                    yield new Type.Simple("i64");
                }
            }
            case Expr.FloatLit fl -> {
                mv.visitLdcInsn(fl.value());
                yield new Type.Simple("f64");
            }
            case Expr.StringLit sl -> {
                mv.visitLdcInsn(sl.value());
                yield new Type.Simple("String");
            }
            case Expr.CharLit cl -> {
                pushInt(cl.value());
                yield new Type.Simple("char");
            }
            case Expr.BoolLit bl -> {
                mv.visitInsn(bl.value() ? ICONST_1 : ICONST_0);
                yield new Type.Simple("bool");
            }
            case Expr.NullLit nl -> {
                mv.visitInsn(ACONST_NULL);
                yield new Type.Simple("null");
            }
            case Expr.Ident id -> generateIdent(id);
            case Expr.SelfExpr se -> {
                mv.visitVarInsn(ALOAD, locals.get("self"));
                yield new Type.Simple(currentClass);
            }
            case Expr.Binary bin -> generateBinary(bin);
            case Expr.Unary un -> generateUnary(un);
            case Expr.Call call -> generateCall(call);
            case Expr.MethodCall mc -> generateMethodCall(mc);
            case Expr.FieldAccess fa -> generateFieldAccess(fa);
            case Expr.StructInit si -> generateStructInit(si);
            case Expr.StaticCall sc -> generateStaticCall(sc);
            case Expr.Assign assign -> generateAssign(assign);
            case Expr.If ifExpr -> generateIf(ifExpr);
            case Expr.While whileExpr -> generateWhile(whileExpr);
            case Expr.Block block -> generateBlock(block);
            case Expr.ForRange fr -> generateForRange(fr);
            case Expr.ForEach fe -> generateForEach(fe);
            case Expr.Match match -> generateMatch(match);
            case Expr.Index idx -> generateIndex(idx);
            case Expr.EnumInit ei -> generateEnumInit(ei);
            case Expr.Throw th -> generateThrow(th);
            case Expr.ArrayLit al -> generateArrayLit(al);
            case Expr.Cast cast -> generateCast(cast);
        };
    }

    private Type generateIdent(Expr.Ident id) {
        // Check locals first
        Integer slot = locals.get(id.name());
        if (slot != null) {
            Type type = localTypes.get(id.name());
            loadLocal(slot, type);
            return type;
        }
        // Check constants
        for (Item.ConstDef cd : constants) {
            if (cd.name().equals(id.name())) {
                mv.visitFieldInsn(GETSTATIC, "Main", cd.name(), typeDescriptor(cd.type()));
                return cd.type();
            }
        }
        throw new RuntimeException("Undefined variable: " + id.name());
    }

    private Type generateBinary(Expr.Binary bin) {
        // Handle null comparisons specially
        if (bin.right() instanceof Expr.NullLit && (bin.op().equals("==") || bin.op().equals("!="))) {
            Type leftType = generateExpr(bin.left());
            Label trueLabel = new Label();
            Label endLabel = new Label();
            mv.visitJumpInsn(bin.op().equals("==") ? IFNULL : IFNONNULL, trueLabel);
            mv.visitInsn(ICONST_0);
            mv.visitJumpInsn(GOTO, endLabel);
            mv.visitLabel(trueLabel);
            mv.visitInsn(ICONST_1);
            mv.visitLabel(endLabel);
            return new Type.Simple("bool");
        }
        if (bin.left() instanceof Expr.NullLit && (bin.op().equals("==") || bin.op().equals("!="))) {
            Type rightType = generateExpr(bin.right());
            Label trueLabel = new Label();
            Label endLabel = new Label();
            mv.visitJumpInsn(bin.op().equals("==") ? IFNULL : IFNONNULL, trueLabel);
            mv.visitInsn(ICONST_0);
            mv.visitJumpInsn(GOTO, endLabel);
            mv.visitLabel(trueLabel);
            mv.visitInsn(ICONST_1);
            mv.visitLabel(endLabel);
            return new Type.Simple("bool");
        }

        Type leftType = generateExpr(bin.left());
        Type rightType = generateExpr(bin.right());

        // String concatenation with +
        if (isString(leftType) && bin.op().equals("+")) {
            if (!isString(rightType)) {
                generateToString(rightType);
            }
            mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "concat",
                    "(Ljava/lang/String;)Ljava/lang/String;", false);
            return new Type.Simple("String");
        }

        // String equality with == / !=
        if (isString(leftType) && isString(rightType) && (bin.op().equals("==") || bin.op().equals("!="))) {
            mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "equals",
                    "(Ljava/lang/Object;)Z", false);
            if (bin.op().equals("!=")) {
                mv.visitInsn(ICONST_1);
                mv.visitInsn(IXOR);
            }
            return new Type.Simple("bool");
        }

        String typeName = resolveSimpleTypeName(leftType);
        return switch (typeName) {
            case "i32", "bool", "char" -> {
                switch (bin.op()) {
                    case "+" -> mv.visitInsn(IADD);
                    case "-" -> mv.visitInsn(ISUB);
                    case "*" -> mv.visitInsn(IMUL);
                    case "/" -> mv.visitInsn(IDIV);
                    case "%" -> mv.visitInsn(IREM);
                    case "|" -> mv.visitInsn(IOR);
                    case "==" -> generateIntComparison(IF_ICMPEQ);
                    case "!=" -> generateIntComparison(IF_ICMPNE);
                    case "<" -> generateIntComparison(IF_ICMPLT);
                    case ">" -> generateIntComparison(IF_ICMPGT);
                    case "<=" -> generateIntComparison(IF_ICMPLE);
                    case ">=" -> generateIntComparison(IF_ICMPGE);
                    case "&&" -> mv.visitInsn(IAND);
                    case "||" -> mv.visitInsn(IOR);
                    default -> throw new RuntimeException("Unknown binary op: " + bin.op());
                }
                if (isComparisonOp(bin.op())) yield new Type.Simple("bool");
                yield leftType;
            }
            case "i64" -> {
                switch (bin.op()) {
                    case "+" -> mv.visitInsn(LADD);
                    case "-" -> mv.visitInsn(LSUB);
                    case "*" -> mv.visitInsn(LMUL);
                    case "/" -> mv.visitInsn(LDIV);
                    case "%" -> mv.visitInsn(LREM);
                    case "==", "!=", "<", ">", "<=", ">=" -> {
                        mv.visitInsn(LCMP);
                        generateIntComparisonZero(bin.op());
                    }
                    default -> throw new RuntimeException("Unknown binary op for i64: " + bin.op());
                }
                if (isComparisonOp(bin.op())) yield new Type.Simple("bool");
                yield leftType;
            }
            case "f64" -> {
                switch (bin.op()) {
                    case "+" -> mv.visitInsn(DADD);
                    case "-" -> mv.visitInsn(DSUB);
                    case "*" -> mv.visitInsn(DMUL);
                    case "/" -> mv.visitInsn(DDIV);
                    case "%" -> mv.visitInsn(DREM);
                    case "==", "!=", "<", ">", "<=", ">=" -> {
                        mv.visitInsn(DCMPL);
                        generateIntComparisonZero(bin.op());
                    }
                    default -> throw new RuntimeException("Unknown binary op for f64: " + bin.op());
                }
                if (isComparisonOp(bin.op())) yield new Type.Simple("bool");
                yield leftType;
            }
            default -> throw new RuntimeException("Cannot apply binary op '" + bin.op() + "' to type: " + typeName);
        };
    }

    private void generateToString(Type type) {
        if (type instanceof Type.Simple s) {
            switch (s.name()) {
                case "i32" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(I)Ljava/lang/String;", false);
                case "i64" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(J)Ljava/lang/String;", false);
                case "f64" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(D)Ljava/lang/String;", false);
                case "bool" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(Z)Ljava/lang/String;", false);
                case "char" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(C)Ljava/lang/String;", false);
                default -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;", false);
            }
        } else {
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/String", "valueOf", "(Ljava/lang/Object;)Ljava/lang/String;", false);
        }
    }

    private void generateIntComparison(int opcode) {
        Label trueLabel = new Label();
        Label endLabel = new Label();
        mv.visitJumpInsn(opcode, trueLabel);
        mv.visitInsn(ICONST_0);
        mv.visitJumpInsn(GOTO, endLabel);
        mv.visitLabel(trueLabel);
        mv.visitInsn(ICONST_1);
        mv.visitLabel(endLabel);
    }

    private void generateIntComparisonZero(String op) {
        Label trueLabel = new Label();
        Label endLabel = new Label();
        int opcode = switch (op) {
            case "==" -> IFEQ;
            case "!=" -> IFNE;
            case "<" -> IFLT;
            case ">" -> IFGT;
            case "<=" -> IFLE;
            case ">=" -> IFGE;
            default -> throw new RuntimeException("Unknown comparison: " + op);
        };
        mv.visitJumpInsn(opcode, trueLabel);
        mv.visitInsn(ICONST_0);
        mv.visitJumpInsn(GOTO, endLabel);
        mv.visitLabel(trueLabel);
        mv.visitInsn(ICONST_1);
        mv.visitLabel(endLabel);
    }

    private boolean isComparisonOp(String op) {
        return switch (op) {
            case "==", "!=", "<", ">", "<=", ">=" -> true;
            default -> false;
        };
    }

    private Type generateUnary(Expr.Unary un) {
        Type type = generateExpr(un.operand());
        String typeName = resolveSimpleTypeName(type);
        switch (un.op()) {
            case "-" -> {
                switch (typeName) {
                    case "i32" -> mv.visitInsn(INEG);
                    case "i64" -> mv.visitInsn(LNEG);
                    case "f64" -> mv.visitInsn(DNEG);
                    default -> throw new RuntimeException("Cannot negate type: " + typeName);
                }
            }
            case "!" -> {
                mv.visitInsn(ICONST_1);
                mv.visitInsn(IXOR);
            }
            default -> throw new RuntimeException("Unknown unary op: " + un.op());
        }
        return type;
    }

    private Type generateCall(Expr.Call call) {
        // Built-in: println
        if (call.name().equals("println")) {
            mv.visitFieldInsn(GETSTATIC, "java/lang/System", "out", "Ljava/io/PrintStream;");
            if (call.args().isEmpty()) {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/io/PrintStream", "println", "()V", false);
            } else {
                Type argType = generateExpr(call.args().get(0));
                String desc = "(" + printStreamDescriptor(argType) + ")V";
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/io/PrintStream", "println", desc, false);
            }
            return new Type.Void();
        }

        // Built-in: print
        if (call.name().equals("print")) {
            mv.visitFieldInsn(GETSTATIC, "java/lang/System", "out", "Ljava/io/PrintStream;");
            if (!call.args().isEmpty()) {
                Type argType = generateExpr(call.args().get(0));
                String desc = "(" + printStreamDescriptor(argType) + ")V";
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/io/PrintStream", "print", desc, false);
            }
            return new Type.Void();
        }

        // Built-in: eprintln
        if (call.name().equals("eprintln")) {
            mv.visitFieldInsn(GETSTATIC, "java/lang/System", "err", "Ljava/io/PrintStream;");
            if (call.args().isEmpty()) {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/io/PrintStream", "println", "()V", false);
            } else {
                Type argType = generateExpr(call.args().get(0));
                String desc = "(" + printStreamDescriptor(argType) + ")V";
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/io/PrintStream", "println", desc, false);
            }
            return new Type.Void();
        }

        // Built-in: panic
        if (call.name().equals("panic")) {
            mv.visitTypeInsn(NEW, "java/lang/RuntimeException");
            mv.visitInsn(DUP);
            if (!call.args().isEmpty()) {
                generateExpr(call.args().get(0));
                mv.visitMethodInsn(INVOKESPECIAL, "java/lang/RuntimeException", "<init>",
                        "(Ljava/lang/String;)V", false);
            } else {
                mv.visitMethodInsn(INVOKESPECIAL, "java/lang/RuntimeException", "<init>",
                        "()V", false);
            }
            mv.visitInsn(ATHROW);
            return new Type.Void();
        }

        // Built-in: exit
        if (call.name().equals("exit")) {
            generateExpr(call.args().get(0));
            mv.visitMethodInsn(INVOKESTATIC, "java/lang/System", "exit", "(I)V", false);
            return new Type.Void();
        }

        // Built-in: read_file
        if (call.name().equals("read_file")) {
            generateExpr(call.args().get(0));
            mv.visitMethodInsn(INVOKESTATIC, "jrust/JRustAsm", "read_file",
                    "(Ljava/lang/String;)Ljava/lang/String;", false);
            return new Type.Simple("String");
        }

        // Built-in: write_file
        if (call.name().equals("write_file")) {
            generateExpr(call.args().get(0));
            generateExpr(call.args().get(1));
            mv.visitMethodInsn(INVOKESTATIC, "jrust/JRustAsm", "write_file",
                    "(Ljava/lang/String;Ljava/lang/String;)V", false);
            return new Type.Void();
        }

        // Built-in: system
        if (call.name().equals("system")) {
            generateExpr(call.args().get(0));
            mv.visitMethodInsn(INVOKESTATIC, "jrust/JRustAsm", "run_command",
                    "(Ljava/lang/String;)I", false);
            return new Type.Simple("i32");
        }

        // Built-in: args_len
        if (call.name().equals("args_len")) {
            mv.visitMethodInsn(INVOKESTATIC, "jrust/JRustAsm", "args_len", "()I", false);
            return new Type.Simple("i32");
        }

        // Built-in: args_get
        if (call.name().equals("args_get")) {
            generateExpr(call.args().get(0));
            mv.visitMethodInsn(INVOKESTATIC, "jrust/JRustAsm", "args_get",
                    "(I)Ljava/lang/String;", false);
            return new Type.Simple("String");
        }

        // Top-level function call on Main class
        Item.FnDef fn = findTopLevelFn(call.name());
        if (fn == null) {
            throw new RuntimeException("Unknown function: " + call.name());
        }

        StringBuilder desc = new StringBuilder("(");
        for (int i = 0; i < call.args().size(); i++) {
            generateExpr(call.args().get(i));
            desc.append(typeDescriptor(fn.params().get(i).type()));
        }
        desc.append(")").append(typeDescriptor(fn.returnType()));

        mv.visitMethodInsn(INVOKESTATIC, "Main", call.name(), desc.toString(), false);
        return fn.returnType();
    }

    private String printStreamDescriptor(Type argType) {
        if (argType instanceof Type.Simple s) {
            return switch (s.name()) {
                case "i32" -> "I";
                case "i64" -> "J";
                case "f64" -> "D";
                case "bool" -> "Z";
                case "char" -> "C";
                case "String" -> "Ljava/lang/String;";
                default -> "Ljava/lang/Object;";
            };
        }
        return "Ljava/lang/Object;";
    }

    private Type generateMethodCall(Expr.MethodCall mc) {
        Type receiverType = generateExpr(mc.receiver());
        String typeName = resolveSimpleTypeName(receiverType);

        // Built-in String methods
        if (typeName.equals("String")) {
            return generateStringMethodCall(mc.method(), mc.args());
        }

        // Built-in Vec methods
        if (isVecType(receiverType)) {
            return generateVecMethodCall(receiverType, mc.method(), mc.args());
        }

        // Built-in Map methods
        if (isMapType(receiverType)) {
            return generateMapMethodCall(receiverType, mc.method(), mc.args());
        }

        // Built-in StringBuilder methods
        if (typeName.equals("StringBuilder")) {
            return generateStringBuilderMethodCall(mc.method(), mc.args());
        }

        // User-defined method
        Item.FnDef method = findImplMethod(typeName, mc.method());
        if (method == null) {
            throw new RuntimeException("Unknown method: " + typeName + "." + mc.method());
        }

        StringBuilder desc = new StringBuilder("(");
        int argIdx = 0;
        for (Item.Param p : method.params()) {
            if (!p.isSelf()) {
                if (argIdx < mc.args().size()) {
                    generateExpr(mc.args().get(argIdx));
                    desc.append(typeDescriptor(p.type()));
                    argIdx++;
                }
            }
        }
        desc.append(")").append(typeDescriptor(method.returnType()));

        // Use the base type for INVOKEVIRTUAL (works for both struct and enum methods)
        mv.visitMethodInsn(INVOKEVIRTUAL, typeName, mc.method(), desc.toString(), false);
        return method.returnType();
    }

    private Type generateStringMethodCall(String method, List<Expr> args) {
        return switch (method) {
            case "len", "length" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "length", "()I", false);
                yield new Type.Simple("i32");
            }
            case "char_at", "charAt" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "charAt", "(I)C", false);
                yield new Type.Simple("char");
            }
            case "contains" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "contains",
                        "(Ljava/lang/CharSequence;)Z", false);
                yield new Type.Simple("bool");
            }
            case "starts_with", "startsWith" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "startsWith",
                        "(Ljava/lang/String;)Z", false);
                yield new Type.Simple("bool");
            }
            case "ends_with", "endsWith" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "endsWith",
                        "(Ljava/lang/String;)Z", false);
                yield new Type.Simple("bool");
            }
            case "substring" -> {
                generateExpr(args.get(0));
                if (args.size() > 1) {
                    generateExpr(args.get(1));
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "substring",
                            "(II)Ljava/lang/String;", false);
                } else {
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "substring",
                            "(I)Ljava/lang/String;", false);
                }
                yield new Type.Simple("String");
            }
            case "equals" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "equals",
                        "(Ljava/lang/Object;)Z", false);
                yield new Type.Simple("bool");
            }
            case "to_string", "toString" -> new Type.Simple("String");
            case "replace" -> {
                generateExpr(args.get(0));
                generateExpr(args.get(1));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "replace",
                        "(Ljava/lang/CharSequence;Ljava/lang/CharSequence;)Ljava/lang/String;", false);
                yield new Type.Simple("String");
            }
            case "trim" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "trim",
                        "()Ljava/lang/String;", false);
                yield new Type.Simple("String");
            }
            case "is_empty", "isEmpty" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "isEmpty", "()Z", false);
                yield new Type.Simple("bool");
            }
            default -> throw new RuntimeException("Unknown String method: " + method);
        };
    }

    // --- Vec<T> methods ---

    private boolean isVecType(Type type) {
        if (type instanceof Type.Generic g) return g.name().equals("Vec");
        if (type instanceof Type.Simple s) return s.name().equals("Vec");
        return false;
    }

    private Type getVecElementType(Type vecType) {
        if (vecType instanceof Type.Generic g && !g.args().isEmpty()) {
            return g.args().get(0);
        }
        return new Type.Simple("Object");
    }

    private Type generateVecMethodCall(Type receiverType, String method, List<Expr> args) {
        Type elemType = getVecElementType(receiverType);
        return switch (method) {
            case "push", "add" -> {
                Type argType = generateExpr(args.get(0));
                boxIfNeeded(argType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "add",
                        "(Ljava/lang/Object;)Z", false);
                mv.visitInsn(POP); // discard boolean return
                yield new Type.Void();
            }
            case "get" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "get",
                        "(I)Ljava/lang/Object;", false);
                castAndUnbox(elemType);
                yield elemType;
            }
            case "set" -> {
                generateExpr(args.get(0)); // index
                Type argType = generateExpr(args.get(1)); // value
                boxIfNeeded(argType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "set",
                        "(ILjava/lang/Object;)Ljava/lang/Object;", false);
                mv.visitInsn(POP); // discard old value
                yield new Type.Void();
            }
            case "len", "size" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "size", "()I", false);
                yield new Type.Simple("i32");
            }
            case "is_empty", "isEmpty" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "isEmpty", "()Z", false);
                yield new Type.Simple("bool");
            }
            case "remove" -> {
                generateExpr(args.get(0));
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "remove",
                        "(I)Ljava/lang/Object;", false);
                castAndUnbox(elemType);
                yield elemType;
            }
            case "contains" -> {
                Type argType = generateExpr(args.get(0));
                boxIfNeeded(argType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "contains",
                        "(Ljava/lang/Object;)Z", false);
                yield new Type.Simple("bool");
            }
            default -> throw new RuntimeException("Unknown Vec method: " + method);
        };
    }

    // --- Map<K,V> methods ---

    private boolean isMapType(Type type) {
        if (type instanceof Type.Generic g) return g.name().equals("Map");
        if (type instanceof Type.Simple s) return s.name().equals("Map");
        return false;
    }

    private Type getMapValueType(Type mapType) {
        if (mapType instanceof Type.Generic g && g.args().size() >= 2) {
            return g.args().get(1);
        }
        return new Type.Simple("Object");
    }

    private Type generateMapMethodCall(Type receiverType, String method, List<Expr> args) {
        Type valueType = getMapValueType(receiverType);
        return switch (method) {
            case "insert", "put" -> {
                Type keyType = generateExpr(args.get(0));
                boxIfNeeded(keyType);
                Type valType = generateExpr(args.get(1));
                boxIfNeeded(valType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "put",
                        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;", false);
                mv.visitInsn(POP); // discard old value
                yield new Type.Void();
            }
            case "get" -> {
                Type keyType = generateExpr(args.get(0));
                boxIfNeeded(keyType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "get",
                        "(Ljava/lang/Object;)Ljava/lang/Object;", false);
                castAndUnbox(valueType);
                yield valueType;
            }
            case "contains_key", "containsKey" -> {
                Type keyType = generateExpr(args.get(0));
                boxIfNeeded(keyType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "containsKey",
                        "(Ljava/lang/Object;)Z", false);
                yield new Type.Simple("bool");
            }
            case "get_or_default", "getOrDefault" -> {
                Type keyType = generateExpr(args.get(0));
                boxIfNeeded(keyType);
                Type defType = generateExpr(args.get(1));
                boxIfNeeded(defType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "getOrDefault",
                        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;", false);
                castAndUnbox(valueType);
                yield valueType;
            }
            case "len", "size" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "size", "()I", false);
                yield new Type.Simple("i32");
            }
            case "is_empty", "isEmpty" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "isEmpty", "()Z", false);
                yield new Type.Simple("bool");
            }
            case "remove" -> {
                Type keyType = generateExpr(args.get(0));
                boxIfNeeded(keyType);
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/HashMap", "remove",
                        "(Ljava/lang/Object;)Ljava/lang/Object;", false);
                castAndUnbox(valueType);
                yield valueType;
            }
            default -> throw new RuntimeException("Unknown Map method: " + method);
        };
    }

    // --- StringBuilder methods ---

    private Type generateStringBuilderMethodCall(String method, List<Expr> args) {
        return switch (method) {
            case "append" -> {
                Type argType = generateExpr(args.get(0));
                String desc = "(" + typeDescriptor(argType) + ")Ljava/lang/StringBuilder;";
                // StringBuilder has overloads for all primitive types and String
                if (isObjectType(argType) && !isString(argType)) {
                    desc = "(Ljava/lang/Object;)Ljava/lang/StringBuilder;";
                }
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/StringBuilder", "append", desc, false);
                yield new Type.Simple("StringBuilder");
            }
            case "to_string", "toString" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/StringBuilder", "toString",
                        "()Ljava/lang/String;", false);
                yield new Type.Simple("String");
            }
            case "len", "length" -> {
                mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/StringBuilder", "length", "()I", false);
                yield new Type.Simple("i32");
            }
            default -> throw new RuntimeException("Unknown StringBuilder method: " + method);
        };
    }

    private Type generateFieldAccess(Expr.FieldAccess fa) {
        Type receiverType = generateExpr(fa.receiver());
        String typeName = resolveSimpleTypeName(receiverType);

        // Check struct fields
        Item.StructDef sd = structs.get(typeName);
        if (sd != null) {
            for (Item.Field field : sd.fields()) {
                if (field.name().equals(fa.field())) {
                    mv.visitFieldInsn(GETFIELD, typeName, field.name(), typeDescriptor(field.type()));
                    return field.type();
                }
            }
        }

        // Check enum variant fields (receiver might be cast to variant)
        // This would be used after a match arm binds a variant

        throw new RuntimeException("Unknown field: " + typeName + "." + fa.field());
    }

    private Type generateStructInit(Expr.StructInit si) {
        Item.StructDef sd = structs.get(si.name());
        if (sd == null) {
            throw new RuntimeException("Unknown struct: " + si.name());
        }

        mv.visitTypeInsn(NEW, si.name());
        mv.visitInsn(DUP);
        mv.visitMethodInsn(INVOKESPECIAL, si.name(), "<init>", "()V", false);

        for (Expr.FieldValue fv : si.fields()) {
            mv.visitInsn(DUP);
            generateExpr(fv.value());
            Type fieldType = null;
            for (Item.Field f : sd.fields()) {
                if (f.name().equals(fv.name())) {
                    fieldType = f.type();
                    break;
                }
            }
            if (fieldType == null) {
                throw new RuntimeException("Unknown field in struct init: " + si.name() + "." + fv.name());
            }
            mv.visitFieldInsn(PUTFIELD, si.name(), fv.name(), typeDescriptor(fieldType));
        }

        return new Type.Simple(si.name());
    }

    private Type generateEnumInit(Expr.EnumInit ei) {
        Item.EnumDef enumDef = enums.get(ei.enumName());
        if (enumDef == null) {
            throw new RuntimeException("Unknown enum: " + ei.enumName());
        }

        Item.EnumVariant variant = null;
        for (Item.EnumVariant v : enumDef.variants()) {
            if (v.name().equals(ei.variant())) {
                variant = v;
                break;
            }
        }
        if (variant == null) {
            throw new RuntimeException("Unknown enum variant: " + ei.enumName() + "::" + ei.variant());
        }

        String variantClass = ei.enumName() + "$" + ei.variant();
        mv.visitTypeInsn(NEW, variantClass);
        mv.visitInsn(DUP);
        mv.visitMethodInsn(INVOKESPECIAL, variantClass, "<init>", "()V", false);

        for (Expr.FieldValue fv : ei.fields()) {
            mv.visitInsn(DUP);
            generateExpr(fv.value());
            Type fieldType = null;
            for (Item.Field f : variant.fields()) {
                if (f.name().equals(fv.name())) {
                    fieldType = f.type();
                    break;
                }
            }
            if (fieldType == null) {
                throw new RuntimeException("Unknown field in enum init: " + ei.enumName() + "::" + ei.variant() + "." + fv.name());
            }
            mv.visitFieldInsn(PUTFIELD, variantClass, fv.name(), typeDescriptor(fieldType));
        }

        return new Type.Simple(ei.enumName());
    }

    private Type generateStaticCall(Expr.StaticCall sc) {
        // Built-in: Vec::new()
        if (sc.typeName().equals("Vec") && sc.method().equals("new")) {
            mv.visitTypeInsn(NEW, "java/util/ArrayList");
            mv.visitInsn(DUP);
            mv.visitMethodInsn(INVOKESPECIAL, "java/util/ArrayList", "<init>", "()V", false);
            return new Type.Simple("Vec");
        }

        // Built-in: Map::new()
        if (sc.typeName().equals("Map") && sc.method().equals("new")) {
            mv.visitTypeInsn(NEW, "java/util/HashMap");
            mv.visitInsn(DUP);
            mv.visitMethodInsn(INVOKESPECIAL, "java/util/HashMap", "<init>", "()V", false);
            return new Type.Simple("Map");
        }

        // Built-in: StringBuilder::new()
        if (sc.typeName().equals("StringBuilder") && sc.method().equals("new")) {
            mv.visitTypeInsn(NEW, "java/lang/StringBuilder");
            mv.visitInsn(DUP);
            if (sc.args().isEmpty()) {
                mv.visitMethodInsn(INVOKESPECIAL, "java/lang/StringBuilder", "<init>", "()V", false);
            } else {
                generateExpr(sc.args().get(0));
                mv.visitMethodInsn(INVOKESPECIAL, "java/lang/StringBuilder", "<init>",
                        "(Ljava/lang/String;)V", false);
            }
            return new Type.Simple("StringBuilder");
        }

        // Built-in: String::from_i32, String::from_i64, String::from_f64 etc
        if (sc.typeName().equals("String") && sc.method().equals("from")) {
            Type argType = generateExpr(sc.args().get(0));
            generateToString(argType);
            return new Type.Simple("String");
        }

        // Check if it's an imported class → infer descriptor from arg types + expected return type
        String jvmClass = imports.get(sc.typeName());
        if (jvmClass != null) {
            StringBuilder desc = new StringBuilder("(");
            for (Expr arg : sc.args()) {
                Type argType = generateExpr(arg);
                desc.append(typeDescriptor(argType));
            }
            desc.append(")");
            Type retType = expectedType != null ? expectedType : new Type.Void();
            desc.append(typeDescriptor(retType));
            mv.visitMethodInsn(INVOKESTATIC, jvmClass, sc.method(), desc.toString(), false);
            return retType;
        }

        // Look up method in impl
        Item.FnDef method = findImplMethod(sc.typeName(), sc.method());
        if (method == null) {
            throw new RuntimeException("Unknown static method: " + sc.typeName() + "::" + sc.method());
        }

        StringBuilder desc = new StringBuilder("(");
        int argIdx = 0;
        for (Item.Param p : method.params()) {
            if (!p.isSelf()) {
                if (argIdx < sc.args().size()) {
                    generateExpr(sc.args().get(argIdx));
                    desc.append(typeDescriptor(p.type()));
                    argIdx++;
                }
            }
        }
        desc.append(")").append(typeDescriptor(method.returnType()));

        mv.visitMethodInsn(INVOKESTATIC, sc.typeName(), sc.method(), desc.toString(), false);
        return method.returnType();
    }

    private Type generateAssign(Expr.Assign assign) {
        if (assign.target() instanceof Expr.Ident id) {
            Integer slot = locals.get(id.name());
            if (slot == null) throw new RuntimeException("Undefined variable: " + id.name());
            Type type = localTypes.get(id.name());
            Type prevExpected = expectedType;
            expectedType = type;
            generateExpr(assign.value());
            expectedType = prevExpected;
            storeLocal(slot, type);
            return new Type.Void();
        }
        if (assign.target() instanceof Expr.FieldAccess fa) {
            Type receiverType = generateExpr(fa.receiver());
            String typeName = resolveSimpleTypeName(receiverType);
            Item.StructDef sd = structs.get(typeName);
            if (sd != null) {
                for (Item.Field f : sd.fields()) {
                    if (f.name().equals(fa.field())) {
                        Type prevExpected = expectedType;
                        expectedType = f.type();
                        generateExpr(assign.value());
                        expectedType = prevExpected;
                        mv.visitFieldInsn(PUTFIELD, typeName, f.name(), typeDescriptor(f.type()));
                        return new Type.Void();
                    }
                }
            }
            throw new RuntimeException("Unknown field: " + typeName + "." + fa.field());
        }
        if (assign.target() instanceof Expr.Index idx) {
            // array[i] = value
            generateExpr(idx.receiver());
            generateExpr(idx.index());
            Type valType = generateExpr(assign.value());
            // For now assume object array
            mv.visitInsn(AASTORE);
            return new Type.Void();
        }
        throw new RuntimeException("Invalid assignment target");
    }

    private Type generateIf(Expr.If ifExpr) {
        generateExpr(ifExpr.condition());
        Label elseLabel = new Label();
        Label endLabel = new Label();

        mv.visitJumpInsn(IFEQ, elseLabel);

        for (Stmt stmt : ifExpr.thenBlock()) {
            generateStmt(stmt);
        }
        mv.visitJumpInsn(GOTO, endLabel);

        mv.visitLabel(elseLabel);
        if (ifExpr.elseBlock() != null) {
            for (Stmt stmt : ifExpr.elseBlock()) {
                generateStmt(stmt);
            }
        }

        mv.visitLabel(endLabel);
        return new Type.Void();
    }

    private Type generateWhile(Expr.While whileExpr) {
        Label loopStart = new Label();
        Label loopEnd = new Label();
        Label prevBreak = breakLabel;
        Label prevContinue = continueLabel;
        breakLabel = loopEnd;
        continueLabel = loopStart;

        mv.visitLabel(loopStart);
        generateExpr(whileExpr.condition());
        mv.visitJumpInsn(IFEQ, loopEnd);

        for (Stmt stmt : whileExpr.body()) {
            generateStmt(stmt);
        }
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        breakLabel = prevBreak;
        continueLabel = prevContinue;
        return new Type.Void();
    }

    private Type generateBlock(Expr.Block block) {
        for (Stmt stmt : block.stmts()) {
            generateStmt(stmt);
        }
        return new Type.Void();
    }

    private Type generateForRange(Expr.ForRange fr) {
        Type startType = generateExpr(fr.start());
        int slot = nextLocal;
        locals.put(fr.var(), slot);
        localTypes.put(fr.var(), new Type.Simple("i32"));
        nextLocal++;
        mv.visitVarInsn(ISTORE, slot);

        Label loopStart = new Label();
        Label loopEnd = new Label();
        Label prevBreak = breakLabel;
        Label prevContinue = continueLabel;
        breakLabel = loopEnd;
        continueLabel = loopStart;

        mv.visitLabel(loopStart);
        mv.visitVarInsn(ILOAD, slot);
        generateExpr(fr.end());
        mv.visitJumpInsn(IF_ICMPGE, loopEnd);

        for (Stmt stmt : fr.body()) {
            generateStmt(stmt);
        }

        mv.visitIincInsn(slot, 1);
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        breakLabel = prevBreak;
        continueLabel = prevContinue;
        return new Type.Void();
    }

    private Type generateForEach(Expr.ForEach fe) {
        // for item in iterable { body }
        // → Iterator iter = iterable.iterator(); while (iter.hasNext()) { T item = (T)iter.next(); body; }
        Type iterableType = generateExpr(fe.iterable());

        // Get element type
        Type elemType = getVecElementType(iterableType);

        // Call .iterator()
        mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "iterator",
                "()Ljava/util/Iterator;", false);
        int iterSlot = nextLocal;
        nextLocal++;
        mv.visitVarInsn(ASTORE, iterSlot);

        // Loop variable slot
        int itemSlot = nextLocal;
        locals.put(fe.var(), itemSlot);
        localTypes.put(fe.var(), elemType);
        nextLocal += localSize(elemType);

        Label loopStart = new Label();
        Label loopEnd = new Label();

        mv.visitLabel(loopStart);
        mv.visitVarInsn(ALOAD, iterSlot);
        mv.visitMethodInsn(INVOKEINTERFACE, "java/util/Iterator", "hasNext", "()Z", true);
        mv.visitJumpInsn(IFEQ, loopEnd);

        // Get next element
        mv.visitVarInsn(ALOAD, iterSlot);
        mv.visitMethodInsn(INVOKEINTERFACE, "java/util/Iterator", "next",
                "()Ljava/lang/Object;", true);
        castAndUnbox(elemType);
        storeLocal(itemSlot, elemType);

        for (Stmt stmt : fe.body()) {
            generateStmt(stmt);
        }
        mv.visitJumpInsn(GOTO, loopStart);

        mv.visitLabel(loopEnd);
        return new Type.Void();
    }

    private Type generateMatch(Expr.Match match) {
        // Evaluate subject and store in temp
        Type subjectType = generateExpr(match.subject());
        int tempSlot = nextLocal;
        nextLocal += localSize(subjectType);
        storeLocal(tempSlot, subjectType);

        Label endLabel = new Label();
        String subjectTypeName = resolveSimpleTypeName(subjectType);

        for (int i = 0; i < match.arms().size(); i++) {
            Expr.MatchArm arm = match.arms().get(i);
            Label nextArm = new Label();
            boolean isWildcard = arm.pattern() instanceof Pattern.Wildcard;

            if (!isWildcard) {
                generateMatchCondition(arm.pattern(), tempSlot, subjectType, nextArm);
            }

            // Bind variables from pattern
            if (arm.pattern() instanceof Pattern.EnumVariant ev && !ev.bindings().isEmpty()) {
                generateEnumBindings(ev, tempSlot);
            }

            // Execute arm body
            for (Stmt stmt : arm.body()) {
                generateStmt(stmt);
            }
            // Skip GOTO if arm ends with terminating statement (return/throw/break/continue)
            if (!armEndsWithTerminator(arm.body())) {
                mv.visitJumpInsn(GOTO, endLabel);
            }

            if (!isWildcard) {
                mv.visitLabel(nextArm);
            }
        }

        mv.visitLabel(endLabel);
        return new Type.Void();
    }

    private boolean armEndsWithTerminator(List<Stmt> body) {
        if (body.isEmpty()) return false;
        Stmt last = body.get(body.size() - 1);
        if (last instanceof Stmt.Return) return true;
        if (last instanceof Stmt.Break) return true;
        if (last instanceof Stmt.Continue) return true;
        // Check for panic() or throw calls as expression statements
        if (last instanceof Stmt.ExprStmt es) {
            Expr expr = es.expr();
            if (expr instanceof Expr.Call call) {
                if (call.name().equals("panic") || call.name().equals("exit")) return true;
            }
            if (expr instanceof Expr.Throw) return true;
        }
        return false;
    }

    private void generateMatchCondition(Pattern pattern, int tempSlot, Type subjectType, Label failLabel) {
        switch (pattern) {
            case Pattern.Wildcard w -> {} // always matches
            case Pattern.Literal lit -> {
                loadLocal(tempSlot, subjectType);
                String typeName = resolveSimpleTypeName(subjectType);
                if (typeName.equals("String")) {
                    // String matching with .equals()
                    generateExpr(lit.expr());
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "equals",
                            "(Ljava/lang/Object;)Z", false);
                    mv.visitJumpInsn(IFEQ, failLabel);
                } else if (typeName.equals("char") || typeName.equals("i32") || typeName.equals("bool")) {
                    generateExpr(lit.expr());
                    mv.visitJumpInsn(IF_ICMPNE, failLabel);
                } else if (typeName.equals("i64")) {
                    generateExpr(lit.expr());
                    mv.visitInsn(LCMP);
                    mv.visitJumpInsn(IFNE, failLabel);
                } else if (lit.expr() instanceof Expr.NullLit) {
                    mv.visitJumpInsn(IFNONNULL, failLabel);
                } else {
                    throw new RuntimeException("Cannot match on type: " + typeName);
                }
            }
            case Pattern.EnumVariant ev -> {
                // instanceof check
                String variantClass = ev.enumName() + "$" + ev.variant();
                loadLocal(tempSlot, subjectType);
                mv.visitTypeInsn(INSTANCEOF, variantClass);
                mv.visitJumpInsn(IFEQ, failLabel);
            }
        }
    }

    private void generateEnumBindings(Pattern.EnumVariant ev, int tempSlot) {
        String variantClass = ev.enumName() + "$" + ev.variant();

        // Find the variant definition to get field types
        Item.EnumDef enumDef = enums.get(ev.enumName());
        Item.EnumVariant variantDef = null;
        for (Item.EnumVariant v : enumDef.variants()) {
            if (v.name().equals(ev.variant())) {
                variantDef = v;
                break;
            }
        }
        if (variantDef == null) {
            throw new RuntimeException("Unknown variant: " + ev.enumName() + "::" + ev.variant());
        }

        // Cast to variant type and store in temp
        mv.visitVarInsn(ALOAD, tempSlot);
        mv.visitTypeInsn(CHECKCAST, variantClass);
        int castSlot = nextLocal;
        nextLocal++;
        mv.visitVarInsn(ASTORE, castSlot);

        // Bind each field
        for (int i = 0; i < ev.bindings().size() && i < variantDef.fields().size(); i++) {
            String bindingName = ev.bindings().get(i);
            Item.Field field = variantDef.fields().get(i);

            mv.visitVarInsn(ALOAD, castSlot);
            mv.visitFieldInsn(GETFIELD, variantClass, field.name(), typeDescriptor(field.type()));

            int bindSlot = nextLocal;
            locals.put(bindingName, bindSlot);
            localTypes.put(bindingName, field.type());
            nextLocal += localSize(field.type());
            storeLocal(bindSlot, field.type());
        }
    }

    private Type generateIndex(Expr.Index idx) {
        Type receiverType = generateExpr(idx.receiver());

        // Vec indexing
        if (isVecType(receiverType)) {
            Type elemType = getVecElementType(receiverType);
            generateExpr(idx.index());
            mv.visitMethodInsn(INVOKEVIRTUAL, "java/util/ArrayList", "get",
                    "(I)Ljava/lang/Object;", false);
            castAndUnbox(elemType);
            return elemType;
        }

        // String indexing → charAt
        if (isString(receiverType)) {
            generateExpr(idx.index());
            mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/String", "charAt", "(I)C", false);
            return new Type.Simple("char");
        }

        // Array indexing
        generateExpr(idx.index());
        if (receiverType instanceof Type.Array at) {
            String elemTypeName = resolveSimpleTypeName(at.element());
            switch (elemTypeName) {
                case "i32" -> { mv.visitInsn(IALOAD); return at.element(); }
                case "i64" -> { mv.visitInsn(LALOAD); return at.element(); }
                case "f64" -> { mv.visitInsn(DALOAD); return at.element(); }
                case "bool" -> { mv.visitInsn(BALOAD); return at.element(); }
                case "char" -> { mv.visitInsn(CALOAD); return at.element(); }
                default -> { mv.visitInsn(AALOAD); return at.element(); }
            }
        }

        // Fallback: assume object array
        mv.visitInsn(AALOAD);
        return new Type.Simple("Object");
    }

    private Type generateThrow(Expr.Throw th) {
        mv.visitTypeInsn(NEW, "java/lang/RuntimeException");
        mv.visitInsn(DUP);
        generateExpr(th.message());
        mv.visitMethodInsn(INVOKESPECIAL, "java/lang/RuntimeException", "<init>",
                "(Ljava/lang/String;)V", false);
        mv.visitInsn(ATHROW);
        return new Type.Void();
    }

    private Type generateArrayLit(Expr.ArrayLit al) {
        if (al.elements().isEmpty()) {
            mv.visitInsn(ICONST_0);
            mv.visitTypeInsn(ANEWARRAY, "java/lang/Object");
            return new Type.Array(new Type.Simple("Object"));
        }

        // Infer element type from first element
        Type elemType = inferType(al.elements().get(0));
        String elemTypeName = resolveSimpleTypeName(elemType);

        pushInt(al.elements().size());

        switch (elemTypeName) {
            case "i32" -> {
                mv.visitIntInsn(NEWARRAY, T_INT);
                for (int i = 0; i < al.elements().size(); i++) {
                    mv.visitInsn(DUP);
                    pushInt(i);
                    generateExpr(al.elements().get(i));
                    mv.visitInsn(IASTORE);
                }
            }
            case "i64" -> {
                mv.visitIntInsn(NEWARRAY, T_LONG);
                for (int i = 0; i < al.elements().size(); i++) {
                    mv.visitInsn(DUP);
                    pushInt(i);
                    generateExpr(al.elements().get(i));
                    mv.visitInsn(LASTORE);
                }
            }
            case "f64" -> {
                mv.visitIntInsn(NEWARRAY, T_DOUBLE);
                for (int i = 0; i < al.elements().size(); i++) {
                    mv.visitInsn(DUP);
                    pushInt(i);
                    generateExpr(al.elements().get(i));
                    mv.visitInsn(DASTORE);
                }
            }
            default -> {
                mv.visitTypeInsn(ANEWARRAY, jvmClassName(elemTypeName));
                for (int i = 0; i < al.elements().size(); i++) {
                    mv.visitInsn(DUP);
                    pushInt(i);
                    generateExpr(al.elements().get(i));
                    mv.visitInsn(AASTORE);
                }
            }
        }

        return new Type.Array(elemType);
    }

    private Type generateCast(Expr.Cast cast) {
        Type srcType = generateExpr(cast.value());
        Type destType = cast.type();
        String srcName = resolveSimpleTypeName(srcType);
        String destName = resolveSimpleTypeName(destType);

        // Numeric casts
        if (srcName.equals("i32") && destName.equals("i64")) mv.visitInsn(I2L);
        else if (srcName.equals("i32") && destName.equals("f64")) mv.visitInsn(I2D);
        else if (srcName.equals("i32") && destName.equals("char")) mv.visitInsn(I2C);
        else if (srcName.equals("i64") && destName.equals("i32")) mv.visitInsn(L2I);
        else if (srcName.equals("i64") && destName.equals("f64")) mv.visitInsn(L2D);
        else if (srcName.equals("f64") && destName.equals("i32")) mv.visitInsn(D2I);
        else if (srcName.equals("f64") && destName.equals("i64")) mv.visitInsn(D2L);
        else if (srcName.equals("char") && destName.equals("i32")) { /* char is already int on stack */ }
        else {
            // Reference cast
            mv.visitTypeInsn(CHECKCAST, jvmClassName(destName));
        }

        return destType;
    }

    // --- Boxing / Unboxing ---

    private void boxIfNeeded(Type type) {
        if (type instanceof Type.Simple s) {
            switch (s.name()) {
                case "i32" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Integer", "valueOf", "(I)Ljava/lang/Integer;", false);
                case "i64" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Long", "valueOf", "(J)Ljava/lang/Long;", false);
                case "f64" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Double", "valueOf", "(D)Ljava/lang/Double;", false);
                case "bool" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Boolean", "valueOf", "(Z)Ljava/lang/Boolean;", false);
                case "char" -> mv.visitMethodInsn(INVOKESTATIC, "java/lang/Character", "valueOf", "(C)Ljava/lang/Character;", false);
                // Objects don't need boxing
            }
        }
    }

    private void castAndUnbox(Type targetType) {
        if (targetType instanceof Type.Simple s) {
            switch (s.name()) {
                case "i32" -> {
                    mv.visitTypeInsn(CHECKCAST, "java/lang/Integer");
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/Integer", "intValue", "()I", false);
                }
                case "i64" -> {
                    mv.visitTypeInsn(CHECKCAST, "java/lang/Long");
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/Long", "longValue", "()J", false);
                }
                case "f64" -> {
                    mv.visitTypeInsn(CHECKCAST, "java/lang/Double");
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/Double", "doubleValue", "()D", false);
                }
                case "bool" -> {
                    mv.visitTypeInsn(CHECKCAST, "java/lang/Boolean");
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/Boolean", "booleanValue", "()Z", false);
                }
                case "char" -> {
                    mv.visitTypeInsn(CHECKCAST, "java/lang/Character");
                    mv.visitMethodInsn(INVOKEVIRTUAL, "java/lang/Character", "charValue", "()C", false);
                }
                case "String" -> mv.visitTypeInsn(CHECKCAST, "java/lang/String");
                case "Object", "null" -> {} // no cast needed
                default -> mv.visitTypeInsn(CHECKCAST, jvmClassName(s.name()));
            }
        } else if (targetType instanceof Type.Generic g) {
            mv.visitTypeInsn(CHECKCAST, jvmClassName(g.name()));
        }
    }

    // --- Type helpers ---

    private String typeDescriptor(Type type) {
        if (type instanceof Type.Void) return "V";
        if (type instanceof Type.Simple s) {
            return switch (s.name()) {
                case "i32" -> "I";
                case "i64" -> "J";
                case "f64" -> "D";
                case "bool" -> "Z";
                case "char" -> "C";
                case "String" -> "Ljava/lang/String;";
                case "Vec" -> "Ljava/util/ArrayList;";
                case "Map" -> "Ljava/util/HashMap;";
                case "StringBuilder" -> "Ljava/lang/StringBuilder;";
                case "Object", "null" -> "Ljava/lang/Object;";
                default -> "L" + s.name() + ";";
            };
        }
        if (type instanceof Type.Generic g) {
            // Generics are erased on JVM
            return switch (g.name()) {
                case "Vec" -> "Ljava/util/ArrayList;";
                case "Map" -> "Ljava/util/HashMap;";
                default -> "L" + g.name() + ";";
            };
        }
        if (type instanceof Type.Array at) {
            return "[" + typeDescriptor(at.element());
        }
        throw new RuntimeException("Unknown type: " + type);
    }

    private int localSize(Type type) {
        if (type instanceof Type.Simple s) {
            return switch (s.name()) {
                case "i64", "f64" -> 2;
                default -> 1;
            };
        }
        return 1;
    }

    private boolean isWide(Type type) {
        if (type instanceof Type.Simple s) {
            return s.name().equals("i64") || s.name().equals("f64");
        }
        return false;
    }

    private boolean isString(Type type) {
        return type instanceof Type.Simple s && s.name().equals("String");
    }

    private boolean isObjectType(Type type) {
        if (type instanceof Type.Simple s) {
            return switch (s.name()) {
                case "i32", "i64", "f64", "bool", "char" -> false;
                default -> true;
            };
        }
        return true;
    }

    private boolean isPrimitive(Type type) {
        return !isObjectType(type);
    }

    private String resolveSimpleTypeName(Type type) {
        if (type instanceof Type.Simple s) return s.name();
        if (type instanceof Type.Generic g) return g.name();
        if (type instanceof Type.Array) return "array";
        if (type instanceof Type.Void) return "void";
        return "unknown";
    }

    private String jvmClassName(String typeName) {
        return switch (typeName) {
            case "String" -> "java/lang/String";
            case "Vec" -> "java/util/ArrayList";
            case "Map" -> "java/util/HashMap";
            case "StringBuilder" -> "java/lang/StringBuilder";
            case "Object" -> "java/lang/Object";
            default -> {
                // Check imports
                String imported = imports.get(typeName);
                yield imported != null ? imported : typeName;
            }
        };
    }

    private void loadLocal(int slot, Type type) {
        if (type instanceof Type.Simple s) {
            switch (s.name()) {
                case "i32", "bool", "char" -> mv.visitVarInsn(ILOAD, slot);
                case "i64" -> mv.visitVarInsn(LLOAD, slot);
                case "f64" -> mv.visitVarInsn(DLOAD, slot);
                default -> mv.visitVarInsn(ALOAD, slot);
            }
        } else {
            mv.visitVarInsn(ALOAD, slot);
        }
    }

    private void storeLocal(int slot, Type type) {
        if (type instanceof Type.Simple s) {
            switch (s.name()) {
                case "i32", "bool", "char" -> mv.visitVarInsn(ISTORE, slot);
                case "i64" -> mv.visitVarInsn(LSTORE, slot);
                case "f64" -> mv.visitVarInsn(DSTORE, slot);
                default -> mv.visitVarInsn(ASTORE, slot);
            }
        } else {
            mv.visitVarInsn(ASTORE, slot);
        }
    }

    private void pushInt(int value) {
        if (value >= -1 && value <= 5) {
            mv.visitInsn(ICONST_0 + value);
        } else if (value >= Byte.MIN_VALUE && value <= Byte.MAX_VALUE) {
            mv.visitIntInsn(BIPUSH, value);
        } else if (value >= Short.MIN_VALUE && value <= Short.MAX_VALUE) {
            mv.visitIntInsn(SIPUSH, value);
        } else {
            mv.visitLdcInsn(value);
        }
    }

    private Type inferType(Expr expr) {
        return switch (expr) {
            case Expr.IntLit il -> {
                if (il.value() > Integer.MAX_VALUE || il.value() < Integer.MIN_VALUE) {
                    yield new Type.Simple("i64");
                }
                yield new Type.Simple("i32");
            }
            case Expr.FloatLit fl -> new Type.Simple("f64");
            case Expr.StringLit sl -> new Type.Simple("String");
            case Expr.CharLit cl -> new Type.Simple("char");
            case Expr.BoolLit bl -> new Type.Simple("bool");
            case Expr.NullLit nl -> new Type.Simple("null");
            case Expr.Ident id -> {
                Type t = localTypes.get(id.name());
                if (t != null) yield t;
                for (Item.ConstDef cd : constants) {
                    if (cd.name().equals(id.name())) yield cd.type();
                }
                yield new Type.Simple("Object");
            }
            case Expr.SelfExpr se -> new Type.Simple(currentClass);
            case Expr.Call call -> {
                if (call.name().equals("println") || call.name().equals("print") ||
                        call.name().equals("eprintln") || call.name().equals("panic") ||
                        call.name().equals("exit")) yield new Type.Void();
                Item.FnDef fn = findTopLevelFn(call.name());
                yield fn != null ? fn.returnType() : new Type.Void();
            }
            case Expr.StaticCall sc -> {
                if (sc.typeName().equals("Vec") && sc.method().equals("new")) {
                    yield new Type.Simple("Vec");
                }
                if (sc.typeName().equals("Map") && sc.method().equals("new")) {
                    yield new Type.Simple("Map");
                }
                if (sc.typeName().equals("StringBuilder") && sc.method().equals("new")) {
                    yield new Type.Simple("StringBuilder");
                }
                Item.FnDef m = findImplMethod(sc.typeName(), sc.method());
                yield m != null ? m.returnType() : new Type.Void();
            }
            case Expr.MethodCall mc -> {
                Type recvType = inferType(mc.receiver());
                String recvName = resolveSimpleTypeName(recvType);
                if (recvName.equals("String")) {
                    yield switch (mc.method()) {
                        case "len", "length" -> new Type.Simple("i32");
                        case "char_at", "charAt" -> new Type.Simple("char");
                        case "contains", "starts_with", "startsWith", "ends_with", "endsWith",
                             "equals", "is_empty", "isEmpty" -> new Type.Simple("bool");
                        default -> new Type.Simple("String");
                    };
                }
                if (recvName.equals("Vec")) {
                    yield switch (mc.method()) {
                        case "len", "size" -> new Type.Simple("i32");
                        case "is_empty", "isEmpty", "contains" -> new Type.Simple("bool");
                        case "get", "remove" -> getVecElementType(recvType);
                        default -> new Type.Void();
                    };
                }
                if (recvName.equals("Map")) {
                    yield switch (mc.method()) {
                        case "len", "size" -> new Type.Simple("i32");
                        case "is_empty", "isEmpty", "contains_key", "containsKey" -> new Type.Simple("bool");
                        case "get", "remove", "get_or_default", "getOrDefault" -> getMapValueType(recvType);
                        default -> new Type.Void();
                    };
                }
                if (recvName.equals("StringBuilder")) {
                    yield switch (mc.method()) {
                        case "append" -> new Type.Simple("StringBuilder");
                        case "to_string", "toString" -> new Type.Simple("String");
                        case "len", "length" -> new Type.Simple("i32");
                        default -> new Type.Void();
                    };
                }
                Item.FnDef m = findImplMethod(recvName, mc.method());
                yield m != null ? m.returnType() : new Type.Void();
            }
            case Expr.FieldAccess fa -> {
                Type recvType = inferType(fa.receiver());
                String recvName = resolveSimpleTypeName(recvType);
                Item.StructDef sd = structs.get(recvName);
                if (sd != null) {
                    for (Item.Field f : sd.fields()) {
                        if (f.name().equals(fa.field())) yield f.type();
                    }
                }
                yield new Type.Simple("Object");
            }
            case Expr.StructInit si -> new Type.Simple(si.name());
            case Expr.EnumInit ei -> new Type.Simple(ei.enumName());
            case Expr.Binary bin -> {
                if (isComparisonOp(bin.op())) yield new Type.Simple("bool");
                if (bin.op().equals("&&") || bin.op().equals("||")) yield new Type.Simple("bool");
                Type lt = inferType(bin.left());
                if (bin.op().equals("+") && isString(lt)) yield new Type.Simple("String");
                yield lt;
            }
            case Expr.Unary un -> {
                if (un.op().equals("!")) yield new Type.Simple("bool");
                yield inferType(un.operand());
            }
            case Expr.Assign a -> new Type.Void();
            case Expr.If i -> new Type.Void();
            case Expr.While w -> new Type.Void();
            case Expr.Block b -> new Type.Void();
            case Expr.ForRange fr -> new Type.Void();
            case Expr.ForEach fe -> new Type.Void();
            case Expr.Match m -> new Type.Void();
            case Expr.Index idx -> {
                Type recvType = inferType(idx.receiver());
                if (isVecType(recvType)) yield getVecElementType(recvType);
                if (isString(recvType)) yield new Type.Simple("char");
                if (recvType instanceof Type.Array at) yield at.element();
                yield new Type.Simple("Object");
            }
            case Expr.Throw t -> new Type.Void();
            case Expr.ArrayLit al -> {
                if (al.elements().isEmpty()) yield new Type.Array(new Type.Simple("Object"));
                yield new Type.Array(inferType(al.elements().get(0)));
            }
            case Expr.Cast c -> c.type();
        };
    }

    // --- Lookup helpers ---

    private Item.FnDef findTopLevelFn(String name) {
        for (Item.FnDef fn : topLevelFns) {
            if (fn.name().equals(name)) return fn;
        }
        return null;
    }

    private Item.FnDef findImplMethod(String typeName, String methodName) {
        List<Item.FnDef> methods = implMethods.getOrDefault(typeName, List.of());
        for (Item.FnDef m : methods) {
            if (m.name().equals(methodName)) return m;
        }
        return null;
    }

    // --- File output ---

    private void writeClass(String name, byte[] bytecode) throws IOException {
        File dir = new File(outputDir);
        dir.mkdirs();
        File file = new File(dir, name + ".class");
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(bytecode);
        }
    }
}
