#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Compiling JRust bootstrap compiler ==="
mkdir -p build
javac -cp asm.jar -d build \
    src/jrust/Token.java \
    src/jrust/Lexer.java \
    src/jrust/ast/Type.java \
    src/jrust/ast/Expr.java \
    src/jrust/ast/Stmt.java \
    src/jrust/ast/Pattern.java \
    src/jrust/ast/Item.java \
    src/jrust/ast/Program.java \
    src/jrust/Parser.java \
    src/jrust/codegen/JvmCodegen.java \
    src/jrust/JRustRuntime.java \
    src/jrust/Main.java

echo "=== Running JRust on sample.jrs ==="
java -cp "asm.jar:build" jrust.Main sample.jrs
