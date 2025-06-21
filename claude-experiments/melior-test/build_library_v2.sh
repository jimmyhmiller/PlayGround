#!/bin/bash

# Script to convert MLIR func dialect to a C library

echo "Converting func dialect to LLVM dialect..."

# First convert func dialect to LLVM dialect
mlir-opt --convert-func-to-llvm hello.mlir > hello_llvm.mlir

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert func dialect to LLVM dialect"
    echo "Make sure you have MLIR tools installed"
    exit 1
fi

echo "Generated LLVM dialect MLIR:"
cat hello_llvm.mlir

echo -e "\nConverting LLVM dialect to LLVM IR..."

# Convert LLVM dialect MLIR to LLVM IR
mlir-translate --mlir-to-llvmir hello_llvm.mlir > hello.ll

if [ $? -ne 0 ]; then
    echo "Error: Failed to convert LLVM dialect to LLVM IR"
    exit 1
fi

echo "Generated LLVM IR:"
cat hello.ll

echo -e "\nCompiling to object file..."

# Compile LLVM IR to object file
llc -filetype=obj hello.ll -o hello.o

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile LLVM IR to object file"
    exit 1
fi

echo "Creating shared library..."

# Create shared library
gcc -shared -o libhello.so hello.o

if [ $? -ne 0 ]; then
    echo "Error: Failed to create shared library"
    exit 1
fi

echo "Successfully created libhello.so!"

# Create a simple header file
cat > hello.h << EOF
#ifndef HELLO_H
#define HELLO_H

void hello(void);

#endif
EOF

echo "Created hello.h header file"

# Create a test program
cat > test_hello.c << EOF
#include <stdio.h>
#include "hello.h"

int main() {
    printf("Calling hello function from our compiler-generated library:\n");
    hello();
    printf("Done!\n");
    return 0;
}
EOF

echo "Created test_hello.c"

echo -e "\nTo test the library, run:"
echo "gcc -L. -lhello test_hello.c -o test_hello"
echo "./test_hello"