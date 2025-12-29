// Shim library to provide _mlir_ciface_* aliases for runner utils functions
// These call through to the original functions in libmlir_c_runner_utils.so

#include <stdio.h>

// Forward declarations - these are in libmlir_c_runner_utils.so
extern void printF32(float x);
extern void printF64(double x);
extern void printI64(long long x);
extern void printU64(unsigned long long x);
extern void printOpen();
extern void printClose();
extern void printComma();
extern void printNewline();
extern void printString(const char* s);

// C-interface aliases that MLIR-generated code expects
void _mlir_ciface_printF32(float x) { printF32(x); }
void _mlir_ciface_printF64(double x) { printF64(x); }
void _mlir_ciface_printI64(long long x) { printI64(x); }
void _mlir_ciface_printU64(unsigned long long x) { printU64(x); }
void _mlir_ciface_printOpen() { printOpen(); }
void _mlir_ciface_printClose() { printClose(); }
void _mlir_ciface_printComma() { printComma(); }
void _mlir_ciface_printNewline() { printNewline(); }
void _mlir_ciface_printString(const char* s) { printString(s); }
