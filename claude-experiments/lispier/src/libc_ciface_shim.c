// Provides _mlir_ciface_ wrappers for libc functions
// Needed when llvm-request-c-wrappers is applied to all func.func

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>

// malloc wrapper
void* _mlir_ciface_malloc(int64_t size) {
    return malloc((size_t)size);
}

// free wrapper
void _mlir_ciface_free(void* ptr) {
    free(ptr);
}

// fopen wrapper
void* _mlir_ciface_fopen(void* path, void* mode) {
    return fopen((const char*)path, (const char*)mode);
}

// fread wrapper
int64_t _mlir_ciface_fread(void* ptr, int64_t size, int64_t nmemb, void* stream) {
    return (int64_t)fread(ptr, (size_t)size, (size_t)nmemb, (FILE*)stream);
}

// fseek wrapper
int32_t _mlir_ciface_fseek(void* stream, int64_t offset, int32_t whence) {
    return (int32_t)fseek((FILE*)stream, (long)offset, (int)whence);
}

// fclose wrapper
int32_t _mlir_ciface_fclose(void* stream) {
    return (int32_t)fclose((FILE*)stream);
}

// printf wrapper (variadic - may not work perfectly but worth trying)
int32_t _mlir_ciface_printf(void* format, ...) {
    // For simple printf calls with 1-2 args
    // This is a simplified version
    va_list args;
    va_start(args, format);
    int result = vprintf((const char*)format, args);
    va_end(args);
    return (int32_t)result;
}
