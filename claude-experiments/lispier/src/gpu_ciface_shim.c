// GPU Runtime C Interface Shim
// Provides _mlir_ciface_ wrappers for GPU runtime functions that take/return memrefs

#include <stdint.h>

// Memref descriptor for 1D memrefs
typedef struct {
    void* allocatedPtr;
    void* alignedPtr;
    int64_t offset;
    int64_t size;
    int64_t stride;
} MemRef1D;

// External GPU runtime functions (from libmlir_rocm_runtime.so)
extern MemRef1D mgpuMemGetDeviceMemRef1dFloat(void* allocatedPtr, void* alignedPtr,
                                               int64_t offset, int64_t size, int64_t stride);

// C interface wrapper - takes output pointer as first arg, input as second
void _mlir_ciface_mgpuMemGetDeviceMemRef1dFloat(MemRef1D* result, MemRef1D* input) {
    *result = mgpuMemGetDeviceMemRef1dFloat(
        input->allocatedPtr,
        input->alignedPtr,
        input->offset,
        input->size,
        input->stride
    );
}
