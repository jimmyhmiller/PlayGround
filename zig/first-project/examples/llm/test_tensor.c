#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "math.h"

typedef struct {
    void (*gelu_forward)(float*, float*, int32_t);
    void (*simple_layernorm)(float*, float*, float*, float*, int32_t);
    void (*simple_softmax)(float*, float*, int32_t);
    void (*test_gelu)();
    void (*test_layernorm)();
    void (*test_softmax)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static void gelu_forward(float*, float*, int32_t);
static void simple_layernorm(float*, float*, float*, float*, int32_t);
static void simple_softmax(float*, float*, int32_t);
static void test_gelu();
static void test_layernorm();
static void test_softmax();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->gelu_forward = &gelu_forward;
    ns->simple_layernorm = &simple_layernorm;
    ns->simple_softmax = &simple_softmax;
    ns->test_gelu = &test_gelu;
    ns->test_layernorm = &test_layernorm;
    ns->test_softmax = &test_softmax;
    ns->main_fn = &main_fn;
}

static void gelu_forward(float* out, float* inp, int32_t N) {
    ({ float s = sqrtf((2 / 3.141592653589793)); int32_t i = 0; ({ while ((i < N)) { ({ float x = inp[i]; float cube = ((0.044715 * x) * (x * x)); float y = ((0.5 * x) * (1 + tanhf((s * (x + cube))))); (out[i] = y); }); i = (i + 1); } }); });
}
static void simple_layernorm(float* out, float* inp, float* weight, float* bias, int32_t N) {
    ({ float eps = 0.00001; ({ float mean = 0; int32_t i = 0; ({ while ((i < N)) { mean = (mean + inp[i]); i = (i + 1); } }); mean = (mean / (N + 0)); ({ float variance = 0; i = 0; ({ while ((i < N)) { ({ float diff = (inp[i] - mean); variance = (variance + (diff * diff)); }); i = (i + 1); } }); variance = (variance / (N + 0)); ({ float rstd = (1 / sqrtf((variance + eps))); i = 0; ({ while ((i < N)) { ({ float normalized = ((inp[i] - mean) * rstd); float w = weight[i]; float b = bias[i]; (out[i] = ((normalized * w) + b)); }); i = (i + 1); } }); 0; }); }); }); });
}
static void simple_softmax(float* out, float* inp, int32_t N) {
    ({ float maxval = inp[0]; int32_t i = 1; ({ while ((i < N)) { ({ float val = inp[i]; ({ if ((val > maxval)) { maxval = val; } else { } }); }); i = (i + 1); } }); ({ float sum = 0; i = 0; ({ while ((i < N)) { ({ float exp_val = expf((inp[i] - maxval)); (out[i] = exp_val); sum = (sum + exp_val); }); i = (i + 1); } }); i = 0; ({ while ((i < N)) { (out[i] = (out[i] / sum)); i = (i + 1); } }); 0; }); });
}
static void test_gelu() {
    printf("\n=== Testing GELU ===\n");
    ({ int32_t n = 5; float* inp = (float*)(float*)malloc(5 * sizeof(float)); float* out = (float*)(float*)malloc(5 * sizeof(float)); (inp[0] = -2); (inp[1] = -1); (inp[2] = 0); (inp[3] = 1); (inp[4] = 2); g_user.gelu_forward(out, inp, n); printf("GELU test completed\n"); free(inp); free(out); 0; });
}
static void test_layernorm() {
    printf("\n=== Testing LayerNorm ===\n");
    ({ int32_t n = 4; float* inp = (float*)(float*)malloc(4 * sizeof(float)); float* out = (float*)(float*)malloc(4 * sizeof(float)); float* weight = (float*)({ float* __arr = (float*)malloc(4 * sizeof(float)); for (size_t __i = 0; __i < 4; __i++) __arr[__i] = 1; __arr; }); float* bias = (float*)({ float* __arr = (float*)malloc(4 * sizeof(float)); for (size_t __i = 0; __i < 4; __i++) __arr[__i] = 0; __arr; }); (inp[0] = 1); (inp[1] = 2); (inp[2] = 3); (inp[3] = 4); g_user.simple_layernorm(out, inp, weight, bias, n); printf("LayerNorm test completed\n"); free(inp); free(out); free(weight); free(bias); 0; });
}
static void test_softmax() {
    printf("\n=== Testing Softmax ===\n");
    ({ int32_t n = 5; float* inp = (float*)(float*)malloc(5 * sizeof(float)); float* out = (float*)(float*)malloc(5 * sizeof(float)); (inp[0] = 1); (inp[1] = 2); (inp[2] = 3); (inp[3] = 4); (inp[4] = 5); g_user.simple_softmax(out, inp, n); ({ float sum = 0; int32_t i = 0; ({ while ((i < n)) { sum = (sum + out[i]); i = (i + 1); } }); printf("Softmax test completed (sum should be ~1.0)\n"); 0; }); free(inp); free(out); 0; });
}
static int32_t main_fn() {
    printf("=================================\n");
    printf("Tensor Operations Test Suite\n");
    printf("=================================\n");
    g_user.test_gelu();
    g_user.test_layernorm();
    g_user.test_softmax();
    printf("\n=================================\n");
    printf("All tests completed!\n");
    printf("=================================\n");
    return 0;
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
