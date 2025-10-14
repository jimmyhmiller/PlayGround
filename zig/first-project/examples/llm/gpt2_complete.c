#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
typedef struct {
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t num_layers;
    int32_t num_heads;
    int32_t channels;
} GPT2Config;


typedef struct {
    void (*layernorm_forward)(float*, float*, float*, float*, int32_t, int32_t, int32_t);
    void (*matmul_forward)(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*gelu_forward)(float*, float*, int32_t);
    void (*residual_forward)(float*, float*, float*, int32_t);
    void (*test_layernorm)();
    void (*test_matmul)();
    void (*test_gelu)();
    void (*test_residual)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static void layernorm_forward(float*, float*, float*, float*, int32_t, int32_t, int32_t);
static void matmul_forward(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
static void gelu_forward(float*, float*, int32_t);
static void residual_forward(float*, float*, float*, int32_t);
static void test_layernorm();
static void test_matmul();
static void test_gelu();
static void test_residual();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->layernorm_forward = &layernorm_forward;
    ns->matmul_forward = &matmul_forward;
    ns->gelu_forward = &gelu_forward;
    ns->residual_forward = &residual_forward;
    ns->test_layernorm = &test_layernorm;
    ns->test_matmul = &test_matmul;
    ns->test_gelu = &test_gelu;
    ns->test_residual = &test_residual;
    ns->main_fn = &main_fn;
}

static void layernorm_forward(float* out, float* inp, float* weight, float* bias, int32_t B, int32_t T, int32_t C) {
    ({ float eps = 0.00001; int32_t bt = 0; ({ while ((bt < (B * T))) { ({ int32_t bt_offset = (bt * C); float mean = 0; int32_t c = 0; ({ while ((c < C)) { mean = (mean + inp[(bt_offset + c)]); c = (c + 1); } }); mean = (mean / (C + 0)); ({ float variance = 0; c = 0; ({ while ((c < C)) { ({ float diff = (inp[(bt_offset + c)] - mean); variance = (variance + (diff * diff)); }); c = (c + 1); } }); variance = (variance / (C + 0)); ({ float rstd = (1 / sqrtf((variance + eps))); c = 0; ({ while ((c < C)) { ({ int32_t idx = (bt_offset + c); float n = ((inp[idx] - mean) * rstd); float w = weight[c]; float b = bias[c]; (out[idx] = ((n * w) + b)); }); c = (c + 1); } }); 0; }); }); }); bt = (bt + 1); } }); });
}
static void matmul_forward(float* out, float* inp, float* weight, float* bias, int32_t B, int32_t T, int32_t C, int32_t OC) {
    ({ int32_t bt = 0; ({ while ((bt < (B * T))) { ({ int32_t bt_out_offset = (bt * OC); int32_t bt_inp_offset = (bt * C); int32_t o = 0; ({ while ((o < OC)) { ({ float val = ((bias == NULL) ? 0 : bias[o]); int32_t wrow_offset = (o * C); int32_t c = 0; ({ while ((c < C)) { val = (val + (inp[(bt_inp_offset + c)] * weight[(wrow_offset + c)])); c = (c + 1); } }); (out[(bt_out_offset + o)] = val); }); o = (o + 1); } }); }); bt = (bt + 1); } }); });
}
static void gelu_forward(float* out, float* inp, int32_t N) {
    ({ float s = sqrtf((2 / 3.14159265)); int32_t i = 0; ({ while ((i < N)) { ({ float x = inp[i]; float cube = ((0.044715 * x) * (x * x)); float y = ((0.5 * x) * (1 + tanhf((s * (x + cube))))); (out[i] = y); }); i = (i + 1); } }); });
}
static void residual_forward(float* out, float* inp1, float* inp2, int32_t N) {
    ({ int32_t i = 0; ({ while ((i < N)) { (out[i] = (inp1[i] + inp2[i])); i = (i + 1); } }); });
}
static void test_layernorm() {
    printf("Testing LayerNorm...\n");
    ({ int32_t B = 2; int32_t T = 3; int32_t C = 4; int32_t size = ((B * T) * C); float* inp = (float*)(float*)malloc(24 * sizeof(float)); float* out = (float*)(float*)malloc(24 * sizeof(float)); float* weight = (float*)({ float* __arr = (float*)malloc(4 * sizeof(float)); for (size_t __i = 0; __i < 4; __i++) __arr[__i] = 1; __arr; }); float* bias = (float*)({ float* __arr = (float*)malloc(4 * sizeof(float)); for (size_t __i = 0; __i < 4; __i++) __arr[__i] = 0; __arr; }); ({ int32_t i = 0; ({ while ((i < size)) { (inp[i] = ((i + 0) + 1)); i = (i + 1); } }); }); g_user.layernorm_forward(out, inp, weight, bias, B, T, C); printf("LayerNorm test completed\n"); free(inp); free(out); free(weight); free(bias); 0; });
}
static void test_matmul() {
    printf("Testing MatMul...\n");
    ({ int32_t B = 1; int32_t T = 2; int32_t C = 3; int32_t OC = 4; float* inp = (float*)(float*)malloc(6 * sizeof(float)); float* weight = (float*)(float*)malloc(12 * sizeof(float)); float* out = (float*)(float*)malloc(8 * sizeof(float)); ({ int32_t i = 0; ({ while ((i < 6)) { (inp[i] = ((i + 0) + 1)); i = (i + 1); } }); }); ({ int32_t i = 0; ({ while ((i < 12)) { (weight[i] = 0.1); i = (i + 1); } }); }); g_user.matmul_forward(out, inp, weight, NULL, B, T, C, OC); printf("MatMul test completed\n"); free(inp); free(weight); free(out); 0; });
}
static void test_gelu() {
    printf("Testing GELU...\n");
    ({ int32_t N = 5; float* inp = (float*)(float*)malloc(5 * sizeof(float)); float* out = (float*)(float*)malloc(5 * sizeof(float)); (inp[0] = -2); (inp[1] = -1); (inp[2] = 0); (inp[3] = 1); (inp[4] = 2); g_user.gelu_forward(out, inp, N); printf("GELU results: [%f, %f, %f, %f, %f]\n", out[0], out[1], out[2], out[3], out[4]); free(inp); free(out); 0; });
}
static void test_residual() {
    printf("Testing Residual...\n");
    ({ int32_t N = 4; float* inp1 = (float*)(float*)malloc(4 * sizeof(float)); float* inp2 = (float*)(float*)malloc(4 * sizeof(float)); float* out = (float*)(float*)malloc(4 * sizeof(float)); ({ int32_t i = 0; ({ while ((i < N)) { (inp1[i] = ((i + 0) + 1)); (inp2[i] = ((i + 0) + 10)); i = (i + 1); } }); }); g_user.residual_forward(out, inp1, inp2, N); printf("Residual results: [%f, %f, %f, %f]\n", out[0], out[1], out[2], out[3]); free(inp1); free(inp2); free(out); 0; });
}
static int32_t main_fn() {
    printf("=================================\n");
    printf("GPT-2 Tensor Operations Test\n");
    printf("=================================\n\n");
    g_user.test_layernorm();
    g_user.test_matmul();
    g_user.test_gelu();
    g_user.test_residual();
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
