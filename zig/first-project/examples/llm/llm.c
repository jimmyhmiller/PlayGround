#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "stdio.h"
#include "stdlib.h"
#include "math.h"
typedef struct {
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t padded_vocab_size;
    int32_t num_layers;
    int32_t num_heads;
    int32_t channels;
} GPT2Config;


typedef struct {
    void (*encoder_forward)(float*, int32_t*, float*, float*, int32_t, int32_t, int32_t);
    void (*layernorm_forward)(float*, float*, float*, float*, float*, float*, int32_t, int32_t, int32_t);
    int32_t (*test_encoder_forward)();
    int32_t (*test_layernorm_forward)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static void encoder_forward(float*, int32_t*, float*, float*, int32_t, int32_t, int32_t);
static void layernorm_forward(float*, float*, float*, float*, float*, float*, int32_t, int32_t, int32_t);
static int32_t test_encoder_forward();
static int32_t test_layernorm_forward();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->encoder_forward = &encoder_forward;
    ns->layernorm_forward = &layernorm_forward;
    ns->test_encoder_forward = &test_encoder_forward;
    ns->test_layernorm_forward = &test_layernorm_forward;
    ns->main_fn = &main_fn;
}

static void encoder_forward(float* out, int32_t* inp, float* wte, float* wpe, int32_t B, int32_t T, int32_t C) {
    ({ int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t out_bt_offset = (((b * T) * C) + (t * C)); int32_t ix = inp[((b * T) + t)]; int32_t wte_ix_offset = (ix * C); int32_t wpe_t_offset = (t * C); int32_t i = 0; ({ while ((i < C)) { (out[(out_bt_offset + i)] = (wte[(wte_ix_offset + i)] + wpe[(wpe_t_offset + i)])); i = (i + 1); } }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int32_t B, int32_t T, int32_t C) {
    ({ float eps = 0.00001; int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t x_offset = (((b * T) * C) + (t * C)); float m = 0; int32_t i = 0; ({ while ((i < C)) { m = (m + inp[(x_offset + i)]); i = (i + 1); } }); m = (m / (C + 0)); ({ float v = 0; i = 0; ({ while ((i < C)) { ({ float xshift = (inp[(x_offset + i)] - m); v = (v + (xshift * xshift)); }); i = (i + 1); } }); v = (v / (C + 0)); ({ float s = (1 / sqrtf((v + eps))); int32_t out_bt_offset = (((b * T) * C) + (t * C)); i = 0; ({ while ((i < C)) { ({ float n = (s * (inp[(x_offset + i)] - m)); float o = ((n * weight[i]) + bias[i]); (out[(out_bt_offset + i)] = o); }); i = (i + 1); } }); (mean[((b * T) + t)] = m); (rstd[((b * T) + t)] = s); 0; }); }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static int32_t test_encoder_forward() {
    printf("Testing encoder_forward...\n");
    return ({ int32_t B = 2; int32_t T = 3; int32_t C = 4; int32_t V = 10; float* out = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); int32_t* inp = (int32_t*)(int32_t*)malloc((B * T) * sizeof(int32_t)); float* wte = (float*)(float*)malloc((V * C) * sizeof(float)); float* wpe = (float*)(float*)malloc((T * C) * sizeof(float)); (inp[0] = 1); (inp[1] = 2); (inp[2] = 3); (inp[3] = 4); (inp[4] = 5); (inp[5] = 6); ({ int32_t i = 0; ({ while ((i < (V * C))) { (wte[i] = (((i + 0) * 0.1) + 1)); i = (i + 1); } }); }); ({ int32_t i = 0; ({ while ((i < (T * C))) { (wpe[i] = (((i + 0) * 0.01) + 0.5)); i = (i + 1); } }); }); g_user.encoder_forward(out, inp, wte, wpe, B, T, C); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  out[3] = %f\n", out[3]); free(out); free(inp); free(wte); free(wpe); printf("encoder_forward test completed!\n"); 0; });
}
static int32_t test_layernorm_forward() {
    printf("Testing layernorm_forward...\n");
    return ({ int32_t B = 2; int32_t T = 2; int32_t C = 3; float* out = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); float* mean = (float*)(float*)malloc((B * T) * sizeof(float)); float* rstd = (float*)(float*)malloc((B * T) * sizeof(float)); float* inp = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); float* weight = (float*)({ float* __arr = (float*)malloc(C * sizeof(float)); for (size_t __i = 0; __i < C; __i++) __arr[__i] = 1; __arr; }); float* bias = (float*)({ float* __arr = (float*)malloc(C * sizeof(float)); for (size_t __i = 0; __i < C; __i++) __arr[__i] = 0; __arr; }); ({ int32_t i = 0; ({ while ((i < ((B * T) * C))) { (inp[i] = ((i + 0) + 1)); i = (i + 1); } }); }); g_user.layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  mean[0] = %f\n", mean[0]); printf("  rstd[0] = %f\n", rstd[0]); free(out); free(mean); free(rstd); free(inp); free(weight); free(bias); printf("layernorm_forward test completed!\n"); 0; });
}
static int32_t main_fn() {
    g_user.test_encoder_forward();
    printf("\n");
    return g_user.test_layernorm_forward();
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
