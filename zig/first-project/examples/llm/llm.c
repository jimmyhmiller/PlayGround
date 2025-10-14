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
    float* wte;
    float* wpe;
    float* ln1w;
    float* ln1b;
    float* qkvw;
    float* qkvb;
    float* attprojw;
    float* attprojb;
    float* ln2w;
    float* ln2b;
    float* fcw;
    float* fcb;
    float* fcprojw;
    float* fcprojb;
    float* lnfw;
    float* lnfb;
} ParameterTensors;

typedef struct {
    float* encoded;
    float* ln1;
    float* ln1_mean;
    float* ln1_rstd;
    float* qkv;
    float* atty;
    float* preatt;
    float* att;
    float* attproj;
    float* residual2;
    float* ln2;
    float* ln2_mean;
    float* ln2_rstd;
    float* fch;
    float* fch_gelu;
    float* fcproj;
    float* residual3;
    float* lnf;
    float* lnf_mean;
    float* lnf_rstd;
    float* logits;
    float* probs;
    float* losses;
} ActivationTensors;


typedef struct {
    void (*encoder_forward)(float*, int32_t*, float*, float*, int32_t, int32_t, int32_t);
    void (*layernorm_forward)(float*, float*, float*, float*, float*, float*, int32_t, int32_t, int32_t);
    void (*matmul_forward_naive)(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*residual_forward)(float*, float*, float*, int32_t);
    void (*gelu_forward)(float*, float*, int32_t);
    void (*attention_forward)(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*softmax_forward)(float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*gpt2_forward)(int32_t*, GPT2Config, ParameterTensors, ActivationTensors, int32_t, int32_t);
    int32_t (*test_encoder_forward)();
    int32_t (*test_layernorm_forward)();
    int32_t (*test_matmul_forward)();
    int32_t (*test_residual_forward)();
    int32_t (*test_gelu_forward)();
    int32_t (*test_attention_forward)();
    int32_t (*test_softmax_forward)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static void encoder_forward(float*, int32_t*, float*, float*, int32_t, int32_t, int32_t);
static void layernorm_forward(float*, float*, float*, float*, float*, float*, int32_t, int32_t, int32_t);
static void matmul_forward_naive(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
static void residual_forward(float*, float*, float*, int32_t);
static void gelu_forward(float*, float*, int32_t);
static void attention_forward(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
static void softmax_forward(float*, float*, int32_t, int32_t, int32_t, int32_t);
static void gpt2_forward(int32_t*, GPT2Config, ParameterTensors, ActivationTensors, int32_t, int32_t);
static int32_t test_encoder_forward();
static int32_t test_layernorm_forward();
static int32_t test_matmul_forward();
static int32_t test_residual_forward();
static int32_t test_gelu_forward();
static int32_t test_attention_forward();
static int32_t test_softmax_forward();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->encoder_forward = &encoder_forward;
    ns->layernorm_forward = &layernorm_forward;
    ns->matmul_forward_naive = &matmul_forward_naive;
    ns->residual_forward = &residual_forward;
    ns->gelu_forward = &gelu_forward;
    ns->attention_forward = &attention_forward;
    ns->softmax_forward = &softmax_forward;
    ns->gpt2_forward = &gpt2_forward;
    ns->test_encoder_forward = &test_encoder_forward;
    ns->test_layernorm_forward = &test_layernorm_forward;
    ns->test_matmul_forward = &test_matmul_forward;
    ns->test_residual_forward = &test_residual_forward;
    ns->test_gelu_forward = &test_gelu_forward;
    ns->test_attention_forward = &test_attention_forward;
    ns->test_softmax_forward = &test_softmax_forward;
    ns->main_fn = &main_fn;
}

static void encoder_forward(float* out, int32_t* inp, float* wte, float* wpe, int32_t B, int32_t T, int32_t C) {
    ({ int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t out_bt_offset = (((b * T) * C) + (t * C)); int32_t ix = inp[((b * T) + t)]; int32_t wte_ix_offset = (ix * C); int32_t wpe_t_offset = (t * C); int32_t i = 0; ({ while ((i < C)) { (out[(out_bt_offset + i)] = (wte[(wte_ix_offset + i)] + wpe[(wpe_t_offset + i)])); i = (i + 1); } }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int32_t B, int32_t T, int32_t C) {
    ({ float eps = 0.00001; int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t x_offset = (((b * T) * C) + (t * C)); float m = 0; int32_t i = 0; ({ while ((i < C)) { m = (m + inp[(x_offset + i)]); i = (i + 1); } }); m = (m / (C + 0)); ({ float v = 0; i = 0; ({ while ((i < C)) { ({ float xshift = (inp[(x_offset + i)] - m); v = (v + (xshift * xshift)); }); i = (i + 1); } }); v = (v / (C + 0)); ({ float s = (1 / sqrtf((v + eps))); int32_t out_bt_offset = (((b * T) * C) + (t * C)); i = 0; ({ while ((i < C)) { ({ float n = (s * (inp[(x_offset + i)] - m)); float o = ((n * weight[i]) + bias[i]); (out[(out_bt_offset + i)] = o); }); i = (i + 1); } }); (mean[((b * T) + t)] = m); (rstd[((b * T) + t)] = s); 0; }); }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static void matmul_forward_naive(float* out, float* inp, float* weight, float* bias, int32_t B, int32_t T, int32_t C, int32_t OC) {
    ({ int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t bt = ((b * T) + t); int32_t o = 0; ({ while ((o < OC)) { ({ float val = ((bias == NULL) ? 0 : bias[o]); int32_t i = 0; ({ while ((i < C)) { val = (val + (inp[((bt * C) + i)] * weight[((o * C) + i)])); i = (i + 1); } }); (out[((bt * OC) + o)] = val); }); o = (o + 1); } }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static void residual_forward(float* out, float* inp1, float* inp2, int32_t N) {
    ({ int32_t i = 0; ({ while ((i < N)) { (out[i] = (inp1[i] + inp2[i])); i = (i + 1); } }); });
}
static void gelu_forward(float* out, float* inp, int32_t N) {
    ({ float pi_value = 3.141592653589793; float GELU_SCALING_FACTOR = sqrtf((2 / pi_value)); int32_t i = 0; ({ while ((i < N)) { ({ float x = inp[i]; float cube = (((0.044715 * x) * x) * x); float result = ((0.5 * x) * (1 + tanhf((GELU_SCALING_FACTOR * (x + cube))))); (out[i] = result); }); i = (i + 1); } }); });
}
static void attention_forward(float* out, float* preatt, float* att, float* inp, int32_t B, int32_t T, int32_t C, int32_t NH) {
    ({ int32_t C3 = (C * 3); int32_t hs = (C / NH); float scale = (1 / sqrtf((hs + 0))); int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t h = 0; ({ while ((h < NH)) { ({ int32_t query_t_offset = ((((b * T) * C3) + (t * C3)) + (h * hs)); int32_t preatt_bth_offset = (((((b * NH) * T) * T) + ((h * T) * T)) + (t * T)); int32_t att_bth_offset = (((((b * NH) * T) * T) + ((h * T) * T)) + (t * T)); float maxval = (0 - 10000); int32_t t2 = 0; ({ while ((t2 <= t)) { ({ int32_t key_t2_offset = (((((b * T) * C3) + (t2 * C3)) + (h * hs)) + C); float val = 0; int32_t i = 0; ({ while ((i < hs)) { val = (val + (inp[(query_t_offset + i)] * inp[(key_t2_offset + i)])); i = (i + 1); } }); val = (val * scale); ({ if ((val > maxval)) { maxval = val; } else { } }); (preatt[(preatt_bth_offset + t2)] = val); }); t2 = (t2 + 1); } }); ({ float expsum = 0; t2 = 0; ({ while ((t2 <= t)) { ({ float expv = expf((preatt[(preatt_bth_offset + t2)] - maxval)); expsum = (expsum + expv); (att[(att_bth_offset + t2)] = expv); }); t2 = (t2 + 1); } }); ({ float expsum_inv = ((expsum == 0) ? 0 : (1 / expsum)); t2 = 0; ({ while ((t2 < T)) { ({ if ((t2 <= t)) { (att[(att_bth_offset + t2)] = (att[(att_bth_offset + t2)] * expsum_inv)); } else { (att[(att_bth_offset + t2)] = 0); } }); t2 = (t2 + 1); } }); ({ int32_t out_bth_offset = ((((b * T) * C) + (t * C)) + (h * hs)); int32_t i = 0; ({ while ((i < hs)) { (out[(out_bth_offset + i)] = 0); i = (i + 1); } }); t2 = 0; ({ while ((t2 <= t)) { ({ int32_t value_t2_offset = (((((b * T) * C3) + (t2 * C3)) + (h * hs)) + (C * 2)); float att_btht2 = att[(att_bth_offset + t2)]; i = 0; ({ while ((i < hs)) { (out[(out_bth_offset + i)] = (out[(out_bth_offset + i)] + (att_btht2 * inp[(value_t2_offset + i)]))); i = (i + 1); } }); }); t2 = (t2 + 1); } }); }); }); }); }); h = (h + 1); } }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static void softmax_forward(float* probs, float* logits, int32_t B, int32_t T, int32_t V, int32_t Vp) {
    ({ int32_t b = 0; ({ while ((b < B)) { ({ int32_t t = 0; ({ while ((t < T)) { ({ int32_t logits_bt_offset = (((b * T) * Vp) + (t * Vp)); int32_t probs_bt_offset = (((b * T) * Vp) + (t * Vp)); float maxval = (0 - 10000); int32_t i = 0; ({ while ((i < V)) { ({ float val = logits[(logits_bt_offset + i)]; ({ if ((val > maxval)) { maxval = val; } else { } }); }); i = (i + 1); } }); ({ float sum = 0; i = 0; ({ while ((i < V)) { ({ float expv = expf((logits[(logits_bt_offset + i)] - maxval)); (probs[(probs_bt_offset + i)] = expv); sum = (sum + expv); }); i = (i + 1); } }); i = 0; ({ while ((i < V)) { (probs[(probs_bt_offset + i)] = (probs[(probs_bt_offset + i)] / sum)); i = (i + 1); } }); i = V; ({ while ((i < Vp)) { (probs[(probs_bt_offset + i)] = 0); i = (i + 1); } }); }); }); t = (t + 1); } }); }); b = (b + 1); } }); });
}
static void gpt2_forward(int32_t* inputs, GPT2Config config, ParameterTensors params, ActivationTensors acts, int32_t B, int32_t T) {
    ({ int32_t C = config.channels; int32_t L = config.num_layers; int32_t NH = config.num_heads; int32_t V = config.vocab_size; int32_t Vp = config.padded_vocab_size; g_user.encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); ({ int32_t l = 0; ({ while ((l < L)) { ({ int32_t BTC = ((B * T) * C); int32_t BT = (B * T); float* residual = (float*)((l == 0) ? acts.encoded : (acts.residual3 + (((l - 1) * BTC) * 1))); float* l_ln1w = (float*)(params.ln1w + (l * C)); float* l_ln1b = (float*)(params.ln1b + (l * C)); float* l_qkvw = (float*)(params.qkvw + (((l * 3) * C) * C)); float* l_qkvb = (float*)(params.qkvb + ((l * 3) * C)); float* l_attprojw = (float*)(params.attprojw + ((l * C) * C)); float* l_attprojb = (float*)(params.attprojb + (l * C)); float* l_ln2w = (float*)(params.ln2w + (l * C)); float* l_ln2b = (float*)(params.ln2b + (l * C)); float* l_fcw = (float*)(params.fcw + (((l * 4) * C) * C)); float* l_fcb = (float*)(params.fcb + ((l * 4) * C)); float* l_fcprojw = (float*)(params.fcprojw + ((l * C) * (4 * C))); float* l_fcprojb = (float*)(params.fcprojb + (l * C)); float* l_ln1 = (float*)(acts.ln1 + (l * BTC)); float* l_ln1_mean = (float*)(acts.ln1_mean + (l * BT)); float* l_ln1_rstd = (float*)(acts.ln1_rstd + (l * BT)); float* l_qkv = (float*)(acts.qkv + (((l * B) * T) * (3 * C))); float* l_atty = (float*)(acts.atty + (l * BTC)); float* l_preatt = (float*)(acts.preatt + ((((l * B) * NH) * T) * T)); float* l_att = (float*)(acts.att + ((((l * B) * NH) * T) * T)); float* l_attproj = (float*)(acts.attproj + (l * BTC)); float* l_residual2 = (float*)(acts.residual2 + (l * BTC)); float* l_ln2 = (float*)(acts.ln2 + (l * BTC)); float* l_ln2_mean = (float*)(acts.ln2_mean + (l * BT)); float* l_ln2_rstd = (float*)(acts.ln2_rstd + (l * BT)); float* l_fch = (float*)(acts.fch + (((l * B) * T) * (4 * C))); float* l_fch_gelu = (float*)(acts.fch_gelu + (((l * B) * T) * (4 * C))); float* l_fcproj = (float*)(acts.fcproj + (l * BTC)); float* l_residual3 = (float*)(acts.residual3 + (l * BTC)); g_user.layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C); g_user.matmul_forward_naive(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, (3 * C)); g_user.attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH); g_user.matmul_forward_naive(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C); g_user.residual_forward(l_residual2, residual, l_attproj, BTC); g_user.layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C); g_user.matmul_forward_naive(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, (4 * C)); g_user.gelu_forward(l_fch_gelu, l_fch, ((B * T) * (4 * C))); g_user.matmul_forward_naive(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, (4 * C), C); g_user.residual_forward(l_residual3, l_residual2, l_fcproj, BTC); }); l = (l + 1); } }); }); ({ float* residual = (float*)(acts.residual3 + ((((L - 1) * B) * T) * C)); g_user.layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C); g_user.matmul_forward_naive(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp); g_user.softmax_forward(acts.probs, acts.logits, B, T, V, Vp); }); });
}
static int32_t test_encoder_forward() {
    printf("Testing encoder_forward...\n");
    return ({ int32_t B = 2; int32_t T = 3; int32_t C = 4; int32_t V = 10; float* out = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); int32_t* inp = (int32_t*)(int32_t*)malloc((B * T) * sizeof(int32_t)); float* wte = (float*)(float*)malloc((V * C) * sizeof(float)); float* wpe = (float*)(float*)malloc((T * C) * sizeof(float)); (inp[0] = 1); (inp[1] = 2); (inp[2] = 3); (inp[3] = 4); (inp[4] = 5); (inp[5] = 6); ({ int32_t i = 0; ({ while ((i < (V * C))) { (wte[i] = (((i + 0) * 0.1) + 1)); i = (i + 1); } }); }); ({ int32_t i = 0; ({ while ((i < (T * C))) { (wpe[i] = (((i + 0) * 0.01) + 0.5)); i = (i + 1); } }); }); g_user.encoder_forward(out, inp, wte, wpe, B, T, C); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  out[3] = %f\n", out[3]); free(out); free(inp); free(wte); free(wpe); printf("encoder_forward test completed!\n"); 0; });
}
static int32_t test_layernorm_forward() {
    printf("Testing layernorm_forward...\n");
    return ({ int32_t B = 2; int32_t T = 2; int32_t C = 3; float* out = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); float* mean = (float*)(float*)malloc((B * T) * sizeof(float)); float* rstd = (float*)(float*)malloc((B * T) * sizeof(float)); float* inp = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); float* weight = (float*)({ float* __arr = (float*)malloc(C * sizeof(float)); for (size_t __i = 0; __i < C; __i++) __arr[__i] = 1; __arr; }); float* bias = (float*)({ float* __arr = (float*)malloc(C * sizeof(float)); for (size_t __i = 0; __i < C; __i++) __arr[__i] = 0; __arr; }); ({ int32_t i = 0; ({ while ((i < ((B * T) * C))) { (inp[i] = ((i + 0) + 1)); i = (i + 1); } }); }); g_user.layernorm_forward(out, mean, rstd, inp, weight, bias, B, T, C); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  mean[0] = %f\n", mean[0]); printf("  rstd[0] = %f\n", rstd[0]); free(out); free(mean); free(rstd); free(inp); free(weight); free(bias); printf("layernorm_forward test completed!\n"); 0; });
}
static int32_t test_matmul_forward() {
    printf("Testing matmul_forward_naive...\n");
    return ({ int32_t B = 2; int32_t T = 2; int32_t C = 3; int32_t OC = 4; float* out = (float*)(float*)malloc(((B * T) * OC) * sizeof(float)); float* inp = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); float* weight = (float*)(float*)malloc((OC * C) * sizeof(float)); float* bias = (float*)(float*)malloc(OC * sizeof(float)); ({ int32_t i = 0; ({ while ((i < ((B * T) * C))) { (inp[i] = ((i + 0) + 1)); i = (i + 1); } }); }); ({ int32_t i = 0; ({ while ((i < (OC * C))) { (weight[i] = ((i + 0) + 1)); i = (i + 1); } }); }); ({ int32_t i = 0; ({ while ((i < OC)) { (bias[i] = ((i + 0) + 0.1)); i = (i + 1); } }); }); g_user.matmul_forward_naive(out, inp, weight, bias, B, T, C, OC); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  out[3] = %f\n", out[3]); free(out); free(inp); free(weight); free(bias); printf("matmul_forward_naive test completed!\n"); 0; });
}
static int32_t test_residual_forward() {
    printf("Testing residual_forward...\n");
    return ({ int32_t N = 6; float* out = (float*)(float*)malloc(N * sizeof(float)); float* inp1 = (float*)(float*)malloc(N * sizeof(float)); float* inp2 = (float*)(float*)malloc(N * sizeof(float)); ({ int32_t i = 0; ({ while ((i < N)) { (inp1[i] = ((i + 0) + 1)); i = (i + 1); } }); }); ({ int32_t i = 0; ({ while ((i < N)) { (inp2[i] = (((i + 0) + 1) * 0.1)); i = (i + 1); } }); }); g_user.residual_forward(out, inp1, inp2, N); printf("First output values:\n"); printf("  out[0] = %f (expected 1.1)\n", out[0]); printf("  out[1] = %f (expected 2.2)\n", out[1]); printf("  out[2] = %f (expected 3.3)\n", out[2]); free(out); free(inp1); free(inp2); printf("residual_forward test completed!\n"); 0; });
}
static int32_t test_gelu_forward() {
    printf("Testing gelu_forward...\n");
    return ({ int32_t N = 4; float* out = (float*)(float*)malloc(N * sizeof(float)); float* inp = (float*)(float*)malloc(N * sizeof(float)); (inp[0] = (0 - 1)); (inp[1] = 0); (inp[2] = 1); (inp[3] = 2); g_user.gelu_forward(out, inp, N); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  out[3] = %f\n", out[3]); free(out); free(inp); printf("gelu_forward test completed!\n"); 0; });
}
static int32_t test_attention_forward() {
    printf("Testing attention_forward...\n");
    return ({ int32_t B = 1; int32_t T = 2; int32_t C = 4; int32_t NH = 2; float* out = (float*)(float*)malloc(((B * T) * C) * sizeof(float)); float* preatt = (float*)(float*)malloc(((((B * NH) * T) * T) * 1) * sizeof(float)); float* att = (float*)(float*)malloc(((((B * NH) * T) * T) * 1) * sizeof(float)); float* inp = (float*)(float*)malloc(((B * T) * (C * 3)) * sizeof(float)); ({ int32_t i = 0; ({ while ((i < ((B * T) * (C * 3)))) { (inp[i] = (((i + 0) + 1) * 0.1)); i = (i + 1); } }); }); g_user.attention_forward(out, preatt, att, inp, B, T, C, NH); printf("First output values:\n"); printf("  out[0] = %f\n", out[0]); printf("  out[1] = %f\n", out[1]); printf("  out[2] = %f\n", out[2]); printf("  out[3] = %f\n", out[3]); free(out); free(preatt); free(att); free(inp); printf("attention_forward test completed!\n"); 0; });
}
static int32_t test_softmax_forward() {
    printf("Testing softmax_forward...\n");
    return ({ int32_t B = 1; int32_t T = 2; int32_t V = 4; int32_t Vp = 8; float* probs = (float*)(float*)malloc(((B * T) * Vp) * sizeof(float)); float* logits = (float*)(float*)malloc(((B * T) * Vp) * sizeof(float)); (logits[0] = 1); (logits[1] = 2); (logits[2] = 3); (logits[3] = 4); (logits[4] = 0); (logits[5] = 0); (logits[6] = 0); (logits[7] = 0); (logits[8] = 2); (logits[9] = 2); (logits[10] = 2); (logits[11] = 2); (logits[12] = 0); (logits[13] = 0); (logits[14] = 0); (logits[15] = 0); g_user.softmax_forward(probs, logits, B, T, V, Vp); printf("First position probabilities:\n"); printf("  probs[0] = %f\n", probs[0]); printf("  probs[1] = %f\n", probs[1]); printf("  probs[2] = %f\n", probs[2]); printf("  probs[3] = %f\n", probs[3]); printf("  Sum = %f (should be 1.0)\n", (((probs[0] + probs[1]) + probs[2]) + probs[3])); printf("Second position probabilities (all equal logits):\n"); printf("  probs[8] = %f\n", probs[8]); printf("  probs[9] = %f\n", probs[9]); free(probs); free(logits); printf("softmax_forward test completed!\n"); 0; });
}
static int32_t main_fn() {
    g_user.test_encoder_forward();
    printf("\n");
    g_user.test_layernorm_forward();
    printf("\n");
    g_user.test_matmul_forward();
    printf("\n");
    g_user.test_residual_forward();
    printf("\n");
    g_user.test_gelu_forward();
    printf("\n");
    g_user.test_attention_forward();
    printf("\n");
    return g_user.test_softmax_forward();
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
