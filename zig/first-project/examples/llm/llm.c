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
    GPT2Config config;
    ParameterTensors params;
} CheckpointData;


typedef struct {
    CheckpointData (*load_gpt2_checkpoint)(uint8_t*);
    void (*encoder_forward)(float*, int32_t*, float*, float*, int32_t, int32_t, int32_t);
    void (*layernorm_forward)(float*, float*, float*, float*, float*, float*, int32_t, int32_t, int32_t);
    void (*matmul_forward_naive)(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*matmul_forward)(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*residual_forward)(float*, float*, float*, int32_t);
    void (*gelu_forward)(float*, float*, int32_t);
    void (*attention_forward)(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
    void (*softmax_forward)(float*, float*, int32_t, int32_t, int32_t, int32_t);
    int32_t (*argmax)(float*, int32_t);
    void (*gpt2_forward)(int32_t*, GPT2Config, ParameterTensors, ActivationTensors, int32_t, int32_t);
    int32_t (*test_encoder_forward)();
    int32_t (*test_layernorm_forward)();
    int32_t (*test_matmul_forward)();
    int32_t (*test_residual_forward)();
    int32_t (*test_gelu_forward)();
    int32_t (*test_attention_forward)();
    int32_t (*test_softmax_forward)();
    ActivationTensors (*allocate_activation_tensors)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
    ParameterTensors (*allocate_parameter_tensors)(int32_t, int32_t, int32_t, int32_t, int32_t);
    int32_t (*test_gpt2_inference)();
    int32_t (*test_real_gpt2_inference)();
    int32_t (*test_autoregressive_generation)();
    int32_t (*main_fn)();
} Namespace_user;

Namespace_user g_user;

static CheckpointData load_gpt2_checkpoint(uint8_t*);
static void encoder_forward(float*, int32_t*, float*, float*, int32_t, int32_t, int32_t);
static void layernorm_forward(float*, float*, float*, float*, float*, float*, int32_t, int32_t, int32_t);
static void matmul_forward_naive(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
static void matmul_forward(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
static void residual_forward(float*, float*, float*, int32_t);
static void gelu_forward(float*, float*, int32_t);
static void attention_forward(float*, float*, float*, float*, int32_t, int32_t, int32_t, int32_t);
static void softmax_forward(float*, float*, int32_t, int32_t, int32_t, int32_t);
static int32_t argmax(float*, int32_t);
static void gpt2_forward(int32_t*, GPT2Config, ParameterTensors, ActivationTensors, int32_t, int32_t);
static int32_t test_encoder_forward();
static int32_t test_layernorm_forward();
static int32_t test_matmul_forward();
static int32_t test_residual_forward();
static int32_t test_gelu_forward();
static int32_t test_attention_forward();
static int32_t test_softmax_forward();
static ActivationTensors allocate_activation_tensors(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
static ParameterTensors allocate_parameter_tensors(int32_t, int32_t, int32_t, int32_t, int32_t);
static int32_t test_gpt2_inference();
static int32_t test_real_gpt2_inference();
static int32_t test_autoregressive_generation();
static int32_t main_fn();

void init_namespace_user(Namespace_user* ns) {
    ns->load_gpt2_checkpoint = &load_gpt2_checkpoint;
    ns->encoder_forward = &encoder_forward;
    ns->layernorm_forward = &layernorm_forward;
    ns->matmul_forward_naive = &matmul_forward_naive;
    ns->matmul_forward = &matmul_forward;
    ns->residual_forward = &residual_forward;
    ns->gelu_forward = &gelu_forward;
    ns->attention_forward = &attention_forward;
    ns->softmax_forward = &softmax_forward;
    ns->argmax = &argmax;
    ns->gpt2_forward = &gpt2_forward;
    ns->test_encoder_forward = &test_encoder_forward;
    ns->test_layernorm_forward = &test_layernorm_forward;
    ns->test_matmul_forward = &test_matmul_forward;
    ns->test_residual_forward = &test_residual_forward;
    ns->test_gelu_forward = &test_gelu_forward;
    ns->test_attention_forward = &test_attention_forward;
    ns->test_softmax_forward = &test_softmax_forward;
    ns->allocate_activation_tensors = &allocate_activation_tensors;
    ns->allocate_parameter_tensors = &allocate_parameter_tensors;
    ns->test_gpt2_inference = &test_gpt2_inference;
    ns->test_real_gpt2_inference = &test_real_gpt2_inference;
    ns->test_autoregressive_generation = &test_autoregressive_generation;
    ns->main_fn = &main_fn;
}

static CheckpointData load_gpt2_checkpoint(uint8_t* checkpoint_path) {
    printf("Loading checkpoint from file...\n");
    return ({ uint8_t* model_file = (uint8_t*)fopen(checkpoint_path, "rb"); ((model_file == NULL) ? ({ int32_t dummy = 0; printf("ERROR: Could not open checkpoint file\n"); exit(1); (CheckpointData){(GPT2Config){0, 0, 0, 0, 0, 0}, (ParameterTensors){NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL}}; }) : ({ int32_t* header = (int32_t*)(int32_t*)malloc(256 * sizeof(int32_t)); int32_t bytes_read = fread(((uint8_t*)header), 4, 256, model_file); ({ int32_t magic = header[0]; int32_t version = header[1]; ({ if ((magic != 20240326)) { ({ int32_t dummy = 0; printf("ERROR: Bad magic number in model file: %d\n", magic); exit(1); }); } else { } }); ({ if ((version != 3)) { ({ int32_t dummy = 0; printf("ERROR: Bad version in model file: %d (expected 3)\n", version); printf("HINT: Re-run 'python train_gpt2.py' to generate correct format\n"); exit(1); }); } else { } }); ({ int32_t maxT = header[2]; int32_t V = header[3]; int32_t L = header[4]; int32_t NH = header[5]; int32_t C = header[6]; int32_t Vp = header[7]; printf("[GPT-2]\n"); printf("max_seq_len: %d\n", maxT); printf("vocab_size: %d\n", V); printf("padded_vocab_size: %d\n", Vp); printf("num_layers: %d\n", L); printf("num_heads: %d\n", NH); printf("channels: %d\n", C); ({ int32_t wte_size = (Vp * C); int32_t wpe_size = (maxT * C); int32_t ln1w_size = (L * C); int32_t ln1b_size = (L * C); int32_t qkvw_size = (((L * 3) * C) * C); int32_t qkvb_size = ((L * 3) * C); int32_t attprojw_size = ((L * C) * C); int32_t attprojb_size = (L * C); int32_t ln2w_size = (L * C); int32_t ln2b_size = (L * C); int32_t fcw_size = (((L * 4) * C) * C); int32_t fcb_size = ((L * 4) * C); int32_t fcprojw_size = ((L * C) * (4 * C)); int32_t fcprojb_size = (L * C); int32_t lnfw_size = C; int32_t lnfb_size = C; int32_t num_parameters = (((wte_size + wpe_size) + ((ln1w_size + ln1b_size) + (qkvw_size + qkvb_size))) + (((attprojw_size + attprojb_size) + (ln2w_size + ln2b_size)) + (((fcw_size + fcb_size) + (fcprojw_size + fcprojb_size)) + (lnfw_size + lnfb_size)))); printf("num_parameters: %d\n", num_parameters); ({ float* params_memory = (float*)(float*)malloc(num_parameters * sizeof(float)); int32_t params_bytes_read = fread(((uint8_t*)params_memory), 4, num_parameters, model_file); fclose(model_file); free(header); ({ float* wte = (float*)params_memory; float* wpe = (float*)(wte + wte_size); float* ln1w = (float*)(wpe + wpe_size); float* ln1b = (float*)(ln1w + ln1w_size); float* qkvw = (float*)(ln1b + ln1b_size); float* qkvb = (float*)(qkvw + qkvw_size); float* attprojw = (float*)(qkvb + qkvb_size); float* attprojb = (float*)(attprojw + attprojw_size); float* ln2w = (float*)(attprojb + attprojb_size); float* ln2b = (float*)(ln2w + ln2w_size); float* fcw = (float*)(ln2b + ln2b_size); float* fcb = (float*)(fcw + fcw_size); float* fcprojw = (float*)(fcb + fcb_size); float* fcprojb = (float*)(fcprojw + fcprojw_size); float* lnfw = (float*)(fcprojb + fcprojb_size); float* lnfb = (float*)(lnfw + lnfw_size); GPT2Config config = (GPT2Config){maxT, V, Vp, L, NH, C}; ParameterTensors params = (ParameterTensors){wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb, lnfw, lnfb}; printf("Successfully loaded checkpoint!\n"); (CheckpointData){config, params}; }); }); }); }); }); })); });
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
static void matmul_forward(float* out, float* inp, float* weight, float* bias, int32_t B, int32_t T, int32_t C, int32_t OC) {
    ({ int32_t LOOP_UNROLL = 8; int32_t BT = (B * T); ({ if (((BT % LOOP_UNROLL) != 0)) { g_user.matmul_forward_naive(out, inp, weight, bias, B, T, C, OC); } else { ({ int32_t obt = 0; ({ while ((obt < BT)) { ({ int32_t o = 0; ({ while ((o < OC)) { ({ float result0 = ((bias == NULL) ? 0 : bias[o]); float result1 = ((bias == NULL) ? 0 : bias[o]); float result2 = ((bias == NULL) ? 0 : bias[o]); float result3 = ((bias == NULL) ? 0 : bias[o]); float result4 = ((bias == NULL) ? 0 : bias[o]); float result5 = ((bias == NULL) ? 0 : bias[o]); float result6 = ((bias == NULL) ? 0 : bias[o]); float result7 = ((bias == NULL) ? 0 : bias[o]); ({ int32_t i = 0; ({ while ((i < C)) { ({ float w = weight[(i + (o * C))]; int32_t bt0 = (obt + 0); int32_t bt1 = (obt + 1); int32_t bt2 = (obt + 2); int32_t bt3 = (obt + 3); int32_t bt4 = (obt + 4); int32_t bt5 = (obt + 5); int32_t bt6 = (obt + 6); int32_t bt7 = (obt + 7); result0 = (result0 + (inp[((bt0 * C) + i)] * w)); result1 = (result1 + (inp[((bt1 * C) + i)] * w)); result2 = (result2 + (inp[((bt2 * C) + i)] * w)); result3 = (result3 + (inp[((bt3 * C) + i)] * w)); result4 = (result4 + (inp[((bt4 * C) + i)] * w)); result5 = (result5 + (inp[((bt5 * C) + i)] * w)); result6 = (result6 + (inp[((bt6 * C) + i)] * w)); result7 = (result7 + (inp[((bt7 * C) + i)] * w)); }); i = (i + 1); } }); }); (out[(((obt + 0) * OC) + o)] = result0); (out[(((obt + 1) * OC) + o)] = result1); (out[(((obt + 2) * OC) + o)] = result2); (out[(((obt + 3) * OC) + o)] = result3); (out[(((obt + 4) * OC) + o)] = result4); (out[(((obt + 5) * OC) + o)] = result5); (out[(((obt + 6) * OC) + o)] = result6); (out[(((obt + 7) * OC) + o)] = result7); }); o = (o + 1); } }); }); obt = (obt + LOOP_UNROLL); } }); }); } }); });
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
static int32_t argmax(float* probs, int32_t n) {
    return ({ float max_val = (0 - 999999); int32_t max_idx = 0; int32_t i = 0; ({ while ((i < n)) { ({ float val = probs[i]; ({ if ((val > max_val)) { ({ int32_t dummy = 0; max_val = val; max_idx = i; }); } else { } }); }); i = (i + 1); } }); max_idx; });
}
static void gpt2_forward(int32_t* inputs, GPT2Config config, ParameterTensors params, ActivationTensors acts, int32_t B, int32_t T) {
    ({ int32_t C = config.channels; int32_t L = config.num_layers; int32_t NH = config.num_heads; int32_t V = config.vocab_size; int32_t Vp = config.padded_vocab_size; g_user.encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); ({ int32_t l = 0; ({ while ((l < L)) { ({ int32_t BTC = ((B * T) * C); int32_t BT = (B * T); float* residual = (float*)((l == 0) ? acts.encoded : (acts.residual3 + (((l - 1) * BTC) * 1))); float* l_ln1w = (float*)(params.ln1w + (l * C)); float* l_ln1b = (float*)(params.ln1b + (l * C)); float* l_qkvw = (float*)(params.qkvw + (((l * 3) * C) * C)); float* l_qkvb = (float*)(params.qkvb + ((l * 3) * C)); float* l_attprojw = (float*)(params.attprojw + ((l * C) * C)); float* l_attprojb = (float*)(params.attprojb + (l * C)); float* l_ln2w = (float*)(params.ln2w + (l * C)); float* l_ln2b = (float*)(params.ln2b + (l * C)); float* l_fcw = (float*)(params.fcw + (((l * 4) * C) * C)); float* l_fcb = (float*)(params.fcb + ((l * 4) * C)); float* l_fcprojw = (float*)(params.fcprojw + ((l * C) * (4 * C))); float* l_fcprojb = (float*)(params.fcprojb + (l * C)); float* l_ln1 = (float*)(acts.ln1 + (l * BTC)); float* l_ln1_mean = (float*)(acts.ln1_mean + (l * BT)); float* l_ln1_rstd = (float*)(acts.ln1_rstd + (l * BT)); float* l_qkv = (float*)(acts.qkv + (((l * B) * T) * (3 * C))); float* l_atty = (float*)(acts.atty + (l * BTC)); float* l_preatt = (float*)(acts.preatt + ((((l * B) * NH) * T) * T)); float* l_att = (float*)(acts.att + ((((l * B) * NH) * T) * T)); float* l_attproj = (float*)(acts.attproj + (l * BTC)); float* l_residual2 = (float*)(acts.residual2 + (l * BTC)); float* l_ln2 = (float*)(acts.ln2 + (l * BTC)); float* l_ln2_mean = (float*)(acts.ln2_mean + (l * BT)); float* l_ln2_rstd = (float*)(acts.ln2_rstd + (l * BT)); float* l_fch = (float*)(acts.fch + (((l * B) * T) * (4 * C))); float* l_fch_gelu = (float*)(acts.fch_gelu + (((l * B) * T) * (4 * C))); float* l_fcproj = (float*)(acts.fcproj + (l * BTC)); float* l_residual3 = (float*)(acts.residual3 + (l * BTC)); g_user.layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C); g_user.matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, (3 * C)); g_user.attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH); g_user.matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C); g_user.residual_forward(l_residual2, residual, l_attproj, BTC); g_user.layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C); g_user.matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, (4 * C)); g_user.gelu_forward(l_fch_gelu, l_fch, ((B * T) * (4 * C))); g_user.matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, (4 * C), C); g_user.residual_forward(l_residual3, l_residual2, l_fcproj, BTC); }); l = (l + 1); } }); }); ({ float* residual = (float*)(acts.residual3 + ((((L - 1) * B) * T) * C)); g_user.layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C); g_user.matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp); g_user.softmax_forward(acts.probs, acts.logits, B, T, V, Vp); }); });
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
static ActivationTensors allocate_activation_tensors(int32_t B, int32_t T, int32_t C, int32_t L, int32_t NH, int32_t Vp) {
    return ({ int32_t encoded_size = ((B * T) * C); int32_t ln1_size = (((L * B) * T) * C); int32_t ln1_mean_size = ((L * B) * T); int32_t ln1_rstd_size = ((L * B) * T); int32_t qkv_size = (((L * B) * T) * (3 * C)); int32_t atty_size = (((L * B) * T) * C); int32_t preatt_size = ((((L * B) * NH) * T) * T); int32_t att_size = ((((L * B) * NH) * T) * T); int32_t attproj_size = (((L * B) * T) * C); int32_t residual2_size = (((L * B) * T) * C); int32_t ln2_size = (((L * B) * T) * C); int32_t ln2_mean_size = ((L * B) * T); int32_t ln2_rstd_size = ((L * B) * T); int32_t fch_size = (((L * B) * T) * (4 * C)); int32_t fch_gelu_size = (((L * B) * T) * (4 * C)); int32_t fcproj_size = (((L * B) * T) * C); int32_t residual3_size = (((L * B) * T) * C); int32_t lnf_size = ((B * T) * C); int32_t lnf_mean_size = (B * T); int32_t lnf_rstd_size = (B * T); int32_t logits_size = ((B * T) * Vp); int32_t probs_size = ((B * T) * Vp); int32_t losses_size = (B * T); (ActivationTensors){(float*)malloc(encoded_size * sizeof(float)), (float*)malloc(ln1_size * sizeof(float)), (float*)malloc(ln1_mean_size * sizeof(float)), (float*)malloc(ln1_rstd_size * sizeof(float)), (float*)malloc(qkv_size * sizeof(float)), (float*)malloc(atty_size * sizeof(float)), (float*)malloc(preatt_size * sizeof(float)), (float*)malloc(att_size * sizeof(float)), (float*)malloc(attproj_size * sizeof(float)), (float*)malloc(residual2_size * sizeof(float)), (float*)malloc(ln2_size * sizeof(float)), (float*)malloc(ln2_mean_size * sizeof(float)), (float*)malloc(ln2_rstd_size * sizeof(float)), (float*)malloc(fch_size * sizeof(float)), (float*)malloc(fch_gelu_size * sizeof(float)), (float*)malloc(fcproj_size * sizeof(float)), (float*)malloc(residual3_size * sizeof(float)), (float*)malloc(lnf_size * sizeof(float)), (float*)malloc(lnf_mean_size * sizeof(float)), (float*)malloc(lnf_rstd_size * sizeof(float)), (float*)malloc(logits_size * sizeof(float)), (float*)malloc(probs_size * sizeof(float)), (float*)malloc(losses_size * sizeof(float))}; });
}
static ParameterTensors allocate_parameter_tensors(int32_t V, int32_t maxT, int32_t C, int32_t L, int32_t Vp) {
    return ({ int32_t wte_size = (V * C); int32_t wpe_size = (maxT * C); int32_t ln1w_size = (L * C); int32_t ln1b_size = (L * C); int32_t qkvw_size = (((L * 3) * C) * C); int32_t qkvb_size = ((L * 3) * C); int32_t attprojw_size = ((L * C) * C); int32_t attprojb_size = (L * C); int32_t ln2w_size = (L * C); int32_t ln2b_size = (L * C); int32_t fcw_size = (((L * 4) * C) * C); int32_t fcb_size = ((L * 4) * C); int32_t fcprojw_size = ((L * C) * (4 * C)); int32_t fcprojb_size = (L * C); int32_t lnfw_size = C; int32_t lnfb_size = C; float* wte = (float*)({ float* __arr = (float*)malloc(wte_size * sizeof(float)); for (size_t __i = 0; __i < wte_size; __i++) __arr[__i] = 0.01; __arr; }); float* wpe = (float*)({ float* __arr = (float*)malloc(wpe_size * sizeof(float)); for (size_t __i = 0; __i < wpe_size; __i++) __arr[__i] = 0.01; __arr; }); float* ln1w = (float*)({ float* __arr = (float*)malloc(ln1w_size * sizeof(float)); for (size_t __i = 0; __i < ln1w_size; __i++) __arr[__i] = 1; __arr; }); float* ln1b = (float*)({ float* __arr = (float*)malloc(ln1b_size * sizeof(float)); for (size_t __i = 0; __i < ln1b_size; __i++) __arr[__i] = 0; __arr; }); float* qkvw = (float*)({ float* __arr = (float*)malloc(qkvw_size * sizeof(float)); for (size_t __i = 0; __i < qkvw_size; __i++) __arr[__i] = 0.01; __arr; }); float* qkvb = (float*)({ float* __arr = (float*)malloc(qkvb_size * sizeof(float)); for (size_t __i = 0; __i < qkvb_size; __i++) __arr[__i] = 0; __arr; }); float* attprojw = (float*)({ float* __arr = (float*)malloc(attprojw_size * sizeof(float)); for (size_t __i = 0; __i < attprojw_size; __i++) __arr[__i] = 0.01; __arr; }); float* attprojb = (float*)({ float* __arr = (float*)malloc(attprojb_size * sizeof(float)); for (size_t __i = 0; __i < attprojb_size; __i++) __arr[__i] = 0; __arr; }); float* ln2w = (float*)({ float* __arr = (float*)malloc(ln2w_size * sizeof(float)); for (size_t __i = 0; __i < ln2w_size; __i++) __arr[__i] = 1; __arr; }); float* ln2b = (float*)({ float* __arr = (float*)malloc(ln2b_size * sizeof(float)); for (size_t __i = 0; __i < ln2b_size; __i++) __arr[__i] = 0; __arr; }); float* fcw = (float*)({ float* __arr = (float*)malloc(fcw_size * sizeof(float)); for (size_t __i = 0; __i < fcw_size; __i++) __arr[__i] = 0.01; __arr; }); float* fcb = (float*)({ float* __arr = (float*)malloc(fcb_size * sizeof(float)); for (size_t __i = 0; __i < fcb_size; __i++) __arr[__i] = 0; __arr; }); float* fcprojw = (float*)({ float* __arr = (float*)malloc(fcprojw_size * sizeof(float)); for (size_t __i = 0; __i < fcprojw_size; __i++) __arr[__i] = 0.01; __arr; }); float* fcprojb = (float*)({ float* __arr = (float*)malloc(fcprojb_size * sizeof(float)); for (size_t __i = 0; __i < fcprojb_size; __i++) __arr[__i] = 0; __arr; }); float* lnfw = (float*)({ float* __arr = (float*)malloc(lnfw_size * sizeof(float)); for (size_t __i = 0; __i < lnfw_size; __i++) __arr[__i] = 1; __arr; }); float* lnfb = (float*)({ float* __arr = (float*)malloc(lnfb_size * sizeof(float)); for (size_t __i = 0; __i < lnfb_size; __i++) __arr[__i] = 0; __arr; }); (ParameterTensors){wte, wpe, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw, fcprojb, lnfw, lnfb}; });
}
static int32_t test_gpt2_inference() {
    printf("Testing complete GPT-2 inference pipeline...\n");
    return ({ GPT2Config config = (GPT2Config){8, 16, 16, 1, 2, 64}; int32_t B = 1; int32_t T = 4; int32_t C = config.channels; int32_t L = config.num_layers; int32_t NH = config.num_heads; int32_t V = config.vocab_size; int32_t Vp = config.padded_vocab_size; int32_t maxT = config.max_seq_len; printf("Config: L=%d, NH=%d, C=%d, V=%d, Vp=%d\n", L, NH, C, V, Vp); ({ ParameterTensors params = g_user.allocate_parameter_tensors(V, maxT, C, L, Vp); ActivationTensors acts = g_user.allocate_activation_tensors(B, T, C, L, NH, Vp); int32_t* inputs = (int32_t*)(int32_t*)malloc((B * T) * sizeof(int32_t)); (inputs[0] = 1); (inputs[1] = 2); (inputs[2] = 3); (inputs[3] = 4); printf("Input tokens: [1, 2, 3, 4]\n"); printf("Running gpt2_forward...\n"); g_user.gpt2_forward(inputs, config, params, acts, B, T); ({ int32_t last_t = (T - 1); int32_t last_probs_offset = (last_t * Vp); float* probs_ptr = (float*)(acts.probs + last_probs_offset); printf("Output probabilities for last position (first 8 values):\n"); ({ int32_t i = 0; ({ while ((i < 8)) { printf("  probs[%d] = %f\n", i, probs_ptr[i]); i = (i + 1); } }); }); ({ int32_t next_token = g_user.argmax(probs_ptr, V); printf("Predicted next token (greedy): %d\n", next_token); }); }); free(inputs); printf("GPT-2 inference test completed!\n"); 0; }); });
}
static int32_t test_real_gpt2_inference() {
    printf("\n=== Testing GPT-2 with REAL pretrained weights ===\n\n");
    return ({ CheckpointData checkpoint = g_user.load_gpt2_checkpoint("gpt2_124M.bin"); GPT2Config config = checkpoint.config; ParameterTensors params = checkpoint.params; printf("\n=== Running inference ===\n\n"); ({ int32_t B = 1; int32_t T = 4; int32_t C = config.channels; int32_t L = config.num_layers; int32_t NH = config.num_heads; int32_t V = config.vocab_size; int32_t Vp = config.padded_vocab_size; ({ ActivationTensors acts = g_user.allocate_activation_tensors(B, T, C, L, NH, Vp); int32_t* inputs = (int32_t*)(int32_t*)malloc((B * T) * sizeof(int32_t)); (inputs[0] = 15496); (inputs[1] = 995); (inputs[2] = 318); (inputs[3] = 1); printf("Input tokens: [15496, 995, 318, 1]\n"); printf("Running full GPT-2 forward pass with 12 layers...\n"); g_user.gpt2_forward(inputs, config, params, acts, B, T); ({ int32_t last_t = (T - 1); int32_t last_probs_offset = (last_t * Vp); float* probs_ptr = (float*)(acts.probs + last_probs_offset); printf("\nOutput probabilities for last position (first 10 values):\n"); ({ int32_t i = 0; ({ while ((i < 10)) { printf("  token[%d] prob = %.6f\n", i, probs_ptr[i]); i = (i + 1); } }); }); ({ int32_t next_token = g_user.argmax(probs_ptr, V); printf("\nPredicted next token (greedy): %d\n", next_token); printf("Probability of predicted token: %.6f\n", probs_ptr[next_token]); }); }); free(inputs); printf("\nReal GPT-2 inference test completed successfully!\n"); 0; }); }); });
}
static int32_t test_autoregressive_generation() {
    printf("\n=== Testing Autoregressive Generation ===\n\n");
    return ({ CheckpointData checkpoint = g_user.load_gpt2_checkpoint("gpt2_124M.bin"); GPT2Config config = checkpoint.config; ParameterTensors params = checkpoint.params; printf("\n=== Generating tokens ===\n\n"); ({ int32_t C = config.channels; int32_t L = config.num_layers; int32_t NH = config.num_heads; int32_t V = config.vocab_size; int32_t Vp = config.padded_vocab_size; int32_t context_window = 8; int32_t num_tokens_to_generate = 100; int32_t* sequence = (int32_t*)(int32_t*)malloc((context_window + num_tokens_to_generate) * sizeof(int32_t)); (sequence[0] = 15496); (sequence[1] = 995); (sequence[2] = 318); printf("Initial prompt tokens: [15496, 995, 318]\n"); printf("Generating 100 new tokens with fixed context window...\n\n"); ({ int32_t B = 1; ActivationTensors acts = g_user.allocate_activation_tensors(B, context_window, C, L, NH, Vp); ({ int32_t gen_count = 0; int32_t total_generated = 3; ({ while ((gen_count < num_tokens_to_generate)) { ({ int32_t window_start = ((total_generated < context_window) ? 0 : (total_generated - context_window)); int32_t window_len = ((total_generated < context_window) ? total_generated : context_window); int32_t* window_input = (int32_t*)(sequence + window_start); g_user.gpt2_forward(window_input, config, params, acts, B, window_len); ({ int32_t last_t = (window_len - 1); int32_t last_probs_offset = (last_t * Vp); float* probs_ptr = (float*)(acts.probs + last_probs_offset); int32_t next_token = g_user.argmax(probs_ptr, V); ({ if ((gen_count == 0)) { ({ int32_t dummy = printf("Starting generation...\n"); 0; }); } else { } }); ({ if ((gen_count == 24)) { ({ int32_t dummy = printf("25 tokens generated...\n"); 0; }); } else { } }); ({ if ((gen_count == 49)) { ({ int32_t dummy = printf("50 tokens generated...\n"); 0; }); } else { } }); ({ if ((gen_count == 74)) { ({ int32_t dummy = printf("75 tokens generated...\n"); 0; }); } else { } }); (sequence[total_generated] = next_token); total_generated = (total_generated + 1); }); }); gen_count = (gen_count + 1); } }); }); }); printf("\nFirst 20 generated tokens:\n"); printf("["); ({ int32_t i = 0; int32_t print_limit = (((3 + num_tokens_to_generate) < 20) ? (3 + num_tokens_to_generate) : 20); ({ while ((i < print_limit)) { ((i < (print_limit - 1)) ? printf("%d, ", sequence[i]) : printf("%d", sequence[i])); i = (i + 1); } }); }); printf(" ...]\n"); free(sequence); printf("\nAutoregressive generation test completed!\n"); 0; }); });
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
    g_user.test_softmax_forward();
    printf("\n");
    g_user.test_gpt2_inference();
    printf("\n");
    g_user.test_real_gpt2_inference();
    printf("\n");
    return g_user.test_autoregressive_generation();
}
int main() {
    init_namespace_user(&g_user);
    g_user.main_fn();
    return 0;
}
