/*
 * GPT-2 forward pass benchmark — llm.c style
 *
 * Single-threaded CPU reference implementation for comparison with WASM+SIMD.
 * Loads weights from gpt2_weights/ (same format as tensor-lang benchmark).
 *
 * Compile (naive):       cc -O3 -march=native -o gpt2_cpu gpt2_cpu.c -lm
 * Compile (Accelerate):  cc -O3 -march=native -DUSE_ACCELERATE -o gpt2_cpu_blas gpt2_cpu.c -lm -framework Accelerate
 * Run:                   ./gpt2_cpu [seq_len] [warmup] [iters]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

/* ---------- config ---------- */

#define VOCAB_SIZE 50257
#define N_EMBD     768
#define N_HEAD     12
#define N_LAYER    12
#define HEAD_SIZE  (N_EMBD / N_HEAD)
#define MLP_HIDDEN (4 * N_EMBD)
#define MAX_SEQ    1024

/* ---------- timing ---------- */

static double wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* ---------- kernels (llm.c style) ---------- */

static void encoder_forward(float *out, const float *tokens, const float *wte,
                            const float *wpe, int B, int T) {
    /* out[b][t] = wte[token] + wpe[t] */
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int tok = (int)tokens[b * T + t];
            const float *emb = wte + tok * N_EMBD;
            const float *pos = wpe + t * N_EMBD;
            float *o = out + (b * T + t) * N_EMBD;
            for (int c = 0; c < N_EMBD; c++) {
                o[c] = emb[c] + pos[c];
            }
        }
    }
}

static void layernorm_forward(float *out, const float *inp, const float *gamma,
                              const float *beta, int B, int T) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *x = inp + (b * T + t) * N_EMBD;
            float *o = out + (b * T + t) * N_EMBD;
            float mean = 0.0f;
            for (int c = 0; c < N_EMBD; c++) mean += x[c];
            mean /= N_EMBD;
            float var = 0.0f;
            for (int c = 0; c < N_EMBD; c++) {
                float d = x[c] - mean;
                var += d * d;
            }
            var /= N_EMBD;
            float rstd = 1.0f / sqrtf(var + 1e-5f);
            for (int c = 0; c < N_EMBD; c++) {
                o[c] = (x[c] - mean) * rstd * gamma[c] + beta[c];
            }
        }
    }
}

/*
 * matmul: out[B,T,OC] = inp[B,T,C] @ weight[C,OC] + bias[OC]
 * Weight is in [C, OC] layout (column-major / Conv1D style).
 */
static void matmul_forward(float *out, const float *inp, const float *weight,
                           const float *bias, int B, int T, int C, int OC) {
#ifdef USE_ACCELERATE
    int M = B * T;
    /* inp[M,C] @ weight[C,OC] -> out[M,OC]
     * cblas_sgemm with row-major: C = alpha*A*B + beta*C
     * A = inp (M x C), B = weight (C x OC), C = out (M x OC) */
    if (bias) {
        /* fill output with bias (broadcast across M rows) */
        for (int m = 0; m < M; m++)
            memcpy(out + m * OC, bias, OC * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, OC, C, 1.0f, inp, C, weight, OC, 1.0f, out, OC);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, OC, C, 1.0f, inp, C, weight, OC, 0.0f, out, OC);
    }
#else
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = out + (b * T + t) * OC;
            const float *inp_bt = inp + (b * T + t) * C;
            /* init with bias */
            if (bias) {
                memcpy(out_bt, bias, OC * sizeof(float));
            } else {
                memset(out_bt, 0, OC * sizeof(float));
            }
            /* accumulate: out[oc] += inp[c] * weight[c*OC + oc] */
            for (int c = 0; c < C; c++) {
                float xval = inp_bt[c];
                const float *wrow = weight + c * OC;
                for (int oc = 0; oc < OC; oc++) {
                    out_bt[oc] += xval * wrow[oc];
                }
            }
        }
    }
#endif
}

static void attention_forward(float *out, float *preatt, float *att,
                              const float *qkv, int B, int T) {
    float scale = 1.0f / sqrtf((float)HEAD_SIZE);
    int C3 = 3 * N_EMBD;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < N_HEAD; h++) {
            for (int t = 0; t < T; t++) {
                /* Q for this position */
                const float *q = qkv + (b * T + t) * C3 + h * HEAD_SIZE;

                /* Compute attention scores for this query against all keys */
                float maxval = -1e9f;
                for (int t2 = 0; t2 <= t; t2++) { /* causal: only attend to past */
                    const float *k = qkv + (b * T + t2) * C3 + N_EMBD + h * HEAD_SIZE;
                    float score = 0.0f;
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        score += q[d] * k[d];
                    }
                    score *= scale;
                    preatt[((b * N_HEAD + h) * T + t) * T + t2] = score;
                    if (score > maxval) maxval = score;
                }

                /* Softmax */
                float *att_row = att + ((b * N_HEAD + h) * T + t) * T;
                float sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float v = expf(preatt[((b * N_HEAD + h) * T + t) * T + t2] - maxval);
                    att_row[t2] = v;
                    sum += v;
                }
                float inv_sum = 1.0f / sum;
                for (int t2 = 0; t2 <= t; t2++) {
                    att_row[t2] *= inv_sum;
                }
                /* zero out future positions */
                for (int t2 = t + 1; t2 < T; t2++) {
                    att_row[t2] = 0.0f;
                }

                /* Weighted sum of values */
                float *o = out + (b * T + t) * N_EMBD + h * HEAD_SIZE;
                memset(o, 0, HEAD_SIZE * sizeof(float));
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *v = qkv + (b * T + t2) * C3 + 2 * N_EMBD + h * HEAD_SIZE;
                    float a = att_row[t2];
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        o[d] += a * v[d];
                    }
                }
            }
        }
    }
}

static void gelu_forward(float *out, const float *inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        /* clamp for numerical stability */
        if (inner > 10.0f) inner = 10.0f;
        if (inner < -10.0f) inner = -10.0f;
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

static void residual_forward(float *out, const float *inp1, const float *inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

/* ---------- weight pointers ---------- */

typedef struct {
    /* global */
    const float *wte;      /* [VOCAB_SIZE, N_EMBD] */
    const float *wpe;      /* [MAX_SEQ, N_EMBD]    */
    /* per layer (pointer arrays) */
    const float *ln1_g[N_LAYER];
    const float *ln1_b[N_LAYER];
    const float *qkv_w[N_LAYER];   /* [N_EMBD, 3*N_EMBD] */
    const float *qkv_b[N_LAYER];   /* [3*N_EMBD] */
    const float *proj_w[N_LAYER];   /* [N_EMBD, N_EMBD] */
    const float *proj_b[N_LAYER];
    const float *ln2_g[N_LAYER];
    const float *ln2_b[N_LAYER];
    const float *fc_w[N_LAYER];     /* [N_EMBD, MLP_HIDDEN] */
    const float *fc_b[N_LAYER];
    const float *fcproj_w[N_LAYER]; /* [MLP_HIDDEN, N_EMBD] */
    const float *fcproj_b[N_LAYER];
    /* final */
    const float *ln_f_g;
    const float *ln_f_b;
} Weights;

/* ---------- scratch buffers ---------- */

typedef struct {
    float *emb;       /* [B, T, C] */
    float *ln_out;    /* [B, T, C] */
    float *qkv;       /* [B, T, 3C] */
    float *preatt;    /* [B, NH, T, T] */
    float *att;       /* [B, NH, T, T] */
    float *attn_out;  /* [B, T, C] */
    float *proj_out;  /* [B, T, C] */
    float *residual;  /* [B, T, C] */
    float *fc_out;    /* [B, T, 4C] */
    float *gelu_out;  /* [B, T, 4C] */
    float *mlp_out;   /* [B, T, C] */
    float *logits;    /* [B, T, VOCAB_SIZE] */
} Scratch;

static Scratch alloc_scratch(int B, int T) {
    Scratch s;
    s.emb       = calloc(B * T * N_EMBD, sizeof(float));
    s.ln_out    = calloc(B * T * N_EMBD, sizeof(float));
    s.qkv       = calloc(B * T * 3 * N_EMBD, sizeof(float));
    s.preatt    = calloc(B * N_HEAD * T * T, sizeof(float));
    s.att       = calloc(B * N_HEAD * T * T, sizeof(float));
    s.attn_out  = calloc(B * T * N_EMBD, sizeof(float));
    s.proj_out  = calloc(B * T * N_EMBD, sizeof(float));
    s.residual  = calloc(B * T * N_EMBD, sizeof(float));
    s.fc_out    = calloc(B * T * MLP_HIDDEN, sizeof(float));
    s.gelu_out  = calloc(B * T * MLP_HIDDEN, sizeof(float));
    s.mlp_out   = calloc(B * T * N_EMBD, sizeof(float));
    s.logits    = calloc(B * T * VOCAB_SIZE, sizeof(float));
    return s;
}

static void free_scratch(Scratch *s) {
    free(s->emb);     free(s->ln_out);   free(s->qkv);
    free(s->preatt);  free(s->att);      free(s->attn_out);
    free(s->proj_out);free(s->residual); free(s->fc_out);
    free(s->gelu_out);free(s->mlp_out);  free(s->logits);
}

/* ---------- forward pass ---------- */

static void gpt2_forward(float *logits, const float *tokens, const Weights *w,
                          Scratch *s, int B, int T) {
    int BTC = B * T * N_EMBD;
    int BT4C = B * T * MLP_HIDDEN;

    /* Embedding */
    encoder_forward(s->emb, tokens, w->wte, w->wpe, B, T);

    /* x = embedding output (copy into residual for in-place updates) */
    float *x = s->emb;  /* residual stream */

    for (int l = 0; l < N_LAYER; l++) {
        /* Pre-attention layernorm */
        layernorm_forward(s->ln_out, x, w->ln1_g[l], w->ln1_b[l], B, T);

        /* QKV projection */
        matmul_forward(s->qkv, s->ln_out, w->qkv_w[l], w->qkv_b[l],
                       B, T, N_EMBD, 3 * N_EMBD);

        /* Multi-head attention */
        attention_forward(s->attn_out, s->preatt, s->att, s->qkv, B, T);

        /* Output projection */
        matmul_forward(s->proj_out, s->attn_out, w->proj_w[l], w->proj_b[l],
                       B, T, N_EMBD, N_EMBD);

        /* Residual add */
        residual_forward(s->residual, x, s->proj_out, BTC);

        /* Pre-MLP layernorm */
        layernorm_forward(s->ln_out, s->residual, w->ln2_g[l], w->ln2_b[l], B, T);

        /* MLP: fc up */
        matmul_forward(s->fc_out, s->ln_out, w->fc_w[l], w->fc_b[l],
                       B, T, N_EMBD, MLP_HIDDEN);

        /* GELU */
        gelu_forward(s->gelu_out, s->fc_out, BT4C);

        /* MLP: fc down */
        matmul_forward(s->mlp_out, s->gelu_out, w->fcproj_w[l], w->fcproj_b[l],
                       B, T, MLP_HIDDEN, N_EMBD);

        /* Residual add */
        residual_forward(s->emb, s->residual, s->mlp_out, BTC);
        x = s->emb;
    }

    /* Final layernorm */
    layernorm_forward(s->ln_out, x, w->ln_f_g, w->ln_f_b, B, T);

    /* LM head: logits = x_norm @ wte^T
     * wte is [VOCAB_SIZE, N_EMBD], we want out = x_norm[B*T, C] @ wte^T[C, V]
     * i.e. out[m, v] = sum_c x[m,c] * wte[v,c] */
#ifdef USE_ACCELERATE
    {
        int M = B * T;
        /* A=x_norm (M x C), B=wte^T -> use CblasTrans on wte (V x C)
         * out (M x V) = A * B^T */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, VOCAB_SIZE, N_EMBD,
                    1.0f, s->ln_out, N_EMBD, w->wte, N_EMBD,
                    0.0f, logits, VOCAB_SIZE);
    }
#else
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *x_bt = s->ln_out + (b * T + t) * N_EMBD;
            float *out_bt = logits + (b * T + t) * VOCAB_SIZE;
            for (int v = 0; v < VOCAB_SIZE; v++) {
                const float *wrow = w->wte + v * N_EMBD;
                float val = 0.0f;
                for (int c = 0; c < N_EMBD; c++) {
                    val += x_bt[c] * wrow[c];
                }
                out_bt[v] = val;
            }
        }
    }
#endif
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    int seq_len = (argc > 1) ? atoi(argv[1]) : 8;
    int warmup  = (argc > 2) ? atoi(argv[2]) : 1;
    int iters   = (argc > 3) ? atoi(argv[3]) : 3;

    if (seq_len < 1 || seq_len > MAX_SEQ) {
        fprintf(stderr, "seq_len must be in [1, %d]\n", MAX_SEQ);
        return 1;
    }

    fprintf(stderr, "GPT-2 CPU forward pass (llm.c style)\n");
    fprintf(stderr, "  vocab=%d, d=%d, heads=%d, layers=%d\n",
            VOCAB_SIZE, N_EMBD, N_HEAD, N_LAYER);
    fprintf(stderr, "  T=%d, warmup=%d, iters=%d\n", seq_len, warmup, iters);

    /* --- Load weights --- */
    fprintf(stderr, "Loading weights...\n");
    FILE *f = fopen("gpt2_weights/weights.bin", "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open gpt2_weights/weights.bin\n");
        fprintf(stderr, "Run: python3 export_gpt2.py\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *blob = malloc(fsize);
    if (!blob) { fprintf(stderr, "malloc failed\n"); return 1; }
    fread(blob, 1, fsize, f);
    fclose(f);
    fprintf(stderr, "  Loaded %.1f MB\n", fsize / (1024.0 * 1024.0));

    /* Parse weights by computing offsets (matches export_gpt2.py order) */
    Weights w;
    float *p = blob;

    w.wte = p;  p += VOCAB_SIZE * N_EMBD;
    w.wpe = p;  p += MAX_SEQ * N_EMBD;

    for (int l = 0; l < N_LAYER; l++) {
        w.ln1_g[l] = p;     p += N_EMBD;
        w.ln1_b[l] = p;     p += N_EMBD;
        w.qkv_w[l] = p;     p += N_EMBD * 3 * N_EMBD;
        w.qkv_b[l] = p;     p += 3 * N_EMBD;
        w.proj_w[l] = p;    p += N_EMBD * N_EMBD;
        w.proj_b[l] = p;    p += N_EMBD;
        w.ln2_g[l] = p;     p += N_EMBD;
        w.ln2_b[l] = p;     p += N_EMBD;
        w.fc_w[l] = p;      p += N_EMBD * MLP_HIDDEN;
        w.fc_b[l] = p;      p += MLP_HIDDEN;
        w.fcproj_w[l] = p;  p += MLP_HIDDEN * N_EMBD;
        w.fcproj_b[l] = p;  p += N_EMBD;
    }

    w.ln_f_g = p;  p += N_EMBD;
    w.ln_f_b = p;  p += N_EMBD;

    long expected = (p - blob) * sizeof(float);
    fprintf(stderr, "  Parsed %ld / %ld bytes of weights\n", expected, fsize);

    /* --- Prepare inputs --- */
    int B = 1;
    float *tokens = calloc(B * seq_len, sizeof(float));

    /* Check for --tokens flag to load from reference_input.bin */
    int use_ref_tokens = 0;
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--tokens") == 0 && a + 1 < argc) {
            FILE *tf = fopen(argv[a + 1], "rb");
            if (tf) {
                fread(tokens, sizeof(float), seq_len, tf);
                fclose(tf);
                use_ref_tokens = 1;
                fprintf(stderr, "Loaded tokens from %s:", argv[a + 1]);
                for (int i = 0; i < seq_len; i++)
                    fprintf(stderr, " %.0f", tokens[i]);
                fprintf(stderr, "\n");
            }
        }
    }
    if (!use_ref_tokens) {
        for (int i = 0; i < seq_len; i++) {
            tokens[i] = (float)(464 + i);  /* same dummy tokens as WASM benchmark */
        }
    }

    /* --- Allocate scratch --- */
    Scratch scratch = alloc_scratch(B, seq_len);

    /* --- Warmup --- */
    fprintf(stderr, "Warming up...\n");
    for (int i = 0; i < warmup; i++) {
        gpt2_forward(scratch.logits, tokens, &w, &scratch, B, seq_len);
    }

    /* --- Benchmark --- */
    fprintf(stderr, "Benchmarking...\n");
    double *times = malloc(iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = wall_ms();
        gpt2_forward(scratch.logits, tokens, &w, &scratch, B, seq_len);
        times[i] = wall_ms() - t0;
    }

    double avg = 0, mn = 1e18, mx = 0;
    for (int i = 0; i < iters; i++) {
        avg += times[i];
        if (times[i] < mn) mn = times[i];
        if (times[i] > mx) mx = times[i];
    }
    avg /= iters;

    /* Print result (stdout, machine-readable) */
#ifdef USE_ACCELERATE
    printf("C Accelerate: avg=%8.1fms  min=%8.1fms  max=%8.1fms\n", avg, mn, mx);
#else
    printf("C native:     avg=%8.1fms  min=%8.1fms  max=%8.1fms\n", avg, mn, mx);
#endif

    /* Print top-5 for last token */
    float *last_logits = scratch.logits + (seq_len - 1) * VOCAB_SIZE;
    int top5[5] = {0};
    for (int k = 0; k < 5; k++) {
        float best = -1e30f;
        int best_idx = 0;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            int skip = 0;
            for (int j = 0; j < k; j++) if (top5[j] == v) { skip = 1; break; }
            if (!skip && last_logits[v] > best) {
                best = last_logits[v];
                best_idx = v;
            }
        }
        top5[k] = best_idx;
        fprintf(stderr, "  top-%d: idx=%d logit=%.4f\n", k + 1, best_idx, last_logits[best_idx]);
    }

    /* Dump logits to binary file if requested */
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--dump") == 0) {
            const char *path = (a + 1 < argc) ? argv[a + 1] : "bench/c_logits.bin";
            FILE *out = fopen(path, "wb");
            if (out) {
                fwrite(scratch.logits, sizeof(float), B * seq_len * VOCAB_SIZE, out);
                fclose(out);
                fprintf(stderr, "Dumped logits to %s (%d floats)\n",
                        path, B * seq_len * VOCAB_SIZE);
            }
            break;
        }
    }

    /* Cleanup */
    free(tokens);
    free(times);
    free_scratch(&scratch);
    free(blob);

    return 0;
}
