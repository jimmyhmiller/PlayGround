"""op_t3: T3 conditioning encoder + Llama-30L autoregressive speech-token generator.

One big op covering:
- T3CondEnc: (speaker_emb_256, cond_prompt_tokens, emotion) → cond_emb (1, 34, 1024)
- BPE-already-tokenized text → text_emb + bos_emb → CFG prefix → t3_generate_cfg_sample

Wrapped together because cond_emb is just intermediate compute used by the
generator. The orchestrator does NOT need to touch cond_emb directly.

Public API:
- init_op(t3_base, cond_enc_base, device_ctx) -> handle
- generate(handle, speaker_emb_256_buf, cond_prompt_tokens_buf,
           emotion, text_ids_list, cfg_weight, temperature, top_p,
           rep_penalty, max_new, rng_seed) -> list[int] of speech tokens
- destroy_op(handle) -> None
"""
from std.os import abort
from std.memory import OpaquePointer
from std.python import Python, PythonObject
from std.io.file import open
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from t3 import T3
from cond_enc import T3CondEnc, t3_cond_enc_forward
from t3_generate import t3_generate_cfg_sample, t3_generate_cfg_sample_aligned
from text_embed import build_text_emb, build_bos_emb, build_rope_tables
from weights_t3 import load_t3, load_t3_cond_enc


comptime EOS = 6562
comptime T_BOS = 2   # upstream prefix has TWO start_speech_tokens (initial_speech + bos), both at pos 0
comptime T_COND = 34
comptime D = 1024
comptime MAX_CTX = 4096


@fieldwise_init
struct OpState(Movable):
    var ctx: DeviceContext
    var t3: T3
    var cond_enc: T3CondEnc
    var cos_t3: DeviceBuffer[DType.float32]
    var sin_t3: DeviceBuffer[DType.float32]


def _ctx_from_python(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("op_t3 requires a GPU device context")
    var opaque = OpaquePointer[MutExternalOrigin](unsafe_from_address=addr)
    return DeviceContextPtr(opaque).get_device_context()


def init_op(
    t3_base: PythonObject,
    cond_enc_base: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var t3p = String(py=t3_base)
    var cep = String(py=cond_enc_base)
    var ctx = _ctx_from_python(device_context_ptr)

    var t3 = load_t3(ctx, t3p)
    var cond_enc = load_t3_cond_enc(ctx, cep)

    # RoPE tables for T3 generation (head_dim=64).
    var cos_t3 = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * 64)
    var sin_t3 = ctx.enqueue_create_buffer[DType.float32](MAX_CTX * 64)
    build_rope_tables(ctx, MAX_CTX, 64, cos_t3, sin_t3)
    ctx.synchronize()

    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(ctx^, t3^, cond_enc^, cos_t3^, sin_t3^))
    return PythonObject(Int(ptr))


def generate(
    handle: PythonObject,
    speaker_emb_buf: PythonObject,        # (1, 256) GPU f32
    cond_prompt_tokens_buf: PythonObject, # (1, 150) GPU i32
    text_ids_list: PythonObject,          # Python list[int]
    config: PythonObject,                 # dict: emotion, cfg_weight, temperature, top_p, rep_penalty, max_new, rng_seed
) raises -> PythonObject:
    """End-to-end T3 generation from a voice profile + tokenized text.

    config is a Python dict with keys:
      emotion (float), cfg_weight (float), temperature (float),
      top_p (float), rep_penalty (float), max_new (int), rng_seed (int).
    """
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var ctx_ref = state_ptr[].ctx

    # Pull text ids.
    var T_TEXT = Int(py=Python.evaluate("len")(text_ids_list))
    var ids = List[Int64]()
    for i in range(T_TEXT):
        ids.append(Int64(py=text_ids_list[i]))

    var emotion_val = Float32(py=config["emotion"])
    var cfg_weight = Float32(py=config["cfg_weight"])
    var temperature = Float32(py=config["temperature"])
    var top_p = Float32(py=config["top_p"])
    var rep_penalty = Float32(py=config["rep_penalty"])
    var max_new = Int(py=config["max_new"])
    var rng_seed = UInt64(Int(py=config["rng_seed"]))
    var min_p: Float32 = 0.0
    try:
        min_p = Float32(py=config["min_p"])
    except:
        min_p = 0.0

    # Wrap inputs.
    var sp_addr = Int(py=speaker_emb_buf._data_ptr())
    var sp_n = Int(py=speaker_emb_buf.num_elements)
    var sp_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=sp_addr)
    var speaker_emb = DeviceBuffer[DType.float32](ctx_ref, sp_ptr, sp_n, owning=False)

    # cond_prompt_tokens come in as int32 from s3tokenizer; cond_enc wants int64.
    # Materialize an int64 buffer (cheap — 150 elements).
    var ct_addr_i32 = Int(py=cond_prompt_tokens_buf._data_ptr())
    var ct_n = Int(py=cond_prompt_tokens_buf.num_elements)
    var ct_ptr_i32 = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=ct_addr_i32)
    var ct_buf_i32 = DeviceBuffer[DType.int32](ctx_ref, ct_ptr_i32, ct_n, owning=False)
    var cond_prompt_tok = ctx_ref.enqueue_create_buffer[DType.int64](ct_n)
    with ct_buf_i32.map_to_host() as src:
        with cond_prompt_tok.map_to_host() as dst:
            for i in range(ct_n):
                dst[i] = Int64(src[i])

    # 1. Cond encoder.
    var emotion = ctx_ref.enqueue_create_buffer[DType.float32](1)
    with emotion.map_to_host() as h:
        h[0] = emotion_val

    var SQ = 32
    var T_token_6s = 150
    var mask_q = ctx_ref.enqueue_create_buffer[DType.float32](SQ * SQ)
    mask_q.enqueue_fill(0.0)
    var mask_qq = ctx_ref.enqueue_create_buffer[DType.float32](SQ * T_token_6s)
    mask_qq.enqueue_fill(0.0)
    var cond_emb = ctx_ref.enqueue_create_buffer[DType.float32](1 * T_COND * D)
    t3_cond_enc_forward(
        ctx_ref, state_ptr[].cond_enc, speaker_emb, cond_prompt_tok, emotion, cond_emb,
        mask_q, mask_qq, 1,
    )
    ctx_ref.synchronize()

    # 2. text_emb + bos_emb.
    var text_emb = ctx_ref.enqueue_create_buffer[DType.float32](1 * T_TEXT * D)
    build_text_emb(ctx_ref, state_ptr[].t3, ids, text_emb)
    var bos_emb = ctx_ref.enqueue_create_buffer[DType.float32](1 * 1 * D)
    build_bos_emb(ctx_ref, state_ptr[].t3, bos_emb)


    # 3. Build CFG prefix (2*B = 2 batches: cond + uncond).
    var B2 = 2
    var T_PREFIX = T_COND + T_TEXT + T_BOS
    var prefix = ctx_ref.enqueue_create_buffer[DType.float32](B2 * T_PREFIX * D)
    var ce = cond_emb.unsafe_ptr()
    var te = text_emb.unsafe_ptr()
    var be = bos_emb.unsafe_ptr()
    var px = prefix.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(ce, te, be, px, B2, T_TEXT, T_PREFIX)
    def cat_func[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PREFIX * D)
        var rem = i - bi * T_PREFIX * D
        var ti = rem // D
        var di = rem - ti * D
        if ti < T_COND:
            px[i] = ce[ti * D + di]
        elif ti < T_COND + T_TEXT:
            var src_t = ti - T_COND
            if bi < 1:
                px[i] = te[src_t * D + di]
            else:
                px[i] = 0.0
        else:
            # bos_emb is (1, D). Repeat it T_BOS times (= 2 for upstream parity).
            px[i] = be[di]
    elementwise[cat_func, simd_width=1, target="gpu"](
        IndexList[1](B2 * T_PREFIX * D), DeviceContextPtr(ctx_ref),
    )

    # 4. T3 causal mask.
    var t3_mask = ctx_ref.enqueue_create_buffer[DType.float32](T_PREFIX * T_PREFIX)
    with t3_mask.map_to_host() as h:
        for r in range(T_PREFIX):
            for c in range(T_PREFIX):
                if c > r:
                    h[r * T_PREFIX + c] = -1.0e30
                else:
                    h[r * T_PREFIX + c] = 0.0

    # 5. Generate. (Upstream English chatterbox does NOT use the alignment
    # analyzer; only multilingual does. So we use the plain sampler — quality
    # difference vs upstream must be in the sampler/forward, not analyzer.)
    var speech_pos = state_ptr[].t3.speech_pos_emb.table
    var generated = t3_generate_cfg_sample(
        ctx_ref, state_ptr[].t3, prefix, state_ptr[].cos_t3, state_ptr[].sin_t3, t3_mask, speech_pos,
        1, T_PREFIX, MAX_CTX, max_new,
        speech_pos_offset=1, eos_token=EOS,
        cfg_weight=cfg_weight,
        temperature=temperature, top_p=top_p, rep_penalty=rep_penalty,
        rng_seed=rng_seed, min_p=min_p,
    )
    ctx_ref.synchronize()

    # Return as Python list.
    var out = Python.list()
    for i in range(len(generated)):
        out.append(Int(generated[i]))
    return out


def destroy_op(handle: PythonObject) raises -> PythonObject:
    var addr = Int(py=handle)
    if addr == 0:
        return PythonObject(None)
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    state_ptr.destroy_pointee()
    state_ptr.free()
    return PythonObject(None)


@export
def PyInit_op_t3() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_t3")
        b.def_function[init_op]("init_op")
        b.def_function[generate]("generate")
        b.def_function[destroy_op]("destroy_op")
        return b.finalize()
    except e:
        abort(String("failed to create op_t3 module: ", e))
