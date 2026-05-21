"""op_flow: UpsampleConformerEncoder + CFM Euler solver.

Combines the flow encoder and the CFM noise-to-mel solver into one op.
Together: speech tokens + voice profile → (B, 80, T_total_mel) clean mel.

Public API:
- init_op(flow_enc_base, cfm_base, device_ctx) -> handle
- forward(handle, tok_buf, spks_buf, prompt_feat_buf, out_mel_buf, B, T_tokens, T_prompt_mel, T_total_mel, N_steps, cfg_rate, noise_seed) -> None
- destroy_op(handle) -> None
"""
from std.os import abort
from std.memory import OpaquePointer
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.gpu.host import DeviceContext, DeviceBuffer
from std.runtime.asyncrt import DeviceContextPtr
from std.algorithm.functional import elementwise, IndexList

from upsample_encoder import UpsampleConformerEncoderReal, upsample_conformer_forward
from cfm_estimator_new import CFMEstimatorReal, cfm_solve_euler, gaussian_noise_fill
from weights import load_upsample_conformer_encoder, load_cfm_estimator_real


comptime MEL = 80


@fieldwise_init
struct OpState(Movable):
    var ctx: DeviceContext
    var enc: UpsampleConformerEncoderReal
    var cfm: CFMEstimatorReal


def _ctx_from_python(device_context_ptr: PythonObject) raises -> DeviceContext:
    var addr = Int(py=device_context_ptr)
    if addr == 0:
        raise Error("op_flow requires a GPU device context")
    var opaque = OpaquePointer[MutExternalOrigin](unsafe_from_address=addr)
    return DeviceContextPtr(opaque).get_device_context()


def init_op(
    flow_enc_base: PythonObject,
    cfm_base: PythonObject,
    device_context_ptr: PythonObject,
) raises -> PythonObject:
    var enc_path = String(py=flow_enc_base)
    var cfm_path = String(py=cfm_base)
    var ctx = _ctx_from_python(device_context_ptr)

    var enc = load_upsample_conformer_encoder(ctx, enc_path)
    var cfm = load_cfm_estimator_real(ctx, cfm_path)
    ctx.synchronize()

    var ptr = alloc[OpState](1)
    ptr.init_pointee_move(OpState(ctx^, enc^, cfm^))
    return PythonObject(Int(ptr))


def forward(
    handle: PythonObject,
    tok_buf_i64: PythonObject,        # (B, T_token) int64 GPU - includes prompt + generated
    spks_buf: PythonObject,           # (B, 80) GPU f32
    prompt_feat_buf: PythonObject,    # (B, T_prompt_mel, 80) GPU f32
    out_mel_buf: PythonObject,        # (B, 80, T_out_mel) GPU f32 — trimmed output
    config: PythonObject,             # dict: B, T_token, T_prompt_mel, T_total_mel, T_out_mel, n_steps, cfg_rate, noise_seed
) raises -> PythonObject:
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](
        unsafe_from_address=Int(py=handle)
    )
    var ctx_ref = state_ptr[].ctx

    var B = Int(py=config["B"])
    var T_TOKEN = Int(py=config["T_token"])
    var T_PROMPT_MEL = Int(py=config["T_prompt_mel"])
    var T_TOTAL_MEL = Int(py=config["T_total_mel"])
    var T_OUT_MEL = Int(py=config["T_out_mel"])
    var N_STEPS = Int(py=config["n_steps"])
    var cfg_rate = Float32(py=config["cfg_rate"])
    var noise_seed = UInt64(Int(py=config["noise_seed"]))

    # Wrap inputs.
    var tok_addr = Int(py=tok_buf_i64._data_ptr())
    var tok_n = Int(py=tok_buf_i64.num_elements)
    var tok_ptr = UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=tok_addr)
    var tok_buf = DeviceBuffer[DType.int64](ctx_ref, tok_ptr, tok_n, owning=False)

    var sp_addr = Int(py=spks_buf._data_ptr())
    var sp_n = Int(py=spks_buf.num_elements)
    var sp_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=sp_addr)
    var spks = DeviceBuffer[DType.float32](ctx_ref, sp_ptr, sp_n, owning=False)

    var pf_addr = Int(py=prompt_feat_buf._data_ptr())
    var pf_n = Int(py=prompt_feat_buf.num_elements)
    var pf_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pf_addr)
    var prompt_feat_tm = DeviceBuffer[DType.float32](ctx_ref, pf_ptr, pf_n, owning=False)

    var out_addr = Int(py=out_mel_buf._data_ptr())
    var out_n = Int(py=out_mel_buf.num_elements)
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=out_addr)
    var mel_out = DeviceBuffer[DType.float32](ctx_ref, out_ptr, out_n, owning=False)

    # 1. Flow encoder: (B, T_token) tokens → (B, 80, T_total_mel) mu.
    var mu = ctx_ref.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    upsample_conformer_forward(ctx_ref, state_ptr[].enc, tok_buf, mu, B, T_TOKEN)
    ctx_ref.synchronize()

    # 2. cond = zeros + prompt_feat at the beginning, transposed.
    var cond = ctx_ref.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    cond.enqueue_fill(0.0)
    var pf_pp = prompt_feat_tm.unsafe_ptr()
    var cond_pp = cond.unsafe_ptr()
    var T_PROMPT_MEL_CAP = T_PROMPT_MEL

    @always_inline
    @parameter
    @__copy_capture(pf_pp, cond_pp, T_TOTAL_MEL, T_PROMPT_MEL_CAP)
    def cond_fill[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (T_PROMPT_MEL_CAP * MEL)
        var rem = i - bi * T_PROMPT_MEL_CAP * MEL
        var ti = rem // MEL
        var ci = rem - ti * MEL
        cond_pp[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + ti] = pf_pp[ti * MEL + ci]
    elementwise[cond_fill, simd_width=1, target="gpu"](
        IndexList[1](B * T_PROMPT_MEL_CAP * MEL), DeviceContextPtr(ctx_ref),
    )

    # 3. Sample noise + run Euler.
    var x = ctx_ref.enqueue_create_buffer[DType.float32](B * MEL * T_TOTAL_MEL)
    gaussian_noise_fill(ctx_ref, x, B * MEL * T_TOTAL_MEL, noise_seed, Float32(1.0))
    var cfm_mask = ctx_ref.enqueue_create_buffer[DType.float32](B * T_TOTAL_MEL)
    cfm_mask.enqueue_fill(1.0)
    cfm_solve_euler(ctx_ref, state_ptr[].cfm, x, mu, spks, cond, cfm_mask,
                     B, T_TOTAL_MEL, N_STEPS, cfg_rate)
    ctx_ref.synchronize()

    # 4. Trim prompt prefix: x[:, :, T_PROMPT_MEL:] → out_mel.
    var x_pp = x.unsafe_ptr()
    var mo_pp = mel_out.unsafe_ptr()

    @always_inline
    @parameter
    @__copy_capture(x_pp, mo_pp, T_TOTAL_MEL, T_OUT_MEL, T_PROMPT_MEL_CAP)
    def trim[width: Int, rank: Int, alignment: Int = 1](idx: IndexList[rank]):
        var i = idx[0]
        var bi = i // (MEL * T_OUT_MEL)
        var rem = i - bi * MEL * T_OUT_MEL
        var ci = rem // T_OUT_MEL
        var ti = rem - ci * T_OUT_MEL
        mo_pp[i] = x_pp[bi * MEL * T_TOTAL_MEL + ci * T_TOTAL_MEL + (T_PROMPT_MEL_CAP + ti)]
    elementwise[trim, simd_width=1, target="gpu"](
        IndexList[1](B * MEL * T_OUT_MEL), DeviceContextPtr(ctx_ref),
    )
    ctx_ref.synchronize()
    return PythonObject(None)


def destroy_op(handle: PythonObject) raises -> PythonObject:
    var addr = Int(py=handle)
    if addr == 0:
        return PythonObject(None)
    var state_ptr = UnsafePointer[OpState, MutExternalOrigin](unsafe_from_address=addr)
    state_ptr.destroy_pointee()
    state_ptr.free()
    return PythonObject(None)


@export
def PyInit_op_flow() -> PythonObject:
    try:
        var b = PythonModuleBuilder("op_flow")
        b.def_function[init_op]("init_op")
        b.def_function[forward]("forward")
        b.def_function[destroy_op]("destroy_op")
        return b.finalize()
    except e:
        abort(String("failed to create op_flow module: ", e))
