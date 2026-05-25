"""Chatterbox Mojo orchestrator — chains 9 Mojo .so ops to synthesize speech.

Usage:
    pixi run python -m chatterbox_mojo voice.wav "text to speak" out.wav

The orchestrator does ZERO compute — all math is in Mojo .so files. Python
holds buffer handles, calls into ops in sequence, and runs the autoregressive
T3 loop counter (well, t3.generate runs the whole loop internally).
"""
import sys
from pathlib import Path

# Bootstrap sys.path so ops/<op>/<op>.mojo is importable.
import chatterbox_mojo  # noqa: F401

import mojo.importer  # noqa: F401 — registers .mojo loader
import op_load_wav
import op_write_wav
import op_text_tokenize
import op_audio_in
import op_campplus
import op_spk_affine
import op_t3
import op_flow
import op_hift

from max.driver import Accelerator, CPU, Buffer
from max.dtype import DType
import numpy as np


# Hyperparameters matching tests/synthesize_from_wav.mojo
T_PROMPT_TOKEN = 250    # 10s of speech tokens
T_PROMPT_MEL = 500      # 10s of 24k mel
N_CFM_STEPS = 10
CFG = 0.7
T3_CFG = 0.5
MAX_CTX = 600
MAX_NEW = 200
EOS = 6562
T_COND = 34
T_BOS = 1
S3_SR = 16000
S3GEN_SR = 24000
DEC_COND_LEN = 240000      # 10s @ 24k
ENC_COND_LEN = 96000       # 6s @ 16k
DEC_COND_LEN_16K = 160000  # 10s @ 16k
MEL = 80
T_TOKEN_FULL = 250
T_TOKEN_6S = 150


def main(argv):
    if len(argv) < 4:
        print(f"usage: python -m chatterbox_mojo <ref.wav> <text> <out.wav>", file=sys.stderr)
        return 2
    ref_wav, text, out_wav = argv[1], argv[2], argv[3]

    print(f"[orch] ref={ref_wav}")
    print(f"[orch] text={text!r}")
    print(f"[orch] out={out_wav}")

    # Set up GPU device context shared across all ops.
    gpu = Accelerator()
    cpu = CPU()
    dctx_ptr = gpu._device_context_ptr()
    print(f"[orch] gpu device_context_ptr=0x{dctx_ptr:x}")

    # ── 0. Init every op once (loads weights).
    print("[orch] initializing ops (loading weights)...")
    tok_h = op_text_tokenize.init_op(
        "../mojo-t3/tests/fixtures/tokenizer/vocab.txt",
        "../mojo-t3/tests/fixtures/tokenizer/merges.txt",
    )
    audio_in_h = op_audio_in.init_op("weights/s3t", "weights/ve", dctx_ptr)
    campplus_h = op_campplus.init_op("weights/s3gen/speaker_encoder", dctx_ptr)
    spk_affine_h = op_spk_affine.init_op("weights/s3gen/flow", dctx_ptr)
    t3_h = op_t3.init_op("weights/t3", "weights/t3", dctx_ptr)
    flow_h = op_flow.init_op("weights/s3gen/flow", "weights/s3gen/flow/decoder/estimator", dctx_ptr)
    hift_h = op_hift.init_op("weights/s3gen/mel2wav", dctx_ptr)
    print(f"[orch] all ops initialized")

    try:
        # ── 1. Load 24k WAV.
        n_24, sr_in = op_load_wav.get_wav_size(ref_wav)
        if sr_in != 24000:
            raise RuntimeError(f"Expected 24kHz ref, got {sr_in}Hz")
        print(f"[orch] loaded {n_24} samples @ {sr_in}Hz")

        # Load the full WAV into a host buffer, then slice/pad to DEC_COND_LEN.
        host_wav_24_full = Buffer(shape=(n_24,), dtype=DType.float32, device=cpu)
        op_load_wav.load_wav_into(host_wav_24_full, ref_wav)
        full_arr = host_wav_24_full.to_numpy()
        # Pad with zeros if shorter, truncate if longer; result is exactly DEC_COND_LEN samples.
        if n_24 >= DEC_COND_LEN:
            arr_24 = full_arr[:DEC_COND_LEN].astype(np.float32).copy()
        else:
            arr_24 = np.zeros(DEC_COND_LEN, dtype=np.float32)
            arr_24[:n_24] = full_arr
        n_24 = DEC_COND_LEN
        host_wav_24 = Buffer.from_numpy(arr_24)
        wav_24 = host_wav_24.to(gpu)

        # ── 2. Resample 24k → 16k via ffmpeg+soxr (on host).
        n_16 = DEC_COND_LEN_16K
        host_wav_16 = Buffer(shape=(n_16,), dtype=DType.float32, device=cpu)
        op_load_wav.resample_into(host_wav_24, host_wav_16, 24000, 16000, n_24, n_16, "/tmp")
        wav_16 = host_wav_16.to(gpu)
        print(f"[orch] resampled to {n_16} samples @ 16kHz")

        # ── 3. s3tokenizer log-mel for full 10s.
        T_mel_full = (n_16 + 2 * 200 - 400) // 160 + 1 - 1   # 1000
        log_mel_full = Buffer(shape=(128, T_mel_full), dtype=DType.float32, device=gpu)
        op_audio_in.compute_log_mel_s3tok(audio_in_h, wav_16, log_mel_full, n_16, T_mel_full)

        T_token_full = T_mel_full // 4   # 250
        prompt_token = Buffer(shape=(1, T_token_full), dtype=DType.int32, device=gpu)
        mp_full = Buffer(shape=(1, T_token_full, 1), dtype=DType.float32, device=gpu)
        # Fill mp_full with 1.0 on host then push.
        mp_arr = np.ones((1, T_token_full, 1), dtype=np.float32)
        mp_full = Buffer.from_numpy(mp_arr).to(gpu)
        am_full = Buffer.from_numpy(np.zeros((T_token_full, T_token_full), dtype=np.float32)).to(gpu)
        op_audio_in.s3tokenize(audio_in_h, log_mel_full, prompt_token, mp_full, am_full, 1, T_mel_full)
        print(f"[orch] prompt_token shape=({T_token_full},)")

        # ── 4. s3tokenizer for 6s prefix (slice wav_16 to first 96000 samples).
        # Easier: copy host_wav_16 to a shorter buffer.
        host_6s = Buffer.from_numpy(np.asarray(host_wav_16.to_numpy()[:ENC_COND_LEN], dtype=np.float32))
        wav_16_6s = host_6s.to(gpu)
        T_mel_6s = (ENC_COND_LEN + 2 * 200 - 400) // 160 + 1 - 1  # 600
        T_token_6s = T_mel_6s // 4   # 150
        log_mel_6s = Buffer(shape=(128, T_mel_6s), dtype=DType.float32, device=gpu)
        op_audio_in.compute_log_mel_s3tok(audio_in_h, wav_16_6s, log_mel_6s, ENC_COND_LEN, T_mel_6s)
        mp_6s = Buffer.from_numpy(np.ones((1, T_token_6s, 1), dtype=np.float32)).to(gpu)
        am_6s = Buffer.from_numpy(np.zeros((T_token_6s, T_token_6s), dtype=np.float32)).to(gpu)
        cond_prompt_tok = Buffer(shape=(1, T_token_6s), dtype=DType.int32, device=gpu)
        op_audio_in.s3tokenize(audio_in_h, log_mel_6s, cond_prompt_tok, mp_6s, am_6s, 1, T_mel_6s)
        print(f"[orch] cond_prompt_tok shape=({T_token_6s},)")

        # ── 5. 24kHz mel for prompt_feat (1, 500, 80).
        prompt_feat_mt = Buffer(shape=(MEL, T_PROMPT_MEL), dtype=DType.float32, device=gpu)
        op_audio_in.compute_mel_24k(audio_in_h, wav_24, prompt_feat_mt, n_24, T_PROMPT_MEL)
        # Re-shape to (1, T_PROMPT_MEL, MEL): transpose on host.
        pfeat_host = prompt_feat_mt.to(cpu).to_numpy()  # (MEL, T_PROMPT_MEL)
        prompt_feat = Buffer.from_numpy(pfeat_host.T.copy().reshape(1, T_PROMPT_MEL, MEL)).to(gpu)
        print(f"[orch] prompt_feat shape=(1, {T_PROMPT_MEL}, {MEL})")

        # ── 6. Kaldi fbank → CAMPPlus → 192-d embed → spks (80).
        T_fbank = (n_16 - 400) // 160 + 1   # 998
        fbank_bft = Buffer(shape=(MEL, T_fbank), dtype=DType.float32, device=gpu)
        op_audio_in.compute_kaldi_fbank(audio_in_h, wav_16, fbank_bft, n_16, T_fbank)

        emb_192 = Buffer(shape=(1, 192), dtype=DType.float32, device=gpu)
        op_campplus.speaker_embedding(campplus_h, fbank_bft, emb_192, 1, T_fbank)

        spks = Buffer(shape=(1, MEL), dtype=DType.float32, device=gpu)
        op_spk_affine.forward(spk_affine_h, emb_192, spks, 1)
        print(f"[orch] spks shape=(1, {MEL})")

        # ── 7. VoiceEncoder mel + forward → 256-d speaker_emb.
        T_ve = (n_16 + 2 * 200 - 400) // 160 + 1   # 1001
        mel_ve_full = Buffer(shape=(T_ve, 40), dtype=DType.float32, device=gpu)
        op_audio_in.compute_ve_mel(audio_in_h, wav_16, mel_ve_full, n_16, T_ve)
        speaker_emb_256 = Buffer(shape=(1, 256), dtype=DType.float32, device=gpu)
        op_audio_in.voice_encode(audio_in_h, mel_ve_full, speaker_emb_256, T_ve)
        print(f"[orch] speaker_emb_256 shape=(256,)")

        # ── 8. Text tokenize → list[int].
        text_ids = op_text_tokenize.tokenize(tok_h, text)
        # Prepend START_TEXT (255) and append STOP_TEXT (0) — see text_embed.text_to_input_ids.
        text_ids_full = [255] + list(text_ids) + [0]
        print(f"[orch] text_ids: {len(text_ids_full)} tokens")

        # ── 9. T3 generate speech tokens.
        t3_config = {
            "emotion": 0.5,
            "cfg_weight": T3_CFG,
            "temperature": 0.8,
            "top_p": 0.95,
            "rep_penalty": 1.2,
            "max_new": MAX_NEW,
            "rng_seed": 0xDEADBEEF,
        }
        print(f"[orch] T3 generate (CFG={T3_CFG}, max_new={MAX_NEW})...")
        raw_tokens = op_t3.generate(t3_h, speaker_emb_256, cond_prompt_tok, text_ids_full, t3_config)
        print(f"[orch] T3 generated {len(raw_tokens)} raw tokens")
        # Filter: stop at EOS, drop tokens >= 6561 (vocab boundary for flow encoder).
        speech_tokens = []
        for tok in raw_tokens:
            if tok == EOS:
                break
            if tok < 6561:
                speech_tokens.append(tok)
        if len(speech_tokens) == 0:
            speech_tokens = [0]
        print(f"[orch] {len(speech_tokens)} valid speech tokens after EOS/vocab filter")

        # ── 10. Build full token sequence: prompt_token + speech_tokens.
        T_GEN_TOKEN = len(speech_tokens)
        T_TOTAL_TOKEN = T_PROMPT_TOKEN + T_GEN_TOKEN
        T_TOTAL_MEL = 2 * T_TOTAL_TOKEN  # upsample 2x in flow encoder
        T_OUT_MEL = T_TOTAL_MEL - T_PROMPT_MEL
        # Combine prompt_token (i32 GPU) + speech_tokens (Python list) into one i64 GPU buf.
        prompt_token_host = prompt_token.to(cpu).to_numpy().reshape(-1)
        tok_combined = np.concatenate([
            prompt_token_host.astype(np.int64),
            np.array(speech_tokens, dtype=np.int64),
        ]).reshape(1, T_TOTAL_TOKEN)
        tok_buf_i64 = Buffer.from_numpy(tok_combined).to(gpu)
        print(f"[orch] T_TOTAL_TOKEN={T_TOTAL_TOKEN}, T_TOTAL_MEL={T_TOTAL_MEL}")

        # ── 11. Flow encoder + CFM → mel_out (B, 80, T_OUT_MEL).
        mel_out = Buffer(shape=(1, MEL, T_OUT_MEL), dtype=DType.float32, device=gpu)
        flow_config = {
            "B": 1,
            "T_token": T_TOTAL_TOKEN,
            "T_prompt_mel": T_PROMPT_MEL,
            "T_total_mel": T_TOTAL_MEL,
            "T_out_mel": T_OUT_MEL,
            "n_steps": N_CFM_STEPS,
            "cfg_rate": CFG,
            "noise_seed": 0xC0FFEE,
        }
        print(f"[orch] Flow encoder + CFM Euler ({N_CFM_STEPS} steps)...")
        op_flow.forward(flow_h, tok_buf_i64, spks, prompt_feat, mel_out, flow_config)

        # ── 12. HiFT mel → audio.
        T_HIFT = T_OUT_MEL * 120 + 1
        T_AUDIO = (T_HIFT - 1) * 4   # HOP = 4
        audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
        print(f"[orch] HiFT mel→audio (T_audio={T_AUDIO})...")
        op_hift.forward(hift_h, mel_out, audio_out, 1, T_OUT_MEL)

        # ── 13. Write audio to WAV.
        host_audio = audio_out.to(cpu)
        op_write_wav.write_wav(host_audio, T_AUDIO, 24000, out_wav)
        arr = host_audio.to_numpy().reshape(-1)
        max_amp = float(np.abs(arr).max())
        rms = float(np.sqrt((arr ** 2).mean()))
        print(f"[orch] audio: max={max_amp:.3f}, rms={rms:.3f}, duration={T_AUDIO/24000:.2f}s")
        print(f"[PASS] wrote {out_wav}")
        return 0
    finally:
        # Clean up op handles.
        op_text_tokenize.destroy_op(tok_h)
        op_audio_in.destroy_op(audio_in_h)
        op_campplus.destroy_op(campplus_h)
        op_spk_affine.destroy_op(spk_affine_h)
        op_t3.destroy_op(t3_h)
        op_flow.destroy_op(flow_h)
        op_hift.destroy_op(hift_h)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
