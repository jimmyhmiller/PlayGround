"""ChatterboxTTS — drop-in replacement for upstream `chatterbox.tts.ChatterboxTTS`.

Mirrors the upstream API surface (from_local / from_pretrained /
prepare_conditionals / generate) but routes all compute through the per-op
Mojo .so files. Caches speaker conditioning across generate() calls.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import chatterbox_mojo  # noqa: F401 — bootstraps ops/ on sys.path
import mojo.importer  # noqa: F401

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


S3_SR = 16000
S3GEN_SR = 24000

T_PROMPT_TOKEN = 250
T_PROMPT_MEL = 500
N_CFM_STEPS = 10   # Upstream default. Reducing below 10 with bf16 weights
                   # causes occasional CFM Euler-integration spikes that the HiFT
                   # vocoder amplifies into ±1.0 sample-to-sample "microphone
                   # thump" artifacts (~2% of chunks). Don't drop below 10
                   # unless you upgrade the solver (Heun, RK4) for stability.
                   # Override via CHATTERBOX_CFM_STEPS env var.
MAX_CTX = 600
EOS = 6562
T_COND = 34
DEC_COND_LEN = 240000
ENC_COND_LEN = 96000
DEC_COND_LEN_16K = 160000
MEL = 80
T_TOKEN_FULL = 250
T_TOKEN_6S = 150

DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEFAULT_TOKENIZER_DIR = Path(__file__).resolve().parent.parent.parent / "mojo-t3" / "tests" / "fixtures" / "tokenizer"


def punc_norm(text: str) -> str:
    """Cleanup punctuation. Mirrors upstream chatterbox.tts.punc_norm."""
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
        ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text


@dataclass
class Conditionals:
    """Cached speaker conditioning produced from a reference wav.

    All buffers live on GPU and are reused across generate() calls.
    """
    # T3 conditioning
    speaker_emb_256: Buffer       # (1, 256)  voice-encoder embedding
    cond_prompt_tok: Buffer       # (1, 150) i32 6s s3tokens
    # S3Gen conditioning
    prompt_token: Buffer          # (1, 250) i32 10s s3tokens
    prompt_feat: Buffer           # (1, 500, 80) 24k mel
    spks: Buffer                  # (1, 80)  campplus → affine speaker
    # Exaggeration is per-call (cheap), not part of cache.


class ChatterboxTTS:
    """Pure-Mojo Chatterbox TTS. API mirrors upstream `ChatterboxTTS`."""

    ENC_COND_LEN = ENC_COND_LEN
    DEC_COND_LEN = DEC_COND_LEN

    def __init__(
        self,
        weights_dir: Path,
        tokenizer_dir: Path,
        device: str = "gpu",
        conds: Optional[Conditionals] = None,
        use_bf16: Optional[bool] = None,
    ):
        """If `use_bf16` is True, load pre-cast bfloat16 weight copies and
        use the AMD GEMM matrix-core fast path (≈40-50% faster on AMD f32-only
        hardware, tiny accuracy loss). When None (default), reads the
        CHATTERBOX_BF16=1 environment variable. Requires the
        `*.bf16.bin` sibling files (run scripts/convert_t3_weights_bf16.py
        and scripts/convert_s3gen_weights_bf16.py to generate them).
        """
        self.sr = S3GEN_SR
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.conds = conds

        # Resolve bf16 flag. We propagate via env var because the Mojo loaders
        # read CHATTERBOX_BF16 at module-load time.
        import os as _os
        if use_bf16 is None:
            use_bf16 = _os.environ.get("CHATTERBOX_BF16", "") == "1"
        else:
            _os.environ["CHATTERBOX_BF16"] = "1" if use_bf16 else "0"
        self.use_bf16 = use_bf16

        # Default-on perf flags: fused QKV + fused MLP in T3 (~5% T3 speedup,
        # 0% WER verified). Set CHATTERBOX_T3_FUSE_{QKV,MLP}=0 to disable.
        _os.environ.setdefault("CHATTERBOX_T3_FUSE_QKV", "1")
        _os.environ.setdefault("CHATTERBOX_T3_FUSE_MLP", "1")

        self._gpu = Accelerator()
        self._cpu = CPU()
        self._dctx_ptr = self._gpu._device_context_ptr()

        # Initialize all ops (loads weights once).
        w = self.weights_dir
        self._tok_h = op_text_tokenize.init_op(
            str(self.tokenizer_dir / "vocab.txt"),
            str(self.tokenizer_dir / "merges.txt"),
        )
        self._audio_in_h = op_audio_in.init_op(str(w / "s3t"), str(w / "ve"), self._dctx_ptr)
        self._campplus_h = op_campplus.init_op(str(w / "s3gen" / "speaker_encoder"), self._dctx_ptr)
        self._spk_affine_h = op_spk_affine.init_op(str(w / "s3gen" / "flow"), self._dctx_ptr)
        self._t3_h = op_t3.init_op(str(w / "t3"), str(w / "t3"), self._dctx_ptr)
        self._flow_h = op_flow.init_op(
            str(w / "s3gen" / "flow"),
            str(w / "s3gen" / "flow" / "decoder" / "estimator"),
            self._dctx_ptr,
        )
        self._hift_h = op_hift.init_op(str(w / "s3gen" / "mel2wav"), self._dctx_ptr)

    def __del__(self):
        for h, mod in [
            (getattr(self, "_tok_h", None), op_text_tokenize),
            (getattr(self, "_audio_in_h", None), op_audio_in),
            (getattr(self, "_campplus_h", None), op_campplus),
            (getattr(self, "_spk_affine_h", None), op_spk_affine),
            (getattr(self, "_t3_h", None), op_t3),
            (getattr(self, "_flow_h", None), op_flow),
            (getattr(self, "_hift_h", None), op_hift),
        ]:
            if h is not None:
                try:
                    mod.destroy_op(h)
                except Exception:
                    pass

    # ── Public API ────────────────────────────────────────────────────────

    @classmethod
    def from_local(cls, ckpt_dir, device: str = "gpu") -> "ChatterboxTTS":
        ckpt_dir = Path(ckpt_dir)
        return cls(weights_dir=ckpt_dir, tokenizer_dir=DEFAULT_TOKENIZER_DIR, device=device)

    @classmethod
    def from_pretrained(cls, device: str = "gpu",
                        use_bf16: Optional[bool] = None) -> "ChatterboxTTS":
        if not DEFAULT_WEIGHTS_DIR.exists():
            raise FileNotFoundError(
                f"weights dir not found at {DEFAULT_WEIGHTS_DIR}. "
                "Run the weight dump scripts under scripts/ first."
            )
        return cls(weights_dir=DEFAULT_WEIGHTS_DIR, tokenizer_dir=DEFAULT_TOKENIZER_DIR,
                   device=device, use_bf16=use_bf16)

    def prepare_conditionals(self, wav_fpath, exaggeration: float = 0.5) -> None:
        """Compute speaker conditioning from a reference wav and cache on `self.conds`."""
        del exaggeration  # exaggeration is applied per-call inside generate()
        self.conds = self._compute_conditionals(str(wav_fpath))

    def generate(
        self,
        text: str,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,            # accepted for API compat; not implemented in Mojo sampler
        top_p: float = 1.0,
        audio_prompt_path=None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 1000,
        rng_seed: int = 0xDEADBEEF,
    ) -> torch.Tensor:
        """Synthesize speech. Returns torch tensor shaped (1, n_samples) at 24kHz."""

        if audio_prompt_path is not None:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        if self.conds is None:
            raise RuntimeError(
                "No conditionals available. Call prepare_conditionals(wav) "
                "or pass audio_prompt_path=... to generate()."
            )

        import time, os
        prof = os.environ.get("CHATTERBOX_PROFILE", "") == "1"

        text = punc_norm(text)
        t_tok0 = time.perf_counter()
        text_ids = op_text_tokenize.tokenize(self._tok_h, text)
        text_ids_full = [255] + list(text_ids) + [0]   # START_TEXT, STOP_TEXT
        t_tok = time.perf_counter() - t_tok0

        t3_config = {
            "emotion": float(exaggeration),
            "cfg_weight": float(cfg_weight),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "rep_penalty": float(repetition_penalty),
            "min_p": float(min_p),
            "max_new": int(max_new_tokens),
            "rng_seed": int(rng_seed),
        }
        t_t3_0 = time.perf_counter()
        raw_tokens = op_t3.generate(
            self._t3_h, self.conds.speaker_emb_256, self.conds.cond_prompt_tok,
            text_ids_full, t3_config,
        )
        t_t3 = time.perf_counter() - t_t3_0

        # Filter EOS + invalid tokens (mirrors drop_invalid_tokens + <6561 filter).
        speech_tokens = []
        for tok in raw_tokens:
            if tok == EOS:
                break
            if tok < 6561:
                speech_tokens.append(tok)
        if len(speech_tokens) == 0:
            speech_tokens = [0]

        t_dec0 = time.perf_counter()
        wav = self._s3gen_decode(speech_tokens)
        t_dec = time.perf_counter() - t_dec0
        if prof:
            n_samples = wav.size
            audio_s = n_samples / self.sr
            print(f"[prof] tok={t_tok*1000:.1f}ms  t3({len(raw_tokens)}tok)={t_t3*1000:.1f}ms  "
                  f"s3gen({len(speech_tokens)}st)={t_dec*1000:.1f}ms  "
                  f"audio={audio_s:.2f}s  total={(t_tok+t_t3+t_dec)*1000:.1f}ms",
                  flush=True)
        return torch.from_numpy(wav).unsqueeze(0)

    # ── Internals ─────────────────────────────────────────────────────────

    def _compute_conditionals(self, ref_wav_path: str) -> Conditionals:
        gpu, cpu = self._gpu, self._cpu

        n_24_in, sr_in = op_load_wav.get_wav_size(ref_wav_path)
        if sr_in != S3GEN_SR:
            raise RuntimeError(f"Expected {S3GEN_SR}Hz ref wav, got {sr_in}Hz")

        # Load full 24k wav.
        host_wav_24_full = Buffer(shape=(n_24_in,), dtype=DType.float32, device=cpu)
        op_load_wav.load_wav_into(host_wav_24_full, ref_wav_path)
        full_arr_24 = host_wav_24_full.to_numpy().astype(np.float32)

        # Resample FULL 24k → FULL 16k. This is the wav fed to the VoiceEncoder
        # (matching upstream which uses the untrimmed-pre-trim 16k ref wav).
        n_16_full = (full_arr_24.size * S3_SR) // S3GEN_SR
        host_full_24 = Buffer.from_numpy(full_arr_24)
        host_full_16 = Buffer(shape=(n_16_full,), dtype=DType.float32, device=cpu)
        op_load_wav.resample_into(host_full_24, host_full_16, S3GEN_SR, S3_SR,
                                  full_arr_24.size, n_16_full, "/tmp")
        full_arr_16 = host_full_16.to_numpy().astype(np.float32).copy()

        # librosa.effects.trim(top_db=20) equivalent: remove leading/trailing
        # silence frames where RMS is more than 20 dB below the peak RMS.
        trimmed_full_16 = _librosa_trim(full_arr_16, top_db=20.0,
                                        frame_length=2048, hop_length=512)
        n_ve = trimmed_full_16.size
        host_ve_16 = Buffer.from_numpy(trimmed_full_16)
        wav_ve_16 = host_ve_16.to(gpu)

        # For s3gen + campplus paths: pad/clip to 10s as before.
        if full_arr_24.size >= DEC_COND_LEN:
            arr_24 = full_arr_24[:DEC_COND_LEN].copy()
        else:
            arr_24 = np.zeros(DEC_COND_LEN, dtype=np.float32)
            arr_24[:full_arr_24.size] = full_arr_24
        n_24 = DEC_COND_LEN
        host_wav_24 = Buffer.from_numpy(arr_24)
        wav_24 = host_wav_24.to(gpu)

        n_16 = DEC_COND_LEN_16K
        host_wav_16 = Buffer(shape=(n_16,), dtype=DType.float32, device=cpu)
        op_load_wav.resample_into(host_wav_24, host_wav_16, S3GEN_SR, S3_SR, n_24, n_16, "/tmp")
        wav_16 = host_wav_16.to(gpu)

        # 10s s3tokens (prompt_token).
        T_mel_full = (n_16 + 2 * 200 - 400) // 160 + 1 - 1
        log_mel_full = Buffer(shape=(128, T_mel_full), dtype=DType.float32, device=gpu)
        op_audio_in.compute_log_mel_s3tok(self._audio_in_h, wav_16, log_mel_full, n_16, T_mel_full)

        T_token_full = T_mel_full // 4
        prompt_token = Buffer(shape=(1, T_token_full), dtype=DType.int32, device=gpu)
        mp_full = Buffer.from_numpy(np.ones((1, T_token_full, 1), dtype=np.float32)).to(gpu)
        am_full = Buffer.from_numpy(np.zeros((T_token_full, T_token_full), dtype=np.float32)).to(gpu)
        op_audio_in.s3tokenize(self._audio_in_h, log_mel_full, prompt_token, mp_full, am_full, 1, T_mel_full)

        # 6s s3tokens (T3 cond prompt).
        host_6s = Buffer.from_numpy(np.asarray(host_wav_16.to_numpy()[:ENC_COND_LEN], dtype=np.float32))
        wav_16_6s = host_6s.to(gpu)
        T_mel_6s = (ENC_COND_LEN + 2 * 200 - 400) // 160 + 1 - 1
        T_token_6s = T_mel_6s // 4
        log_mel_6s = Buffer(shape=(128, T_mel_6s), dtype=DType.float32, device=gpu)
        op_audio_in.compute_log_mel_s3tok(self._audio_in_h, wav_16_6s, log_mel_6s, ENC_COND_LEN, T_mel_6s)
        mp_6s = Buffer.from_numpy(np.ones((1, T_token_6s, 1), dtype=np.float32)).to(gpu)
        am_6s = Buffer.from_numpy(np.zeros((T_token_6s, T_token_6s), dtype=np.float32)).to(gpu)
        cond_prompt_tok = Buffer(shape=(1, T_token_6s), dtype=DType.int32, device=gpu)
        op_audio_in.s3tokenize(self._audio_in_h, log_mel_6s, cond_prompt_tok, mp_6s, am_6s, 1, T_mel_6s)

        # 24k prompt mel.
        prompt_feat_mt = Buffer(shape=(MEL, T_PROMPT_MEL), dtype=DType.float32, device=gpu)
        op_audio_in.compute_mel_24k(self._audio_in_h, wav_24, prompt_feat_mt, n_24, T_PROMPT_MEL)
        pfeat_host = prompt_feat_mt.to(cpu).to_numpy()
        prompt_feat = Buffer.from_numpy(pfeat_host.T.copy().reshape(1, T_PROMPT_MEL, MEL)).to(gpu)

        # CAMPPlus → spks.
        T_fbank = (n_16 - 400) // 160 + 1
        fbank_bft = Buffer(shape=(MEL, T_fbank), dtype=DType.float32, device=gpu)
        op_audio_in.compute_kaldi_fbank(self._audio_in_h, wav_16, fbank_bft, n_16, T_fbank)
        emb_192 = Buffer(shape=(1, 192), dtype=DType.float32, device=gpu)
        op_campplus.speaker_embedding(self._campplus_h, fbank_bft, emb_192, 1, T_fbank)
        spks = Buffer(shape=(1, MEL), dtype=DType.float32, device=gpu)
        op_spk_affine.forward(self._spk_affine_h, emb_192, spks, 1)

        # VoiceEncoder → 256-d speaker_emb (T3 input).
        # Uses FULL trimmed 16k wav (not the 10s-padded one) to match upstream.
        T_ve = (n_ve + 2 * 200 - 400) // 160 + 1
        mel_ve_full = Buffer(shape=(T_ve, 40), dtype=DType.float32, device=gpu)
        op_audio_in.compute_ve_mel(self._audio_in_h, wav_ve_16, mel_ve_full, n_ve, T_ve)
        speaker_emb_256 = Buffer(shape=(1, 256), dtype=DType.float32, device=gpu)
        op_audio_in.voice_encode(self._audio_in_h, mel_ve_full, speaker_emb_256, T_ve)

        # DEBUG: dump conditioning for parity comparison vs upstream.
        import os
        if os.environ.get("CHATTERBOX_DEBUG_DUMP_COND"):
            os.makedirs("/tmp/t3_dump", exist_ok=True)
            spk_host = speaker_emb_256.to(cpu).to_numpy().copy()
            tok_host = cond_prompt_tok.to(cpu).to_numpy().copy()
            np.savez("/tmp/t3_dump/mojo_t3_cond.npz",
                     speaker_emb=spk_host, cond_prompt=tok_host)
            print(f"[dump] saved Mojo T3 conditioning")

        return Conditionals(
            speaker_emb_256=speaker_emb_256,
            cond_prompt_tok=cond_prompt_tok,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            spks=spks,
        )

    def generate_batch(
        self,
        texts: list[str],
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        audio_prompt_path=None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 1000,
        rng_seed: int = 0xDEADBEEF,
    ) -> list[torch.Tensor]:
        """Generate audio for a list of texts. NOTE: single-instance batching
        does NOT give a speedup on our hardware — we're compute-bound at B=1
        due to large mel-time dimension. For cross-utterance throughput, use
        the WorkerPool class instead (separate processes). This method
        currently runs sequentially.
        """
        if audio_prompt_path is not None:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        results = []
        for i, text in enumerate(texts):
            results.append(self.generate(
                text, repetition_penalty=repetition_penalty, min_p=min_p,
                top_p=top_p, exaggeration=exaggeration, cfg_weight=cfg_weight,
                temperature=temperature, max_new_tokens=max_new_tokens,
                rng_seed=rng_seed + i,
            ))
        return results

    def _s3gen_decode(self, speech_tokens: list[int], cfg_rate: float = 0.7,
                      n_steps: int = 0, noise_seed: int = 0xC0FFEE) -> np.ndarray:
        # Allow override via CHATTERBOX_CFM_STEPS env var (1..10). Default 10.
        import os as _os
        if n_steps == 0:
            n_steps = int(_os.environ.get("CHATTERBOX_CFM_STEPS", N_CFM_STEPS))
        gpu, cpu = self._gpu, self._cpu
        c = self.conds
        assert c is not None

        T_GEN_TOKEN = len(speech_tokens)
        T_TOTAL_TOKEN = T_PROMPT_TOKEN + T_GEN_TOKEN
        T_TOTAL_MEL = 2 * T_TOTAL_TOKEN
        T_OUT_MEL = T_TOTAL_MEL - T_PROMPT_MEL

        prompt_token_host = c.prompt_token.to(cpu).to_numpy().reshape(-1)
        tok_combined = np.concatenate([
            prompt_token_host.astype(np.int64),
            np.array(speech_tokens, dtype=np.int64),
        ]).reshape(1, T_TOTAL_TOKEN)
        tok_buf_i64 = Buffer.from_numpy(tok_combined).to(gpu)

        mel_out = Buffer(shape=(1, MEL, T_OUT_MEL), dtype=DType.float32, device=gpu)
        flow_config = {
            "B": 1,
            "T_token": T_TOTAL_TOKEN,
            "T_prompt_mel": T_PROMPT_MEL,
            "T_total_mel": T_TOTAL_MEL,
            "T_out_mel": T_OUT_MEL,
            "n_steps": int(n_steps),
            "cfg_rate": float(cfg_rate),
            "noise_seed": int(noise_seed),
        }
        import time, os
        prof = os.environ.get("CHATTERBOX_PROFILE", "") == "1"
        t_fl0 = time.perf_counter()
        op_flow.forward(self._flow_h, tok_buf_i64, c.spks, c.prompt_feat, mel_out, flow_config)
        t_fl = time.perf_counter() - t_fl0

        T_HIFT = T_OUT_MEL * 120 + 1
        T_AUDIO = (T_HIFT - 1) * 4
        audio_out = Buffer(shape=(1, T_AUDIO), dtype=DType.float32, device=gpu)
        t_hf0 = time.perf_counter()
        op_hift.forward(self._hift_h, mel_out, audio_out, 1, T_OUT_MEL)
        t_hf = time.perf_counter() - t_hf0

        host_audio = audio_out.to(cpu)
        if prof:
            print(f"[prof]   flow={t_fl*1000:.1f}ms  hift={t_hf*1000:.1f}ms  "
                  f"T_out_mel={T_OUT_MEL} T_audio={T_AUDIO}", flush=True)
        return host_audio.to_numpy().reshape(-1).copy()


def _librosa_trim(y: np.ndarray, top_db: float = 20.0,
                  frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """numpy port of librosa.effects.trim — strip leading/trailing silence.

    Frames the signal at (frame_length, hop_length), computes per-frame RMS,
    converts to dB ref=max(rms), keeps frames > -top_db, returns y sliced to
    the [first, last+1] non-silent frames. Matches librosa's pad-center=True
    framing exactly.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = y.shape[0]
    if n == 0:
        return y

    # librosa's util.frame with default pad_mode='reflect', pad=True (centered).
    # The centering pads y by frame_length//2 on each side before framing.
    pad = frame_length // 2
    y_padded = np.pad(y, pad, mode="reflect")
    n_frames = 1 + (y_padded.shape[0] - frame_length) // hop_length
    if n_frames <= 0:
        return y
    # Build frame matrix (n_frames, frame_length).
    idx = (np.arange(frame_length)[None, :]
           + (np.arange(n_frames) * hop_length)[:, None])
    frames = y_padded[idx]

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    ref = rms.max()
    if ref <= 0:
        return y
    db = 20.0 * np.log10(rms / ref + 1e-10)
    non_silent = db > -top_db
    nz = np.flatnonzero(non_silent)
    if nz.size == 0:
        return y[:0]
    start = int(nz[0]) * hop_length
    end = min(n, (int(nz[-1]) + 1) * hop_length)
    return y[start:end].copy()
