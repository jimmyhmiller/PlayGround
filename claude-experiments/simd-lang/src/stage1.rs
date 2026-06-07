//! Stage-1: the SIMD token-boundary classifier, in pure Rust.
//!
//! Byte-for-byte the same contract as the MLIR `examples/js_stage1.simd`:
//! for each 64-byte chunk it emits one `u64` into each of two bitmaps —
//!
//! - `start_masks[c]` bit `i` ⇔ byte `c*64+i` **begins a token**: a word char
//!   whose predecessor is not a word char, OR any non-word/non-whitespace/
//!   non-NUL byte.
//! - `word_masks[c]`  bit `i` ⇔ byte is a word char `[A-Za-z0-9_$]`.
//!
//! "previous byte was a word char" is carried exactly across the 64-byte
//! boundary (`word_bits << 1 | carry`). On aarch64 the per-byte classification
//! and the 16→bit movemask run on NEON; the scalar path is the portable
//! reference and the correctness oracle.

/// Run stage-1 over `src`, returning `(start_masks, word_masks)`.
pub fn stage1(src: &[u8]) -> (Vec<u64>, Vec<u64>) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON is available on every aarch64 target we run on; the
            // detection above gates it regardless.
            return unsafe { stage1_neon(src) };
        }
    }
    stage1_scalar(src)
}

/// Convenience alias matching the doc's suggested entry point.
pub fn lex(src: &[u8]) -> (Vec<u64>, Vec<u64>) {
    stage1(src)
}

#[inline]
fn n_words(n: usize) -> usize {
    (n + 63) / 64 + 1 // one word per 64-byte chunk + a trailing guard word
}

#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}
#[inline]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, 32 | 9 | 10 | 13 | 11 | 12)
}

/// Combine the per-chunk class bitmasks into the two output words, carrying the
/// cross-chunk "previous byte was a word char" bit. Shared by both backends.
#[inline]
fn finish_chunk(word_bits: u64, ws_bits: u64, nul_bits: u64, prev_word: &mut u64) -> u64 {
    let prev_word_bits = (word_bits << 1) | *prev_word;
    let word_start = word_bits & !prev_word_bits;
    let punct = !word_bits & !ws_bits & !nul_bits;
    *prev_word = word_bits >> 63;
    word_start | punct
}

/// Portable scalar reference.
pub fn stage1_scalar(src: &[u8]) -> (Vec<u64>, Vec<u64>) {
    let n = src.len();
    let words = n_words(n);
    let mut start_masks = vec![0u64; words];
    let mut word_masks = vec![0u64; words];
    let mut prev_word = 0u64;
    for c in 0..words {
        let base = c * 64;
        let mut word_bits = 0u64;
        let mut ws_bits = 0u64;
        let mut nul_bits = 0u64;
        for i in 0..64 {
            let idx = base + i;
            let b = if idx < n { src[idx] } else { 0 };
            if is_word_byte(b) {
                word_bits |= 1u64 << i;
            } else if is_ws_byte(b) {
                ws_bits |= 1u64 << i;
            } else if b == 0 {
                nul_bits |= 1u64 << i;
            }
        }
        word_masks[c] = word_bits;
        start_masks[c] = finish_chunk(word_bits, ws_bits, nul_bits, &mut prev_word);
    }
    (start_masks, word_masks)
}

// ───────────────────────────── NEON ─────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn stage1_neon(src: &[u8]) -> (Vec<u64>, Vec<u64>) {
    use std::arch::aarch64::*;

    let n = src.len();
    let words = n_words(n);
    let mut start_masks = vec![0u64; words];
    let mut word_masks = vec![0u64; words];

    // Pad into a 64-aligned scratch so NEON can read whole 16-byte lanes without
    // over-reading `src`. Padding bytes are NUL (→ not word, not ws, not start).
    let padded_len = (words - 1) * 64; // covers every real byte's chunk
    let mut buf = vec![0u8; padded_len.max(64)];
    buf[..n].copy_from_slice(src);

    // Per-lane bit weights for the 16→u16 movemask (vaddv trick).
    let powers: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let powv = vld1q_u8(powers.as_ptr());

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn movemask16(m: uint8x16_t, powv: uint8x16_t) -> u64 {
        // m has 0xFF/0x00 lanes; AND with bit-weights then horizontally add each
        // 8-lane half to assemble the low and high bytes of a 16-bit mask.
        let masked = vandq_u8(m, powv);
        let lo = vaddv_u8(vget_low_u8(masked)) as u64;
        let hi = vaddv_u8(vget_high_u8(masked)) as u64;
        lo | (hi << 8)
    }

    let mut prev_word = 0u64;
    for c in 0..(words - 1) {
        let mut word_bits = 0u64;
        let mut ws_bits = 0u64;
        let mut nul_bits = 0u64;
        for sub in 0..4 {
            let p = buf.as_ptr().add(c * 64 + sub * 16);
            let v = vld1q_u8(p);
            // word: [0-9A-Za-z_$]
            let digit = vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'0')), vcleq_u8(v, vdupq_n_u8(b'9')));
            let upper = vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'A')), vcleq_u8(v, vdupq_n_u8(b'Z')));
            let lower = vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'a')), vcleq_u8(v, vdupq_n_u8(b'z')));
            let us = vceqq_u8(v, vdupq_n_u8(b'_'));
            let dollar = vceqq_u8(v, vdupq_n_u8(b'$'));
            let word = vorrq_u8(vorrq_u8(vorrq_u8(digit, upper), vorrq_u8(lower, us)), dollar);
            // ws: 32 9 10 13 11 12
            let ws = vorrq_u8(
                vorrq_u8(
                    vorrq_u8(vceqq_u8(v, vdupq_n_u8(32)), vceqq_u8(v, vdupq_n_u8(9))),
                    vorrq_u8(vceqq_u8(v, vdupq_n_u8(10)), vceqq_u8(v, vdupq_n_u8(13))),
                ),
                vorrq_u8(vceqq_u8(v, vdupq_n_u8(11)), vceqq_u8(v, vdupq_n_u8(12))),
            );
            let nul = vceqq_u8(v, vdupq_n_u8(0));
            let shift = (sub * 16) as u64;
            word_bits |= movemask16(word, powv) << shift;
            ws_bits |= movemask16(ws, powv) << shift;
            nul_bits |= movemask16(nul, powv) << shift;
        }
        // The trailing partial-chunk padding read 0x00 → counted as NUL above,
        // but bytes past `n` within this chunk must be NUL not "real": they are,
        // since `buf` is zero-padded. Re-mask NUL for indices >= n in this chunk.
        let chunk_base = c * 64;
        if chunk_base + 64 > n {
            let valid = n.saturating_sub(chunk_base).min(64);
            let valid_mask = if valid >= 64 { !0u64 } else { (1u64 << valid) - 1 };
            // Outside valid: force not-word, not-ws, and NUL (so no token starts).
            word_bits &= valid_mask;
            ws_bits &= valid_mask;
            nul_bits |= !valid_mask;
        }
        word_masks[c] = word_bits;
        start_masks[c] = finish_chunk(word_bits, ws_bits, nul_bits, &mut prev_word);
    }
    (start_masks, word_masks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_src(seed: u64, len: usize) -> Vec<u8> {
        // cheap xorshift over a representative byte alphabet
        let mut x = seed | 1;
        let alpha = b"abcXYZ_$0123 \t\n+=-*/(){};,.\"'<>!&|^%~?:";
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                alpha[(x as usize) % alpha.len()]
            })
            .collect()
    }

    #[test]
    fn neon_matches_scalar() {
        for len in [0usize, 1, 15, 16, 17, 63, 64, 65, 200, 1000, 4096] {
            for seed in [1u64, 2, 99, 123456] {
                let src = random_src(seed, len);
                let s = stage1_scalar(&src);
                let n = stage1(&src); // NEON on aarch64
                assert_eq!(s, n, "len={len} seed={seed}");
            }
        }
    }

    #[test]
    fn token_starts_simple() {
        let src = b"let x = foo + 42;";
        let (start, _) = stage1(src);
        let mut got = Vec::new();
        for (c, &w) in start.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                got.push(c * 64 + w.trailing_zeros() as usize);
                w &= w - 1;
            }
        }
        assert_eq!(got, vec![0, 4, 6, 8, 12, 14, 16]);
    }
}
