//! Native side of `internalBinding('buffer')`.
//!
//! Mirrors the 24 names destructured from `internalBinding('buffer')` in
//! nodejs/node `lib/buffer.js`. Each primitive is implemented in Rust; the
//! JS layer (`lib/buffer.js`, to be lifted in a later pass) will call into
//! them as if it were running inside Node.

use base64::Engine;
use rquickjs::function::Func;
use rquickjs::{ArrayBuffer, Ctx, Error, Object, Result, TypedArray, Value};

// -------------------------------------------------------------------------
// ascii/latin1 WriteStatic (utf8WriteStatic lives in JS — see make()).
// `*WriteStatic(buf, string, offset, length) -> bytesWritten`
// -------------------------------------------------------------------------

fn ascii_write_static<'js>(
    buf: TypedArray<'js, u8>,
    string: String,
    offset: u32,
    length: u32,
) -> Result<u32> {
    let dst = ta_as_bytes_mut(&buf)?;
    let off = offset as usize;
    let cap = dst.len().saturating_sub(off).min(length as usize);
    let mut written = 0;
    for c in string.chars() {
        if written >= cap { break; }
        dst[off + written] = (c as u32) as u8 & 0x7F; // mask to 7-bit
        written += 1;
    }
    Ok(written as u32)
}

fn latin1_write_static<'js>(
    buf: TypedArray<'js, u8>,
    string: String,
    offset: u32,
    length: u32,
) -> Result<u32> {
    let dst = ta_as_bytes_mut(&buf)?;
    let off = offset as usize;
    let cap = dst.len().saturating_sub(off).min(length as usize);
    let mut written = 0;
    for c in string.chars() {
        if written >= cap { break; }
        dst[off + written] = (c as u32) as u8; // truncate to low byte
        written += 1;
    }
    Ok(written as u32)
}

fn create_unsafe_array_buffer(ctx: Ctx<'_>, size: u32) -> Result<ArrayBuffer<'_>> {
    // "Unsafe" in Node means uninitialized memory. We just allocate zero-filled;
    // it's safe and the difference is invisible to JS that overwrites it anyway.
    let zeros = vec![0u8; size as usize];
    ArrayBuffer::new(ctx, zeros)
}

fn set_detach_key<'js>(_ab: Value<'js>, _key: Value<'js>) -> Result<()> {
    // Node uses this for crypto-grade transferables. We don't implement
    // transferables yet; no-op is safe for non-worker code paths.
    Ok(())
}

// -------------------------------------------------------------------------
// TypedArray helpers
// -------------------------------------------------------------------------

/// View a Uint8Array as an immutable byte slice (respecting the view's
/// byteOffset/length within its backing ArrayBuffer).
fn ta_as_bytes<'js>(arr: &TypedArray<'js, u8>) -> Result<&'js [u8]> {
    arr.as_bytes()
        .ok_or_else(|| Error::new_from_js("TypedArray", "bytes (detached?)"))
        .map(|b| {
            // Extend lifetime: the underlying ArrayBuffer is GC-rooted by the
            // TypedArray we hold; QuickJS is single-threaded so no aliasing
            // problem during a synchronous call.
            unsafe { std::slice::from_raw_parts(b.as_ptr(), b.len()) }
        })
}

/// View a Uint8Array as a mutable byte slice. Unsafe in principle (we punch
/// through the immutable API) but safe in practice because QuickJS executes
/// JS single-threaded — no other handler can observe the buffer while we
/// hold this slice during a synchronous binding call.
fn ta_as_bytes_mut<'js>(arr: &TypedArray<'js, u8>) -> Result<&'js mut [u8]> {
    let bytes = arr
        .as_bytes()
        .ok_or_else(|| Error::new_from_js("TypedArray", "bytes (detached?)"))?;
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_ptr() as *mut u8, bytes.len()) })
}

fn throw<'js, T>(ctx: &Ctx<'js>, msg: &str) -> Result<T> {
    let err = rquickjs::Exception::from_message(ctx.clone(), msg)?;
    Err(ctx.throw(err.into_value()))
}

// -------------------------------------------------------------------------
// Encodings: *Slice (bytes -> string) and *Write (string -> bytes)
// -------------------------------------------------------------------------

fn utf8_slice<'js>(buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<String> {
    let bytes = ta_as_bytes(&buf)?;
    let slice = &bytes[start as usize..end as usize];
    Ok(String::from_utf8_lossy(slice).into_owned())
}

fn latin1_slice<'js>(buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<String> {
    let bytes = ta_as_bytes(&buf)?;
    Ok(bytes[start as usize..end as usize]
        .iter()
        .map(|&b| b as char)
        .collect())
}

fn ascii_slice<'js>(buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<String> {
    // Node's asciiSlice masks the high bit (treats as 7-bit).
    let bytes = ta_as_bytes(&buf)?;
    Ok(bytes[start as usize..end as usize]
        .iter()
        .map(|&b| (b & 0x7F) as char)
        .collect())
}

fn base64_slice<'js>(buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<String> {
    let bytes = ta_as_bytes(&buf)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(&bytes[start as usize..end as usize]))
}

fn base64url_slice<'js>(buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<String> {
    let bytes = ta_as_bytes(&buf)?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(&bytes[start as usize..end as usize]))
}

fn hex_slice<'js>(buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<String> {
    let bytes = ta_as_bytes(&buf)?;
    let slice = &bytes[start as usize..end as usize];
    let mut s = String::with_capacity(slice.len() * 2);
    for &b in slice {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0F) as usize] as char);
    }
    Ok(s)
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Returns the raw UTF-16 code units (as `Vec<u16>` → `Uint16Array`) so the
/// JS wrapper can build a string via `String.fromCharCode` and *preserve*
/// lone surrogates. Going through Rust's `String::from_utf16_lossy` would
/// silently replace them with U+FFFD, which breaks chunked utf16le decoding
/// in `string_decoder`.
fn ucs2_slice_codes<'js>(ctx: Ctx<'js>, buf: TypedArray<'js, u8>, start: u32, end: u32) -> Result<TypedArray<'js, u16>> {
    let bytes = ta_as_bytes(&buf)?;
    let slice = &bytes[start as usize..end as usize];
    let mut units: Vec<u16> = Vec::with_capacity(slice.len() / 2);
    let mut i = 0;
    while i + 1 < slice.len() {
        units.push(u16::from_le_bytes([slice[i], slice[i + 1]]));
        i += 2;
    }
    TypedArray::<u16>::new(ctx, units)
}

// Writes ---------------------------------------------------------------

// Node's binding-level Write signature is (buf, string, offset, length).
fn base64_write<'js>(
    buf: TypedArray<'js, u8>,
    string: String,
    offset: u32,
    length: u32,
) -> Result<u32> {
    let cleaned: String = string.chars().filter(|c| !c.is_whitespace()).collect();
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(cleaned.as_bytes()))
        .unwrap_or_default();
    let dst = ta_as_bytes_mut(&buf)?;
    let off = offset as usize;
    let max = (dst.len().saturating_sub(off))
        .min(length as usize)
        .min(decoded.len());
    dst[off..off + max].copy_from_slice(&decoded[..max]);
    Ok(max as u32)
}

fn base64url_write<'js>(
    buf: TypedArray<'js, u8>,
    string: String,
    offset: u32,
    length: u32,
) -> Result<u32> {
    let cleaned: String = string.chars().filter(|c| !c.is_whitespace()).collect();
    let decoded = base64::engine::general_purpose::URL_SAFE
        .decode(cleaned.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(cleaned.as_bytes()))
        .or_else(|_| base64::engine::general_purpose::STANDARD.decode(cleaned.as_bytes()))
        .unwrap_or_default();
    let dst = ta_as_bytes_mut(&buf)?;
    let off = offset as usize;
    let max = (dst.len().saturating_sub(off))
        .min(length as usize)
        .min(decoded.len());
    dst[off..off + max].copy_from_slice(&decoded[..max]);
    Ok(max as u32)
}

fn hex_write<'js>(
    buf: TypedArray<'js, u8>,
    string: String,
    offset: u32,
    length: u32,
) -> Result<u32> {
    let dst = ta_as_bytes_mut(&buf)?;
    let off = offset as usize;
    let cap = dst.len().saturating_sub(off).min(length as usize);
    let src = string.as_bytes();
    let mut written = 0usize;
    let mut i = 0usize;
    while i + 1 < src.len() && written < cap {
        let hi = hex_digit(src[i]);
        let lo = hex_digit(src[i + 1]);
        match (hi, lo) {
            (Some(h), Some(l)) => {
                dst[off + written] = (h << 4) | l;
                written += 1;
                i += 2;
            }
            _ => break,
        }
    }
    Ok(written as u32)
}

fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

// -------------------------------------------------------------------------
// atob / btoa (legacy aliases for base64 over latin1 strings)
// -------------------------------------------------------------------------

fn atob<'js>(ctx: Ctx<'js>, s: String) -> Result<String> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(cleaned.as_bytes()))
        .map_err(|_| {
            let _ = throw::<()>(&ctx, "atob: invalid base64");
            Error::new_from_js("string", "base64")
        })?;
    Ok(decoded.iter().map(|&b| b as char).collect())
}

fn btoa<'js>(ctx: Ctx<'js>, s: String) -> Result<String> {
    // Input must be latin1 (chars 0..=255). Reject otherwise (Node throws
    // InvalidCharacterError; we just throw a plain Error).
    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) > 0xFF {
            return throw(&ctx, "btoa: argument contains non-latin1 chars");
        }
        bytes.push(c as u8);
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(&bytes))
}

// -------------------------------------------------------------------------
// compare / compareOffset / copy / fill
// -------------------------------------------------------------------------

fn compare<'js>(a: TypedArray<'js, u8>, b: TypedArray<'js, u8>) -> Result<i32> {
    let ab = ta_as_bytes(&a)?;
    let bb = ta_as_bytes(&b)?;
    Ok(cmp_i32(ab, bb))
}

fn compare_offset<'js>(
    source: TypedArray<'js, u8>,
    target: TypedArray<'js, u8>,
    target_start: u32,
    source_start: u32,
    target_end: u32,
    source_end: u32,
) -> Result<i32> {
    let s = ta_as_bytes(&source)?;
    let t = ta_as_bytes(&target)?;
    let s_slice = &s[source_start as usize..source_end as usize];
    let t_slice = &t[target_start as usize..target_end as usize];
    Ok(cmp_i32(s_slice, t_slice))
}

fn cmp_i32(a: &[u8], b: &[u8]) -> i32 {
    match a.cmp(b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

fn copy<'js>(
    source: TypedArray<'js, u8>,
    target: TypedArray<'js, u8>,
    target_start: u32,
    source_start: u32,
    source_end: u32,
) -> Result<u32> {
    let src = ta_as_bytes(&source)?;
    let dst = ta_as_bytes_mut(&target)?;
    let ts = target_start as usize;
    let ss = source_start as usize;
    let se = source_end as usize;
    let avail_src = se.saturating_sub(ss).min(src.len().saturating_sub(ss));
    let avail_dst = dst.len().saturating_sub(ts);
    let n = avail_src.min(avail_dst);
    // memmove semantics — handle overlap when source and target are the same.
    let src_ptr = src[ss..].as_ptr();
    let dst_ptr = dst[ts..].as_mut_ptr();
    unsafe {
        std::ptr::copy(src_ptr, dst_ptr, n);
    }
    Ok(n as u32)
}

/// `fill(buf, value, offset, end, encoding)` — repeat `value`'s bytes into
/// `buf[offset..end]`. `value` may be a Uint8Array (bytes already encoded
/// by the JS layer) or a single number 0..=255.
fn fill<'js>(
    ctx: Ctx<'js>,
    buf: TypedArray<'js, u8>,
    value: Value<'js>,
    offset: u32,
    end: u32,
    _encoding: Value<'js>,
) -> Result<()> {
    let dst = ta_as_bytes_mut(&buf)?;
    let o = (offset as usize).min(dst.len());
    let e = (end as usize).min(dst.len());
    if e <= o {
        return Ok(());
    }
    // Pattern bytes
    let pattern: Vec<u8> = if let Some(n) = value.as_number() {
        vec![(n as i64).rem_euclid(256) as u8]
    } else if value.is_int() {
        vec![(value.as_int().unwrap() as i64).rem_euclid(256) as u8]
    } else if let Some(s) = value.clone().into_string() {
        s.to_string()?.into_bytes()
    } else if value.is_object() {
        // First try to view as a TypedArray (Uint8Array / Buffer). If that
        // fails, treat as a generic array-like (plain Array, Array-of-bytes).
        let obj = value.into_object().unwrap();
        match TypedArray::<'js, u8>::from_object(obj.clone()) {
            Ok(ta) => ta_as_bytes(&ta)?.to_vec(),
            Err(_) => {
                // Iterate as JS array-like.
                let len_v = obj.get::<_, Value<'js>>("length")
                    .ok().and_then(|v| v.as_number()).unwrap_or(0.0);
                let len = len_v as usize;
                let mut bytes = Vec::with_capacity(len);
                for i in 0..len {
                    let v = obj.get::<_, Value<'js>>(i as u32).ok();
                    let b = v.and_then(|x| x.as_number()).unwrap_or(0.0) as i64;
                    bytes.push(b.rem_euclid(256) as u8);
                }
                bytes
            }
        }
    } else {
        return throw(&ctx, "fill: unsupported value type");
    };
    // Empty pattern → zero-fill the range. Matches Node's behavior for
    // \`Buffer.alloc(n, '')\` and \`Buffer.alloc(n, [])\`.
    if pattern.is_empty() {
        for i in o..e { dst[i] = 0; }
        return Ok(());
    }
    let mut i = o;
    let mut pi = 0;
    while i < e {
        dst[i] = pattern[pi];
        i += 1;
        pi = (pi + 1) % pattern.len();
    }
    Ok(())
}

// -------------------------------------------------------------------------
// isAscii / isUtf8
// -------------------------------------------------------------------------

fn is_ascii<'js>(buf: TypedArray<'js, u8>) -> Result<bool> {
    Ok(ta_as_bytes(&buf)?.iter().all(|&b| b < 0x80))
}

fn is_utf8<'js>(buf: TypedArray<'js, u8>) -> Result<bool> {
    Ok(std::str::from_utf8(ta_as_bytes(&buf)?).is_ok())
}

// -------------------------------------------------------------------------
// indexOf*
// -------------------------------------------------------------------------

fn index_of_number<'js>(
    buf: TypedArray<'js, u8>,
    val: u32, // Node always calls with `val >>> 0` → uint32
    byte_offset: f64,
    dir: bool,
    end: Option<f64>,
) -> Result<i32> {
    let full = ta_as_bytes(&buf)?;
    // Truncate at `end`; negative end → empty range (no match possible).
    let bytes: &[u8] = match end {
        Some(e) if e.is_finite() => {
            if e <= 0.0 { &full[..0] } else { &full[..(e as usize).min(full.len())] }
        }
        _ => full,
    };
    let needle = (val % 256) as u8;
    let len = bytes.len();
    if len == 0 {
        return Ok(-1);
    }
    // Reverse search: offset before the buffer start → no positions to scan.
    if !dir && !byte_offset.is_nan() {
        if byte_offset == f64::NEG_INFINITY { return Ok(-1); }
        if byte_offset < 0.0 && (-byte_offset) > len as f64 { return Ok(-1); }
    }
    let start = clamp_offset_f64(byte_offset, len, dir);
    if dir {
        for i in start..len {
            if bytes[i] == needle {
                return Ok(i as i32);
            }
        }
    } else {
        let s = start.min(len.saturating_sub(1));
        for i in (0..=s).rev() {
            if bytes[i] == needle {
                return Ok(i as i32);
            }
        }
    }
    Ok(-1)
}

fn index_of_buffer<'js>(
    haystack: TypedArray<'js, u8>,
    needle: TypedArray<'js, u8>,
    byte_offset: f64,
    encoding: Value<'js>,
    dir: bool,
    end: Option<f64>,
) -> Result<i32> {
    let full = ta_as_bytes(&haystack)?;
    let h: &[u8] = match end {
        Some(e) if e.is_finite() => {
            if e <= 0.0 { &[] } else { &full[..(e as usize).min(full.len())] }
        }
        _ => full,
    };
    let n = ta_as_bytes(&needle)?;
    let unit = encoding_alignment(&encoding);
    Ok(search_bytes_aligned(h, n, byte_offset, dir, unit))
}

/// Returns the search alignment in bytes for a given encoding hint.
/// ucs2/utf16le require positions to be even (2-byte aligned). Accepts both
/// the numeric form (Node's binding internal enum) and string form.
fn encoding_alignment<'js>(encoding: &Value<'js>) -> usize {
    if let Some(s) = encoding.as_string() {
        if let Ok(s) = s.to_string() {
            return match s.as_str() {
                "utf16le" | "utf-16le" | "ucs2" | "ucs-2" => 2,
                _ => 1,
            };
        }
    }
    // Numeric: 1 = utf16le per encodingsMap.
    if let Some(n) = encoding.as_int() {
        return if n == 1 { 2 } else { 1 };
    }
    if let Some(n) = encoding.as_float() {
        return if (n as i32) == 1 { 2 } else { 1 };
    }
    1
}

fn index_of_string<'js>(
    haystack: TypedArray<'js, u8>,
    needle: String,
    byte_offset: f64,
    encoding: Value<'js>,
    dir: bool,
    end: Option<f64>,
) -> Result<i32> {
    // encoding can be a number (Node's numeric enum) OR a string. Map either
    // shape into a stable name we can switch on.
    let encoding: String = if let Some(s) = encoding.as_string() {
        s.to_string()?
    } else if let Some(n) = encoding.as_int() {
        match n { 0 => "utf8", 1 => "utf16le", 2 => "latin1",
                  3 => "base64", 4 => "base64url", 5 => "ascii",
                  6 => "hex", _ => "utf8" }.to_string()
    } else if let Some(n) = encoding.as_float() {
        match n as i32 { 0 => "utf8", 1 => "utf16le", 2 => "latin1",
                          3 => "base64", 4 => "base64url", 5 => "ascii",
                          6 => "hex", _ => "utf8" }.to_string()
    } else {
        "utf8".to_string()
    };
    let full = ta_as_bytes(&haystack)?;
    let h: &[u8] = match end {
        Some(e) if e.is_finite() => {
            if e <= 0.0 { &[] } else { &full[..(e as usize).min(full.len())] }
        }
        _ => full,
    };
    let unit = match encoding.as_str() {
        "ucs2" | "ucs-2" | "utf16le" | "utf-16le" => 2,
        _ => 1,
    };
    let n_bytes: Vec<u8> = match encoding.as_str() {
        "utf8" | "utf-8" => needle.into_bytes(),
        "latin1" | "binary" => needle.chars().map(|c| c as u8).collect(),
        "ascii" => needle.chars().map(|c| (c as u8) & 0x7F).collect(),
        "ucs2" | "ucs-2" | "utf16le" | "utf-16le" => {
            let mut v = Vec::with_capacity(needle.len() * 2);
            for u in needle.encode_utf16() {
                v.extend_from_slice(&u.to_le_bytes());
            }
            v
        }
        "hex" => {
            let mut v = Vec::with_capacity(needle.len() / 2);
            let s = needle.as_bytes();
            let mut i = 0;
            while i + 1 < s.len() {
                match (hex_digit(s[i]), hex_digit(s[i + 1])) {
                    (Some(h), Some(l)) => {
                        v.push((h << 4) | l);
                        i += 2;
                    }
                    _ => break,
                }
            }
            v
        }
        "base64" => base64::engine::general_purpose::STANDARD
            .decode(needle.as_bytes())
            .unwrap_or_default(),
        _ => needle.into_bytes(),
    };
    Ok(search_bytes_aligned(h, &n_bytes, byte_offset, dir, unit))
}

// Convert a JS-number-style offset (may be NaN/±Infinity/negative/large) to a
// clamped [0, len] index. Direction matters for NaN: forward → 0, reverse → len.
fn clamp_offset_f64(byte_offset: f64, len: usize, dir: bool) -> usize {
    if byte_offset.is_nan() {
        return if dir { 0 } else { len };
    }
    if byte_offset == f64::INFINITY { return len; }
    if byte_offset == f64::NEG_INFINITY { return 0; }
    if byte_offset < 0.0 {
        let from_end = (-byte_offset) as usize;
        len.saturating_sub(from_end)
    } else {
        (byte_offset as usize).min(len)
    }
}

fn search_bytes_aligned(haystack: &[u8], needle: &[u8], byte_offset: f64, dir: bool, unit: usize) -> i32 {
    let len = haystack.len();
    // Reverse search: if the requested offset points before the buffer
    // start, there are no positions to scan — Node returns -1 (not 0).
    if !dir && !byte_offset.is_nan() {
        if byte_offset == f64::NEG_INFINITY { return -1; }
        if byte_offset < 0.0 && (-byte_offset) > len as f64 { return -1; }
    }
    if needle.is_empty() {
        return clamp_offset_f64(byte_offset, len, dir) as i32;
    }
    if needle.len() > len || needle.len() % unit != 0 {
        return -1;
    }
    let last = len - needle.len();
    let mut start = clamp_offset_f64(byte_offset, len, dir);
    if unit > 1 {
        // Round forward to next aligned position; round reverse down.
        if dir {
            let rem = start % unit;
            if rem != 0 { start += unit - rem; }
        } else {
            start -= start % unit;
        }
    }
    if dir {
        if start > last { return -1; }
        let mut i = start;
        while i <= last {
            if &haystack[i..i + needle.len()] == needle {
                return i as i32;
            }
            i += unit;
        }
    } else {
        let mut i = start.min(last);
        i -= i % unit;
        loop {
            if &haystack[i..i + needle.len()] == needle {
                return i as i32;
            }
            if i < unit { break; }
            i -= unit;
        }
    }
    -1
}

// search_bytes (unit=1) is just search_bytes_aligned with unit=1.

// -------------------------------------------------------------------------
// swap16 / swap32 / swap64 (in-place byte swap)
// -------------------------------------------------------------------------

fn swap_n<'js>(ctx: &Ctx<'js>, buf: &TypedArray<'js, u8>, unit: usize) -> Result<()> {
    let dst = ta_as_bytes_mut(buf)?;
    if dst.len() % unit != 0 {
        return throw(ctx, &format!("swap{}: buffer length not a multiple of {unit}", unit * 8));
    }
    let mut i = 0;
    while i < dst.len() {
        dst[i..i + unit].reverse();
        i += unit;
    }
    Ok(())
}

fn swap16<'js>(ctx: Ctx<'js>, buf: TypedArray<'js, u8>) -> Result<()> {
    swap_n(&ctx, &buf, 2)
}
fn swap32<'js>(ctx: Ctx<'js>, buf: TypedArray<'js, u8>) -> Result<()> {
    swap_n(&ctx, &buf, 4)
}
fn swap64<'js>(ctx: Ctx<'js>, buf: TypedArray<'js, u8>) -> Result<()> {
    swap_n(&ctx, &buf, 8)
}

// -------------------------------------------------------------------------
// Binding object
// -------------------------------------------------------------------------

pub fn make(ctx: Ctx<'_>) -> Result<Object<'_>> {
    let b = Object::new(ctx.clone())?;

    // utf8WriteStatic and byteLengthUtf8 are implemented in JS rather than
    // Rust because QuickJS exposes JS strings to Rust as CESU-8 — lone
    // surrogates emit 0xED 0xA0..0xBF byte sequences that `Rust` String /
    // &str conversion rejects as "invalid UTF-8". The web-spec rule is to
    // replace lone surrogates with U+FFFD, which we do in JS where we can
    // iterate UTF-16 code units directly.
    let js_helpers: rquickjs::Object<'_> = ctx.eval(r#"
        ({
          // ucs2Write is also JS-implemented: it stores the JS string's
          // UTF-16 code units verbatim (including lone surrogates), which
          // Rust's `String` type cannot represent.
          ucs2Write: function(buf, str, offset, length) {
            offset |= 0;
            length |= 0;
            const cap = Math.min(buf.length - offset, length);
            let written = 0;
            for (let i = 0; i < str.length && written + 2 <= cap; i++) {
              const u = str.charCodeAt(i);
              buf[offset + written]     = u & 0xFF;
              buf[offset + written + 1] = (u >> 8) & 0xFF;
              written += 2;
            }
            return written;
          },
          utf8WriteStatic: function _JS_UTF8_WRITE_STATIC(buf, str, offset, length) {
            offset |= 0;
            length |= 0;
            const cap = Math.min(buf.length - offset, length);
            let written = 0;
            for (let i = 0; i < str.length; i++) {
              let cp = str.charCodeAt(i);
              if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (i + 1 < str.length) {
                  const next = str.charCodeAt(i + 1);
                  if (next >= 0xDC00 && next <= 0xDFFF) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (next - 0xDC00);
                    i++;
                  } else {
                    cp = 0xFFFD;
                  }
                } else {
                  cp = 0xFFFD;
                }
              } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                cp = 0xFFFD;
              }
              const nbytes = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
              if (written + nbytes > cap) break;
              if (nbytes === 1) {
                buf[offset + written] = cp;
              } else if (nbytes === 2) {
                buf[offset + written]     = 0xC0 | (cp >> 6);
                buf[offset + written + 1] = 0x80 | (cp & 0x3F);
              } else if (nbytes === 3) {
                buf[offset + written]     = 0xE0 | (cp >> 12);
                buf[offset + written + 1] = 0x80 | ((cp >> 6) & 0x3F);
                buf[offset + written + 2] = 0x80 | (cp & 0x3F);
              } else {
                buf[offset + written]     = 0xF0 | (cp >> 18);
                buf[offset + written + 1] = 0x80 | ((cp >> 12) & 0x3F);
                buf[offset + written + 2] = 0x80 | ((cp >> 6) & 0x3F);
                buf[offset + written + 3] = 0x80 | (cp & 0x3F);
              }
              written += nbytes;
            }
            return written;
          },
          byteLengthUtf8: function(str) {
            let n = 0;
            for (let i = 0; i < str.length; i++) {
              let cp = str.charCodeAt(i);
              if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < str.length) {
                const next = str.charCodeAt(i + 1);
                if (next >= 0xDC00 && next <= 0xDFFF) {
                  n += 4;
                  i++;
                  continue;
                }
              }
              if (cp >= 0xD800 && cp <= 0xDFFF) { n += 3; continue; } // lone surrogate -> U+FFFD
              n += cp < 0x80 ? 1 : cp < 0x800 ? 2 : 3;
            }
            return n;
          },
        })
    "#)?;
    b.set("utf8WriteStatic", js_helpers.get::<_, rquickjs::Function>("utf8WriteStatic")?)?;
    b.set("byteLengthUtf8",  js_helpers.get::<_, rquickjs::Function>("byteLengthUtf8")?)?;
    b.set("ucs2Write",       js_helpers.get::<_, rquickjs::Function>("ucs2Write")?)?;

    b.set("kMaxLength", u32::MAX as u64)?;
    b.set("kStringMaxLength", (1u64 << 30) - 1)?;

    b.set("utf8Slice",      Func::from(utf8_slice))?;
    b.set("asciiSlice",     Func::from(ascii_slice))?;
    b.set("latin1Slice",    Func::from(latin1_slice))?;
    b.set("base64Slice",    Func::from(base64_slice))?;
    b.set("base64urlSlice", Func::from(base64url_slice))?;
    b.set("hexSlice",       Func::from(hex_slice))?;
    // ucs2Slice: thin JS wrapper around the raw u16-codes function so we can
    // preserve lone surrogates. Going Rust→String would force lossy UTF-16
    // decoding (Rust strings are valid UTF-8; lone surrogates → U+FFFD).
    b.set("_ucs2SliceCodes", Func::from(ucs2_slice_codes))?;
    let wrap: rquickjs::Function = ctx.eval(r#"
      (function (binding) {
        const codesFn = binding._ucs2SliceCodes;
        const fromCharCode = String.fromCharCode;
        binding.ucs2Slice = function ucs2Slice(buf, start, end) {
          const codes = codesFn(buf, start | 0, end | 0);
          // Apply chunks to avoid 'apply' arg-count limits.
          let s = '';
          const CHUNK = 0x4000;
          for (let i = 0; i < codes.length; i += CHUNK) {
            s += fromCharCode.apply(null, codes.subarray(i, Math.min(codes.length, i + CHUNK)));
          }
          return s;
        };
      })
    "#)?;
    let _: () = wrap.call((b.clone(),))?;

    b.set("base64Write",    Func::from(base64_write))?;
    b.set("base64urlWrite", Func::from(base64url_write))?;
    b.set("hexWrite",       Func::from(hex_write))?;
    // ucs2Write is JS-implemented above (next to utf8 helpers).

    b.set("atob",           Func::from(atob))?;
    b.set("btoa",           Func::from(btoa))?;

    b.set("compare",        Func::from(compare))?;
    b.set("compareOffset",  Func::from(compare_offset))?;
    b.set("copy",           Func::from(copy))?;
    b.set("fill",           Func::from(fill))?;

    b.set("isAscii",        Func::from(is_ascii))?;
    b.set("isUtf8",         Func::from(is_utf8))?;

    b.set("indexOfBuffer",  Func::from(index_of_buffer))?;
    b.set("indexOfNumber",  Func::from(index_of_number))?;
    b.set("indexOfString",  Func::from(index_of_string))?;

    b.set("swap16",         Func::from(swap16))?;
    b.set("swap32",         Func::from(swap32))?;
    b.set("swap64",         Func::from(swap64))?;

    // asciiWriteStatic/latin1WriteStatic don't have the surrogate issue —
    // they take latin1/ascii chars (truncated to 1 byte each). utf8 is
    // attached above via the JS helpers block.
    b.set("asciiWriteStatic",  Func::from(ascii_write_static))?;
    b.set("latin1WriteStatic", Func::from(latin1_write_static))?;
    b.set("createUnsafeArrayBuffer", Func::from(create_unsafe_array_buffer))?;
    b.set("setDetachKey",      Func::from(set_detach_key))?;

    let _ = ctx; // ctx consumed via Func::from registrations
    Ok(b)
}
