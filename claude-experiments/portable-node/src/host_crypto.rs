//! Portable crypto primitives — exposed as `__host.crypto`.
//!
//! Cryptographic hashes (SHA-1, SHA-256, SHA-512, MD5), HMAC, OS-level
//! CSPRNG, and constant-time equality. The JS shim at `shimSources['crypto']`
//! wraps these into Node's `crypto.createHash` / `createHmac` /
//! `randomBytes` / `randomUUID` / `timingSafeEqual` surface.
//!
//! Every host language ecosystem has equivalents (Go: `crypto/sha256`,
//! `crypto/hmac`, `crypto/rand`; Python: `hashlib`, `hmac`, `secrets`;
//! Java: `MessageDigest`, `Mac`, `SecureRandom`; C: OpenSSL / libsodium).

use digest::Digest;
use hmac::{Hmac, Mac};
use rquickjs::function::Func;
use rquickjs::{Ctx, Object, Result, TypedArray};

pub fn install<'js>(ctx: Ctx<'js>, host: &Object<'js>) -> Result<()> {
    let c = Object::new(ctx.clone())?;
    c.set("random_bytes",      Func::from(random_bytes))?;
    c.set("hash",              Func::from(hash))?;
    c.set("hmac",              Func::from(hmac_fn))?;
    c.set("timing_safe_equal", Func::from(timing_safe_equal))?;
    c.set("supported_hashes",  Func::from(supported_hashes))?;
    host.set("crypto", c)?;
    Ok(())
}

fn random_bytes<'js>(ctx: Ctx<'js>, n: u32) -> Result<TypedArray<'js, u8>> {
    let n = (n as usize).min(64 * 1024 * 1024); // 64MB cap
    let mut buf = vec![0u8; n];
    if getrandom::getrandom(&mut buf).is_err() {
        let err = rquickjs::Exception::from_message(ctx.clone(), "random_bytes: OS RNG failure")?;
        return Err(ctx.throw(err.into_value()));
    }
    TypedArray::<u8>::new(ctx, buf)
}

/// `hash(algorithm, data)` → Uint8Array(digest).
fn hash<'js>(
    ctx: Ctx<'js>,
    alg: String,
    data: TypedArray<'js, u8>,
) -> Result<TypedArray<'js, u8>> {
    let bytes = data.as_bytes().unwrap_or(&[]);
    let out: Vec<u8> = match alg.to_ascii_lowercase().as_str() {
        "sha1"     => sha1::Sha1::digest(bytes).to_vec(),
        "sha224"   => sha2::Sha224::digest(bytes).to_vec(),
        "sha256"   => sha2::Sha256::digest(bytes).to_vec(),
        "sha384"   => sha2::Sha384::digest(bytes).to_vec(),
        "sha512"   => sha2::Sha512::digest(bytes).to_vec(),
        "md5"      => md5::Md5::digest(bytes).to_vec(),
        other => {
            let msg = format!("hash: unsupported algorithm {other:?}");
            let err = rquickjs::Exception::from_message(ctx.clone(), &msg)?;
            return Err(ctx.throw(err.into_value()));
        }
    };
    TypedArray::<u8>::new(ctx, out)
}

/// `hmac(algorithm, key, data)` → Uint8Array(MAC).
fn hmac_fn<'js>(
    ctx: Ctx<'js>,
    alg: String,
    key: TypedArray<'js, u8>,
    data: TypedArray<'js, u8>,
) -> Result<TypedArray<'js, u8>> {
    let k = key.as_bytes().unwrap_or(&[]);
    let d = data.as_bytes().unwrap_or(&[]);
    let out: Vec<u8> = match alg.to_ascii_lowercase().as_str() {
        "sha1"   => { let mut m = <Hmac<sha1::Sha1>   as KeyInit>::new_from_slice(k).unwrap(); m.update(d); m.finalize().into_bytes().to_vec() }
        "sha224" => { let mut m = <Hmac<sha2::Sha224> as KeyInit>::new_from_slice(k).unwrap(); m.update(d); m.finalize().into_bytes().to_vec() }
        "sha256" => { let mut m = <Hmac<sha2::Sha256> as KeyInit>::new_from_slice(k).unwrap(); m.update(d); m.finalize().into_bytes().to_vec() }
        "sha384" => { let mut m = <Hmac<sha2::Sha384> as KeyInit>::new_from_slice(k).unwrap(); m.update(d); m.finalize().into_bytes().to_vec() }
        "sha512" => { let mut m = <Hmac<sha2::Sha512> as KeyInit>::new_from_slice(k).unwrap(); m.update(d); m.finalize().into_bytes().to_vec() }
        "md5"    => { let mut m = <Hmac<md5::Md5>     as KeyInit>::new_from_slice(k).unwrap(); m.update(d); m.finalize().into_bytes().to_vec() }
        other => {
            let msg = format!("hmac: unsupported algorithm {other:?}");
            let err = rquickjs::Exception::from_message(ctx.clone(), &msg)?;
            return Err(ctx.throw(err.into_value()));
        }
    };
    TypedArray::<u8>::new(ctx, out)
}

use hmac::digest::KeyInit;

fn timing_safe_equal<'js>(a: TypedArray<'js, u8>, b: TypedArray<'js, u8>) -> bool {
    let ab = a.as_bytes().unwrap_or(&[]);
    let bb = b.as_bytes().unwrap_or(&[]);
    if ab.len() != bb.len() { return false; }
    // Constant-time XOR-and-OR comparison.
    let mut diff: u8 = 0;
    for i in 0..ab.len() {
        diff |= ab[i] ^ bb[i];
    }
    diff == 0
}

fn supported_hashes() -> Vec<String> {
    ["sha1","sha224","sha256","sha384","sha512","md5"]
        .iter().map(|s| s.to_string()).collect()
}
