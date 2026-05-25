//! Portable host interface — the irreducible OS primitive surface.
//!
//! This is what any host (Rust, C, Go, browser shim, ...) must provide.
//! Everything else in the Node stdlib is built on top of these calls in JS.
//! The JS layer reaches in via `__host.os.*`, `__host.file.*`, etc.
//!
//! Today: just `os.*` since that's what we're proving with `node:os`.
//! Tier 1 (file + socket + process + time + random) gets added per the
//! roadmap; each section here grows additively.

use rquickjs::function::Func;
use rquickjs::{Array, Ctx, Object, Result};

/// Install `globalThis.__host` carrying every host primitive.
pub fn install<'js>(ctx: Ctx<'js>) -> Result<()> {
    let host = Object::new(ctx.clone())?;
    host.set("os", make_os(ctx.clone())?)?;
    host.set("process", make_process(ctx.clone())?)?;
    host.set("file", make_file(ctx.clone())?)?;
    host.set("zlib", make_zlib(ctx.clone())?)?;
    host.set("time", make_time(ctx.clone())?)?;
    // Tier-2 async I/O (skeleton; ops land in subsequent commits).
    crate::io_loop::install(ctx.clone(), &host)?;
    // HTTP/1.x parser (robust, RFC-correct via httparse + a body-framing
    // state machine — see src/http_parser.rs).
    crate::http_parser::install(ctx.clone(), &host)?;
    // Cryptographic primitives (SHA family, HMAC, CSPRNG, constant-time eq).
    crate::host_crypto::install(ctx.clone(), &host)?;
    ctx.globals().set("__host", host)?;
    Ok(())
}

// =========================================================================
// time.* — the Tier-1 event-loop primitive. Just two functions; every
// language ecosystem has equivalents.
// =========================================================================

fn make_time<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let t = Object::new(ctx.clone())?;
    // now_ms — monotonic-ish ms since Unix epoch (we use system clock for
    // simplicity; real Node uses libuv's monotonic clock for timers).
    t.set("now_ms", Func::from(|| -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }))?;
    // sleep_ms — blocking sleep. Capped at 1s per call so the outer loop
    // stays responsive (a host with signal handling could check during
    // sleep; we don't).
    t.set("sleep_ms", Func::from(|ms: f64| {
        if ms <= 0.0 { return; }
        let ms_capped = ms.min(1000.0) as u64;
        std::thread::sleep(std::time::Duration::from_millis(ms_capped));
    }))?;
    Ok(t)
}

// =========================================================================
// zlib.* — compression primitives. Built on the `flate2` crate.
// All sync; streams happen at the JS level on top.
// =========================================================================

fn make_zlib<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    use flate2::{Compression, write::{DeflateEncoder, GzEncoder, ZlibEncoder},
                 read::{DeflateDecoder, GzDecoder, ZlibDecoder}};
    use std::io::{Read, Write};

    let z = Object::new(ctx.clone())?;

    // deflate(bytes, level) → bytes  (raw deflate, no zlib/gzip header)
    z.set("deflate_raw", Func::from(|input: rquickjs::TypedArray<'_, u8>, level: u32| -> Vec<u8> {
        let mut enc = DeflateEncoder::new(Vec::new(),
            Compression::new(level.min(9)));
        let _ = enc.write_all(input.as_bytes().unwrap_or(&[]));
        enc.finish().unwrap_or_default()
    }))?;
    z.set("inflate_raw", Func::from(|input: rquickjs::TypedArray<'_, u8>| -> Vec<u8> {
        let mut dec = DeflateDecoder::new(input.as_bytes().unwrap_or(&[]));
        let mut out = Vec::new();
        let _ = dec.read_to_end(&mut out);
        out
    }))?;

    // zlib-wrapped (RFC 1950 — has 2-byte header + adler32 trailer)
    z.set("deflate", Func::from(|input: rquickjs::TypedArray<'_, u8>, level: u32| -> Vec<u8> {
        let mut enc = ZlibEncoder::new(Vec::new(),
            Compression::new(level.min(9)));
        let _ = enc.write_all(input.as_bytes().unwrap_or(&[]));
        enc.finish().unwrap_or_default()
    }))?;
    z.set("inflate", Func::from(|input: rquickjs::TypedArray<'_, u8>| -> Vec<u8> {
        let mut dec = ZlibDecoder::new(input.as_bytes().unwrap_or(&[]));
        let mut out = Vec::new();
        let _ = dec.read_to_end(&mut out);
        out
    }))?;

    // gzip-wrapped (RFC 1952)
    z.set("gzip", Func::from(|input: rquickjs::TypedArray<'_, u8>, level: u32| -> Vec<u8> {
        let mut enc = GzEncoder::new(Vec::new(),
            Compression::new(level.min(9)));
        let _ = enc.write_all(input.as_bytes().unwrap_or(&[]));
        enc.finish().unwrap_or_default()
    }))?;
    z.set("gunzip", Func::from(|input: rquickjs::TypedArray<'_, u8>| -> Vec<u8> {
        let mut dec = GzDecoder::new(input.as_bytes().unwrap_or(&[]));
        let mut out = Vec::new();
        let _ = dec.read_to_end(&mut out);
        out
    }))?;

    // CRC-32 (handy enough that flate2 exposes it as a free function).
    z.set("crc32", Func::from(|input: rquickjs::TypedArray<'_, u8>| -> u32 {
        let mut hasher = flate2::Crc::new();
        hasher.update(input.as_bytes().unwrap_or(&[]));
        hasher.sum()
    }))?;

    Ok(z)
}

// =========================================================================
// file.* — sync I/O. The Tier-1 surface for `node:fs`.
// Host signatures intentionally stay close to POSIX semantics so any
// implementer (Rust, C, Go, Python, ...) can map them 1:1.
// =========================================================================

fn make_file<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let f = Object::new(ctx.clone())?;

    // open(path, flags, mode) → fd ; throws Error with code on failure
    f.set("open", Func::from(file_open))?;
    f.set("close", Func::from(file_close))?;
    f.set("read", Func::from(file_read))?;
    f.set("write", Func::from(file_write))?;
    f.set("stat", Func::from(file_stat))?;
    f.set("fstat", Func::from(file_fstat))?;
    f.set("lstat", Func::from(file_lstat))?;
    f.set("readdir", Func::from(file_readdir))?;
    f.set("realpath", Func::from(file_realpath))?;
    f.set("unlink", Func::from(file_unlink))?;
    f.set("mkdir", Func::from(file_mkdir))?;
    f.set("rmdir", Func::from(file_rmdir))?;
    f.set("rename", Func::from(file_rename))?;
    f.set("access", Func::from(file_access))?;
    // Convenience: read entire file into a UTF-8 string (errors as a Node
    // Error with code+path). Used by the CommonJS resolver.
    f.set("read_to_string", Func::from(file_read_to_string))?;
    // exists(path) → boolean. Stat-or-false; doesn't throw.
    f.set("exists",         Func::from(file_exists))?;
    f.set("is_file",        Func::from(file_is_file))?;
    f.set("is_dir",         Func::from(file_is_dir))?;

    // Flag constants — match POSIX values on unix. JS shim composes these
    // into the form Node's fs expects.
    let flags = Object::new(ctx.clone())?;
    flags.set("O_RDONLY", libc::O_RDONLY)?;
    flags.set("O_WRONLY", libc::O_WRONLY)?;
    flags.set("O_RDWR", libc::O_RDWR)?;
    flags.set("O_CREAT", libc::O_CREAT)?;
    flags.set("O_EXCL", libc::O_EXCL)?;
    flags.set("O_TRUNC", libc::O_TRUNC)?;
    flags.set("O_APPEND", libc::O_APPEND)?;
    flags.set("O_NOFOLLOW", libc::O_NOFOLLOW)?;
    flags.set("O_DIRECTORY", libc::O_DIRECTORY)?;
    flags.set("O_NONBLOCK", libc::O_NONBLOCK)?;
    flags.set("F_OK", libc::F_OK)?;
    flags.set("R_OK", libc::R_OK)?;
    flags.set("W_OK", libc::W_OK)?;
    flags.set("X_OK", libc::X_OK)?;
    f.set("flags", flags)?;

    Ok(f)
}

fn throw_io<'js, T>(ctx: &Ctx<'js>, syscall: &str, path: &str) -> Result<T> {
    let errno = std::io::Error::last_os_error();
    let code = errno_to_code(errno.raw_os_error().unwrap_or(0));
    let msg = format!(
        "{code}: {}, {syscall} '{path}'",
        errno.to_string()
    );
    let err_value: rquickjs::Value = ctx.eval(format!(
        r#"(function () {{
            const e = new Error({msg:?});
            e.code = {code:?};
            e.syscall = {syscall:?};
            e.path = {path:?};
            e.errno = {errno_num};
            return e;
        }})()"#,
        errno_num = -(errno.raw_os_error().unwrap_or(0).abs())
    ).as_str())?;
    Err(ctx.throw(err_value))
}

fn errno_to_code(errno: i32) -> &'static str {
    match errno {
        libc::ENOENT => "ENOENT",
        libc::EACCES => "EACCES",
        libc::EEXIST => "EEXIST",
        libc::EISDIR => "EISDIR",
        libc::ENOTDIR => "ENOTDIR",
        libc::ENOTEMPTY => "ENOTEMPTY",
        libc::EMFILE => "EMFILE",
        libc::ENFILE => "ENFILE",
        libc::EBADF => "EBADF",
        libc::EINVAL => "EINVAL",
        libc::EPERM => "EPERM",
        libc::EIO => "EIO",
        libc::ELOOP => "ELOOP",
        libc::ENAMETOOLONG => "ENAMETOOLONG",
        libc::ENOSPC => "ENOSPC",
        libc::ENOSYS => "ENOSYS",
        libc::ENOMEM => "ENOMEM",
        _ => "UNKNOWN",
    }
}

fn cstring(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap_or_else(|_| std::ffi::CString::new("").unwrap())
}

fn file_open<'js>(ctx: Ctx<'js>, path: String, flags: i32, mode: i32) -> Result<i32> {
    let c = cstring(&path);
    let fd = unsafe { libc::open(c.as_ptr(), flags, mode as libc::c_uint) };
    if fd < 0 { return throw_io(&ctx, "open", &path); }
    Ok(fd)
}

fn file_close<'js>(ctx: Ctx<'js>, fd: i32) -> Result<()> {
    let rc = unsafe { libc::close(fd) };
    if rc < 0 { return throw_io(&ctx, "close", ""); }
    Ok(())
}

fn file_read<'js>(
    ctx: Ctx<'js>,
    fd: i32,
    buf: rquickjs::TypedArray<'js, u8>,
    offset: u32,
    length: u32,
    position: f64,
) -> Result<u32> {
    let bytes = buf.as_bytes()
        .ok_or_else(|| rquickjs::Error::new_from_js("TypedArray", "detached"))?;
    let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_ptr() as *mut u8, bytes.len()) };
    let off = offset as usize;
    let len = (length as usize).min(dst.len().saturating_sub(off));

    let n = if position < 0.0 {
        unsafe { libc::read(fd, dst[off..].as_mut_ptr() as *mut _, len) }
    } else {
        unsafe { libc::pread(fd, dst[off..].as_mut_ptr() as *mut _, len, position as libc::off_t) }
    };
    if n < 0 { return throw_io(&ctx, "read", ""); }
    Ok(n as u32)
}

fn file_write<'js>(
    ctx: Ctx<'js>,
    fd: i32,
    buf: rquickjs::TypedArray<'js, u8>,
    offset: u32,
    length: u32,
    position: f64,
) -> Result<u32> {
    let bytes = buf.as_bytes()
        .ok_or_else(|| rquickjs::Error::new_from_js("TypedArray", "detached"))?;
    let off = offset as usize;
    let len = (length as usize).min(bytes.len().saturating_sub(off));

    let n = if position < 0.0 {
        unsafe { libc::write(fd, bytes[off..].as_ptr() as *const _, len) }
    } else {
        unsafe { libc::pwrite(fd, bytes[off..].as_ptr() as *const _, len, position as libc::off_t) }
    };
    if n < 0 { return throw_io(&ctx, "write", ""); }
    Ok(n as u32)
}

fn stat_to_object<'js>(ctx: Ctx<'js>, s: &libc::stat) -> Result<Object<'js>> {
    let o = Object::new(ctx.clone())?;
    o.set("dev", s.st_dev as f64)?;
    o.set("ino", s.st_ino as f64)?;
    o.set("mode", s.st_mode as u32)?;
    o.set("nlink", s.st_nlink as f64)?;
    o.set("uid", s.st_uid as u32)?;
    o.set("gid", s.st_gid as u32)?;
    o.set("rdev", s.st_rdev as f64)?;
    o.set("size", s.st_size as f64)?;
    o.set("blksize", s.st_blksize as f64)?;
    o.set("blocks", s.st_blocks as f64)?;
    // Timestamps as ms since epoch.
    #[cfg(target_os = "macos")]
    {
        o.set("atime_ms", s.st_atime as f64 * 1000.0 + (s.st_atime_nsec as f64) / 1e6)?;
        o.set("mtime_ms", s.st_mtime as f64 * 1000.0 + (s.st_mtime_nsec as f64) / 1e6)?;
        o.set("ctime_ms", s.st_ctime as f64 * 1000.0 + (s.st_ctime_nsec as f64) / 1e6)?;
        o.set("birthtime_ms", s.st_birthtime as f64 * 1000.0 + (s.st_birthtime_nsec as f64) / 1e6)?;
    }
    #[cfg(not(target_os = "macos"))]
    {
        o.set("atime_ms", s.st_atime as f64 * 1000.0)?;
        o.set("mtime_ms", s.st_mtime as f64 * 1000.0)?;
        o.set("ctime_ms", s.st_ctime as f64 * 1000.0)?;
        o.set("birthtime_ms", s.st_mtime as f64 * 1000.0)?;
    }
    Ok(o)
}

fn file_stat<'js>(ctx: Ctx<'js>, path: String) -> Result<Object<'js>> {
    let c = cstring(&path);
    let mut s: libc::stat = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::stat(c.as_ptr(), &mut s) };
    if rc < 0 { return throw_io(&ctx, "stat", &path); }
    stat_to_object(ctx, &s)
}

fn file_lstat<'js>(ctx: Ctx<'js>, path: String) -> Result<Object<'js>> {
    let c = cstring(&path);
    let mut s: libc::stat = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::lstat(c.as_ptr(), &mut s) };
    if rc < 0 { return throw_io(&ctx, "lstat", &path); }
    stat_to_object(ctx, &s)
}

fn file_fstat<'js>(ctx: Ctx<'js>, fd: i32) -> Result<Object<'js>> {
    let mut s: libc::stat = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::fstat(fd, &mut s) };
    if rc < 0 { return throw_io(&ctx, "fstat", ""); }
    stat_to_object(ctx, &s)
}

fn file_readdir<'js>(ctx: Ctx<'js>, path: String) -> Result<Array<'js>> {
    use std::fs;
    let entries = match fs::read_dir(&path) {
        Ok(e) => e,
        Err(_) => return throw_io(&ctx, "readdir", &path),
    };
    let arr = Array::new(ctx.clone())?;
    let mut i = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let ent = Object::new(ctx.clone())?;
        ent.set("name", name)?;
        // type: 1=file, 2=dir, 3=symlink, others=4
        let ty = entry.file_type().ok().map(|t| {
            if t.is_dir() { 2u8 } else if t.is_symlink() { 3u8 }
            else if t.is_file() { 1u8 } else { 4u8 }
        }).unwrap_or(4);
        ent.set("type", ty)?;
        arr.set(i, ent)?;
        i += 1;
    }
    Ok(arr)
}

fn file_realpath<'js>(ctx: Ctx<'js>, path: String) -> Result<String> {
    let c = cstring(&path);
    let mut resolved = vec![0u8; libc::PATH_MAX as usize];
    let p = unsafe { libc::realpath(c.as_ptr(), resolved.as_mut_ptr() as *mut _) };
    if p.is_null() { return throw_io(&ctx, "realpath", &path); }
    let cstr = unsafe { std::ffi::CStr::from_ptr(p) };
    Ok(cstr.to_string_lossy().into_owned())
}

fn file_unlink<'js>(ctx: Ctx<'js>, path: String) -> Result<()> {
    let c = cstring(&path);
    if unsafe { libc::unlink(c.as_ptr()) } < 0 {
        return throw_io(&ctx, "unlink", &path);
    }
    Ok(())
}

fn file_mkdir<'js>(ctx: Ctx<'js>, path: String, mode: u32) -> Result<()> {
    let c = cstring(&path);
    if unsafe { libc::mkdir(c.as_ptr(), mode as libc::mode_t) } < 0 {
        return throw_io(&ctx, "mkdir", &path);
    }
    Ok(())
}

fn file_rmdir<'js>(ctx: Ctx<'js>, path: String) -> Result<()> {
    let c = cstring(&path);
    if unsafe { libc::rmdir(c.as_ptr()) } < 0 {
        return throw_io(&ctx, "rmdir", &path);
    }
    Ok(())
}

fn file_rename<'js>(ctx: Ctx<'js>, from: String, to: String) -> Result<()> {
    let cf = cstring(&from);
    let ct = cstring(&to);
    if unsafe { libc::rename(cf.as_ptr(), ct.as_ptr()) } < 0 {
        return throw_io(&ctx, "rename", &from);
    }
    Ok(())
}

fn file_read_to_string<'js>(ctx: Ctx<'js>, path: String) -> Result<String> {
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s),
        Err(_) => throw_io(&ctx, "open", &path),
    }
}

fn file_exists(path: String) -> bool {
    std::fs::metadata(&path).is_ok()
}

fn file_is_file(path: String) -> bool {
    std::fs::metadata(&path).map(|m| m.is_file()).unwrap_or(false)
}

fn file_is_dir(path: String) -> bool {
    std::fs::metadata(&path).map(|m| m.is_dir()).unwrap_or(false)
}

fn file_access<'js>(ctx: Ctx<'js>, path: String, mode: i32) -> Result<()> {
    let c = cstring(&path);
    if unsafe { libc::access(c.as_ptr(), mode) } < 0 {
        return throw_io(&ctx, "access", &path);
    }
    Ok(())
}

// =========================================================================
// process.* — argv, env, platform, arch, cwd, exit, hrtime
// =========================================================================

/// Coerce a JS value to a byte vector for stdio writes. Accepts strings
/// (UTF-8 encoded) and Uint8Array / Buffer (raw bytes).
fn value_to_bytes<'js>(v: &rquickjs::Value<'js>) -> Vec<u8> {
    if let Some(s) = v.as_string() {
        return s.to_string().unwrap_or_default().into_bytes();
    }
    if let Ok(ta) = rquickjs::TypedArray::<u8>::from_value(v.clone()) {
        return ta.as_bytes().unwrap_or(&[]).to_vec();
    }
    Vec::new()
}

fn make_process<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let p = Object::new(ctx.clone())?;
    // Data properties (don't change during process lifetime).
    p.set("platform", platform_name())?;
    p.set("arch", arch_name())?;
    p.set("pid", std::process::id())?;

    // env as a plain object snapshot. Real Node makes mutations visible
    // back to the OS via setenv — we just return a snapshot for now.
    let env = Object::new(ctx.clone())?;
    for (k, v) in std::env::vars() {
        let _ = env.set(k.as_str(), v.as_str());
    }
    p.set("env", env)?;

    // argv as an array.
    let argv = Array::new(ctx.clone())?;
    for (i, a) in std::env::args().enumerate() {
        argv.set(i, a)?;
    }
    p.set("argv", argv)?;

    p.set("cwd", Func::from(|| -> String {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "/".into())
    }))?;

    p.set("chdir", Func::from(|path: String| -> bool {
        std::env::set_current_dir(&path).is_ok()
    }))?;

    p.set("exit", Func::from(|code: i32| -> () {
        std::process::exit(code);
    }))?;

    // stdout / stderr — raw writes that flush. We accept either a string
    // or a Uint8Array; strings are written as UTF-8, byte arrays go through
    // verbatim. console.log lands here via the lifted node:console module.
    use rquickjs::Value;
    use std::io::Write;
    let stdout_write = Func::from(|v: Value<'_>| -> bool {
        let bytes = value_to_bytes(&v);
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(&bytes);
        let _ = out.flush();
        true
    });
    p.set("stdout_write", stdout_write)?;
    let stderr_write = Func::from(|v: Value<'_>| -> bool {
        let bytes = value_to_bytes(&v);
        let mut err = std::io::stderr().lock();
        let _ = err.write_all(&bytes);
        let _ = err.flush();
        true
    });
    p.set("stderr_write", stderr_write)?;

    // hrtime — high-resolution monotonic nanoseconds since some fixed start.
    p.set("hrtime_ns", Func::from(|| -> f64 {
        // f64 is fine for ~285 years of ns precision.
        let now = std::time::Instant::now();
        // Without a static base, every call is relative to itself which is
        // useless. Lazy-init a base instead.
        use std::sync::OnceLock;
        static BASE: OnceLock<std::time::Instant> = OnceLock::new();
        let base = BASE.get_or_init(|| now);
        now.duration_since(*base).as_nanos() as f64
    }))?;

    Ok(p)
}

// =========================================================================
// os.*
// =========================================================================

fn make_os<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let o = Object::new(ctx.clone())?;

    o.set("hostname", Func::from(|| -> String {
        hostname::get()
            .ok()
            .and_then(|s| s.into_string().ok())
            .unwrap_or_else(|| "localhost".into())
    }))?;

    o.set("uptime", Func::from(uptime_seconds))?;

    o.set("totalmem", Func::from(totalmem_bytes))?;
    o.set("freemem",  Func::from(freemem_bytes))?;

    o.set("loadavg", Func::from(loadavg))?;

    o.set("homedir", Func::from(homedir))?;
    o.set("tmpdir",  Func::from(tmpdir))?;

    // Platform / arch / endianness are compile-time constants.
    o.set("platform", Func::from(|| -> &'static str { platform_name() }))?;
    o.set("arch",     Func::from(|| -> &'static str { arch_name() }))?;
    o.set("endianness", Func::from(|| -> &'static str {
        if cfg!(target_endian = "big") { "BE" } else { "LE" }
    }))?;

    // Uname-derived info.
    o.set("osType",    Func::from(|| -> String { uname_field(UnameField::Sysname) }))?;
    o.set("osRelease", Func::from(|| -> String { uname_field(UnameField::Release) }))?;
    o.set("osVersion", Func::from(|| -> String { uname_field(UnameField::Version) }))?;

    o.set("availableParallelism", Func::from(available_parallelism))?;

    // CPUs returns an array of objects so the JS layer can pass it through.
    o.set("cpus", Func::from(make_cpus))?;

    // userInfo returns { username, uid, gid, shell, homedir }.
    o.set("userInfo", Func::from(make_user_info))?;

    // networkInterfaces returns a flat array; the JS shim groups it by name
    // and converts to the Node-shape (with CIDR etc.).
    o.set("networkInterfaces", Func::from(make_network_interfaces))?;

    // priority — not implemented on every host; returning Err here would
    // throw, but we just return a sensible default of 0.
    o.set("getPriority", Func::from(|_pid: i32| -> i32 { 0 }))?;
    o.set("setPriority", Func::from(|_pid: i32, _prio: i32| -> bool { true }))?;

    Ok(o)
}

// ---- platform / arch ----

fn platform_name() -> &'static str {
    // Match Node's process.platform values.
    if cfg!(target_os = "macos") { "darwin" }
    else if cfg!(target_os = "linux") { "linux" }
    else if cfg!(target_os = "windows") { "win32" }
    else if cfg!(target_os = "freebsd") { "freebsd" }
    else if cfg!(target_os = "openbsd") { "openbsd" }
    else if cfg!(target_os = "netbsd") { "netbsd" }
    else if cfg!(target_os = "ios") { "darwin" }
    else { "unknown" }
}

fn arch_name() -> &'static str {
    // Match Node's process.arch values.
    if cfg!(target_arch = "x86_64") { "x64" }
    else if cfg!(target_arch = "aarch64") { "arm64" }
    else if cfg!(target_arch = "x86") { "ia32" }
    else if cfg!(target_arch = "arm") { "arm" }
    else if cfg!(target_arch = "powerpc64") { "ppc64" }
    else if cfg!(target_arch = "riscv64") { "riscv64" }
    else { "unknown" }
}

// ---- env-based dirs ----

fn homedir() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/".into())
}

fn tmpdir() -> String {
    std::env::var("TMPDIR")
        .or_else(|_| std::env::var("TMP"))
        .or_else(|_| std::env::var("TEMP"))
        .unwrap_or_else(|_| {
            if cfg!(windows) { "C:\\Windows\\Temp".into() } else { "/tmp".into() }
        })
}

// ---- uname ----

enum UnameField { Sysname, Release, Version }

fn uname_field(f: UnameField) -> String {
    let mut u: libc::utsname = unsafe { std::mem::zeroed() };
    if unsafe { libc::uname(&mut u) } != 0 {
        return String::new();
    }
    unsafe {
        let p = match f {
            UnameField::Sysname => u.sysname.as_ptr(),
            UnameField::Release => u.release.as_ptr(),
            UnameField::Version => u.version.as_ptr(),
        };
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

// ---- uptime / mem / loadavg ----

fn uptime_seconds() -> f64 {
    // sysctl kern.boottime on macOS/BSD; /proc/uptime on Linux.
    #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd"))]
    {
        let mut mib = [libc::CTL_KERN, libc::KERN_BOOTTIME];
        let mut boottime: libc::timeval = unsafe { std::mem::zeroed() };
        let mut size = std::mem::size_of::<libc::timeval>();
        let rc = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(), mib.len() as u32,
                &mut boottime as *mut _ as *mut _, &mut size,
                std::ptr::null_mut(), 0,
            )
        };
        if rc != 0 { return 0.0; }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        return now - boottime.tv_sec as f64 - (boottime.tv_usec as f64 / 1e6);
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/uptime") {
            if let Some(field) = s.split_whitespace().next() {
                return field.parse().unwrap_or(0.0);
            }
        }
        return 0.0;
    }
    #[allow(unreachable_code)]
    0.0
}

fn totalmem_bytes() -> f64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        let mut v: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        let rc = libc::sysctl(
            mib.as_mut_ptr(), mib.len() as u32,
            &mut v as *mut _ as *mut _, &mut size,
            std::ptr::null_mut(), 0,
        );
        return if rc == 0 { v as f64 } else { 0.0 };
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    let kb: u64 = rest.split_whitespace().next()
                        .and_then(|n| n.parse().ok()).unwrap_or(0);
                    return (kb * 1024) as f64;
                }
            }
        }
        return 0.0;
    }
    #[allow(unreachable_code)]
    0.0
}

fn freemem_bytes() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    let kb: u64 = rest.split_whitespace().next()
                        .and_then(|n| n.parse().ok()).unwrap_or(0);
                    return (kb * 1024) as f64;
                }
            }
        }
    }
    // macOS: vm_statistics — complex. Approximate by reporting totalmem so
    // the JS layer doesn't divide by zero. Real impl can land later.
    totalmem_bytes() / 2.0
}

fn loadavg<'js>(ctx: Ctx<'js>) -> Result<Array<'js>> {
    let mut v = [0.0f64; 3];
    let n = unsafe { libc::getloadavg(v.as_mut_ptr(), 3) };
    if n < 0 { v = [0.0; 3]; }
    let arr = Array::new(ctx)?;
    arr.set(0, v[0])?;
    arr.set(1, v[1])?;
    arr.set(2, v[2])?;
    Ok(arr)
}

// ---- cpus ----

fn available_parallelism() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn make_cpus<'js>(ctx: Ctx<'js>) -> Result<Array<'js>> {
    let n = available_parallelism() as usize;
    let arr = Array::new(ctx.clone())?;
    for i in 0..n {
        let cpu = Object::new(ctx.clone())?;
        cpu.set("model", "unknown")?;
        cpu.set("speed", 0u32)?;
        let times = Object::new(ctx.clone())?;
        times.set("user", 0u64)?;
        times.set("nice", 0u64)?;
        times.set("sys",  0u64)?;
        times.set("idle", 0u64)?;
        times.set("irq",  0u64)?;
        cpu.set("times", times)?;
        arr.set(i, cpu)?;
    }
    Ok(arr)
}

// ---- userInfo ----

fn make_user_info<'js>(ctx: Ctx<'js>) -> Result<Object<'js>> {
    let o = Object::new(ctx.clone())?;
    // getpwuid(getuid()) on unix.
    #[cfg(unix)]
    unsafe {
        let uid = libc::getuid();
        let pw = libc::getpwuid(uid);
        if !pw.is_null() {
            let name = std::ffi::CStr::from_ptr((*pw).pw_name).to_string_lossy().into_owned();
            let home = std::ffi::CStr::from_ptr((*pw).pw_dir).to_string_lossy().into_owned();
            let shell = std::ffi::CStr::from_ptr((*pw).pw_shell).to_string_lossy().into_owned();
            o.set("username", name)?;
            o.set("homedir",  home)?;
            o.set("shell",    shell)?;
            o.set("uid",      uid as i32)?;
            o.set("gid",      libc::getgid() as i32)?;
            return Ok(o);
        }
    }
    // Fallback.
    o.set("username", std::env::var("USER").unwrap_or_else(|_| "unknown".into()))?;
    o.set("homedir",  homedir())?;
    o.set("shell",    std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into()))?;
    o.set("uid",      -1i32)?;
    o.set("gid",      -1i32)?;
    Ok(o)
}

// ---- network interfaces ----

/// Stub: returns just loopback. Real getifaddrs is OS-specific; the JS shim
/// can already format what we return into Node's shape, so adding more
/// interfaces later is purely a Rust-side expansion.
fn make_network_interfaces<'js>(ctx: Ctx<'js>) -> Result<Array<'js>> {
    let arr = Array::new(ctx.clone())?;
    let lo4 = Object::new(ctx.clone())?;
    lo4.set("name",      "lo0")?;
    lo4.set("address",   "127.0.0.1")?;
    lo4.set("netmask",   "255.0.0.0")?;
    lo4.set("family",    "IPv4")?;
    lo4.set("mac",       "00:00:00:00:00:00")?;
    lo4.set("internal",  true)?;
    lo4.set("scopeid",   0u32)?;
    arr.set(0, lo4)?;

    let lo6 = Object::new(ctx.clone())?;
    lo6.set("name",      "lo0")?;
    lo6.set("address",   "::1")?;
    lo6.set("netmask",   "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff")?;
    lo6.set("family",    "IPv6")?;
    lo6.set("mac",       "00:00:00:00:00:00")?;
    lo6.set("internal",  true)?;
    lo6.set("scopeid",   0u32)?;
    arr.set(1, lo6)?;
    Ok(arr)
}
