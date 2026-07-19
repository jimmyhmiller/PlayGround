// Vercel function: a GENERIC `at` node that stays warm. On the first request
// per instance we spawn a persistent `ai-lang at-serve` (stdlib-only, no baked
// app) and remember its port. Each request ships its own program in a
// self-contained KIND_BUNDLE frame; the resident node installs that code on
// demand (a no-op once warm) and runs it, keeping JIT'd code resident across
// calls. Every request — including concurrent ones on the same warm instance
// (Fluid in-function concurrency) — is proxied to that node over a fresh
// localhost connection: HTTP body (a wire frame) in, reply frame out. The
// proxy is byte-transparent, so the bundle flows through unchanged.

use http_body_util::BodyExt;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use vercel_runtime::{run, service_fn, Error, Request, Response, ResponseBody};

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(handler)).await
}

/// A bundled file sits next to this handler binary (`includeFiles`), with
/// /var/task as a fallback.
fn bundled(name: &str) -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join(name);
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from("/var/task").join(name)
}

// The resident node: kept alive for the instance's lifetime.
static NODE_CHILD: Mutex<Option<Child>> = Mutex::new(None);
static NODE_PORT: OnceLock<u16> = OnceLock::new();
static SPAWN_LOCK: Mutex<()> = Mutex::new(());

/// Port of the resident `at-serve` node, spawning it once (per instance).
fn node_port() -> Result<u16, String> {
    if let Some(p) = NODE_PORT.get() {
        return Ok(*p);
    }
    let _g = SPAWN_LOCK.lock().unwrap();
    if let Some(p) = NODE_PORT.get() {
        return Ok(*p); // another request won the race
    }

    let bin = bundled("ai-lang");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&bin) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            let _ = std::fs::set_permissions(&bin, perms);
        }
    }

    let mut child = Command::new(&bin)
        .arg("at-serve")
        .current_dir("/tmp") // /var/task is read-only
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // node logs → function logs
        .spawn()
        .map_err(|e| format!("spawn at-serve ({:?}): {}", bin, e))?;

    // Read the `READY <port>` line the node prints once compiled.
    let mut out = child.stdout.take().ok_or("no stdout from at-serve")?;
    let mut line = Vec::new();
    let mut b = [0u8; 1];
    loop {
        match out.read(&mut b) {
            Ok(0) => break,
            Ok(_) => {
                if b[0] == b'\n' {
                    break;
                }
                line.push(b[0]);
            }
            Err(e) => return Err(format!("reading READY: {}", e)),
        }
    }
    let line = String::from_utf8_lossy(&line);
    let port: u16 = line
        .strip_prefix("READY ")
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| format!("at-serve did not report READY (got {:?})", line))?;

    // Drain any further stdout so the child never blocks on a full pipe.
    std::thread::spawn(move || {
        let mut sink = [0u8; 1024];
        while out.read(&mut sink).unwrap_or(0) > 0 {}
    });

    *NODE_CHILD.lock().unwrap() = Some(child);
    let _ = NODE_PORT.set(port);
    Ok(port)
}

/// Send one length-prefixed frame to the resident node, read its reply.
fn proxy(port: u16, frame: &[u8]) -> Result<Vec<u8>, String> {
    let mut s =
        TcpStream::connect(("127.0.0.1", port)).map_err(|e| format!("connect node: {}", e))?;
    s.set_nodelay(true).ok();
    s.write_all(&(frame.len() as u32).to_be_bytes())
        .and_then(|_| s.write_all(frame))
        .and_then(|_| s.flush())
        .map_err(|e| format!("send frame: {}", e))?;
    let mut lenb = [0u8; 4];
    s.read_exact(&mut lenb).map_err(|e| format!("read reply len: {}", e))?;
    let len = u32::from_be_bytes(lenb) as usize;
    let mut reply = vec![0u8; len];
    s.read_exact(&mut reply).map_err(|e| format!("read reply: {}", e))?;
    Ok(reply)
}

async fn handler(req: Request) -> Result<Response<ResponseBody>, Error> {
    let body: Vec<u8> = req.into_body().collect().await?.to_bytes().to_vec();

    let err500 = |msg: String| -> Result<Response<ResponseBody>, Error> {
        Ok(Response::builder()
            .status(500)
            .header("content-type", "text/plain")
            .body(ResponseBody::from(msg))?)
    };

    let port = match node_port() {
        Ok(p) => p,
        Err(e) => return err500(format!("node start failed: {}", e)),
    };
    match proxy(port, &body) {
        Ok(reply) => Ok(Response::builder()
            .status(200)
            .header("content-type", "application/octet-stream")
            .body(ResponseBody::from(reply))?),
        Err(e) => err500(format!("proxy failed: {}", e)),
    }
}
