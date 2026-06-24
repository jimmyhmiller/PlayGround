//! `memscope-agent` — the in-process transport server. It runs on its own
//! thread (excluded from allocation tracking), listens on a Unix socket, and
//! answers a consumer (the CLI or the eventual UI) with newline-delimited JSON.
//!
//! Why in-process type resolution: `backtrace::resolve` (and thus our DWARF
//! join) only works for the *current* process's own addresses, so the agent —
//! which lives inside the traced program — owns the [`TypeOracle`] and ships
//! already-typed snapshots. The consumer is a thin viewer and needs no access
//! to the target's binary.
//!
//! Protocol: request/reply, one JSON object per line. A consumer polls
//! [`ClientMsg::GetStats`] / [`ClientMsg::PollEvents`] for the live view and
//! asks for [`ClientMsg::GetSnapshot`] to get a full, type-resolved heap dump.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};

use memscope_core as mem;
use memscope_proto::{ClientMsg, ServerMsg, SiteInfo, StatsView, TypeId, TypeInfo};
use memscope_symbols::TypeOracle;

const AGENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Lazily-built type oracle, shared across connections.
enum OracleState {
    NotBuilt,
    Built(TypeOracle),
    Failed(String),
}

struct Shared {
    oracle: Mutex<OracleState>,
}

impl Shared {
    /// Run `f` with a built oracle, building it on first use. Returns an error
    /// string if the oracle can't be built (e.g. missing debuginfo).
    fn with_oracle<R>(&self, f: impl FnOnce(&TypeOracle) -> R) -> Result<R, String> {
        let mut guard = self.oracle.lock().unwrap();
        if let OracleState::NotBuilt = &*guard {
            match TypeOracle::for_current_process() {
                Ok(o) => *guard = OracleState::Built(o),
                Err(e) => *guard = OracleState::Failed(e.to_string()),
            }
        }
        match &*guard {
            OracleState::Built(o) => Ok(f(o)),
            OracleState::Failed(e) => Err(e.clone()),
            OracleState::NotBuilt => unreachable!(),
        }
    }
}

/// Default socket path for this process (`/tmp/memscope-<pid>.sock`), overridable
/// with the `MEMSCOPE_SOCK` environment variable.
pub fn default_socket_path() -> String {
    std::env::var("MEMSCOPE_SOCK").unwrap_or_else(|_| format!("/tmp/memscope-{}.sock", std::process::id()))
}

/// Start the agent on a background thread listening at [`default_socket_path`].
/// Returns the socket path. The workload thread is never blocked.
pub fn start() -> std::io::Result<String> {
    let path = default_socket_path();
    start_at(&path)?;
    Ok(path)
}

/// Start the agent at a specific socket path.
pub fn start_at(path: &str) -> std::io::Result<()> {
    let _ = std::fs::remove_file(path); // clear any stale socket
    let listener = UnixListener::bind(path)?;
    let path_owned = path.to_string();

    std::thread::Builder::new()
        .name("memscope-agent".into())
        .spawn(move || {
            // This thread's own allocations must never be tracked.
            mem::exclude_current_thread();

            let shared = Arc::new(Shared {
                oracle: Mutex::new(OracleState::NotBuilt),
            });

            eprintln!("[memscope] agent listening on {path_owned}");
            for conn in listener.incoming() {
                match conn {
                    Ok(stream) => {
                        // Handle one consumer at a time (UI is a single client).
                        if let Err(e) = handle_conn(stream, &shared) {
                            eprintln!("[memscope] connection ended: {e}");
                        }
                    }
                    Err(e) => {
                        eprintln!("[memscope] accept error: {e}");
                        break;
                    }
                }
            }
        })?;
    Ok(())
}

fn handle_conn(stream: UnixStream, shared: &Arc<Shared>) -> std::io::Result<()> {
    let mut writer = stream.try_clone()?;
    send(&mut writer, &ServerMsg::Hello {
        pid: std::process::id(),
        agent_version: AGENT_VERSION.to_string(),
    })?;

    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: ClientMsg = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                send(&mut writer, &ServerMsg::Error(format!("bad request: {e}")))?;
                continue;
            }
        };
        handle_msg(msg, shared, &mut writer)?;
    }
    Ok(())
}

fn handle_msg(msg: ClientMsg, shared: &Arc<Shared>, w: &mut UnixStream) -> std::io::Result<()> {
    match msg {
        ClientMsg::SetMode(m) => {
            mem::set_mode(match m {
                1 => mem::Mode::Full,
                2 => mem::Mode::Sampled,
                _ => mem::Mode::Off,
            });
            send(w, &ServerMsg::Stats(stats_view()))
        }
        ClientMsg::SetSampleRate(r) => {
            mem::set_sample_rate(r);
            send(w, &ServerMsg::Stats(stats_view()))
        }
        ClientMsg::SetCaptureSites(on) => {
            mem::set_capture_sites(on);
            send(w, &ServerMsg::Stats(stats_view()))
        }
        ClientMsg::GetStats => send(w, &ServerMsg::Stats(stats_view())),
        ClientMsg::PollEvents { max } => {
            // Turn on the (otherwise-skipped) event stream on first poll.
            mem::set_event_streaming(true);
            let mut evs = Vec::new();
            mem::drain_events(&mut evs, max);
            send(w, &ServerMsg::Events(evs))
        }
        ClientMsg::GetSnapshot => {
            let mut snap = mem::snapshot();
            match shared.with_oracle(|o| o.resolve_snapshot(&mut snap)) {
                Ok(()) => {}
                Err(e) => {
                    // Still return the (untyped) snapshot, plus an error note.
                    send(w, &ServerMsg::Error(format!("type resolution unavailable: {e}")))?;
                }
            }
            send(w, &ServerMsg::Snapshot(Box::new(snap)))
        }
        ClientMsg::ResolveSites { ids } => {
            let mut sites: Vec<SiteInfo> = Vec::new();
            let mut types: Vec<TypeInfo> = Vec::new();
            let mut type_ids: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

            let result = shared.with_oracle(|o| {
                for id in &ids {
                    let Some(ips) = mem::site_frames(memscope_proto::SiteId(*id)) else {
                        continue;
                    };
                    let (frames, elem, shape) = o.resolve_site_ips(&ips);
                    let ty = match elem {
                        Some(name) => {
                            let tid = *type_ids.entry(name.clone()).or_insert_with(|| {
                                let tid = types.len() as u32;
                                types.push(TypeInfo { id: tid, name, size: None });
                                tid
                            });
                            TypeId(tid)
                        }
                        None => TypeId::UNKNOWN,
                    };
                    sites.push(SiteInfo { id: *id, frames, ty, shape });
                }
            });
            if let Err(e) = result {
                return send(w, &ServerMsg::Error(format!("type resolution unavailable: {e}")));
            }
            send(w, &ServerMsg::Types(types))?;
            send(w, &ServerMsg::Sites(sites))
        }
        ClientMsg::GetGraph => {
            let mut snap = mem::snapshot();
            let built = shared.with_oracle(|o| {
                o.resolve_snapshot(&mut snap);
                let type_name: std::collections::HashMap<u32, String> =
                    snap.types.iter().map(|t| (t.id, t.name.clone())).collect();
                let site: std::collections::HashMap<u32, &memscope_proto::SiteInfo> =
                    snap.sites.iter().map(|s| (s.id, s)).collect();
                let nodes: Vec<memscope_graph::NodeInput> = snap
                    .live
                    .iter()
                    .map(|l| {
                        let (ty, shape) = site
                            .get(&l.site.0)
                            .map(|s| (type_name.get(&s.ty.0).cloned(), s.shape))
                            .unwrap_or((None, None));
                        memscope_graph::NodeInput {
                            addr: l.addr,
                            size: l.size,
                            type_name: ty,
                            shape,
                        }
                    })
                    .collect();
                memscope_graph::build(&nodes, o.layout(), &memscope_graph::InProcessReader)
            });
            match built {
                Ok(g) => send(w, &ServerMsg::Graph(Box::new(g))),
                Err(e) => send(w, &ServerMsg::Error(format!("graph unavailable: {e}"))),
            }
        }
    }
}

fn stats_view() -> StatsView {
    let s = mem::stats();
    StatsView {
        live_bytes: s.live_bytes,
        total_allocs: s.total_allocs,
        total_alloc_bytes: s.total_alloc_bytes,
        dropped_events: s.dropped_events,
        mode: s.mode as u8,
        sample_rate: s.sample_rate,
    }
}

fn send(w: &mut UnixStream, msg: &ServerMsg) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(msg).map_err(std::io::Error::other)?;
    line.push(b'\n');
    w.write_all(&line)?;
    w.flush()
}
