//! Portable HTTP/1.x parser primitive — exposed as `__host.http.parser`.
//!
//! Header / request-line / status-line parsing uses `httparse` (the same
//! parser hyper / actix / axum sit on). Body framing — Content-Length,
//! chunked transfer-encoding, and connection-close identity bodies — is a
//! small state machine in this file. The result is a robust HTTP/1.1 parser
//! that any host language with an equivalent battle-tested parser can drop
//! in (Go: `net/http`, Python: `h11`, C: `llhttp`, etc.).
//!
//! Interface presented to JS:
//!
//!   __host.http.parser.create(kind)            → handle (u32). kind is
//!                                                 'request' | 'response'.
//!   __host.http.parser.execute(h, buf, off, n) → { nread, events[], error? }
//!   __host.http.parser.finish(h)               → { events[], error? }
//!                                                 (signals EOF — used for
//!                                                 identity bodies / HTTP/1.0)
//!   __host.http.parser.reset(h, kind)          → ()  (reuse parser)
//!   __host.http.parser.free(h)                 → ()
//!
//! Event shape (objects in the `events` array):
//!   { kind: 'headers',
//!     method: string|null, url: string|null,
//!     status_code: number|null, status_message: string|null,
//!     http_major: number, http_minor: number,
//!     headers: string[],         // flat name/value pairs
//!     upgrade: boolean, should_keep_alive: boolean }
//!   { kind: 'body', data: Uint8Array }
//!   { kind: 'message_complete' }
//!
//! `httparse` validates header characters and rejects multiple Content-Length
//! / conflicting framing — the well-known smuggling defenses. We additionally
//! reject mixing Transfer-Encoding: chunked with Content-Length (chunked
//! wins, per RFC 9112 §6.1, but we reject because that combination is almost
//! always an attack).

use rquickjs::function::Func;
use rquickjs::{Array, Ctx, Object, Result, TypedArray};
use std::collections::HashMap;
use std::sync::Mutex;

const MAX_HEADERS: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind { Request, Response }

#[derive(Debug)]
enum State {
    /// Accumulating bytes until we have a full header block.
    Headers,
    /// Body of known length, N bytes remaining.
    ContentLength(u64),
    /// Chunked transfer-encoding.
    Chunked(ChunkedState),
    /// Identity body — read until EOF (HTTP/1.0 or close-delimited).
    Identity,
    /// No body for this message (e.g. 1xx/204/304 response, or request with
    /// no Content-Length and no chunked TE).
    NoBody,
    /// Headers parsed but not yet handed to caller (we buffer until execute()
    /// returns so the events array stays consistent).
    AfterHeaders,
    /// Terminal — parser has emitted message_complete; reset() to reuse.
    Done,
    /// Parser hit an unrecoverable error. Stays in this state until reset.
    Error,
}

#[derive(Debug)]
enum ChunkedState {
    /// Reading the hex size + optional extensions until CRLF.
    Size { line: Vec<u8> },
    /// Reading `remaining` data bytes.
    Data { remaining: u64 },
    /// Just consumed the data; expecting `\r\n` after it.
    DataCrlf { saw_cr: bool },
    /// Last chunk (size 0) emitted; consuming trailer headers until empty line.
    Trailers { line: Vec<u8>, blank_after: bool },
}

struct Parser {
    kind: Kind,
    state: State,
    /// Accumulator for the header block, filled until httparse says Complete.
    /// Capped at MAX_HEADER_BYTES so a malicious peer can't OOM us.
    header_buf: Vec<u8>,
    /// Latched after headers so finish() and reset()-on-keepalive know.
    should_keep_alive: bool,
}

const MAX_HEADER_BYTES: usize = 80 * 1024;
const MAX_CHUNK_SIZE_DIGITS: usize = 16; // 2^64 fits in 16 hex digits

impl Parser {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            state: State::Headers,
            header_buf: Vec::with_capacity(2048),
            should_keep_alive: false,
        }
    }

    fn reset(&mut self, kind: Kind) {
        self.kind = kind;
        self.state = State::Headers;
        self.header_buf.clear();
        self.should_keep_alive = false;
    }
}

// =========================================================================
// Global state (single-threaded JS host; one Mutex is fine).
// =========================================================================

struct GlobalState {
    next: u32,
    parsers: HashMap<u32, Parser>,
}

static STATE: Mutex<Option<GlobalState>> = Mutex::new(None);

fn with_state<F, R>(f: F) -> R
where F: FnOnce(&mut GlobalState) -> R,
{
    let mut g = STATE.lock().unwrap();
    if g.is_none() {
        *g = Some(GlobalState { next: 1, parsers: HashMap::new() });
    }
    f(g.as_mut().unwrap())
}

// =========================================================================
// Public install: __host.http.parser.*
// =========================================================================

pub fn install<'js>(ctx: Ctx<'js>, host: &Object<'js>) -> Result<()> {
    let http = Object::new(ctx.clone())?;
    let parser = Object::new(ctx.clone())?;
    parser.set("create", Func::from(p_create))?;
    parser.set("execute", Func::from(p_execute))?;
    parser.set("finish",  Func::from(p_finish))?;
    parser.set("reset",   Func::from(p_reset))?;
    parser.set("free",    Func::from(p_free))?;
    http.set("parser", parser)?;
    host.set("http", http)?;
    Ok(())
}

fn parse_kind(s: &str) -> std::result::Result<Kind, &'static str> {
    match s {
        "request"  | "REQUEST"  => Ok(Kind::Request),
        "response" | "RESPONSE" => Ok(Kind::Response),
        _ => Err("kind must be 'request' or 'response'"),
    }
}

fn p_create<'js>(ctx: Ctx<'js>, kind: String) -> Result<u32> {
    let k = parse_kind(&kind).map_err(|m| {
        let err = rquickjs::Exception::from_message(ctx.clone(), m).unwrap();
        ctx.throw(err.into_value())
    })?;
    with_state(|s| {
        let id = s.next;
        s.next = s.next.wrapping_add(1).max(1);
        s.parsers.insert(id, Parser::new(k));
        Ok(id)
    })
}

fn p_reset<'js>(ctx: Ctx<'js>, handle: u32, kind: String) -> Result<()> {
    let k = parse_kind(&kind).map_err(|m| {
        let err = rquickjs::Exception::from_message(ctx.clone(), m).unwrap();
        ctx.throw(err.into_value())
    })?;
    with_state(|s| {
        if let Some(p) = s.parsers.get_mut(&handle) { p.reset(k); }
    });
    Ok(())
}

fn p_free(handle: u32) {
    with_state(|s| { s.parsers.remove(&handle); });
}

fn p_execute<'js>(
    ctx: Ctx<'js>,
    handle: u32,
    buf: TypedArray<'js, u8>,
    off: u32,
    len: u32,
) -> Result<Object<'js>> {
    let bytes = buf.as_bytes().unwrap_or(&[]);
    let off = off as usize;
    let len = (len as usize).min(bytes.len().saturating_sub(off));
    let slice = &bytes[off..off + len];

    let (events, nread, err) = with_state(|s| {
        if let Some(p) = s.parsers.get_mut(&handle) {
            run_parser(p, slice, false)
        } else {
            (Vec::new(), 0, Some("bad handle".to_string()))
        }
    });

    build_result(ctx, events, nread, err)
}

fn p_finish<'js>(ctx: Ctx<'js>, handle: u32) -> Result<Object<'js>> {
    let (events, _, err) = with_state(|s| {
        if let Some(p) = s.parsers.get_mut(&handle) {
            run_parser(p, &[], true)
        } else {
            (Vec::new(), 0, Some("bad handle".to_string()))
        }
    });
    build_result(ctx, events, 0, err)
}

// =========================================================================
// Parser core
// =========================================================================

#[derive(Debug)]
enum Event {
    Headers {
        method: Option<String>,
        url: Option<String>,
        status_code: Option<u16>,
        status_message: Option<String>,
        http_major: u8,
        http_minor: u8,
        headers: Vec<(String, String)>,
        upgrade: bool,
        should_keep_alive: bool,
    },
    /// Owned because we may need to slice/concatenate; lifetime simpler.
    Body(Vec<u8>),
    MessageComplete,
}

/// Returns (events, bytes_consumed, error_message).
/// `eof` is true when called from finish(); identity-body messages complete
/// at EOF.
fn run_parser(p: &mut Parser, mut input: &[u8], eof: bool) -> (Vec<Event>, usize, Option<String>) {
    let mut events: Vec<Event> = Vec::new();
    let mut consumed = 0usize;

    loop {
        match &mut p.state {
            State::Error | State::Done => return (events, consumed, None),

            State::Headers => {
                // Accumulate into header_buf. We feed httparse the whole
                // header_buf each pass — it's cheap relative to network I/O
                // and avoids edge cases with header splits across calls.
                if !input.is_empty() {
                    let space = MAX_HEADER_BYTES.saturating_sub(p.header_buf.len());
                    if space == 0 {
                        p.state = State::Error;
                        return (events, consumed, Some("header block too large".to_string()));
                    }
                    let take = input.len().min(space);
                    p.header_buf.extend_from_slice(&input[..take]);
                    input = &input[take..];
                    consumed += take;
                }

                match try_parse_headers(p) {
                    Ok(None) => {
                        // Need more bytes; if EOF, that's an error unless
                        // header_buf is empty (idle keep-alive).
                        if eof && !p.header_buf.is_empty() {
                            p.state = State::Error;
                            return (events, consumed, Some("EOF mid-headers".to_string()));
                        }
                        return (events, consumed, None);
                    }
                    Ok(Some((headers_consumed, ev, next_state))) => {
                        // Trim consumed header bytes; remainder rejoins input.
                        let remainder = p.header_buf.split_off(headers_consumed);
                        p.header_buf.clear();
                        // Prepend remainder back to input.
                        if !remainder.is_empty() {
                            // Concatenate remainder + input by stashing.
                            // input is shorter or we already consumed it.
                            let mut joined = remainder;
                            joined.extend_from_slice(input);
                            // We can't reassign input to a temporary; loop
                            // again with a side-buffer.
                            p.state = next_state;
                            events.push(ev);
                            // Drain joined through the body state below by
                            // recursing once. We do this by setting input to
                            // a borrowed slice of joined and re-entering the
                            // body branches via continue. We use a Cow-like
                            // approach: write the joined back into header_buf
                            // as a scratch area (it's empty now), then point
                            // input at it.
                            p.header_buf = joined;
                            // SAFETY: we don't touch p.header_buf again until
                            // the next iteration consumes it.
                            let input_owned: *const [u8] = &p.header_buf[..];
                            // Unsafe-free alternative: copy bytes to a fresh
                            // Vec and run body through it iteratively.
                            // Simpler: process body using header_buf, drain.
                            let _ = input_owned;
                            // Process body from header_buf, then continue.
                            let mut buf = std::mem::take(&mut p.header_buf);
                            let (more_events, body_consumed, err) =
                                run_body_loop(p, &mut buf, eof);
                            events.extend(more_events);
                            // body_consumed bytes from buf. If buf has leftover,
                            // those are bytes belonging to the NEXT message
                            // (pipelined). We leave them by counting them as
                            // not-consumed-from-input.
                            let body_used_from_remainder =
                                body_consumed.min(buf.len());
                            // Anything left in buf belongs to either the next
                            // message or unparsed body; we *did* claim it from
                            // input already, so adjust consumed downwards.
                            let unused = buf.len().saturating_sub(body_used_from_remainder);
                            consumed = consumed.saturating_sub(unused);
                            if err.is_some() { return (events, consumed, err); }
                            return (events, consumed, None);
                        } else {
                            p.state = next_state;
                            events.push(ev);
                            // fall through to body processing on the same loop
                        }
                    }
                    Err(msg) => {
                        p.state = State::Error;
                        return (events, consumed, Some(msg));
                    }
                }
            }

            State::AfterHeaders => {
                // unreachable; AfterHeaders is set transiently inside parser
                // setup. Treat as NoBody if we ever land here.
                p.state = State::NoBody;
            }

            State::NoBody => {
                events.push(Event::MessageComplete);
                p.state = if p.should_keep_alive { State::Headers } else { State::Done };
                return (events, consumed, None);
            }

            State::ContentLength(remaining) => {
                if *remaining == 0 {
                    events.push(Event::MessageComplete);
                    p.state = if p.should_keep_alive { State::Headers } else { State::Done };
                    return (events, consumed, None);
                }
                if input.is_empty() {
                    if eof {
                        let short = *remaining;
                        p.state = State::Error;
                        return (events, consumed, Some(
                            format!("EOF in content-length body, {short} bytes short")
                        ));
                    }
                    return (events, consumed, None);
                }
                let take = (*remaining as usize).min(input.len());
                events.push(Event::Body(input[..take].to_vec()));
                *remaining -= take as u64;
                input = &input[take..];
                consumed += take;
            }

            State::Identity => {
                if input.is_empty() {
                    if eof {
                        events.push(Event::MessageComplete);
                        p.state = State::Done;
                    }
                    return (events, consumed, None);
                }
                events.push(Event::Body(input.to_vec()));
                consumed += input.len();
                input = &[];
            }

            State::Chunked(_) => {
                let (more_events, used, err, done) =
                    drive_chunked(p, input, eof);
                events.extend(more_events);
                consumed += used;
                input = &input[used..];
                if let Some(m) = err {
                    p.state = State::Error;
                    return (events, consumed, Some(m));
                }
                if done {
                    p.state = if p.should_keep_alive { State::Headers } else { State::Done };
                    return (events, consumed, None);
                }
                if input.is_empty() { return (events, consumed, None); }
            }
        }
    }
}

/// Body-only inner loop, used when remainder bytes spill out of header parsing.
fn run_body_loop(p: &mut Parser, buf: &mut Vec<u8>, eof: bool) -> (Vec<Event>, usize, Option<String>) {
    let mut events = Vec::new();
    let mut used = 0;
    let mut remaining: &[u8] = &buf[..];
    loop {
        match &mut p.state {
            State::Error | State::Done => return (events, used, None),
            State::NoBody => {
                events.push(Event::MessageComplete);
                p.state = if p.should_keep_alive { State::Headers } else { State::Done };
                return (events, used, None);
            }
            State::ContentLength(rem) => {
                if *rem == 0 {
                    events.push(Event::MessageComplete);
                    p.state = if p.should_keep_alive { State::Headers } else { State::Done };
                    return (events, used, None);
                }
                if remaining.is_empty() {
                    if eof {
                        let short = *rem;
                        p.state = State::Error;
                        return (events, used, Some(format!("EOF in content-length body, {short} short")));
                    }
                    return (events, used, None);
                }
                let take = (*rem as usize).min(remaining.len());
                events.push(Event::Body(remaining[..take].to_vec()));
                *rem -= take as u64;
                remaining = &remaining[take..];
                used += take;
            }
            State::Identity => {
                if remaining.is_empty() {
                    if eof {
                        events.push(Event::MessageComplete);
                        p.state = State::Done;
                    }
                    return (events, used, None);
                }
                events.push(Event::Body(remaining.to_vec()));
                used += remaining.len();
                remaining = &[];
            }
            State::Chunked(_) => {
                let (more_events, u, err, done) = drive_chunked(p, remaining, eof);
                events.extend(more_events);
                remaining = &remaining[u..];
                used += u;
                if let Some(m) = err { p.state = State::Error; return (events, used, Some(m)); }
                if done {
                    p.state = if p.should_keep_alive { State::Headers } else { State::Done };
                    return (events, used, None);
                }
                if remaining.is_empty() { return (events, used, None); }
            }
            State::Headers | State::AfterHeaders => {
                // We shouldn't land here from run_body_loop, but if we do, bail
                // and let the outer loop reprocess.
                return (events, used, None);
            }
        }
    }
}

/// Try to parse the full header block from p.header_buf. Returns:
///   Ok(None)                                  — need more bytes
///   Ok(Some((consumed, Headers-event, next_state))) — headers fully parsed
///   Err(msg)                                  — parse error
fn try_parse_headers(p: &mut Parser) -> std::result::Result<
    Option<(usize, Event, State)>, String
> {
    let mut hdrs = [httparse::EMPTY_HEADER; MAX_HEADERS];
    match p.kind {
        Kind::Request => {
            let mut req = httparse::Request::new(&mut hdrs);
            match req.parse(&p.header_buf) {
                Ok(httparse::Status::Partial) => Ok(None),
                Ok(httparse::Status::Complete(n)) => {
                    let method = req.method.unwrap_or("").to_string();
                    let url = req.path.unwrap_or("").to_string();
                    let v = req.version.unwrap_or(1); // 0=HTTP/1.0, 1=HTTP/1.1
                    let http_minor = v;
                    let http_major = 1;
                    let headers_vec: Vec<(String, String)> = req.headers.iter()
                        .map(|h| (
                            h.name.to_string(),
                            String::from_utf8_lossy(h.value).into_owned()
                        ))
                        .collect();

                    let (state, ka, upgrade) = post_headers_state(
                        &headers_vec, http_major, http_minor, &method, None,
                    )?;
                    p.should_keep_alive = ka;
                    Ok(Some((n, Event::Headers {
                        method: Some(method),
                        url: Some(url),
                        status_code: None,
                        status_message: None,
                        http_major, http_minor,
                        headers: headers_vec,
                        upgrade,
                        should_keep_alive: ka,
                    }, state)))
                }
                Err(e) => Err(format!("parse error: {e}")),
            }
        }
        Kind::Response => {
            let mut res = httparse::Response::new(&mut hdrs);
            match res.parse(&p.header_buf) {
                Ok(httparse::Status::Partial) => Ok(None),
                Ok(httparse::Status::Complete(n)) => {
                    let code = res.code.unwrap_or(0);
                    let reason = res.reason.unwrap_or("").to_string();
                    let v = res.version.unwrap_or(1);
                    let http_major = 1;
                    let http_minor = v;
                    let headers_vec: Vec<(String, String)> = res.headers.iter()
                        .map(|h| (
                            h.name.to_string(),
                            String::from_utf8_lossy(h.value).into_owned()
                        ))
                        .collect();

                    let (state, ka, upgrade) = post_headers_state(
                        &headers_vec, http_major, http_minor, "", Some(code),
                    )?;
                    p.should_keep_alive = ka;
                    Ok(Some((n, Event::Headers {
                        method: None,
                        url: None,
                        status_code: Some(code),
                        status_message: Some(reason),
                        http_major, http_minor,
                        headers: headers_vec,
                        upgrade,
                        should_keep_alive: ka,
                    }, state)))
                }
                Err(e) => Err(format!("parse error: {e}")),
            }
        }
    }
}

/// Decide which body state we enter after headers, plus keep-alive + upgrade.
fn post_headers_state(
    headers: &[(String, String)],
    http_major: u8, http_minor: u8,
    method: &str,
    status: Option<u16>,
) -> std::result::Result<(State, bool, bool), String> {
    let mut content_length: Option<u64> = None;
    let mut transfer_encoding_chunked = false;
    let mut has_te = false;
    let mut connection_close = false;
    let mut connection_keepalive = false;
    let mut upgrade = false;

    for (name, value) in headers {
        if name.eq_ignore_ascii_case("content-length") {
            // Reject duplicate Content-Length with differing values (smuggling).
            let trimmed = value.trim();
            let n: u64 = trimmed.parse().map_err(|_| {
                format!("invalid Content-Length: {trimmed:?}")
            })?;
            if let Some(prev) = content_length {
                if prev != n {
                    return Err("conflicting Content-Length headers".to_string());
                }
            }
            content_length = Some(n);
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            has_te = true;
            // RFC 9112: chunked must be the final coding.
            for coding in value.split(',') {
                let c = coding.trim();
                if c.eq_ignore_ascii_case("chunked") {
                    transfer_encoding_chunked = true;
                } else if c.eq_ignore_ascii_case("identity") {
                    // identity = no transform, no framing change
                } else {
                    return Err(format!("unsupported transfer-encoding: {c}"));
                }
            }
        } else if name.eq_ignore_ascii_case("connection") {
            for tok in value.split(',') {
                let t = tok.trim();
                if t.eq_ignore_ascii_case("close") { connection_close = true; }
                else if t.eq_ignore_ascii_case("keep-alive") { connection_keepalive = true; }
                else if t.eq_ignore_ascii_case("upgrade") { upgrade = true; }
            }
        } else if name.eq_ignore_ascii_case("upgrade") {
            // Presence of Upgrade header alone doesn't switch us; Connection: upgrade does.
        }
    }

    // Smuggling defense: reject TE + CL together (some servers prefer CL,
    // some TE — attacker leverages disagreement).
    if has_te && content_length.is_some() {
        return Err("both Transfer-Encoding and Content-Length present".to_string());
    }

    // Determine if this message has a body.
    let has_body = match status {
        Some(code) => {
            // Responses: 1xx, 204, 304 have no body. HEAD/CONNECT responses
            // would also have no body, but we'd need the request method to
            // know — JS layer handles that with parser.reset_no_body() if
            // needed in the future. For now assume body unless code says no.
            !(code < 200 || code == 204 || code == 304)
        }
        None => {
            // Request: body exists only if CL>0 or chunked.
            // GET/HEAD/DELETE can technically have a body if framed, so we
            // just check framing.
            let _ = method;
            content_length.unwrap_or(0) > 0 || transfer_encoding_chunked
        }
    };

    // Default keep-alive based on version.
    let default_ka = http_minor >= 1;
    let keep_alive = if connection_close { false }
                     else if connection_keepalive { true }
                     else { default_ka };

    let state = if upgrade {
        // After Upgrade we hand the socket to the upgrade handler; no more
        // HTTP parsing on this connection.
        State::NoBody
    } else if !has_body {
        State::NoBody
    } else if transfer_encoding_chunked {
        State::Chunked(ChunkedState::Size { line: Vec::new() })
    } else if let Some(n) = content_length {
        State::ContentLength(n)
    } else {
        // Response with no framing → identity body (read to EOF).
        State::Identity
    };

    Ok((state, keep_alive, upgrade))
}

/// Returns (events, consumed, error, done).
fn drive_chunked(p: &mut Parser, mut input: &[u8], eof: bool) -> (Vec<Event>, usize, Option<String>, bool) {
    let mut events = Vec::new();
    let mut consumed = 0;
    loop {
        let cs = match &mut p.state {
            State::Chunked(cs) => cs,
            _ => return (events, consumed, None, false),
        };
        match cs {
            ChunkedState::Size { line } => {
                while let Some((&b, rest)) = input.split_first() {
                    input = rest;
                    consumed += 1;
                    if b == b'\n' {
                        // Parse line as hex; ignore extensions after ';'.
                        let s = std::str::from_utf8(line).unwrap_or("");
                        let s = s.trim_end_matches('\r');
                        let size_str = s.split(';').next().unwrap_or("").trim();
                        if size_str.is_empty() || size_str.len() > MAX_CHUNK_SIZE_DIGITS {
                            return (events, consumed, Some(format!("invalid chunk size {size_str:?}")), false);
                        }
                        let n = match u64::from_str_radix(size_str, 16) {
                            Ok(v) => v,
                            Err(_) => return (events, consumed, Some(format!("invalid chunk size {size_str:?}")), false),
                        };
                        if n == 0 {
                            // Move to trailers.
                            p.state = State::Chunked(ChunkedState::Trailers {
                                line: Vec::new(), blank_after: false,
                            });
                        } else {
                            p.state = State::Chunked(ChunkedState::Data { remaining: n });
                        }
                        break;
                    } else {
                        if line.len() >= 256 {
                            return (events, consumed, Some("chunk-size line too long".to_string()), false);
                        }
                        line.push(b);
                    }
                }
                if matches!(p.state, State::Chunked(ChunkedState::Size { .. })) {
                    // didn't break out — need more bytes
                    if eof {
                        return (events, consumed, Some("EOF in chunk size".to_string()), false);
                    }
                    return (events, consumed, None, false);
                }
            }
            ChunkedState::Data { remaining } => {
                if *remaining == 0 {
                    p.state = State::Chunked(ChunkedState::DataCrlf { saw_cr: false });
                    continue;
                }
                if input.is_empty() {
                    if eof { return (events, consumed, Some("EOF in chunk data".to_string()), false); }
                    return (events, consumed, None, false);
                }
                let take = (*remaining as usize).min(input.len());
                events.push(Event::Body(input[..take].to_vec()));
                *remaining -= take as u64;
                input = &input[take..];
                consumed += take;
            }
            ChunkedState::DataCrlf { saw_cr } => {
                while let Some((&b, rest)) = input.split_first() {
                    input = rest;
                    consumed += 1;
                    if !*saw_cr {
                        if b == b'\r' { *saw_cr = true; continue; }
                        if b == b'\n' {
                            p.state = State::Chunked(ChunkedState::Size { line: Vec::new() });
                            break;
                        }
                        return (events, consumed, Some("expected CRLF after chunk".to_string()), false);
                    } else {
                        if b == b'\n' {
                            p.state = State::Chunked(ChunkedState::Size { line: Vec::new() });
                            break;
                        }
                        return (events, consumed, Some("expected LF after CR".to_string()), false);
                    }
                }
                if matches!(p.state, State::Chunked(ChunkedState::DataCrlf { .. })) {
                    if eof { return (events, consumed, Some("EOF after chunk".to_string()), false); }
                    return (events, consumed, None, false);
                }
            }
            ChunkedState::Trailers { line, blank_after } => {
                // Consume bytes until we see CRLF on an empty line.
                while let Some((&b, rest)) = input.split_first() {
                    input = rest;
                    consumed += 1;
                    if b == b'\n' {
                        // End of a line.
                        let was_empty = line.is_empty() || line == b"\r";
                        line.clear();
                        if was_empty {
                            // Final CRLF; message complete.
                            events.push(Event::MessageComplete);
                            return (events, consumed, None, true);
                        }
                        // Otherwise it was a real trailer header; ignore body
                        // and keep collecting.
                        *blank_after = false;
                    } else {
                        if line.len() >= 1024 {
                            return (events, consumed, Some("trailer line too long".to_string()), false);
                        }
                        line.push(b);
                    }
                }
                if eof { return (events, consumed, Some("EOF in trailers".to_string()), false); }
                return (events, consumed, None, false);
            }
        }
    }
}

// =========================================================================
// Build a JS Object from Rust events.
// =========================================================================

fn build_result<'js>(
    ctx: Ctx<'js>,
    events: Vec<Event>,
    nread: usize,
    err: Option<String>,
) -> Result<Object<'js>> {
    let out = Object::new(ctx.clone())?;
    out.set("nread", nread as u32)?;
    let evs = Array::new(ctx.clone())?;
    for (i, ev) in events.into_iter().enumerate() {
        evs.set(i, event_to_js(ctx.clone(), ev)?)?;
    }
    out.set("events", evs)?;
    if let Some(e) = err { out.set("error", e)?; }
    Ok(out)
}

fn event_to_js<'js>(ctx: Ctx<'js>, ev: Event) -> Result<Object<'js>> {
    let o = Object::new(ctx.clone())?;
    match ev {
        Event::Headers { method, url, status_code, status_message,
                         http_major, http_minor, headers, upgrade, should_keep_alive } => {
            o.set("kind", "headers")?;
            o.set("method", method)?;
            o.set("url", url)?;
            o.set("status_code", status_code)?;
            o.set("status_message", status_message)?;
            o.set("http_major", http_major)?;
            o.set("http_minor", http_minor)?;
            let arr = Array::new(ctx.clone())?;
            let mut i = 0;
            for (n, v) in headers {
                arr.set(i, n)?;     i += 1;
                arr.set(i, v)?;     i += 1;
            }
            o.set("headers", arr)?;
            o.set("upgrade", upgrade)?;
            o.set("should_keep_alive", should_keep_alive)?;
        }
        Event::Body(data) => {
            o.set("kind", "body")?;
            let ta = TypedArray::<u8>::new(ctx.clone(), data)?;
            o.set("data", ta)?;
        }
        Event::MessageComplete => {
            o.set("kind", "message_complete")?;
        }
    }
    Ok(o)
}
