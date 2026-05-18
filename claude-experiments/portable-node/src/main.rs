use rquickjs::function::Func;
use rquickjs::{Context, Ctx, Function, Object, Result, Runtime};

mod buffer_binding;
mod host;
mod host_crypto;
mod http_parser;
mod io_loop;
mod util_binding;

// -------------------------------------------------------------------------
// Registry: every Node source (lib and test) we want require()-loadable.
// Adding a new module = add a row here.
// -------------------------------------------------------------------------

/// Modules loadable via JS `require()`. Keys match the strings Node source
/// passes to `require(...)` — e.g. `require('buffer')` resolves to "buffer".
const NODE_SOURCES: &[(&str, &str)] = &[
    // node:buffer
    ("buffer",                  include_str!("../js/node-src/buffer.js")),
    ("internal/buffer",         include_str!("../js/node-src/internal-buffer.js")),
    // node:querystring
    ("querystring",             include_str!("../js/node-src/querystring.js")),
    ("internal/querystring",    include_str!("../js/node-src/internal-querystring.js")),
    // node:os
    ("os",                      include_str!("../js/node-src/os.js")),
    // node:path
    ("path",                    include_str!("../js/node-src/path.js")),
    // node:events
    ("events",                  include_str!("../js/node-src/events.js")),
    // node:punycode (deprecated but still in core)
    ("punycode",                include_str!("../js/node-src/punycode.js")),
    // node:string_decoder
    ("string_decoder",          include_str!("../js/node-src/string_decoder.js")),
    // node:fs — Node's real lib/fs.js verbatim (replaces the portable shim)
    ("fs",                      include_str!("../js/node-src/fs.js")),
    ("internal/fs/utils",       include_str!("../js/node-src/internal-fs-utils.js")),
    // Broad-strokes lifts — many of these will fail at runtime on missing
    // bindings/deps but the goal here is to surface what's actually required.
    ("util",                    include_str!("../js/node-src/util.js")),
    ("console",                 include_str!("../js/node-src/console.js")),
    ("timers",                  include_str!("../js/node-src/timers.js")),
    ("domain",                  include_str!("../js/node-src/domain.js")),
    ("url",                     include_str!("../js/node-src/url.js")),
    ("dns",                     include_str!("../js/node-src/dns.js")),
    // ('crypto' is a portable shim in bootstrap.js — Node's lib/crypto.js
    //  drags in 15+ internal/crypto/* files that all assume libcrypto. We
    //  expose the API surface Express + friends actually use on top of the
    //  __host.crypto.* primitives.)
    ("zlib",                    include_str!("../js/node-src/zlib.js")),
    ("stream",                  include_str!("../js/node-src/stream.js")),
    // internal deps that the above pull in
    ("internal/util/debuglog",  include_str!("../js/node-src/internal-util-debuglog.js")),
    ("internal/console/global", include_str!("../js/node-src/internal-console-global.js")),
    ("internal/console/constructor", include_str!("../js/node-src/internal-console-constructor.js")),
    ("internal/streams/utils",      include_str!("../js/node-src/internal-streams-utils.js")),
    ("internal/streams/readable",   include_str!("../js/node-src/internal-streams-readable.js")),
    ("internal/streams/writable",   include_str!("../js/node-src/internal-streams-writable.js")),
    ("internal/streams/duplex",     include_str!("../js/node-src/internal-streams-duplex.js")),
    ("internal/streams/transform",  include_str!("../js/node-src/internal-streams-transform.js")),
    ("internal/streams/passthrough",include_str!("../js/node-src/internal-streams-passthrough.js")),
    ("internal/streams/destroy",    include_str!("../js/node-src/internal-streams-destroy.js")),
    ("internal/streams/state",      include_str!("../js/node-src/internal-streams-state.js")),
    ("internal/streams/pipeline",   include_str!("../js/node-src/internal-streams-pipeline.js")),
    ("internal/streams/end-of-stream", include_str!("../js/node-src/internal-streams-end-of-stream.js")),
    ("internal/streams/compose",    include_str!("../js/node-src/internal-streams-compose.js")),
    ("internal/streams/from",       include_str!("../js/node-src/internal-streams-from.js")),
    ("internal/streams/operators",  include_str!("../js/node-src/internal-streams-operators.js")),
    ("internal/streams/legacy",     include_str!("../js/node-src/internal-streams-legacy.js")),
    ("internal/streams/add-abort-signal", include_str!("../js/node-src/internal-streams-add-abort-signal.js")),
    // node:net + its eager internals (for the http stack)
    ("net",                                  include_str!("../js/node-src/net.js")),
    ("internal/net",                         include_str!("../js/node-src/internal-net.js")),
    ("internal/stream_base_commons",         include_str!("../js/node-src/internal-stream_base_commons.js")),
    ("internal/process/task_queues",         include_str!("../js/node-src/internal-process-task_queues.js")),
    ("internal/timers",                      include_str!("../js/node-src/internal-timers.js")),
    ("internal/socketaddress",               include_str!("../js/node-src/internal-socketaddress.js")),
    ("internal/blocklist",                   include_str!("../js/node-src/internal-blocklist.js")),
    ("internal/priority_queue",              include_str!("../js/node-src/internal-priority_queue.js")),
    ("internal/fixed_queue",                 include_str!("../js/node-src/internal-fixed_queue.js")),
    // node:http — lifted verbatim from nodejs/node, sitting on the http_parser
    // binding (powered by the pure-JS http-parser-js vendored under js/).
    ("http",                                 include_str!("../js/node-src/http.js")),
    ("_http_common",                         include_str!("../js/node-src/_http_common.js")),
    ("_http_incoming",                       include_str!("../js/node-src/_http_incoming.js")),
    ("_http_outgoing",                       include_str!("../js/node-src/_http_outgoing.js")),
    ("_http_server",                         include_str!("../js/node-src/_http_server.js")),
    ("_http_client",                         include_str!("../js/node-src/_http_client.js")),
    ("_http_agent",                          include_str!("../js/node-src/_http_agent.js")),
    ("internal/http",                        include_str!("../js/node-src/internal-http.js")),
    // node:assert — Node's real assert.js + its helpers
    ("assert",                              include_str!("../js/node-src/assert.js")),
    ("internal/assert/assertion_error",     include_str!("../js/node-src/internal-assert-assertion_error.js")),
    ("internal/assert/utils",               include_str!("../js/node-src/internal-assert-utils.js")),
    ("internal/assert/myers_diff",          include_str!("../js/node-src/internal-assert-myers_diff.js")),
    ("internal/util/colors",                include_str!("../js/node-src/internal-util-colors.js")),
];

/// Node test files. One row = one test name + source. The list of names to
/// actually invoke is derived from this — keeps the two in lockstep.
const NODE_TEST_SOURCES: &[(&str, &str)] = &[
    // buffer
    ("test-buffer-tostring",     include_str!("../js/node-src/test-buffer-tostring.js")),
    ("test-buffer-from",         include_str!("../js/node-src/test-buffer-from.js")),
    ("test-buffer-isencoding",   include_str!("../js/node-src/test-buffer-isencoding.js")),
    ("test-buffer-compare",      include_str!("../js/node-src/test-buffer-compare.js")),
    ("test-buffer-indexof",      include_str!("../js/node-src/test-buffer-indexof.js")),
    ("test-buffer-alloc",        include_str!("../js/node-src/test-buffer-alloc.js")),
    ("test-buffer-slice",        include_str!("../js/node-src/test-buffer-slice.js")),
    ("test-buffer-equals",       include_str!("../js/node-src/test-buffer-equals.js")),
    ("test-buffer-includes",     include_str!("../js/node-src/test-buffer-includes.js")),
    ("test-buffer-concat",       include_str!("../js/node-src/test-buffer-concat.js")),
    ("test-buffer-fill",         include_str!("../js/node-src/test-buffer-fill.js")),
    ("test-buffer-arraybuffer",  include_str!("../js/node-src/test-buffer-arraybuffer.js")),
    ("test-buffer-readuint",     include_str!("../js/node-src/test-buffer-readuint.js")),
    ("test-buffer-write",        include_str!("../js/node-src/test-buffer-write.js")),
    ("test-buffer-bytelength",   include_str!("../js/node-src/test-buffer-bytelength.js")),
    // querystring
    ("test-querystring",         include_str!("../js/node-src/test-querystring.js")),
    ("test-querystring-escape",  include_str!("../js/node-src/test-querystring-escape.js")),
    // path
    ("test-path-basename",       include_str!("../js/node-src/test-path-basename.js")),
    ("test-path-dirname",        include_str!("../js/node-src/test-path-dirname.js")),
    ("test-path-extname",        include_str!("../js/node-src/test-path-extname.js")),
    ("test-path-isabsolute",     include_str!("../js/node-src/test-path-isabsolute.js")),
    ("test-path-join",           include_str!("../js/node-src/test-path-join.js")),
    ("test-path-normalize",      include_str!("../js/node-src/test-path-normalize.js")),
    ("test-path-parse-format",   include_str!("../js/node-src/test-path-parse-format.js")),
    ("test-path-relative",       include_str!("../js/node-src/test-path-relative.js")),
    ("test-path-resolve",        include_str!("../js/node-src/test-path-resolve.js")),
    ("test-path-zero-length-strings", include_str!("../js/node-src/test-path-zero-length-strings.js")),
    // events
    ("test-event-emitter-once",                   include_str!("../js/node-src/test-event-emitter-once.js")),
    ("test-event-emitter-add-listeners",          include_str!("../js/node-src/test-event-emitter-add-listeners.js")),
    ("test-event-emitter-error-monitor",          include_str!("../js/node-src/test-event-emitter-error-monitor.js")),
    ("test-event-emitter-listener-count",         include_str!("../js/node-src/test-event-emitter-listener-count.js")),
    ("test-event-emitter-method-names",           include_str!("../js/node-src/test-event-emitter-method-names.js")),
    ("test-event-emitter-remove-listeners",       include_str!("../js/node-src/test-event-emitter-remove-listeners.js")),
    ("test-event-emitter-special-event-names",    include_str!("../js/node-src/test-event-emitter-special-event-names.js")),
    // assert
    ("test-assert",                              include_str!("../js/node-src/test-assert.js")),
    ("test-assert-deep",                         include_str!("../js/node-src/test-assert-deep.js")),
    ("test-assert-typedarray-deepequal",         include_str!("../js/node-src/test-assert-typedarray-deepequal.js")),
    ("test-assert-fail",                         include_str!("../js/node-src/test-assert-fail.js")),
    // string_decoder + punycode
    ("test-string-decoder",      include_str!("../js/node-src/test-string-decoder.js")),
    ("test-string-decoder-end",  include_str!("../js/node-src/test-string-decoder-end.js")),
    ("test-punycode",            include_str!("../js/node-src/test-punycode.js")),
];

// -------------------------------------------------------------------------

/// Drive the JS event loop until idle.
///
/// Ordering matches Node's "phases" coarsely:
///   1. drain microtasks (process.nextTick + Promise.then)
///   2. look at JS timer queue; fire any due
///   3. sleep until the next due timer (or exit if none and microtasks idle)
///
/// Anything Tier 2 (async I/O) would slot into the wait step — we'd block
/// on `__host.io.poll(timeout_ms)` instead of sleep. For now: pure timers.
fn run_event_loop<'js>(ctx: &Ctx<'js>, rt: &Runtime) -> std::result::Result<(), String> {
    use rquickjs::Error as RErr;
    loop {
        // 1. Drain ALL pending microtasks (process.nextTick + Promise.then chains).
        // rquickjs exposes execute_pending_job() returning Ok(true) when a job
        // ran, Ok(false) when the queue's empty.
        loop {
            match rt.execute_pending_job() {
                Ok(true) => continue,         // ran a job; keep draining
                Ok(false) => break,            // queue empty
                Err(_e) => break,              // job threw; let next eval surface it
            }
        }

        // 2. Anything pending in the timer queue?
        let has_work: bool = ctx.eval("__eventLoopHasWork()")
            .map_err(|e| format!("has-work check: {e:?}"))?;
        if !has_work { break; }

        // 3. Next due time vs now.
        let next_due: f64 = ctx.eval("__eventLoopNextDueMs()")
            .map_err(|e| format!("next-due check: {e:?}"))?;
        if !next_due.is_finite() { break; }
        let now: f64 = ctx.eval("__host.time.now_ms()")
            .map_err(|e| format!("now: {e:?}"))?;
        let sleep_ms = (next_due - now).max(0.0);
        if sleep_ms > 0.0 {
            // Bound sleep so we stay responsive (and so this loop terminates
            // for adversarial inputs).
            let s = sleep_ms.min(60_000.0);
            std::thread::sleep(std::time::Duration::from_millis(s as u64));
        }
        // 4. Fire whatever's due.
        let _ : () = match ctx.eval::<(), _>("__eventLoopFireDue()") {
            Ok(_) => (),
            Err(RErr::Exception) => {
                let exc = ctx.catch();
                let msg = exc.as_object()
                    .and_then(|o| o.get::<_, String>("message").ok())
                    .unwrap_or_else(|| format!("{exc:?}"));
                eprintln!("event loop: timer callback threw: {msg}");
            }
            Err(e) => return Err(format!("fire-due: {e:?}")),
        };
    }
    Ok(())
}

fn dispatch_internal_binding<'js>(ctx: Ctx<'js>, name: String) -> Result<Object<'js>> {
    // Rust-implemented bindings: short list of things that genuinely benefit
    // from native code (buffer's fast byte ops, util's introspection, config).
    match name.as_str() {
        "buffer" => return buffer_binding::make(ctx),
        "util"   => return util_binding::make(ctx),
        "config" => return util_binding::make_config(ctx),
        _ => {}
    }

    // Fallback: try a JS-implemented binding module registered under
    // `__binding/<name>` in shimSources. This is the "portable" path —
    // these bindings are pure JS calling into __host.* primitives.
    let require: Function<'js> = ctx.globals().get("require")?;
    let module_name = format!("__binding/{name}");
    match require.call::<_, Object<'js>>((module_name.as_str(),)) {
        Ok(o) => Ok(o),
        Err(_) => {
            let msg = format!(
                "portable-node: internalBinding({name:?}) not implemented (no Rust impl and no JS shim '__binding/{name}')"
            );
            let err = rquickjs::Exception::from_message(ctx.clone(), &msg)?;
            Err(ctx.throw(err.into_value()))
        }
    }
}

/// Run `f`; if it throws a JS exception, format the JS error (name, message,
/// stack) into a readable string instead of the opaque "Exception generated by
/// QuickJS" that rquickjs returns by default.
fn run_with_diagnostics<'js, F, T>(ctx: &Ctx<'js>, f: F) -> std::result::Result<T, String>
where
    F: FnOnce() -> Result<T>,
{
    match f() {
        Ok(v) => Ok(v),
        Err(rquickjs::Error::Exception) => {
            let exc = ctx.catch();
            let name = exc.as_object().and_then(|o| o.get::<_, String>("name").ok())
                .unwrap_or_else(|| "Error".into());
            let message = exc.as_object().and_then(|o| o.get::<_, String>("message").ok())
                .unwrap_or_else(|| format!("{exc:?}"));
            let stack = exc.as_object().and_then(|o| o.get::<_, String>("stack").ok());
            let mut s = format!("{name}: {message}");
            if let Some(stk) = stack { s.push('\n'); s.push_str(&stk); }
            Err(s)
        }
        Err(other) => Err(format!("{other}")),
    }
}

fn drive_event_loop(context: &Context, rt: &Runtime) -> std::result::Result<(), String> {
    drive_event_loop_for(context, rt, None)
}

fn drive_event_loop_for(
    context: &Context,
    rt: &Runtime,
    max_ms: Option<u64>,
) -> std::result::Result<(), String> {
    let deadline = max_ms.map(|m| std::time::Instant::now() + std::time::Duration::from_millis(m));
    loop {
        if let Some(d) = deadline { if std::time::Instant::now() >= d { break; } }
        // 1. Drain microtasks (process.nextTick + Promise.then). Must be
        // outside Context::with — rquickjs's runtime is locked by the guard.
        loop {
            match rt.execute_pending_job() {
                Ok(true) => continue,
                Ok(false) => break,
                Err(_e) => break,
            }
        }

        // 2. Inside with: compute next wake time. Three sources of work:
        //    timers (due in N ms), pending I/O ops (any), pending completions
        //    (immediate dispatch). Exit when none.
        let (has_work, sleep_ms): (bool, f64) = context.with(|ctx| {
            let has_timers: bool = ctx.eval("__eventLoopHasWork()").unwrap_or(false);
            let has_io: bool     = ctx.eval("__ioHasPending()").unwrap_or(false);
            if !has_timers && !has_io { return (false, 0.0); }
            // Sleep until next timer due, or 60s if only I/O pending.
            let timeout_ms = if has_timers {
                let due: f64 = ctx.eval("__eventLoopNextDueMs()").unwrap_or(f64::INFINITY);
                let now: f64 = ctx.eval("__host.time.now_ms()").unwrap_or(0.0);
                (due - now).max(0.0).min(60_000.0)
            } else {
                60_000.0
            };
            (true, timeout_ms)
        });
        if !has_work { break; }

        // Clamp sleep_ms by remaining wall-clock budget if a deadline is set.
        let sleep_ms = if let Some(d) = deadline {
            let rem = d.saturating_duration_since(std::time::Instant::now()).as_millis() as f64;
            sleep_ms.min(rem).max(0.0)
        } else { sleep_ms };

        // 3. Wait for either I/O completion or the timer deadline.
        //    __host.io.poll(timeout) sleeps inside mio; cheaper and lower-
        //    latency than std::thread::sleep + a separate poll pass.
        //    Then dispatch any completions to their JS callbacks.
        context.with(|ctx| {
            let _ = ctx.eval::<u32, _>(format!("__ioDrain({})", sleep_ms).as_str());
        });

        // 4. Fire any timers that are now due.
        context.with(|ctx| {
            let _ = ctx.eval::<(), _>("__eventLoopFireDue()");
        });
    }
    Ok(())
}

fn run() -> std::result::Result<(), String> {
    let rt = Runtime::new().map_err(|e| e.to_string())?;
    let context = Context::full(&rt).map_err(|e| e.to_string())?;

    // Phase 1: setup + smokes (synchronous; scheduling for async work).
    let phase1 = context.with(|ctx| -> std::result::Result<(), String> {
        run_with_diagnostics(&ctx, || -> Result<()> {
            ctx.globals().set("__internalBinding", Func::from(dispatch_internal_binding))?;

            // Install the portable host interface on `globalThis.__host`.
            host::install(ctx.clone())?;

            // Register every Node source (lib + test) under __nodeSourceFiles.
            let registry: Object<'_> = ctx.eval("({})")?;
            for (name, src) in NODE_SOURCES.iter().chain(NODE_TEST_SOURCES.iter()) {
                registry.set(*name, *src)?;
            }
            ctx.globals().set("__nodeSourceFiles", registry)?;

            ctx.eval::<(), _>(include_str!("../js/bootstrap.js"))?;

            println!("--- hand-written smoke (buffer) ---");
            let smoke: String = ctx.eval(include_str!("../js/smoke-buffer.js"))?;
            println!("{smoke}");

            println!("\n--- hand-written smoke (querystring) ---");
            let qs_smoke: String = ctx.eval(include_str!("../js/smoke-querystring.js"))?;
            println!("{qs_smoke}");

            println!("\n--- hand-written smoke (os) ---");
            let os_smoke: String = ctx.eval(include_str!("../js/smoke-os.js"))?;
            println!("{os_smoke}");

            println!("\n--- host primitives smoke (file I/O direct) ---");
            let file_smoke: String = ctx.eval(include_str!("../js/smoke-host-file.js"))?;
            println!("{file_smoke}");

            println!("\n--- dns isolation probe ---");
            let _ = ctx.eval::<(), _>(r#"
              globalThis._dnsLog = [];
              const dns = require('dns');
              dns.lookup('127.0.0.1', { all: true }, (err, addrs) => {
                globalThis._dnsLog.push('dns cb: err=' + (err && err.code) + ' addrs=' + JSON.stringify(addrs));
              });
              dns.lookup('127.0.0.1', (err, addr, fam) => {
                globalThis._dnsLog.push('dns cb (no all): err=' + (err && err.code) + ' addr=' + addr + ' fam=' + fam);
              });
            "#)?;

            println!("\n--- vendor http-parser-js probe ---");
            let vendor_probe: String = ctx.eval(r#"
              (function() {
                try {
                  const v = require('__vendor/http_parser_js');
                  return 'vendor OK; HTTPParser=' + (typeof v.HTTPParser) + ' methods len=' + v.methods.length;
                } catch (e) {
                  return 'vendor FAIL: ' + (e.message || e) + '\n' + (e.stack||'').split('\n').slice(0,5).join('\n  ');
                }
              })()
            "#)?;
            println!("{vendor_probe}");

            println!("\n--- __binding/http_parser probe ---");
            let bind_probe: String = ctx.eval(r#"
              (function() {
                try {
                  const v = require('__binding/http_parser');
                  return 'binding OK; HTTPParser=' + (typeof v.HTTPParser) + ' REQUEST=' + v.HTTPParser.REQUEST;
                } catch (e) {
                  return 'binding FAIL: ' + (e.message || e) + '\n' + (e.stack||'').split('\n').slice(0,7).join('\n  ');
                }
              })()
            "#)?;
            println!("{bind_probe}");

            // Express probe — only runs if PORTABLE_NODE_PROJECT points at a
            // project with `node_modules/express` (npm install express).
            if std::env::var("PORTABLE_NODE_PROJECT").is_ok() {
                println!("\n--- express load probe ---");
                let express_probe: String = ctx.eval(r#"
                  (function() {
                    try {
                      const express = require('express');
                      return 'LOADS — typeof=' + typeof express +
                             ' keys=' + Object.keys(express).slice(0, 8).join(',');
                    } catch (e) {
                      return 'FAIL — ' + (e.message || e) + '\n  ' +
                             ((e.stack||'').split('\n').slice(0, 10).join('\n  '));
                    }
                  })()
                "#)?;
                println!("{express_probe}");

                println!("\n--- express app setup ---");
                let express_setup: String = ctx.eval(r#"
                  (function() {
                    try {
                      const express = require('express');
                      globalThis._expLog = [];
                      const app = express();
                      app.get('/', (req, res) => {
                        globalThis._expLog.push('GET / from ' + req.ip);
                        res.send('Hello from Express on portable-node!\n');
                      });
                      app.get('/json', (req, res) => {
                        globalThis._expLog.push('GET /json');
                        res.json({ ok: true, msg: 'hello', engine: 'QuickJS' });
                      });
                      app.post('/echo', express.json(), (req, res) => {
                        globalThis._expLog.push('POST /echo body=' + JSON.stringify(req.body));
                        res.json({ youSent: req.body });
                      });
                      globalThis._expSrv = app.listen(
                        +(globalThis.process.env.PORTABLE_NODE_HTTP_PORT || 0),
                        '127.0.0.1',
                        () => {
                          globalThis._expPort = globalThis._expSrv.address().port;
                          globalThis._expLog.push('listening on ' + globalThis._expPort);
                        });
                      globalThis._expSrv.on('error', (e) => globalThis._expLog.push('error: ' + e.message));
                      return 'express setup OK';
                    } catch (e) {
                      return 'express setup FAIL: ' + (e.message || e) + '\n  ' +
                             ((e.stack||'').split('\n').slice(0, 12).join('\n  '));
                    }
                  })()
                "#)?;
                println!("{express_setup}");
            }

            // Debug: probe individual path edge cases to find which assert
            // in test-path-zero-length-strings is failing.
            // Run the test-path-zero-length-strings logic step by step using
            // assert.strictEqual so we know which assertion blows up.
            println!("\n--- path test via assert.strictEqual ---");
            let path_assert_probe: String = ctx.eval(r#"
              (function() {
                const assert = require('assert');
                const path = require('path');
                const pwd = process.cwd();
                const cases = [
                  () => assert.strictEqual(path.posix.join(''), '.'),
                  () => assert.strictEqual(path.posix.join('', ''), '.'),
                  () => assert.strictEqual(path.win32.join(''), '.'),
                  () => assert.strictEqual(path.win32.join('', ''), '.'),
                  () => assert.strictEqual(path.join(pwd), pwd),
                  () => assert.strictEqual(path.join(pwd, ''), pwd),
                  () => assert.strictEqual(path.posix.normalize(''), '.'),
                  () => assert.strictEqual(path.win32.normalize(''), '.'),
                  () => assert.strictEqual(path.normalize(pwd), pwd),
                  () => assert.strictEqual(path.posix.isAbsolute(''), false),
                  () => assert.strictEqual(path.win32.isAbsolute(''), false),
                  () => assert.strictEqual(path.resolve(''), pwd),
                  () => assert.strictEqual(path.resolve('', ''), pwd),
                  () => assert.strictEqual(path.relative('', pwd), ''),
                  () => assert.strictEqual(path.relative(pwd, ''), ''),
                  () => assert.strictEqual(path.relative(pwd, pwd), ''),
                ];
                const out = [];
                for (let i = 0; i < cases.length; i++) {
                  try { cases[i](); out.push('OK ' + i); }
                  catch (e) { out.push('FAIL ' + i + ': ' + (e.message || e).split('\n').slice(0,4).join(' | ')); }
                }
                return out.join('\n');
              })()
            "#)?;
            println!("{path_assert_probe}");

            println!("\n--- path edge-case probe ---");
            let path_probe: String = ctx.eval(r#"
              (function() {
                const path = require('path');
                const pwd = process.cwd();
                const tests = [
                  ['posix.join("")', path.posix.join(''), '.'],
                  ['posix.join("","")', path.posix.join('', ''), '.'],
                  ['win32.join("")', path.win32.join(''), '.'],
                  ['win32.join("","")', path.win32.join('', ''), '.'],
                  ['join(pwd)', path.join(pwd), pwd],
                  ['join(pwd,"")', path.join(pwd, ''), pwd],
                  ['posix.normalize("")', path.posix.normalize(''), '.'],
                  ['win32.normalize("")', path.win32.normalize(''), '.'],
                  ['normalize(pwd)', path.normalize(pwd), pwd],
                  ['resolve("")', path.resolve(''), pwd],
                  ['resolve("","")', path.resolve('', ''), pwd],
                  ['relative("",pwd)', path.relative('', pwd), ''],
                  ['relative(pwd,"")', path.relative(pwd, ''), ''],
                  ['relative(pwd,pwd)', path.relative(pwd, pwd), ''],
                ];
                return tests.map(([n,got,exp]) => (got === exp ? 'OK   ' : 'FAIL ') + n + ' got=' + JSON.stringify(got) + ' exp=' + JSON.stringify(exp)).join('\n');
              })()
            "#)?;
            println!("{path_probe}");

            println!("\n--- node:http load probe ---");
            let http_probe: String = ctx.eval(r#"
              (function() {
                try {
                  const http = require('http');
                  return 'LOADS — keys: ' + Object.keys(http).slice(0, 12).join(', ');
                } catch (e) {
                  return 'FAIL — ' + (e.message || e) + '\n' + (e.stack || '').split('\n').slice(0, 7).join('\n  ');
                }
              })()
            "#)?;
            println!("{http_probe}");

            println!("\n--- node:net load probe ---");
            let net_probe: String = ctx.eval(r#"
              (function() {
                try {
                  const net = require('net');
                  return 'LOADS — keys: ' + Object.keys(net).slice(0, 10).join(', ');
                } catch (e) {
                  return 'FAIL — ' + (e.message || e) + '\n' + (e.stack || '').split('\n').slice(0, 5).join('\n  ');
                }
              })()
            "#)?;
            println!("{net_probe}");

            println!("\n--- http.createServer setup ---");
            let http_setup: String = ctx.eval(r#"
              (function() {
                try {
                  const http = require('http');
                  globalThis._httpLog = [];
                  const push = (m) => globalThis._httpLog.push(m);
                  globalThis._httpSrv = http.createServer((req, res) => {
                    push('handled: ' + req.method + ' ' + req.url + ' host=' + (req.headers && req.headers.host));
                    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
                    res.statusCode = 200;
                    if (req.url === '/echo') {
                      // Echo request body back. Demonstrates Content-Length
                      // and chunked body framing on the parser side.
                      const chunks = [];
                      req.on('data', (c) => { chunks.push(c); push('body chunk ' + c.length + 'B'); });
                      req.on('end',  () => {
                        const body = Buffer.concat(chunks);
                        push('echo end: ' + body.length + 'B');
                        res.end('echo: ' + body.toString('utf8'));
                      });
                    } else {
                      res.end('Hello from portable-node — Node\'s real _http_server.js on QuickJS\n');
                    }
                  });
                  globalThis._httpSrv.on('listening', () => {
                    globalThis._httpPort = globalThis._httpSrv.address().port;
                    push('listening on ' + globalThis._httpPort);
                  });
                  globalThis._httpSrv.on('error', (e) => push('error: ' + e.message));
                  const port = (globalThis.process && globalThis.process.env &&
                                +globalThis.process.env.PORTABLE_NODE_HTTP_PORT) || 0;
                  globalThis._httpSrv.listen(port, '127.0.0.1');
                  return 'http setup OK';
                } catch (e) {
                  return 'http setup FAIL: ' + (e.message || e) + '\n  ' +
                         (e.stack||'').split('\n').slice(0,8).join('\n  ');
                }
              })()
            "#)?;
            println!("{http_setup}");

            println!("\n--- net.createServer round trip ---");
            let net_port: i64 = ctx.eval(r#"
              (function() {
                try {
                const net = require('net');
                globalThis._netLog = [];
                globalThis._netServer = net.createServer((sock) => {
                  globalThis._netLog.push('server got connection');
                  let received = '';
                  sock.on('data', (chunk) => {
                    received += chunk.toString('utf8');
                    globalThis._netLog.push('server data: ' + JSON.stringify(received));
                  });
                  sock.on('end', () => {
                    globalThis._netLog.push('server end; replying');
                    sock.write('Pong: ' + received);
                    sock.end();
                  });
                });
                let port = 0;
                globalThis._netServer.on('listening', () => {
                  globalThis._netPort = globalThis._netServer.address().port;
                  globalThis._netLog.push('listening event: port=' + globalThis._netPort);
                });
                globalThis._netServer.on('error', (e) => {
                  globalThis._netLog.push('server error: ' + (e && e.message));
                  globalThis._netLog.push('stack: ' + (e && e.stack || '').split('\n').slice(0,5).join(' / '));
                });
                try {
                  globalThis._netServer.listen(0, '127.0.0.1');
                  globalThis._netLog.push('listen() returned');
                } catch (e) {
                  globalThis._netLog.push('listen threw: ' + e.message);
                }
                return -1; // ignored; port comes from log post-loop
                } catch (e) {
                  globalThis._netLog = globalThis._netLog || [];
                  globalThis._netLog.push('SETUP THREW: ' + e.message + '\nstack:\n' + (e.stack || '').split('\n').slice(0,8).join('\n  '));
                  return -2;
                }
              })()
            "#)?;
            println!("listen port (immediate): {net_port}");

            println!("\n--- io loop TCP round-trip ---");
            // Set up: JS listens on a port; we spawn a thread that connects
            // to that port and sends bytes; JS polls for the accept + read
            // completions; verifies the bytes round-tripped.
            let listen_port: u32 = ctx.eval(r#"
              (function() {
                const T = __host.tcp;
                globalThis._srv = T.create_tcp();
                const status = T.listen(globalThis._srv, '127.0.0.1', 0, 128);
                if (status !== 0) throw new Error('listen failed: ' + status);
                return T.local_addr(globalThis._srv).port;
              })()
            "#)?;
            println!("listen port: {listen_port}");

            // Spawn a connector thread that hits the listener 100ms later.
            let port_copy = listen_port;
            let connector = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(100));
                use std::io::Write;
                let mut stream = match std::net::TcpStream::connect(
                    format!("127.0.0.1:{port_copy}")
                ) {
                    Ok(s) => s,
                    Err(e) => { eprintln!("connector: {e}"); return; }
                };
                let _ = stream.write_all(b"hello world");
                // Close cleanly so server-side read sees EOF after the bytes.
                let _ = stream.shutdown(std::net::Shutdown::Write);
                std::thread::sleep(std::time::Duration::from_millis(50));
            });

            // JS round-trip: accept → read → verify.
            let round_trip: String = ctx.eval(r#"
              (function() {
                const T = __host.tcp;
                const I = __host.io;
                const out = [];
                // Submit accept.
                const acceptOp = T.accept(globalThis._srv);
                out.push('accept op_id: ' + acceptOp);
                // Poll until accept completes (block up to 2s total).
                let clientHandle = -1;
                const deadline = Date.now() + 2000;
                while (Date.now() < deadline && clientHandle < 0) {
                  const cs = I.poll(200);
                  for (const c of cs) {
                    out.push('completion: ' + JSON.stringify(c));
                    if (c.kind === 'accept' && c.op_id === acceptOp) {
                      clientHandle = c.handle;
                    }
                  }
                }
                if (clientHandle < 0) { out.push('TIMEOUT waiting for accept'); return out.join('\n'); }
                out.push('accepted handle: ' + clientHandle);

                // Read from the accepted client into a Buffer.
                const buf = new Uint8Array(64);
                const readOp = T.read(clientHandle, buf, 0, buf.length);
                out.push('read op_id: ' + readOp);

                // Poll for read completion.
                let bytesRead = -1;
                const rdeadline = Date.now() + 2000;
                while (Date.now() < rdeadline && bytesRead < 0) {
                  const cs = I.poll(200);
                  for (const c of cs) {
                    out.push('completion: ' + JSON.stringify(c));
                    if (c.kind === 'read' && c.op_id === readOp) {
                      bytesRead = c.n;
                    }
                  }
                }
                if (bytesRead < 0) { out.push('TIMEOUT waiting for read'); return out.join('\n'); }
                // Decode the bytes we filled.
                let s = '';
                for (let i = 0; i < bytesRead; i++) s += String.fromCharCode(buf[i]);
                out.push('read ' + bytesRead + ' bytes: ' + JSON.stringify(s));

                T.close(clientHandle);
                T.close(globalThis._srv);
                return out.join('\n');
              })()
            "#)?;
            println!("{round_trip}");

            // Make sure the connector thread is done.
            let _ = connector.join();

            // Now the second proof: same TCP round-trip but using JS-level
            // CALLBACKS through the integrated event loop. This is the API
            // shape Node's net.js will sit on top of.
            println!("\n--- callback-based TCP round-trip (event-loop driven) ---");
            let listen_port2: u32 = ctx.eval(r#"
              (function() {
                globalThis._cbSrv = tcpCreate();
                if (tcpListen(globalThis._cbSrv, '127.0.0.1', 0, 128) !== 0)
                  throw new Error('listen failed');
                globalThis._cbLog = [];
                tcpAccept(globalThis._cbSrv, (c) => {
                  globalThis._cbLog.push('accept: status=' + c.status + ' h=' + c.handle);
                  if (c.status !== 0) return;
                  const buf = new Uint8Array(64);
                  tcpRead(c.handle, buf, 0, buf.length, (cr) => {
                    let s = '';
                    for (let i = 0; i < cr.n; i++) s += String.fromCharCode(buf[i]);
                    globalThis._cbLog.push('read: n=' + cr.n + ' s=' + JSON.stringify(s));
                    tcpClose(c.handle);
                    tcpClose(globalThis._cbSrv);
                  });
                });
                return tcpLocalAddr(globalThis._cbSrv).port;
              })()
            "#)?;
            println!("listen port: {listen_port2}");

            let port_copy2 = listen_port2;
            let connector2 = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                use std::io::Write;
                if let Ok(mut s) = std::net::TcpStream::connect(format!("127.0.0.1:{port_copy2}")) {
                    let _ = s.write_all(b"GET / HTTP/1.0\r\n\r\n");
                    let _ = s.shutdown(std::net::Shutdown::Write);
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            });
            let _ = connector2.join();

            println!("\n--- node:stream smoke ---");
            let stream_smoke: String = ctx.eval(include_str!("../js/smoke-stream.js"))?;
            println!("{stream_smoke}");

            println!("\n--- event loop smoke (schedule) ---");
            let el_pre: String = ctx.eval(include_str!("../js/smoke-eventloop.js"))?;
            println!("{el_pre}");

            println!("\n--- node:zlib smoke ---");
            let zlib_smoke: String = ctx.eval(include_str!("../js/smoke-zlib.js"))?;
            println!("{zlib_smoke}");

            println!("\n--- node:fs sync API smoke ---");
            let fs_smoke: String = ctx.eval(include_str!("../js/smoke-fs.js"))?;
            println!("{fs_smoke}");

            println!("\n--- punycode probe ---");
            let pn_probe: String = ctx.eval(r#"
                (function () {
                  try {
                    const p = require('punycode');
                    return 'loaded; encode("Bücher")=' + p.encode('Bücher') + ', decode("Bcher-kva")=' + p.decode('Bcher-kva');
                  } catch (e) {
                    return 'FAIL: ' + (e.message || e) + '\n' + (e.stack || '').split('\n').slice(0,5).join('\n');
                  }
                })()
            "#)?;
            println!("{pn_probe}");

            println!("\n--- string_decoder smoke ---");
            let sd_smoke: String = ctx.eval(include_str!("../js/smoke-string-decoder.js"))?;
            println!("{sd_smoke}");

            println!("\n--- node:assert load probe ---");
            // Trace which module fails by trying each dep in order.
            let probe: String = ctx.eval(r#"
                (function () {
                  const tries = [
                    'internal/util/colors',
                    'internal/assert/myers_diff',
                    'internal/assert/assertion_error',
                    'internal/assert/utils',
                    'assert',
                  ];
                  const out = [];
                  for (const m of tries) {
                    try {
                      delete moduleCache[m]; // force reload to catch fresh errors
                      const r = require(m);
                      out.push('OK   ' + m + ' (' + (typeof r) + ')');
                    } catch (e) {
                      out.push('FAIL ' + m + ': ' + (e.message || e) + '\n' + (e.stack || '').split('\n').slice(0, 5).join('\n'));
                      break;
                    }
                  }
                  return out.join('\n');
                })()
            "#)?;
            println!("{probe}");

            println!("\n--- broad-strokes module load probe ---");
            let probe: String = ctx.eval(r#"
              (function () {
                const modules = [
                  'buffer','querystring','os','path','events','assert',
                  'punycode','string_decoder','fs',
                  'util','console','timers','domain','url','dns','crypto','zlib','stream',
                ];
                const out = [];
                for (const m of modules) {
                  try {
                    const r = require(m);
                    const keys = Object.keys(r).slice(0, 4).join(',');
                    out.push('LOADS ' + m.padEnd(16) + ' typeof=' + typeof r + ' keys=[' + keys + ']');
                  } catch (e) {
                    const msg = (e.message || String(e)).split('\n')[0].slice(0, 90);
                    out.push('ERROR ' + m.padEnd(16) + ' ' + msg);
                  }
                }
                return out.join('\n');
              })()
            "#)?;
            println!("{probe}");

            // (Event loop runs between phases — see end of run())
            Ok(())
        })
    });
    phase1?;

    println!("\n--- driving event loop (≤500ms cap) ---");
    drive_event_loop_for(&context, &rt, Some(500))?;

    // Phase 3a: read state after first event-loop pass.
    let port_from_js: u32 = context.with(|ctx| -> std::result::Result<u32, String> {
        run_with_diagnostics(&ctx, || -> Result<u32> {
            println!("\n--- net.createServer (after event loop) ---");
            let netresult: String = ctx.eval(r#"
              (function() {
                return JSON.stringify({
                  port: globalThis._netPort || null,
                  log: globalThis._netLog || [],
                });
              })()
            "#)?;
            println!("{netresult}");
            let p: u32 = ctx.eval(r#"globalThis._netPort || 0"#)?;
            Ok(p)
        })
    })?;

    // Phase 3b: external curl-style client → server round-trip, driving the
    // event loop OUTSIDE Context::with so re-entry doesn't double-borrow.
    if port_from_js > 0 {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            std::thread::sleep(std::time::Duration::from_millis(80));
            match std::net::TcpStream::connect(format!("127.0.0.1:{port_from_js}")) {
                Ok(mut s) => {
                    s.set_read_timeout(Some(std::time::Duration::from_millis(2000))).ok();
                    let _ = s.write_all(b"PING");
                    let _ = s.shutdown(std::net::Shutdown::Write);
                    let mut response = String::new();
                    let _ = s.read_to_string(&mut response);
                    let _ = tx.send(response);
                }
                Err(e) => { let _ = tx.send(format!("CONNECT_FAILED: {e}")); }
            }
        });
        // Drive the event loop while the client does its thing. Cap at 3s.
        drive_event_loop_for(&context, &rt, Some(3000))?;
        match rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(r)  => println!("external client read: {r:?}"),
            Err(_) => println!("external client read: TIMEOUT (no reply within deadline)"),
        }
        // One more short drain in case completions are still in-flight.
        drive_event_loop_for(&context, &rt, Some(200))?;
    }

    // Phase 3b': HTTP round-trip — read the http port, fire a real HTTP GET
    // from an external thread, drive the event loop, print the response.
    let http_port: u32 = context.with(|ctx| -> std::result::Result<u32, String> {
        run_with_diagnostics(&ctx, || -> Result<u32> {
            println!("\n--- http.createServer (after event loop) ---");
            let log: String = ctx.eval(r#"JSON.stringify(globalThis._httpLog || [])"#)?;
            println!("log: {log}");
            let p: u32 = ctx.eval(r#"globalThis._httpPort || 0"#)?;
            Ok(p)
        })
    })?;
    if http_port > 0 {
        println!("\n--- external HTTP GET against http.createServer on port {http_port} ---");
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            std::thread::sleep(std::time::Duration::from_millis(80));
            match std::net::TcpStream::connect(format!("127.0.0.1:{http_port}")) {
                Ok(mut s) => {
                    s.set_read_timeout(Some(std::time::Duration::from_millis(3000))).ok();
                    let req = format!("GET / HTTP/1.1\r\nHost: 127.0.0.1:{http_port}\r\nConnection: close\r\n\r\n");
                    let _ = s.write_all(req.as_bytes());
                    let mut response = String::new();
                    let _ = s.read_to_string(&mut response);
                    let _ = tx.send(response);
                }
                Err(e) => { let _ = tx.send(format!("CONNECT_FAILED: {e}")); }
            }
        });
        drive_event_loop_for(&context, &rt, Some(3000))?;
        match rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(r)  => println!("HTTP response:\n{r}"),
            Err(_) => println!("HTTP response: TIMEOUT"),
        }
        drive_event_loop_for(&context, &rt, Some(200))?;
    } else {
        println!("\n--- http.createServer — no port assigned (server did not reach 'listening') ---");
    }

    // Phase 3b'': express round-trip.
    let express_port: u32 = context.with(|ctx| -> std::result::Result<u32, String> {
        run_with_diagnostics(&ctx, || -> Result<u32> {
            println!("\n--- express app (after event loop) ---");
            let log: String = ctx.eval(r#"JSON.stringify(globalThis._expLog || [])"#)?;
            println!("log: {log}");
            let p: u32 = ctx.eval(r#"globalThis._expPort || 0"#)?;
            Ok(p)
        })
    })?;
    if express_port > 0 {
        println!("\n--- external HTTP GET against express on port {express_port} ---");
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            std::thread::sleep(std::time::Duration::from_millis(80));
            match std::net::TcpStream::connect(format!("127.0.0.1:{express_port}")) {
                Ok(mut s) => {
                    s.set_read_timeout(Some(std::time::Duration::from_millis(3000))).ok();
                    let req = format!("GET / HTTP/1.1\r\nHost: 127.0.0.1:{express_port}\r\nConnection: close\r\n\r\n");
                    let _ = s.write_all(req.as_bytes());
                    let mut response = String::new();
                    let _ = s.read_to_string(&mut response);
                    let _ = tx.send(response);
                }
                Err(e) => { let _ = tx.send(format!("CONNECT_FAILED: {e}")); }
            }
        });
        drive_event_loop_for(&context, &rt, Some(3000))?;
        match rx.recv_timeout(std::time::Duration::from_millis(500)) {
            Ok(r) => println!("EXPRESS HTTP response:\n{r}"),
            Err(_) => println!("EXPRESS HTTP response: TIMEOUT"),
        }
        drive_event_loop_for(&context, &rt, Some(200))?;
    }

    // Phase 3c: read final state + run Node tests.
    context.with(|ctx| -> std::result::Result<(), String> {
        run_with_diagnostics(&ctx, || -> Result<()> {
            println!("\n--- http.createServer (final log) ---");
            let final_http: String = ctx.eval(r#"(globalThis._httpLog || []).map(l => '  ' + l).join('\n')"#)?;
            println!("{final_http}");

            println!("\n--- dns isolation log ---");
            let dnsl: String = ctx.eval(r#"JSON.stringify(globalThis._dnsLog || [])"#)?;
            println!("{dnsl}");

            println!("\n--- net.createServer (final log) ---");
            let final_net: String = ctx.eval(r#"JSON.stringify(globalThis._netLog || [])"#)?;
            println!("{final_net}");

            println!("\n--- callback-based TCP (post-loop log) ---");
            let cb_log: String = ctx.eval(r#"
              (globalThis._cbLog || []).map(l => '  ' + l).join('\n') || '  (no entries)'
            "#)?;
            println!("{cb_log}");

            println!("\n--- event loop smoke (post-loop results) ---");
            let el_post: String = ctx.eval(r#"
                (function() {
                  const r = globalThis.__asyncResults || {};
                  const keys = Object.keys(r);
                  const lines = [];
                  for (const k of keys) {
                    let v = r[k];
                    if (Array.isArray(v)) v = '[' + v.join(', ') + ']';
                    lines.push('  ' + k.padEnd(14) + ': ' + JSON.stringify(v));
                  }
                  return lines.join('\n');
                })()
            "#)?;
            println!("{el_post}");

            println!("\n--- AB diagnosis ---");
            let abd: String = ctx.eval(r#"
              (function() {
                function AB() {}
                Object.setPrototypeOf(AB, ArrayBuffer);
                Object.setPrototypeOf(AB.prototype, ArrayBuffer.prototype);
                const inst = new AB();
                const tag = Object.prototype.toString.call(inst);
                const isInst = inst instanceof ArrayBuffer;
                const types = require('util/types');
                const isAny = types.isAnyArrayBuffer(inst);
                return 'tag=' + tag + ' isInstance=' + isInst + ' isAny=' + isAny;
              })()
            "#)?;
            println!("{abd}");

            println!("\n--- exact AB test probe ---");
            let abt: String = ctx.eval(r#"
              (function() {
                try {
                  function AB() { }
                  Object.setPrototypeOf(AB, ArrayBuffer);
                  Object.setPrototypeOf(AB.prototype, ArrayBuffer.prototype);
                  Buffer.from(new AB());
                  return 'NO THROW';
                } catch (e) {
                  return 'THREW: code=' + e.code + ' name=' + e.name + ' msg=' + JSON.stringify(e.message) +
                         ' codeIn=' + ('code' in e) + ' msgIn=' + ('message' in e) + ' nameIn=' + ('name' in e) +
                         ' ownKeys=' + JSON.stringify(Object.getOwnPropertyNames(e));
                }
              })()
            "#)?;
            println!("{abt}");

            println!("\n--- Buffer.write(-1) probe ---");
            let bw: String = ctx.eval(r#"
              try {
                Buffer.alloc(9).write('foo', -1);
                'NO THROW';
              } catch (e) {
                _safeForUtf8('code=' + e.code + ' name=' + e.name + ' message=' + e.message);
              }
            "#)?;
            println!("{bw}");

            println!("\n--- typed array errors probe ---");
            let tae: String = ctx.eval(r#"
              (function() {
                const tests = [
                  ['new Uint8Array(4294967296)', () => new Uint8Array(4294967296)],
                  ['Buffer.allocUnsafe(9).write(1, -1)', () => Buffer.allocUnsafe(9).write('a', -1)],
                  ['Buffer.concat with -2', () => Buffer.concat([Buffer.from('a')], -2)],
                  ['Buffer.from(AB-like)', () => { class AB {}; return Buffer.from(new AB()); }],
                ];
                const out = [];
                for (const [name, fn] of tests) {
                  try { fn(); out.push('NOTHROW ' + name); }
                  catch (e) {
                    out.push('THREW ' + name + ': name=' + e.name + ' code=' + e.code + ' message=' + JSON.stringify(e.message));
                  }
                }
                return out.join('\n');
              })()
            "#)?;
            println!("{tae}");

            println!("\n--- buffer.fill probe ---");
            let bfp: String = ctx.eval(r#"
              (function() {
                const out = [];
                const cases = [
                  ['Buffer.allocUnsafe(28).fill("abc")', () => Buffer.allocUnsafe(28).fill('abc')],
                  ['Buffer.alloc(64,10)', () => Buffer.alloc(64, 10)],
                  ['buf.fill.apply', () => { const b = Buffer.allocUnsafe(28); return b.fill.apply(b, ['abc']); }],
                  ['Buffer.alloc(10,"abc").toString()', () => Buffer.alloc(10, 'abc').toString()],
                  ['Buffer.alloc(10,"abc").fill("ե")', () => { const b=Buffer.alloc(10,'abc'); b.fill('է'); return b.toString(); }],
                ];
                for (const [name, fn] of cases) {
                  try { const r = fn(); out.push('OK ' + name + ' length=' + (r && r.length ? r.length : '?')); }
                  catch (e) { out.push('FAIL ' + name + ': ' + e.message); }
                }
                return out.join(' || ');
              })()
            "#)?;
            println!("{bfp}");

            println!("\n--- binding writeBuffer probe ---");
            let bwp: String = ctx.eval(r#"
              (function() {
                try {
                  const binding = internalBinding('fs');
                  const flags = binding.flagsForString ? binding.flagsForString('w') : 1;
                  const fd = binding.open('/tmp/portable-node-bw-' + Date.now(), flags, 0o644);
                  const buf = Buffer.from([0x66, 0x6f, 0x6f]);
                  const written = binding.writeBuffer(fd, buf, 0, buf.byteLength, -1, undefined, {});
                  binding.close(fd);
                  return 'binding.writeBuffer ok, wrote ' + written;
                } catch (e) {
                  return 'FAIL: ' + e.message + '\nstack:\n' + (e.stack||'<empty>');
                }
              })()
            "#)?;
            println!("{bwp}");

            println!("\n--- fs.writeFileSync(Buffer) probe ---");
            let fsw: String = ctx.eval(r#"
              (function() {
                try {
                  const fs = require('fs');
                  const tmp = '/tmp/portable-node-test-' + Date.now();
                  fs.writeFileSync(tmp, Buffer.from([0x66, 0x6f, 0x6f]));
                  const r = fs.readFileSync(tmp, 'utf8');
                  return 'OK: ' + r;
                } catch (e) {
                  return 'FAIL: ' + e.message + '\nstack:\n' + (e.stack||'<empty>');
                }
              })()
            "#)?;
            println!("{fsw}");

            println!("\n--- isolated test-path-zero-length ---");
            let ipz: String = ctx.eval(r#"
              (function() {
                delete moduleCache['builtin:test-path-zero-length-strings'];
                try { require('test-path-zero-length-strings'); return 'PASS'; }
                catch (e) {
                  return _safeForUtf8('FAIL: ' + (e.message||e) +
                         '\nactual=' + JSON.stringify(e.actual) + ' expected=' + JSON.stringify(e.expected) +
                         '\n' + (e.stack||''));
                }
              })()
            "#)?;
            println!("{ipz}");

            println!("\n--- compile-only test-buffer-fill ---");
            let coc: String = ctx.eval(r#"
              (function() {
                const src = __nodeSourceFiles['test-buffer-fill'];
                if (!src) return 'no source';
                try {
                  // Compile only, don't run.
                  new Function('exports','require','module','__filename','__dirname', src);
                  return 'COMPILE OK (' + src.length + ' bytes)';
                } catch (e) {
                  return 'COMPILE FAIL: ' + (e && e.message);
                }
              })()
            "#)?;
            println!("{coc}");

            println!("\n--- Error.stackTraceLimit + prepare probe ---");
            let limit: String = ctx.eval(r#"
              'stackTraceLimit=' + Error.stackTraceLimit + ' prepare=' + (typeof Error.prepareStackTrace) +
              ' freshStack=' + ((new Error()).stack || '<EMPTY>')
            "#)?;
            println!("{limit}");

            println!("\n--- AllocUnsafeSlow probe ---");
            let aus: String = ctx.eval(r#"
              (function() {
                const out = [];
                const cases = [
                  ['allocUnsafeSlow(16)', () => Buffer.allocUnsafeSlow(16)],
                  ['allocUnsafeSlow(16).fill("a", "utf16le")', () => Buffer.allocUnsafeSlow(16).fill('a', 'utf16le')],
                  ['fill("Љ","utf16le")', () => Buffer.allocUnsafeSlow(16).fill('Љ', 'utf16le')],
                  ['fill("ab","utf16le")', () => Buffer.allocUnsafeSlow(16).fill('ab', 'utf16le')],
                  ['fill("","encoding")', () => { const b=Buffer.alloc(5, ''); return b.toString(); }],
                  ['fill([],"")', () => { const b=Buffer.alloc(5, []); return b.toString(); }],
                  ['fill(Buffer.alloc(5))', () => { const b=Buffer.alloc(5, Buffer.alloc(5)); return b.toString(); }],
                  ['Bad-hex throws ERR_INVALID_ARG_VALUE', () => Buffer.from('a'.repeat(1000)).fill('not-hex', 'hex')],
                ];
                for (const [name, fn] of cases) {
                  try { const r = fn(); out.push('OK ' + name + ' len=' + (r && r.length)); }
                  catch (e) { out.push('FAIL ' + name + ': ' + e.message); }
                }
                return out.join(' | ');
              })()
            "#)?;
            println!("{aus}");

            println!("\n--- step-by-step buffer-fill ---");
            let stp: String = ctx.eval(r#"
              (function() {
                const out = [];
                const log = (s) => out.push(s);
                try {
                  delete moduleCache['builtin:test-buffer-fill'];
                  // Build a sandbox where we evaluate the test source line by line is hard.
                  // Instead, run the FIRST few statements of the test manually and see what fails.
                  const common = require('common');
                  const assert = require('assert');
                  const { codes: { ERR_OUT_OF_RANGE } } = require('internal/errors');
                  const { internalBinding } = require('internal/test/binding');
                  log('imports OK');
                  const SIZE = 28;
                  const buf1 = Buffer.allocUnsafe(SIZE);
                  const buf2 = Buffer.allocUnsafe(SIZE);
                  log('bufs OK');
                  // Try testBufs equivalent manually with 'abc'
                  buf1.fill('abc');
                  log('buf1.fill ok');
                  // buf1.fill.apply
                  buf1.fill.apply(buf1, ['abc']);
                  log('fill.apply ok');
                  // assert.deepStrictEqual
                  const b1 = Buffer.from('abc').fill('abc');
                  log('Buffer.from.fill ok');
                  assert.deepStrictEqual(buf1, buf1);
                  log('deepStrictEqual ok');
                  // testBufs uses arguments-as-array
                  function testBufs(string, offset, length, encoding) {
                    buf1.fill.apply(buf1, arguments);
                    assert.deepStrictEqual(buf1.fill.apply(buf1, arguments),
                                           buf1.fill.apply(buf1, arguments));
                  }
                  testBufs('abc');
                  log('testBufs("abc") ok');
                  testBufs('Ȣaa');
                  log('testBufs unicode ok');
                  // Now assert.throws with fill('a', -1)
                  assert.throws(() => Buffer.allocUnsafe(8).fill('a', -1), { code: 'ERR_OUT_OF_RANGE' });
                  log('throws fill(-1) ok');
                  return out.join(' | ');
                } catch (e) {
                  out.push('FAIL: ' + (e && e.message || e));
                  return out.join(' | ');
                }
              })()
            "#)?;
            println!("{stp}");

            println!("\n--- direct test-buffer-fill probe ---");
            // Force a re-run with Error captured by global catch so we get the
            // raw stack.
            let _: () = ctx.eval(r#"
              delete moduleCache['builtin:test-buffer-fill'];
              try { require('test-buffer-fill'); globalThis._tbfErr = null; }
              catch (e) { globalThis._tbfErr = e; }
            "#)?;
            let raw_stack: String = ctx.eval(r#"
              (function() {
                const e = globalThis._tbfErr;
                if (!e) return 'PASS';
                return _safeForUtf8('msg=' + (e.message || e) + '\nstack=' + (e.stack || '<empty>'));
              })()
            "#)?;
            println!("{raw_stack}");

            println!("\n--- string_decoder views probe ---");
            let views_probe: String = ctx.eval(r#"
              (function() {
                const { StringDecoder } = require('string_decoder');
                const common = require('common');
                const out = [];
                const inputBuffer = Buffer.from('Hello, world!\n'.repeat(8), 'utf8');
                const views = common.getArrayBufferViews(inputBuffer);
                for (const view of views) {
                  try {
                    const d = new StringDecoder('utf8');
                    const s = d.write(view);
                    out.push('OK ' + view.constructor.name + ': ' + (s === inputBuffer.toString('utf8') ? 'matches' : 'mismatch len=' + s.length));
                  } catch (e) {
                    out.push('FAIL ' + view.constructor.name + ': ' + e.message);
                  }
                }
                return out.join('\n');
              })()
            "#)?;
            println!("{views_probe}");

            println!("\n--- lone surrogate buffer probe ---");
            let lone: String = ctx.eval(r#"
              (function() {
                const out = [];
                // High surrogate as utf16le bytes: 3D D8
                const hi = Buffer.from('3DD8', 'hex').toString('utf16le');
                out.push('hi length=' + hi.length + ' code=' + hi.charCodeAt(0).toString(16));
                // Low surrogate as utf16le bytes: 4D DC
                const lo = Buffer.from('4DDC', 'hex').toString('utf16le');
                out.push('lo length=' + lo.length + ' code=' + lo.charCodeAt(0).toString(16));
                // Combined
                out.push('hi+lo equal=' + ((hi+lo) === '👍') + ' codes=' + (hi+lo).charCodeAt(0).toString(16) + ',' + (hi+lo).charCodeAt(1).toString(16));
                return out.join('\n');
              })()
            "#)?;
            println!("{lone}");

            println!("\n--- surrogate-pair split probe ---");
            let surr: String = ctx.eval(r#"
              (function() {
                const { StringDecoder } = require('string_decoder');
                const out = [];
                const input = Buffer.from('3DD84DDC', 'hex');
                // Try all 1-byte-at-a-time + 2+2 + 1+3 + 3+1 splits.
                const splits = [
                  [[0,1],[1,2],[2,3],[3,4]],
                  [[0,2],[2,4]],
                  [[0,1],[1,4]],
                  [[0,3],[3,4]],
                  [[0,4]],
                ];
                for (const seq of splits) {
                  const d = new StringDecoder('utf16le');
                  let s = '';
                  for (const [a, b] of seq) s += d.write(input.slice(a, b));
                  s += d.end();
                  const codes = [];
                  for (let i = 0; i < s.length; i++) codes.push(s.charCodeAt(i).toString(16));
                  out.push(JSON.stringify(seq) + ' => codes=[' + codes.join(',') + ']' +
                           ' ok=' + (s === '👍'));
                }
                return out.join('\n');
              })()
            "#)?;
            println!("{surr}");

            // Run test-path-zero-length-strings directly and capture full stack.
            println!("\n--- direct test-path probe ---");
            let direct: String = ctx.eval(r#"
              (function() {
                delete moduleCache['builtin:test-path-zero-length-strings'];
                try {
                  require('test-path-zero-length-strings');
                  return 'PASS';
                } catch (e) {
                  return 'FAIL: ' + (e && e.message) + '\nactual=' + JSON.stringify(e.actual) +
                         ' expected=' + JSON.stringify(e.expected) +
                         '\nstack=\n' + ((e && e.stack) || '<no stack>');
                }
              })()
            "#)?;
            println!("{direct}");

            // Step through test-path-zero-length-strings's assertions one by
            // one to see which fails when run in this context.
            println!("\n--- path test in-context probe ---");
            let in_ctx_probe: String = ctx.eval(r#"
              (function() {
                const assert = require('assert');
                const path = require('path');
                const pwd = process.cwd();
                const out = [];
                const cases = [
                  ['posix.join("")', () => path.posix.join(''), '.'],
                  ['posix.join("","")', () => path.posix.join('', ''), '.'],
                  ['win32.join("")', () => path.win32.join(''), '.'],
                  ['win32.join("","")', () => path.win32.join('', ''), '.'],
                  ['join(pwd)', () => path.join(pwd), pwd],
                  ['join(pwd,"")', () => path.join(pwd, ''), pwd],
                  ['posix.normalize("")', () => path.posix.normalize(''), '.'],
                  ['win32.normalize("")', () => path.win32.normalize(''), '.'],
                  ['normalize(pwd)', () => path.normalize(pwd), pwd],
                  ['posix.isAbsolute("")', () => path.posix.isAbsolute(''), false],
                  ['win32.isAbsolute("")', () => path.win32.isAbsolute(''), false],
                  ['resolve("")', () => path.resolve(''), pwd],
                  ['resolve("","")', () => path.resolve('', ''), pwd],
                  ['relative("",pwd)', () => path.relative('', pwd), ''],
                  ['relative(pwd,"")', () => path.relative(pwd, ''), ''],
                  ['relative(pwd,pwd)', () => path.relative(pwd, pwd), ''],
                ];
                for (const [name, fn, exp] of cases) {
                  let got;
                  try { got = fn(); } catch (e) { got = '<threw: ' + e.message + '>'; }
                  out.push((got === exp ? 'OK   ' : 'FAIL ') + name + ' got=' + JSON.stringify(got) + ' exp=' + JSON.stringify(exp));
                }
                return out.join('\n');
              })()
            "#)?;
            println!("{in_ctx_probe}");

            println!("\n--- Node's own tests ---");
            let mut runner = String::from("[");
            for (name, _) in NODE_TEST_SOURCES {
                runner.push_str(&format!("__runNodeTest({name:?}),"));
            }
            runner.push_str("].join('\\n')");
            let test_out: String = ctx.eval(runner.as_str())?;
            let passed = test_out.lines().filter(|l| l.starts_with("PASS")).count();
            let failed = test_out.lines().filter(|l| l.starts_with("FAIL")).count();
            // Show all PASS lines (one-liners), then any FAIL with details.
            for line in test_out.lines() {
                if line.starts_with("PASS") {
                    println!("{}", line);
                }
            }
            if failed > 0 {
                println!("\n--- failures ---");
                let mut in_fail = false;
                for line in test_out.lines() {
                    if line.starts_with("FAIL") { println!("{}", line); in_fail = true; }
                    else if in_fail && (line.starts_with("      at") || line.starts_with("  ")) {
                        println!("{}", line);
                    } else if line.starts_with("PASS") { in_fail = false; }
                }
            }
            println!("\n=== {} passed, {} failed ({} total) ===", passed, failed, passed + failed);
            Ok(())
        })
    })?;

    // Phase 4: --serve mode. If PORTABLE_NODE_SERVE is set, keep the embedded
    // http.createServer alive forever (or up to PORTABLE_NODE_SERVE seconds)
    // so a human can `curl http://127.0.0.1:<port>/` from another terminal.
    if let Ok(serve) = std::env::var("PORTABLE_NODE_SERVE") {
        let port: u32 = context.with(|ctx| ctx.eval(r#"globalThis._httpPort || 0"#).unwrap_or(0));
        if port == 0 {
            eprintln!("PORTABLE_NODE_SERVE set but http server has no port assigned");
        } else {
            let secs: u64 = serve.parse().unwrap_or(0);
            if secs == 0 {
                println!("\n*** SERVING http://127.0.0.1:{port}/ — Ctrl-C to stop ***\n");
                drive_event_loop_for(&context, &rt, None)?;
            } else {
                println!("\n*** SERVING http://127.0.0.1:{port}/ for {secs}s ***\n");
                drive_event_loop_for(&context, &rt, Some(secs * 1000))?;
            }
        }
    }
    Ok(())
}

/// Standalone script-runner mode: `portable-node <script.js> [args...]`.
/// Bootstraps the runtime exactly like the self-test path, then `require()`s
/// the user's script (absolute path), then drives the event loop until idle
/// (which, for a listening HTTP server, is "forever").
fn run_script(script_path: &str) -> std::result::Result<(), String> {
    let abs_path = std::fs::canonicalize(script_path)
        .map_err(|e| format!("cannot resolve script {script_path:?}: {e}"))?;
    let abs_str = abs_path.to_string_lossy().to_string();

    let rt = Runtime::new().map_err(|e| e.to_string())?;
    let context = Context::full(&rt).map_err(|e| e.to_string())?;

    context.with(|ctx| -> std::result::Result<(), String> {
        run_with_diagnostics(&ctx, || -> Result<()> {
            ctx.globals().set("__internalBinding", Func::from(dispatch_internal_binding))?;
            host::install(ctx.clone())?;

            // Register Node sources. Test sources are skipped — they're not
            // needed for running user code and would bloat the binary's
            // startup cost.
            let registry: Object<'_> = ctx.eval("({})")?;
            for (name, src) in NODE_SOURCES.iter() {
                registry.set(*name, *src)?;
            }
            ctx.globals().set("__nodeSourceFiles", registry)?;

            ctx.eval::<(), _>(include_str!("../js/bootstrap.js"))?;

            // require() the user's script by absolute path. The resolver
            // treats absolute paths as step 2 of the CJS algorithm.
            ctx.globals().set("__userScriptPath", abs_str.as_str())?;
            ctx.eval::<(), _>("require(__userScriptPath)")?;
            Ok(())
        })
    })?;

    // Run the event loop with no deadline. drive_event_loop_for exits when
    // there's no I/O or timer work pending — a listening server is pending
    // forever, so this blocks until the script winds down all its work.
    drive_event_loop_for(&context, &rt, None)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // First positional arg = script path. The legacy self-test harness is
    // still reachable with no args or `--self-test`.
    let want_script = args.len() >= 2
        && args[1] != "--self-test"
        && !args[1].is_empty();
    let result = if want_script {
        run_script(&args[1])
    } else {
        run()
    };
    if let Err(e) = result {
        eprintln!("portable-node fatal:\n{e}");
        std::process::exit(1);
    }
}
