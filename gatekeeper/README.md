# gatekeeper

A small, fail-closed HTTP front gate. Put it in front of your services so a
single shared-secret token gates every request — and so you **can't accidentally
expose something you didn't mean to**.

The core property: **a route is private unless it explicitly sets `public =
true`.** Forgetting the flag fails safe (the route stays private). Tools like
Caddy/nginx are the opposite — auth is opt-in per route, so a forgotten directive
exposes a service. This makes the unsafe direction the one you have to ask for.

It's deliberately small: synchronous, thread-per-request (`tiny_http`), ~6 source
files, no async runtime. The whole request path — match route, normalize path,
check token, serve/proxy — is linear and auditable.

## Quick start

```toml
# gatekeeper.toml
bind = "0.0.0.0:443"
tls_cert = "/etc/gatekeeper/cert.pem"   # see "TLS and certificates" below
tls_key  = "/etc/gatekeeper/key.pem"
unmatched_status = 404            # or 403

[[route]]
path = "/blog"
static = "./site"
public = true                     # explicit opt-in to public

[[route]]
path = "/metrics"
proxy = "127.0.0.1:5557"          # 'public' omitted -> PRIVATE (default-deny)
```

```sh
export GATEKEEPER_TOKEN=$(head -c 32 /dev/urandom | base64)
gatekeeper --config gatekeeper.toml
```

- Public:  `curl https://computer.jimmyhmiller.com/blog/`
- Private: `curl -H "Authorization: Bearer $GATEKEEPER_TOKEN" https://computer.jimmyhmiller.com/metrics`
  (or a `gatekeeper=<token>` cookie for browsers)

## Config reference

Top level:

| key                | default        | meaning |
|--------------------|----------------|---------|
| `bind`             | `0.0.0.0:443`  | listen address |
| `tls_cert`/`tls_key` | none         | PEM files; set **both** for HTTPS, omit both for plain HTTP |
| `unmatched_status` | `404`          | response for a path matching no route (`404` hides existence; `403` says forbidden). Either way: denied. |
| `[[route]]`        | —              | one or more routes |

Per route:

| key      | meaning |
|----------|---------|
| `path`     | path prefix, e.g. `/blog`. Must start with `/`, no trailing `/`. |
| `static`   | serve this directory (exactly one of `static`/`proxy`/`function`) |
| `proxy`    | reverse-proxy to this `host:port` (exactly one target kind) |
| `function` | invoke a Rust function dylib in process (exactly one target kind) — see [Serverless functions](#serverless-functions) |
| `public`   | `true` = no auth. **Default `false`.** |

The **token is never in the config file** — it comes from `$GATEKEEPER_TOKEN` or
`--token-file <path>`. Boot fails if any private route exists and no token is set.

## Serverless functions

Besides serving static files and proxying to a long-running upstream, a route can
invoke a **Rust function compiled as a dynamic library**, loaded into the gate on
first request and kept warm after. No port to manage, no separate process, no
service to keep running — you write a handler, the gate runs it behind the same
default-deny auth.

The app shouldn't have to worry about any of this, so it doesn't. You write one
function against the `gatekeeper-fn` crate:

```rust
// funcs/hello/src/lib.rs  (crate-type = ["cdylib"])
use gatekeeper_fn::{handler, Request, Response};

#[handler]
fn app(req: Request) -> Response {
    match req.path() {
        "/health" => Response::text("ok"),
        _ => Response::json(r#"{"hello":"world"}"#),
    }
}
```

```sh
cargo build -p hello-fn --release
```

and point a route at the resulting dylib:

```toml
[[route]]
path = "/api"
function = "target/release/libhello_fn.so"   # .dylib on macOS, .dll on Windows
# 'public' omitted -> PRIVATE, like any other route (default-deny holds)
```

`req.path()` is the path **after** the route prefix (so `/api/health` arrives as
`/health`), already normalized and traversal-checked by the gate. You get
`method()`, `query()`, `headers()`/`header(name)`, and `body()`/`text()`;
`Response` has `text`/`json`/`html`/`status`/`new` plus a `header(...)` builder.

### How it works (and why it's safe to run in process)

The `#[handler]` macro generates a tiny C-ABI surface (`gk_handle` / `gk_free` /
`gk_abi_version`) — defined in the `gatekeeper-abi` crate, the *entire* contract
across the boundary. The gate `dlopen`s the dylib, **checks its ABI version and
refuses to call a mismatch**, marshals the request into `#[repr(C)]` structs,
calls the handler, copies the response out, and frees it back through the dylib's
own deallocator (each side frees only what it allocated — no shared allocator
assumption).

- **A panic in your handler becomes a 500, not a crash.** The SDK catches the
  unwind on the function side so it never crosses the C ABI; the gate stays up.
  (Verified by a test that panics on purpose and asserts the gate survives.)
- **A failed load fails closed** (502), like an unreachable proxy upstream.
- The unsafe marshalling/free path is covered by an integration test run under
  valgrind: **0 errors, 0 bytes definitely lost** across many invocations.

Functions are *trusted native code* you deploy yourself — the same trust level as
a proxy upstream. The gate does not sandbox arbitrary dylibs; don't point a
`function` route at code you wouldn't run.

### The crates

| crate                  | who depends on it | what it is |
|------------------------|-------------------|------------|
| `gatekeeper-abi`       | gate **and** function | the `#[repr(C)]` request/response + ABI version. Tiny, no deps. |
| `gatekeeper-fn`        | your function     | the `Request`/`Response` types + `#[handler]` macro. The only crate your app needs. |
| `gatekeeper-fn-macro`  | (re-exported)     | the proc-macro behind `#[handler]`. |

`funcs/hello` is a complete worked example. After `cargo build -p hello-fn`, run
the gate with a route pointing at `target/debug/libhello_fn.so` and `curl` it.

## TLS and certificates

For a public `https://` domain, browsers require a certificate signed by a
trusted Certificate Authority. **Let's Encrypt** is a free CA that issues one
after you prove you control the domain; the result is two PEM files (a
certificate chain and a private key) that gatekeeper's `tls_cert`/`tls_key`
point at.

gatekeeper terminates TLS with rustls but does **not** do ACME in-process (that
would roughly double the dependency tree and re-add an async runtime). Instead an
external tool gets and auto-renews the cert, and gatekeeper **hot-reloads it on
`SIGHUP`** — no restart, no dropped connections.

### One-time setup with acme.sh (zero extra Rust deps)

[`acme.sh`](https://github.com/acmesh-official/acme.sh) is a small, widely-used
shell ACME client that installs its own renewal cron.

```sh
# install
curl https://get.acme.sh | sh -s email=jimmyhmiller@gmail.com

# issue a cert for the domain (standalone briefly binds :80 to prove control;
# alternatively use a DNS challenge — see acme.sh docs)
acme.sh --issue --standalone -d computer.jimmyhmiller.com

# install the cert to a stable path AND tell acme.sh to SIGHUP gatekeeper after
# every renewal, so the new cert is picked up live:
acme.sh --install-cert -d computer.jimmyhmiller.com \
  --fullchain-file /etc/gatekeeper/cert.pem \
  --key-file       /etc/gatekeeper/key.pem \
  --reloadcmd      "kill -HUP \$(pidof gatekeeper)"
```

Then in `gatekeeper.toml`:

```toml
bind = "0.0.0.0:443"
tls_cert = "/etc/gatekeeper/cert.pem"
tls_key  = "/etc/gatekeeper/key.pem"
```

Let's Encrypt certs last 90 days; acme.sh's cron renews them automatically and
the `--reloadcmd` SIGHUPs gatekeeper, which re-reads the files in place. On boot
gatekeeper prints its PID and the exact reload command.

**Fail-safe:** if a reload finds a missing or invalid cert, gatekeeper logs the
error and **keeps serving the current cert** rather than going down — a botched
renewal can't take the site offline.

### Without a public CA

If you don't have a public domain — e.g. you're behind a Cloudflare Tunnel,
Tailscale, or a reverse proxy that already does TLS — omit `tls_cert`/`tls_key`
and run gatekeeper as plain HTTP on localhost. The terminator in front handles
certificates.

## The safety guarantees

1. **Default-deny.** `public` defaults false. Unmatched paths are denied. A
   private route with no/invalid token → 401.
2. **No path-trick bypass.** Request paths are percent-decoded and walked
   component-by-component; any `..` (encoded or not) → **400**, before routing.
   Prefix matching only at `/` boundaries, so `/admin` never matches
   `/administrator`. Case-sensitive. Static serving additionally canonicalizes
   and confirms the file stays within the served root (catches symlink escapes).
3. **Longest-prefix wins**, so a public subpath can sit under a private parent
   (`/admin/docs` public under `/admin` private) — the more specific route wins.
4. **Constant-time token check** (`subtle`), so the token can't be recovered by
   timing. Header (`Authorization: Bearer`) takes precedence over cookie; a
   present-but-wrong header is not silently bypassed by a good cookie.
5. **Loud exposure report at every boot** listing every public and private route,
   so you can eyeball "did I mean to make these public?". Use `--check` to print
   it and validate the config without binding.

These are enforced in one place (`Router::decide`) and verified by a property
test (`tests/safety.rs`) that asserts, over thousands of generated configs and
paths, that **no private route is ever allowed without a valid token**.

## CLI

```
gatekeeper --config <file> [--token-file <file>] [--check]

  --config       config TOML (default ./gatekeeper.toml)
  --token-file   read shared token from a file (else $GATEKEEPER_TOKEN)
  --check        validate config + print exposure report, then exit
```

## Not included (by design)

Rate limiting, multiple/per-route tokens, request logging to a sink, hot config
reload, OIDC/accounts, in-process ACME. Add a tunnel or a terminator in front if
you need more. This stays a small, legible gate.
