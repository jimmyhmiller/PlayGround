//! Native TanStack Start route manifest generation.
//!
//! TanStack Start's server renderer imports a build-generated virtual module,
//! `tanstack-start-manifest:v`, from `@tanstack/start-server-core`'s
//! `router-manifest.js`. That module exports `tsrStartManifest()`, whose
//! `routes` map tells the server which client asset URLs to `<link rel=preload>`
//! and script-inject for each route. It is *build-output-dependent*: the preload
//! URLs are the client build's own emitted chunk URLs, so it cannot be read from
//! a package — it must be generated from Diffpack's own client chunk graph.
//!
//! This is the cross-environment coordination point:
//!
//! * The **client** build persists a route -> chunk mapping
//!   ([`ClientRouteManifest`]) as `.diffpack-output/client-manifest.json`. The
//!   bundler derives it from the same dynamic-import roots it emits as chunks, so
//!   the recorded chunk URLs are exactly the files on disk.
//! * The **server** build reads that JSON and generates the
//!   `tanstack-start-manifest:v` module source natively
//!   ([`ClientRouteManifest::to_start_manifest_source`]), which the bundler loads
//!   as a virtual module and emits as its own server chunk.
//!
//! The mapping is honest about Diffpack's chunking model: each route's `preloads`
//! are the client chunk(s) that hold its code-split properties (its
//! `?tsr-split=*` chunks), and `__root__`'s preload/script is the entry chunk,
//! which statically bundles the root route and all shared code. It is not a
//! transliteration of the reference's Vite/Rollup chunk layout (which shares
//! vendor chunks under `/assets/`); it reflects Diffpack's actual emitted files.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// The name of the JSON artifact the client build writes and the server build
/// reads to coordinate route -> chunk preload mapping across environments.
pub const CLIENT_MANIFEST_FILE: &str = "client-manifest.json";

/// The virtual module specifier TanStack Start's `router-manifest.js` imports.
pub const START_MANIFEST_SPECIFIER: &str = "tanstack-start-manifest:v";

/// The virtual module specifier `@tanstack/start-server-core`'s
/// `loadVirtualModule.js` references for dev-time head scripts. `router-manifest.js`
/// only reads it under `process.env.TSS_DEV_SERVER === "true"`, so it is never
/// used at runtime in a production build; but the `import("...")` literal sits in
/// the switch, so static resolution needs it to exist. (Some `react-start`
/// versions reference it, e.g. 1.133, while others do not, e.g. 1.169.)
pub const INJECTED_HEAD_SCRIPTS_SPECIFIER: &str = "tanstack-start-injected-head-scripts:v";

/// The native source for `tanstack-start-injected-head-scripts:v`. A production
/// build injects no head scripts (those are a dev-server concern, and Diffpack's
/// own dev server does its live-reload injection separately), so the exported
/// value is `undefined` — the consumer guards on `if (injectedHeadScripts)`.
pub fn injected_head_scripts_module_source() -> String {
    "// Generated natively by Diffpack. Injected head scripts are a dev-server-only\n\
     // concern (TSS_DEV_SERVER); a production build injects none.\n\
     export const injectedHeadScripts = undefined;\n\
     export default { injectedHeadScripts };\n"
        .to_string()
}

/// The route id TanStack uses for the root route (`createRootRoute`). Its
/// preloads are the entry chunk, which statically bundles the root and all shared
/// code in Diffpack's chunking model.
pub const ROOT_ROUTE_ID: &str = "__root__";

/// A route -> client-chunk mapping derived from the client build, plus the URL
/// base and entry chunk. Round-trips through `client-manifest.json` so the server
/// build can consume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientRouteManifest {
    /// The public URL base the client chunks are served from (e.g. `/`).
    pub base: String,
    /// The entry chunk file name (e.g. `client.js`), served at `<base><entry>`.
    pub entry: String,
    /// Route id -> the client chunk file names that must be preloaded for it. The
    /// `__root__` key maps to the entry chunk.
    pub routes: BTreeMap<String, Vec<String>>,
}

impl ClientRouteManifest {
    /// Serializes to the `client-manifest.json` artifact.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let value = serde_json::json!({
            "base": self.base,
            "entry": self.entry,
            "routes": self.routes,
        });
        let text = serde_json::to_string_pretty(&value)
            .map_err(|error| format!("cannot serialize client manifest: {error}"))?;
        fs::write(path, text)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))
    }

    /// Reads the `client-manifest.json` artifact the client build produced.
    ///
    /// A missing or malformed file is a hard, specific error — never a silent
    /// empty manifest — so the server build fails loudly when the client build
    /// has not run first.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|error| {
            format!(
                "cannot read client route manifest {}: {error}; \
                 run `diffpack build-app <root> client` before the server build",
                path.display()
            )
        })?;
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
        let base = value["base"]
            .as_str()
            .ok_or_else(|| format!("{}: missing string `base`", path.display()))?
            .to_string();
        let entry = value["entry"]
            .as_str()
            .ok_or_else(|| format!("{}: missing string `entry`", path.display()))?
            .to_string();
        let routes_object = value["routes"]
            .as_object()
            .ok_or_else(|| format!("{}: missing object `routes`", path.display()))?;
        let mut routes = BTreeMap::new();
        for (route_id, chunks) in routes_object {
            let files = chunks
                .as_array()
                .ok_or_else(|| format!("{}: route `{route_id}` is not an array", path.display()))?
                .iter()
                .map(|chunk| {
                    chunk.as_str().map(str::to_string).ok_or_else(|| {
                        format!("{}: route `{route_id}` has a non-string chunk", path.display())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            routes.insert(route_id.clone(), files);
        }
        Ok(Self {
            base,
            entry,
            routes,
        })
    }

    /// The public URL of a chunk file (`<base><file>`), with a single separating
    /// slash regardless of whether `base` has a trailing one.
    fn url_of(&self, file: &str) -> String {
        format!("{}/{}", self.base.trim_end_matches('/'), file)
    }

    /// Generates the native `tanstack-start-manifest:v` module source: the
    /// `tsrStartManifest` factory the server's `router-manifest.js` consumes.
    ///
    /// Matches the contract `getStartManifest` reads: a `routes` map keyed by
    /// route id, each entry carrying `preloads` (and, for `__root__`, the entry
    /// `scripts` tag). URLs are Diffpack's own emitted client chunk URLs.
    pub fn to_start_manifest_source(&self) -> String {
        let mut source = String::new();
        source.push_str(
            "// Generated natively by Diffpack from the client build's route/chunk graph.\n",
        );
        // `clientEntry` is how some react-start versions (e.g. 1.133) inject the
        // browser entry: `import('${startManifest.clientEntry}')`. Others (e.g.
        // 1.169) use the per-route `scripts` array below instead. Emit both so the
        // manifest drives hydration across versions; an unused field is ignored.
        let client_entry = json_string(&self.url_of(&self.entry));
        source.push_str(&format!(
            "const tsrStartManifest = () => ({{ clientEntry: {client_entry}, routes: {{\n"
        ));
        for (route_id, chunks) in &self.routes {
            let key = json_string(route_id);
            let preloads = chunks
                .iter()
                .map(|file| json_string(&self.url_of(file)))
                .collect::<Vec<_>>()
                .join(", ");
            source.push_str(&format!("  {key}: {{ preloads: [{preloads}]"));
            if route_id == ROOT_ROUTE_ID {
                let src = json_string(&self.url_of(&self.entry));
                source.push_str(&format!(
                    ", scripts: [{{ attrs: {{ type: \"module\", async: true, src: {src} }} }}]"
                ));
            }
            source.push_str(" },\n");
        }
        source.push_str("} });\n");
        // Exposed two ways so both consumption paths resolve `tsrStartManifest`:
        // a named export for a native ESM `import { tsrStartManifest }`, and a
        // default export carrying it for Diffpack's runtime dynamic-import glue,
        // whose `require.dynamic` resolves an internal chunk to its `.default`.
        source.push_str("export { tsrStartManifest };\n");
        source.push_str("export default { tsrStartManifest };\n");
        source
    }
}

/// JSON-encodes a string (quotes + escapes) so route ids and URLs embed safely in
/// the generated JavaScript source.
fn json_string(value: &str) -> String {
    serde_json::Value::String(value.to_string()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injected_head_scripts_module_exports_the_guarded_name() {
        let source = injected_head_scripts_module_source();
        // The consumer destructures `const { injectedHeadScripts } = await import(...)`
        // and guards on `if (injectedHeadScripts)`, so a named `undefined` export is
        // exactly what a production build needs.
        assert!(source.contains("export const injectedHeadScripts = undefined;"), "{source}");
        assert!(INJECTED_HEAD_SCRIPTS_SPECIFIER.ends_with(":v"));
    }

    fn sample() -> ClientRouteManifest {
        let mut routes = BTreeMap::new();
        routes.insert(ROOT_ROUTE_ID.to_string(), vec!["client.js".to_string()]);
        routes.insert("/".to_string(), vec!["client.chunk-3.js".to_string()]);
        routes.insert(
            "/posts/$postId".to_string(),
            vec![
                "client.chunk-7.js".to_string(),
                "client.chunk-8.js".to_string(),
            ],
        );
        ClientRouteManifest {
            base: "/".to_string(),
            entry: "client.js".to_string(),
            routes,
        }
    }

    #[test]
    fn round_trips_through_json() {
        let manifest = sample();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CLIENT_MANIFEST_FILE);
        manifest.write(&path).unwrap();
        let read = ClientRouteManifest::read(&path).unwrap();
        assert_eq!(manifest, read);
    }

    #[test]
    fn a_missing_client_manifest_is_a_specific_error() {
        let dir = tempfile::tempdir().unwrap();
        let error = ClientRouteManifest::read(&dir.path().join("nope.json")).unwrap_err();
        assert!(error.contains("client route manifest"), "{error}");
        assert!(error.contains("build-app"), "{error}");
    }

    #[test]
    fn generates_the_tsr_start_manifest_contract() {
        let source = sample().to_start_manifest_source();
        // The exact export the server's router-manifest.js consumes, with the
        // top-level `clientEntry` some versions inject via `import(...)`.
        assert!(
            source.contains("const tsrStartManifest = () => ({ clientEntry: \"/client.js\", routes: {"),
            "{source}"
        );
        assert!(source.contains("export { tsrStartManifest };"), "{source}");
        // The root route preloads and scripts the entry chunk URL.
        assert!(
            source.contains("\"__root__\": { preloads: [\"/client.js\"], scripts: [{ attrs: { type: \"module\", async: true, src: \"/client.js\" } }] }"),
            "{source}"
        );
        // A leaf route preloads its own client chunk URL.
        assert!(source.contains("\"/\": { preloads: [\"/client.chunk-3.js\"] }"), "{source}");
        // A route with several split chunks lists them all.
        assert!(
            source.contains("\"/posts/$postId\": { preloads: [\"/client.chunk-7.js\", \"/client.chunk-8.js\"] }"),
            "{source}"
        );
    }
}
