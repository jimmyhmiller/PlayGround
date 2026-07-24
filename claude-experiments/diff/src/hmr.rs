//! Dev-only Hot Module Replacement (HMR) with React Fast Refresh.
//!
//! This module holds the JavaScript that turns Diffpack's registry runtime into a
//! hot-reloadable one, the browser-side HMR client (a WebSocket channel plus the
//! React Fast Refresh preamble), and the source-to-source Fast Refresh
//! instrumentation applied to client component modules.
//!
//! EVERYTHING here is DEV-ONLY. It is emitted only when the dev server threads
//! `EmitOptions { hmr: true, .. }` (which `build-app` never sets), so production
//! output is byte-for-byte unaffected. The bundler asserts this by gating each
//! injection on the `hmr` flag; there is no code path from `build-app` into any of
//! it.
//!
//! Design:
//!
//! * The singleton runtime (`bundler::render_runtime`) keeps its `register`/
//!   `require` behaviour and, in HMR mode, also installs a per-module
//!   `module.hot` (accept/dispose/invalidate), a `replace(id, factory, map)`, a
//!   `hmrApply(ids)` that runs the accept/dispose propagation for the browser, and
//!   a `serverInvalidate(ids, chunks)` the Node control endpoint calls. Dynamic
//!   `import()` of a chunk is made version-aware so a re-emitted server chunk is
//!   re-fetched fresh (Node caches ESM by URL) instead of returning stale code.
//! * A re-imported chunk carrying the `__diffpack_hmr` marker in its URL registers
//!   its new factories WITHOUT eager-executing (the register-only guard), so the
//!   browser can then drive the update through the accept protocol.
//! * Fast Refresh is two cooperating pieces. (1) The per-component instrumentation
//!   is oxc's NATIVE React Refresh transform (`transform::transform_module_with_
//!   options(refresh=true)`) — the `react-refresh/babel` equivalent done in Rust,
//!   no Node — injecting the `$RefreshReg$` registrations and `$RefreshSig$` hook
//!   signatures the runtime needs to detect a compatible edit and preserve state.
//!   (2) [`fast_refresh_footer`], gated by [`is_refresh_boundary`], appends the HMR
//!   accept wiring: it self-accepts and, on update, runs the runtime's boundary
//!   check. That check is fed PLAIN data-property copies of the exports, because
//!   Diffpack's registry exposes named exports as live-binding GETTERS and some
//!   react-refresh versions reject getter descriptors (a real ESM namespace reports
//!   data descriptors there) — without the copy, every `export const Foo` component
//!   would force a full reload. The dev client build also runs the DEVELOPMENT
//!   React (the client preamble sets `NODE_ENV=development` before the entry)
//!   because production React exposes no Fast Refresh renderer hook.

use std::path::Path;

use crate::transform::Target;

/// The well-known global the HMR client reads to reach the live runtime, so the
/// client never needs to know the per-entry runtime key.
pub const RUNTIME_GLOBAL: &str = "__diffpack_hmr_runtime";

/// URL marker a re-imported chunk carries so its tail registers factories but does
/// NOT eager-execute (the browser drives the update afterwards).
pub const REIMPORT_MARKER: &str = "__diffpack_hmr";

/// Dynamic-import require for HMR ESM builds: version-aware so a re-emitted chunk
/// (same path) is re-fetched with a fresh `?v=` query rather than served from the
/// host's module cache.
///
/// Like the production form in `bundler::render_runtime`, the chunk is imported
/// only for its REGISTRATION side effect and the requested module is then resolved
/// by runtime id out of the shared registry. Reading the chunk's default export
/// instead would assume the chunk holds exactly the requested root, which stopped
/// being true once chunks became a partition and started carrying shared code.
pub const REQUIRE_DYNAMIC_ESM_HMR: &str = r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null){const __v=__hmrVersions[chunk[0]];const __u=__v?chunk[0]+(chunk[0].indexOf("?")>=0?"&v=":"?v=")+__v:chunk[0];return import(__u).then(()=>__require(chunk[1]));}return __require(chunk[1]);};"#;

/// The HMR bookkeeping + apply/propagate/invalidate methods, injected into the
/// singleton runtime IIFE (main chunk, HMR builds only) right before it returns.
/// Everything here closes over the runtime internals (`__modules`, `__maps`,
/// `__chunks`, `__cache`, `__require`).
pub const RUNTIME_METHODS: &str = r#"
const __hmrVersions=Object.create(null);
const __hmrData=Object.create(null);
const __hmrEntries=Object.create(null);
function __hmrEntry(id){return __hmrEntries[id]||(__hmrEntries[id]={selfAccept:false,selfCallbacks:[],depCallbacks:[],disposers:[]});}
function __makeHot(id){
  const data=__hmrData[id]||(__hmrData[id]=Object.create(null));
  return {
    data,
    accept(dep,cb){
      const entry=__hmrEntry(id);
      if(typeof dep==="function"||dep===undefined){entry.selfAccept=true;if(typeof dep==="function")entry.selfCallbacks.push(dep);}
      else{const deps=Array.isArray(dep)?dep:[dep];for(const d of deps)entry.depCallbacks.push([d,cb]);entry.selfAccept=entry.selfAccept;}
    },
    dispose(cb){__hmrEntry(id).disposers.push(cb);},
    invalidate(){__hmrReload("module "+id+" invalidated");},
    decline(){},
    on(){},
  };
}
function __importers(id){const out=[];for(const k in __maps){const m=__maps[k];for(const s in m){if(m[s]===id){out.push(+k);break;}}}return out;}
// Modules that reach `id` through a dynamic import (a code-split boundary): the
// chunk map records `spec -> [chunkPath, targetId]`.
function __dynamicImporters(id){const out=[];for(const k in __chunks){const m=__chunks[k];for(const s in m){const c=m[s];if(c&&c[1]===id){out.push(+k);break;}}}return out;}
function __disposeModule(id){const e=__hmrEntries[id];if(e){for(const d of e.disposers){try{d(__hmrData[id]);}catch(err){console.error(err);}}e.selfCallbacks=[];e.disposers=[];e.depCallbacks=[];e.selfAccept=false;}}
function __hmrReload(reason){if(typeof location!=="undefined"&&location.reload){console.log("[diffpack hmr] full reload: "+reason);location.reload();}else{console.warn("[diffpack hmr] full reload required ("+reason+")");}}
function __replace(id,factory,map){__modules[id]=factory;if(map)__maps[id]=map;return __hmrApply([id]);}
function __bumpVersion(chunk){__hmrVersions[chunk]=(__hmrVersions[chunk]||0)+1;}
// Apply an update for a set of changed module ids whose new factories are already
// registered. Walks up to accepting boundaries; a leaf with no accepting importer
// (reaching the entry) triggers a full reload. Returns true when applied hot.
function __hmrApply(ids){
  const boundaries=[];const seen=new Set();const queue=ids.slice();
  while(queue.length){
    const id=queue.shift();
    if(seen.has(id))continue;seen.add(id);
    const e=__hmrEntries[id];
    if(e&&e.selfAccept){boundaries.push({id,depCb:null});continue;}
    const importers=__importers(id);
    if(importers.length===0){__hmrReload("no accepting boundary for module "+id);return false;}
    for(const imp of importers){
      const ie=__hmrEntries[imp];
      const dep=ie&&ie.depCallbacks.find(([d])=>__maps[imp]&&__maps[imp][d]===id);
      if(dep){boundaries.push({id:imp,depCb:dep[1],changed:id});}
      else queue.push(imp);
    }
  }
  for(const b of boundaries){
    const prev=__cache[b.id]?Object.assign(Object.create(null),__cache[b.id].exports):undefined;
    __disposeModule(b.id);
    delete __cache[b.id];
    if(b.changed!==undefined)delete __cache[b.changed];
    let next;
    try{next=__require(b.id);}catch(err){console.error(err);__hmrReload("re-run of boundary "+b.id+" threw");return false;}
    const e=__hmrEntries[b.id];
    try{
      if(b.depCb)b.depCb(next);
      if(e)for(const cb of e.selfCallbacks)cb(next,prev);
    }catch(err){console.error(err);__hmrReload("accept handler for "+b.id+" threw");return false;}
  }
  return true;
}
// Server-side invalidation (Increment A): hot-reload the changed subtree in-process
// WITHOUT restarting Node. TanStack Start rebuilds its router per request, loading
// route chunks through the runtime's version-aware dynamic `import()`. So it is
// enough to (1) clear the runtime cache for the changed modules and everything that
// imports them up to the entry, and (2) bump the version of EVERY chunk that hosts
// one of those dirty modules — including the intermediate chunks on the path, not
// just the changed leaf, since Node caches each chunk by URL. The next SSR request
// then re-imports the whole dirty chain fresh (each level gets a new `?v=` URL) and
// re-runs exactly those factories, while every unchanged chunk (React, react-dom,
// shared libs) stays cached, preserving the React singleton. The PID never changes.
function __hmrServerInvalidate(ids,chunks){
  const dirty=new Set();const queue=ids.slice();
  while(queue.length){
    const id=queue.shift();
    if(dirty.has(id))continue;dirty.add(id);
    for(const imp of __importers(id))queue.push(imp);
    for(const imp of __dynamicImporters(id))queue.push(imp);
  }
  // Version-bump every chunk that hosts a dirty module (as a dynamic-import
  // target), so each dynamic import along the chain re-fetches a fresh URL.
  const dirtyChunks=new Set(chunks||[]);
  for(const k in __chunks){const m=__chunks[k];for(const s in m){const c=m[s];if(c&&c[0]&&dirty.has(c[1]))dirtyChunks.add(c[0]);}}
  for(const c of dirtyChunks)__bumpVersion(c);
  dirty.add(__entryId);
  for(const id of dirty){__disposeModule(id);delete __cache[id];}
  // Rebuild the app handler in-process by re-running the entry. Only the dirty
  // modules re-execute (React, react-dom, shared libs stay cached, so the React
  // singleton is preserved); the rebuilt router carries fresh lazy component
  // loaders that re-fetch the version-bumped chunks on the next request. The fresh
  // fetch handler is published for the SSR entry to pick up.
  try{
    const fresh=__require(__entryId);
    globalThis.__diffpack_ssr_entry=fresh;
  }catch(err){console.error("[diffpack hmr] server rebuild failed",err);}
  return dirty.size;
}
"#;

/// The register-only guard placed in a chunk tail (HMR builds). A chunk re-imported
/// with the `__diffpack_hmr` URL marker registers its factories and returns without
/// eager-executing, so the browser can drive the update through the accept
/// protocol.
pub const REIMPORT_GUARD: &str = r#"if(import.meta&&import.meta.url&&import.meta.url.indexOf("__diffpack_hmr")>=0)return __runtime;"#;

/// The Node control endpoint, injected into the server (Esm) main chunk in HMR
/// builds. It listens on `DIFFPACK_HMR_CONTROL_PORT` and, on `POST
/// /__diffpack_hmr` with a JSON body `{ids:[...],chunks:[...]}`, invalidates the
/// live runtime in-process — so the dev server never restarts Node.
pub const SERVER_CONTROL: &str = r#"
(()=>{try{
  const __port=process&&process.env&&process.env.DIFFPACK_HMR_CONTROL_PORT;
  if(!__port)return;
  import("node:http").then(({default:http})=>{
    http.createServer((req,res)=>{
      if(req.method!=="POST"||req.url!=="/__diffpack_hmr"){res.writeHead(404);res.end();return;}
      let body="";req.on("data",c=>body+=c);req.on("end",()=>{
        try{const msg=JSON.parse(body||"{}");const n=__runtime.serverInvalidate(msg.ids||[],msg.chunks||[]);res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({ok:true,invalidated:n}));}
        catch(err){res.writeHead(500);res.end(String(err&&err.stack||err));}
      });
    }).listen(Number(__port),"127.0.0.1",()=>{console.log("[diffpack hmr] server control on 127.0.0.1:"+__port);});
  });
}catch(err){console.error("[diffpack hmr] control endpoint failed",err);}})();
"#;

/// Whether a client module is a React Fast Refresh boundary (all exports are
/// components, so it can re-run in place and swap component types while preserving
/// state). Two cases:
///
/// * A route-component split (`?tsr-split=component` / `errorComponent` / ...): the
///   virtual module holds exactly the extracted component, exported under its
///   canonical (lowercase) property name — always a boundary.
/// * A plain `.jsx`/`.tsx` module that exports only likely components (uppercase or
///   `default`). A ROUTE reference file is explicitly excluded: it exports the
///   TanStack `Route` object (not a component), so making it a boundary would make
///   every edit invalidate and full-reload. Its component is split out separately
///   and instrumented via the split module above.
pub fn is_refresh_boundary(path: &Path, exports: &[String], source: &str) -> bool {
    let path_str = path.to_string_lossy();
    if let Some(rest) = path_str.split("?tsr-split=").nth(1) {
        // Component-kind splits are refresh boundaries; a `loader` split is not.
        let kind = rest.split(['&', '=']).next().unwrap_or("");
        return kind.to_ascii_lowercase().ends_with("component");
    }
    // Only the real source extension counts; a `?tsr-split` query would otherwise
    // make `extension()` include the query.
    let is_jsx = ["jsx", "tsx"].iter().any(|ext| {
        path_str
            .split('?')
            .next()
            .unwrap_or(&path_str)
            .ends_with(&format!(".{ext}"))
    });
    if !is_jsx {
        return false;
    }
    // A route file defines its route via `createFileRoute`/`createRootRoute` and
    // exports a `Route`; it is not a component boundary (its component is split).
    if source.contains("createFileRoute") || source.contains("createRootRoute") {
        return false;
    }
    if exports.is_empty() {
        return false;
    }
    exports
        .iter()
        .all(|name| name == "default" || name.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
}

/// Appends the Fast Refresh + self-accept footer to a client component module's
/// lowered factory body. Runs INSIDE the module factory, where `module`, `exports`,
/// and the per-module `module.hot` are all in scope. On update, the module re-runs
/// (re-registering its components into their families), then
/// `validateRefreshBoundaryAndEnqueueUpdate` swaps the component types in the live
/// React tree while preserving hook state (a debounced `performReactRefresh`).
///
/// `module_key` is a stable, per-module string used as the Fast Refresh family
/// namespace: stable across edits of the same module (so the family is reused and
/// state preserved) and unique across modules (so two `App`s never collide).
pub fn fast_refresh_footer(module_key: &str) -> String {
    let key = json_string(module_key);
    format!(
        r#"
;(function(){{
  if(typeof window==="undefined")return;
  var RT=window.$RefreshRuntime$;
  if(!RT||!module.hot)return;
  // Diffpack's registry exposes each named export as a live-binding GETTER. Some
  // react-refresh versions (e.g. @vitejs/plugin-react v4) reject a boundary whose
  // exports have getter descriptors (`if(desc&&desc.get)return key`), because a
  // real ESM namespace reports DATA descriptors there — so they wrongly treat every
  // `export const Foo` component as a "new export" and force a full reload. Passing
  // plain data-property copies (getter values read into own data props) makes the
  // boundary check see the same shape it sees for a native ESM module, so state is
  // preserved across versions. Identity/component checks are unaffected (the copied
  // values are the same component references).
  var __flat=function(o){{return o?Object.assign({{}},o):o;}};
  RT.registerExportsForReactRefresh({key},module.exports);
  module.hot.accept(function(next,prev){{
    if(!next)return;
    var msg=RT.validateRefreshBoundaryAndEnqueueUpdate({key},__flat(prev||module.exports),__flat(next));
    if(msg)module.hot.invalidate(msg);
  }});
}})();
"#
    )
}

/// The React Fast Refresh preamble + WebSocket HMR client. This is a CLASSIC (not
/// module) inline script, injected into `<head>` right after a blocking classic
/// `<script src>` that loads the Fast Refresh runtime as `window.$RefreshRuntime$`.
/// Both run SYNCHRONOUSLY during parse — before the app's deferred/async entry
/// module — so `injectIntoGlobalHook` patches the DevTools hook and the Refresh
/// globals are set before React commits its first render. The script then removes
/// both injected nodes so React 19 hydrates a `<head>` identical to what it
/// server-rendered (no hydration mismatch). It opens the WS channel and applies
/// `update`/`reload` messages.
pub fn client_script(ws_path: &str) -> String {
    format!(
        r#"(function(){{
  var self=document.currentScript;
  // React Fast Refresh requires the DEVELOPMENT React/React-DOM (the production
  // build's renderer exposes no `scheduleRefresh`). Diffpack's entry selects the
  // React build at runtime from `process.env.NODE_ENV`, so set development BEFORE
  // the entry module runs (this preamble is a classic head script; the entry is a
  // deferred module). The entry's own default is `||"production"`, so this wins.
  globalThis.process=globalThis.process||{{}};
  globalThis.process.env=globalThis.process.env||{{}};
  if(!globalThis.process.env.NODE_ENV)globalThis.process.env.NODE_ENV="development";
  var RT=window.$RefreshRuntime$;
  if(RT){{
    RT.injectIntoGlobalHook(window);
    window.$RefreshReg$=function(type,id){{RT.register(type,id);}};
    window.$RefreshSig$=RT.createSignatureFunctionForTransform;
  }}
  var scheme=location.protocol==="https:"?"wss":"ws";
  function connect(){{
    var socket=new WebSocket(scheme+"://"+location.host+{ws});
    socket.addEventListener("message",async function(ev){{
      var msg;try{{msg=JSON.parse(ev.data);}}catch(_){{return;}}
      if(msg.type==="connected")return;
      if(msg.type==="reload"){{location.reload();return;}}
      if(msg.type==="rsc-refresh"){{
        // Server-component edit: refetch the CURRENT route's flight (?__rsc=1) and
        // diff-render it in place through the client Router — no full document
        // reload, and client-island state is preserved by React reconciliation.
        // Falls back to a reload pre-hydration (before the Router registers).
        if(typeof window.__diffpack_navigate==="function"){{window.__diffpack_navigate(location.pathname+location.search,{{push:false}});}}
        else{{location.reload();}}
        return;
      }}
      if(msg.type==="css"){{
        // Swap each changed stylesheet in place: clone the matching <link>, point
        // it at a cache-busted URL, and remove the old node once the new sheet has
        // loaded. No reload, so all component + DOM state is preserved.
        for(var c=0;c<msg.hrefs.length;c++){{(function(href){{
          var links=document.querySelectorAll('link[rel="stylesheet"]');
          for(var j=0;j<links.length;j++){{
            var link=links[j];var u;
            try{{u=new URL(link.href,location.href);}}catch(_){{continue;}}
            if(u.pathname!==href)continue;
            var next=link.cloneNode(false);
            next.href=href+(href.indexOf("?")>=0?"&":"?")+"__hmr_t="+Date.now();
            next.addEventListener("load",function(){{if(link.parentNode)link.parentNode.removeChild(link);}});
            link.parentNode.insertBefore(next,link.nextSibling);
          }}
        }})(msg.hrefs[c]);}}
        return;
      }}
      if(msg.type==="update"){{
        var rt=globalThis[{global}];
        if(!rt){{location.reload();return;}}
        try{{
          for(var i=0;i<msg.chunks.length;i++){{
            var url=msg.chunks[i];
            await import(url+(url.indexOf("?")>=0?"&":"?")+"__diffpack_hmr=1&t="+Date.now());
          }}
          rt.hmrApply(msg.ids);
          // react-refresh's boundary accept ENQUEUES a debounced performReactRefresh
          // (~30ms timer), which is otherwise the dominant browser-side HMR latency.
          // Flush it synchronously now so the DOM updates this task, not a frame+ later.
          if(RT&&RT.performReactRefresh)RT.performReactRefresh();
        }}catch(err){{console.error("[diffpack hmr]",err);location.reload();}}
      }}
    }});
    socket.addEventListener("close",function(){{
      setTimeout(function(){{fetch(location.href,{{method:"HEAD"}}).then(function(){{location.reload();}}).catch(connect);}},1000);
    }});
  }}
  connect();
  // Remove the runtime <script src> and this inline node synchronously (during
  // parse, before hydration) so React sees a clean <head>.
  if(self){{var prev=self.previousElementSibling;if(prev&&prev.tagName==="SCRIPT"&&prev.src&&prev.src.indexOf("refresh-runtime")>=0)prev.remove();self.remove();}}
}})();
"#,
        ws = json_string(ws_path),
        global = json_string(RUNTIME_GLOBAL),
    )
}

/// JSON-encode a string for safe embedding as a JS string literal.
fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

/// The Fast Refresh runtime as a CLASSIC IIFE that assigns `window.$RefreshRuntime$`
/// (so a blocking `<script src>` can install it synchronously before the app entry).
/// Adapted from `@vitejs/plugin-react`'s bundled `react-refresh` runtime (Meta,
/// MIT): the ESM `export`s are stripped and the public API is re-exposed on the
/// global. We do NOT reimplement the runtime.
pub fn refresh_runtime_source(raw: &str) -> String {
    let mut src = raw.replace(
        "__README_URL__",
        "https://github.com/vitejs/vite-plugin-react",
    );
    // Strip the ESM export surface so the body runs as a classic script.
    src = src.replace("export default { injectIntoGlobalHook }", "");
    src = src.replace("export function", "function");
    src = src.replace("export const", "const");
    format!(
        "(function(){{\n{src}\nwindow.$RefreshRuntime$={{register:register,injectIntoGlobalHook:injectIntoGlobalHook,createSignatureFunctionForTransform:createSignatureFunctionForTransform,registerExportsForReactRefresh:registerExportsForReactRefresh,validateRefreshBoundaryAndEnqueueUpdate:validateRefreshBoundaryAndEnqueueUpdate,performReactRefresh:performReactRefresh,__hmr_import:__hmr_import}};\n}})();\n"
    )
}

/// Composes the split (@vitejs/plugin-react >= 4.3) Fast Refresh runtime: the
/// react-refresh CORE (CJS, `react-refresh/cjs/...`) followed by the plugin's
/// `refreshUtils.js` (CJS — `registerExportsForReactRefresh`,
/// `validateRefreshBoundaryAndEnqueueUpdate`, `__hmr_import`). Both files read and
/// write a shared `exports`, so they run in order under one provided
/// `module`/`exports` in a classic IIFE, and the fully-populated object is
/// published on `window.$RefreshRuntime$`. Neither file is reimplemented; each runs
/// verbatim in the CommonJS environment it expects.
fn composed_cjs_runtime(core: &str, utils: &str) -> String {
    let readme = "https://github.com/vitejs/vite-plugin-react";
    let core = core.replace("__README_URL__", readme);
    let utils = utils.replace("__README_URL__", readme);
    // The runtime is a blocking `<script src>` that loads BEFORE the client
    // preamble installs the global `process` shim, and the raw react-refresh CORE
    // guards its dev warnings on `process.env.NODE_ENV`. Give the IIFE its own local
    // `process` so the runtime never touches the (not-yet-shimmed) global — a bare
    // `process` reference here would throw and leave `$RefreshRuntime$` unset,
    // cascading into `$RefreshSig$ is not defined` when the app's instrumented
    // modules run. The self-contained (older) runtime pre-inlines NODE_ENV, so it
    // needs no such shim.
    format!(
        "(function(){{\nvar process={{env:{{NODE_ENV:\"development\"}}}};\nvar module={{exports:{{}}}};var exports=module.exports;\n{core}\n{utils}\nwindow.$RefreshRuntime$=module.exports;\n}})();\n"
    )
}

/// Locates and prepares the React Fast Refresh runtime under the project's
/// `node_modules`, handling both @vitejs/plugin-react layouts:
///
/// * **Self-contained** (older plugin-react / the v6 line): a single
///   `dist/refresh-runtime.js` that already bundles the react-refresh core and the
///   plugin's boundary helpers — handled by [`refresh_runtime_source`].
/// * **Split** (plugin-react >= 4.3): the core `react-refresh` CJS runtime plus the
///   plugin's `dist/refreshUtils.js` — composed by [`composed_cjs_runtime`].
///
/// The raw `react-refresh` package ALONE is intentionally not accepted: it lacks
/// `registerExportsForReactRefresh` / `validateRefreshBoundaryAndEnqueueUpdate`
/// (those are plugin-react additions the HMR footer calls), so serving it would
/// leave `window.$RefreshRuntime$` half-formed and fail at update time.
pub fn find_refresh_runtime(project_root: &Path) -> Result<String, String> {
    let nm = project_root.join("node_modules");
    let read = |path: &Path| {
        std::fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
    };

    let bundled = nm.join("@vitejs/plugin-react/dist/refresh-runtime.js");
    if bundled.is_file() {
        return Ok(refresh_runtime_source(&read(&bundled)?));
    }

    let core = nm.join("react-refresh/cjs/react-refresh-runtime.development.js");
    let utils = nm.join("@vitejs/plugin-react/dist/refreshUtils.js");
    if core.is_file() && utils.is_file() {
        return Ok(composed_cjs_runtime(&read(&core)?, &read(&utils)?));
    }

    Err(format!(
        "React Fast Refresh runtime not found under {}. Looked for the self-contained \
         @vitejs/plugin-react/dist/refresh-runtime.js, and for the split layout \
         (react-refresh/cjs/react-refresh-runtime.development.js + \
         @vitejs/plugin-react/dist/refreshUtils.js). Install @vitejs/plugin-react so \
         dev-mode Fast Refresh has its client runtime.",
        nm.display()
    ))
}

/// Rewrites `import.meta.hot` references to the per-module `module.hot` object so a
/// module authored against the standard HMR API works inside Diffpack's registry
/// factory (where `import.meta` is the chunk's meta, not the module's). DEV-only;
/// a production build never calls this, so `import.meta.hot` there is left as a
/// plain `undefined` member access (a falsy no-op), keeping output unaffected.
pub fn rewrite_import_meta_hot(code: &str, target: Target) -> String {
    let _ = target;
    if !code.contains("import.meta.hot") {
        return code.to_string();
    }
    code.replace("import.meta.hot", "module.hot")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn footer_passes_flattened_exports_to_the_boundary_check() {
        // The boundary validator must receive plain data-property copies, not the
        // registry's live-binding GETTER exports — react-refresh v4 rejects getter
        // descriptors and forces a full reload for every `export const Foo`
        // component. Regression guard for the state-preservation fix.
        let footer = fast_refresh_footer("/app/src/components/Navbar.tsx");
        assert!(footer.contains("__flat"), "footer must define a flatten helper: {footer}");
        assert!(
            footer.contains("Object.assign({},o)"),
            "flatten must copy getter values into own data props: {footer}"
        );
        assert!(
            footer.contains("validateRefreshBoundaryAndEnqueueUpdate({key},__flat("
                .replace("{key}", &json_string("/app/src/components/Navbar.tsx"))
                .as_str())
                || footer.contains("__flat(prev||module.exports),__flat(next)"),
            "both boundary-check arguments must be flattened: {footer}"
        );
    }

    #[test]
    fn composed_runtime_shims_process_and_publishes_global() {
        // @vitejs/plugin-react >= 4.3 split layout: the react-refresh core (CJS,
        // guards on process.env.NODE_ENV) + the plugin's refreshUtils, run under one
        // shared `exports`, published on window.$RefreshRuntime$. The local process
        // shim is what keeps the blocking runtime script from throwing before the
        // client preamble installs the global process shim.
        let out = composed_cjs_runtime(
            "var m=process.env.NODE_ENV;exports.register=1;",
            "exports.validateRefreshBoundaryAndEnqueueUpdate=2;",
        );
        assert!(
            out.contains("var process={env:{NODE_ENV:\"development\"}}"),
            "must shim process locally: {out}"
        );
        assert!(
            out.contains("window.$RefreshRuntime$=module.exports"),
            "must publish the composed exports as the global: {out}"
        );
        assert!(
            out.contains("exports.register=1")
                && out.contains("exports.validateRefreshBoundaryAndEnqueueUpdate=2"),
            "must include both the core and the utils verbatim: {out}"
        );
    }

    #[test]
    fn a_single_named_component_export_is_a_refresh_boundary() {
        // `export const Navbar = () => {...}` — the shape that broke on real apps.
        let boundary = is_refresh_boundary(
            Path::new("/app/src/components/Navbar.tsx"),
            &["Navbar".to_string()],
            "export const Navbar = () => null",
        );
        assert!(boundary, "a lone uppercase named component export must be a boundary");
    }
}
