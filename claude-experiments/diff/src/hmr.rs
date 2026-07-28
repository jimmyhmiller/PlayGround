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

use std::path::{Path, PathBuf};

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

/// [`REQUIRE_DYNAMIC_ESM_HMR`] for a build that contains an ASYNC module (one with
/// a top-level `await`). Identical except that the requested module is resolved
/// through `__requireAsync`, so the promise `import()` already hands back does not
/// settle until the target module's own top-level `await` has — the same guarantee
/// the production form gives, kept while the chunk URL stays version-aware.
pub const REQUIRE_DYNAMIC_ESM_HMR_ASYNC: &str = r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null){const __v=__hmrVersions[chunk[0]];const __u=__v?chunk[0]+(chunk[0].indexOf("?")>=0?"&v=":"?v=")+__v:chunk[0];return import(__u).then(()=>__requireAsync(chunk[1]));}return __requireAsync(chunk[1]);};"#;

/// The HMR bookkeeping + apply/propagate/invalidate methods, injected into the
/// singleton runtime IIFE (main chunk, HMR builds only) right before it returns.
/// Everything here closes over the runtime internals (`__modules`, `__maps`,
/// `__chunks`, `__cache`, `__require`).
pub const RUNTIME_METHODS: &str = r#"
const __hmrVersions=Object.create(null);
const __hmrData=Object.create(null);
const __hmrEntries=Object.create(null);
// Per-module HMR bookkeeping. `data` is NOT stored here: it lives in `__hmrData[id]`
// and deliberately outlives dispose/re-run so a self-accepting module can carry
// state across a hot update (the whole point of `import.meta.hot.data`).
function __hmrEntry(id){return __hmrEntries[id]||(__hmrEntries[id]={selfAccept:false,selfCallbacks:[],depCallbacks:[],disposers:[],pruners:[],listeners:Object.create(null),declined:false});}
// Global HMR event bus (Vite parity). `import.meta.hot.on(event,cb)` registers a
// listener; the runtime emits the standard `vite:*` lifecycle events. Vite's emitter
// is SHARED across every module, so an event fires every registered listener
// regardless of which module registered it. Listeners are stored per module so they
// are discarded automatically when that module is pruned.
function __hmrEmit(event,payload){for(const k in __hmrEntries){const l=__hmrEntries[k].listeners[event];if(l)for(const cb of l.slice()){try{cb(payload,event);}catch(err){console.error("[diffpack hmr] "+event+" listener threw",err);}}}}
function __makeHot(id){
  const data=__hmrData[id]||(__hmrData[id]=Object.create(null));
  return {
    data,
    accept(dep,cb){
      const entry=__hmrEntry(id);
      // accept() / accept(cb): self-accept (re-run this module in place). accept(dep,cb)
      // / accept([dep,...],cb): accept named dependency updates and receive their fresh
      // module namespace(s) in `cb`.
      if(typeof dep==="function"||dep===undefined){entry.selfAccept=true;if(typeof dep==="function")entry.selfCallbacks.push(dep);}
      else{const deps=Array.isArray(dep)?dep:[dep];for(const d of deps)entry.depCallbacks.push([d,cb]);}
    },
    // Vite's acceptExports(names, cb): a self-accept scoped to named exports. Diffpack
    // re-runs the whole module factory on self-accept, so the export filter is advisory
    // (accepting the module is a superset of accepting some of its exports); we
    // self-accept and forward the callback, which preserves the module's intent.
    acceptExports(_exports,cb){const entry=__hmrEntry(id);entry.selfAccept=true;if(typeof cb==="function")entry.selfCallbacks.push(cb);},
    dispose(cb){__hmrEntry(id).disposers.push(cb);},
    prune(cb){__hmrEntry(id).pruners.push(cb);},
    invalidate(message){__hmrInvalidate(id,message);},
    // decline(): this module cannot be hot-updated; any update touching it forces a
    // full page reload instead of a hot swap.
    decline(){__hmrEntry(id).declined=true;},
    on(event,cb){const e=__hmrEntry(id);(e.listeners[event]||(e.listeners[event]=[])).push(cb);},
    off(event,cb){const e=__hmrEntries[id];if(!e)return;const l=e.listeners[event];if(!l)return;const i=l.indexOf(cb);if(i>=0)l.splice(i,1);},
    // Client->server custom messaging needs a duplex dev channel diffpack does not
    // expose; be explicit (a clear throw) rather than silently swallowing the message.
    send(){throw new Error("import.meta.hot.send is not supported by diffpack's dev server");},
  };
}
// REVERSE IMPORT INDEX: for a module id, who imports it (statically, and through a
// dynamic import — the chunk map records `spec -> [chunkPath, targetId]`).
//
// Built once and reused, because both lookups sit inside a graph WALK. Scanning
// `__maps`/`__chunks` per lookup made each walk O(modules x specifiers per module) PER
// NODE: fine for an example app, quadratic blow-up on a real one. cal.com's client
// graph is ~6,000 modules, and a single edit to a component whose boundary does not
// self-accept walked far enough to do hundreds of millions of comparisons on the main
// thread — the browser tab stopped responding rather than showing the update.
//
// `__register` invalidates it (see the runtime's exported `register`), so a chunk
// loaded after the index was built cannot leave it stale.
let __importerIndex=null;
function __buildImporterIndex(){
  const statics=Object.create(null),dynamics=Object.create(null);
  for(const k in __maps){
    const importer=+k,m=__maps[k],seen=new Set();
    for(const s in m){const t=m[s];if(t===undefined||seen.has(t))continue;seen.add(t);(statics[t]||(statics[t]=[])).push(importer);}
  }
  for(const k in __chunks){
    const importer=+k,m=__chunks[k],seen=new Set();
    for(const s in m){const c=m[s];if(!c||seen.has(c[1]))continue;seen.add(c[1]);(dynamics[c[1]]||(dynamics[c[1]]=[])).push(importer);}
  }
  __importerIndex={statics,dynamics};
  return __importerIndex;
}
function __hmrInvalidateImporterIndex(){__importerIndex=null;}
function __importers(id){return (__importerIndex||__buildImporterIndex()).statics[id]||[];}
function __dynamicImporters(id){return (__importerIndex||__buildImporterIndex()).dynamics[id]||[];}
function __disposeModule(id){const e=__hmrEntries[id];if(e){for(const d of e.disposers){try{d(__hmrData[id]);}catch(err){console.error(err);}}e.selfCallbacks=[];e.disposers=[];e.depCallbacks=[];e.selfAccept=false;}}
function __hmrReload(reason){__hmrEmit("vite:beforeFullReload",{path:"*",reason:reason});if(typeof location!=="undefined"&&location.reload){console.log("[diffpack hmr] full reload: "+reason);location.reload();}else{console.warn("[diffpack hmr] full reload required ("+reason+")");}}
// hot.invalidate(): the boundary gives up on this update. Emit the event, then force a
// full reload so the change is still reflected (diffpack's client graph roots at the
// entry, so an invalidated boundary that no importer re-accepts reaches the entry ->
// reload; matching that, invalidate reloads directly rather than leaving stale code).
function __hmrInvalidate(id,message){__hmrEmit("vite:invalidate",{path:String(id),message:message});__hmrReload("module "+id+" invalidated"+(message?": "+message:""));}
// Fire prune callbacks for modules removed from the graph, then forget them. Exposed
// as the runtime's `prune(ids)` and reachable from the dev client's "prune" message.
function __hmrPrune(ids){
  if(!ids||!ids.length)return 0;
  __hmrEmit("vite:beforePrune",{});
  for(const id of ids){
    const e=__hmrEntries[id];
    if(e){for(const p of e.pruners){try{p(__hmrData[id]);}catch(err){console.error("[diffpack hmr] prune handler threw",err);}}}
    __disposeModule(id);
    delete __cache[id];
    delete __hmrEntries[id];
    delete __hmrData[id];
  }
  return ids.length;
}
function __replace(id,factory,map){__modules[id]=factory;if(map)__maps[id]=map;return __hmrApply([id]);}
// Re-run module `id` and, if it is an ASYNC module (it top-level-`await`s, or
// statically imports something that does), wait for its initialisation to settle
// before the caller reads its exports. `__require` returns the module's exports
// object synchronously in both cases — the object's identity is established by the
// factory's synchronous prefix — so only the WAIT differs. `__hmrPending` is a
// constant `undefined` in a build with no async module, which keeps this a plain
// synchronous re-run there.
async function __hmrRerun(id){const exports=__require(id);const pending=__hmrPending(id);if(pending)await pending;return exports;}
function __bumpVersion(chunk){__hmrVersions[chunk]=(__hmrVersions[chunk]||0)+1;}
// Apply an update for a set of changed module ids whose new factories are already
// registered. Walks up to accepting boundaries; a leaf with no accepting importer
// (reaching the entry) triggers a full reload. Returns true when applied hot.
// Async because a re-run boundary may be an ASYNC module, whose accept/dep
// callbacks must not see a half-initialised namespace. Callers (`__replace`, the
// dev client's `update` handler, `serverInvalidate`) await the returned promise.
async function __hmrApply(ids){
  __hmrEmit("vite:beforeUpdate",{type:"update",updates:ids.map(id=>({type:"js-update",path:String(id),acceptedPath:String(id)}))});
  const boundaries=[];const seen=new Set();const queue=ids.slice();
  while(queue.length){
    const id=queue.shift();
    if(seen.has(id))continue;seen.add(id);
    const e=__hmrEntries[id];
    // A module that declined hot updates forces a full reload for any update that
    // reaches it (either directly changed or on the propagation path).
    if(e&&e.declined){__hmrReload("module "+id+" declined hot updates");return false;}
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
    try{next=await __hmrRerun(b.id);}catch(err){console.error(err);__hmrReload("re-run of boundary "+b.id+" threw");return false;}
    const e=__hmrEntries[b.id];
    try{
      if(b.depCb)b.depCb(next);
      if(e)for(const cb of e.selfCallbacks)cb(next,prev);
    }catch(err){console.error(err);__hmrReload("accept handler for "+b.id+" threw");return false;}
  }
  __hmrEmit("vite:afterUpdate",{type:"update",updates:ids.map(id=>({type:"js-update",path:String(id),acceptedPath:String(id)}))});
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
async function __hmrServerInvalidate(ids,chunks){
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
  // `__hmrRerun` (not `__require`): when the entry is an ASYNC module — anywhere in
  // its static import graph a module top-level-`await`s — its exports are not
  // populated until that settles, and publishing them early would hand the SSR entry
  // a handler that is still `undefined`.
  try{
    const fresh=await __hmrRerun(__entryId);
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
      let body="";req.on("data",c=>body+=c);req.on("end",async()=>{
        try{const msg=JSON.parse(body||"{}");const n=await __runtime.serverInvalidate(msg.ids||[],msg.chunks||[]);res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({ok:true,invalidated:n}));}
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
/// * A module in a JSX-capable extension that exports only likely components
///   (uppercase or `default`). Which extensions those are is the PROJECT's rule
///   (`jsx`): under Next a `.js` component is as much a boundary as a `.jsx` one,
///   and treating it as plain JS would give every Next `.js` page a full page
///   reload on each edit. A ROUTE reference file is explicitly excluded: it exports
///   the TanStack `Route` object (not a component), so making it a boundary would
///   make every edit invalidate and full-reload. Its component is split out
///   separately and instrumented via the split module above.
pub fn is_refresh_boundary(
    path: &Path,
    exports: &[String],
    source: &str,
    jsx: crate::parser::JsxExtensions,
) -> bool {
    let path_str = path.to_string_lossy();
    if let Some(rest) = path_str.split("?tsr-split=").nth(1) {
        // Component-kind splits are refresh boundaries; a `loader` split is not.
        let kind = rest.split(['&', '=']).next().unwrap_or("");
        return kind.to_ascii_lowercase().ends_with("component");
    }
    // Only the real source extension counts; a `?tsr-split` query would otherwise
    // make `extension()` include the query. The single JSX rule decides, so this
    // never drifts from what the module was actually parsed as.
    let real_path = Path::new(path_str.split('?').next().unwrap_or(&path_str));
    if !crate::parser::source_type_for(real_path, jsx).is_jsx() {
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

/// Whether `code` carries any of oxc's React Refresh instrumentation, and so needs
/// the module-scoped [`fast_refresh_preamble`] in front of it.
pub fn needs_fast_refresh_preamble(code: &str) -> bool {
    code.contains("$RefreshReg$") || code.contains("$RefreshSig$")
}

/// The module-scoped `$RefreshReg$`/`$RefreshSig$` shims, prepended to the lowered
/// factory body of every client module oxc's React Refresh transform instrumented.
///
/// WHY THIS EXISTS. The transform emits `$RefreshReg$(_c, "Foo")` — the LOCAL
/// component name and nothing else, exactly as `react-refresh/babel` does. That id
/// is the react-refresh FAMILY key, and react-refresh keys families in one global
/// map: two modules that both register a `"Foo"` land in the SAME family, and the
/// second registration is recorded as a PENDING HOT UPDATE of the first. Bundling a
/// real app into one runtime therefore mis-identifies unrelated components as
/// versions of each other before a single edit has happened — cal.com's client graph
/// registers 1,996 components under 14 different `"SkeletonLoader"`s, 5 `"Error"`s,
/// and so on. The first `performReactRefresh` then swaps hundreds of live component
/// types for unrelated ones, and React's `scheduleRefresh` -> `flushSyncWork` loop
/// never reaches a fixpoint: the browser tab wedges in an unbounded synchronous
/// render and the hot update never lands. (A small example app has no colliding
/// names, which is why this only shows up at scale.)
///
/// The fix is the one Vite's and Next's React Fast Refresh integrations both make:
/// give each module its OWN `$RefreshReg$` that namespaces the id with the module
/// key, so a family is per module + per export name. `var` (not `const`) so the
/// binding is hoisted over the whole factory body — the transform emits its
/// `$RefreshSig$()` calls at the top of a component and its `$RefreshReg$` calls at
/// the end of the module, and both must see the shim.
///
/// The shims are also DEFENSIVE about a missing runtime: the globals are installed
/// only when the refresh runtime script loaded, so without this an instrumented
/// module would throw `ReferenceError: $RefreshSig$ is not defined` at load and take
/// the whole client bundle down. The `$RefreshSig$` fallback is the identity
/// signature function react-refresh itself returns when it has nothing to record.
pub fn fast_refresh_preamble(module_key: &str) -> String {
    let key = json_string(module_key);
    format!(
        "var __diffpackRefreshRuntime=typeof window===\"undefined\"?null:window.$RefreshRuntime$;\
         var $RefreshReg$=function(type,id){{if(__diffpackRefreshRuntime)__diffpackRefreshRuntime.register(type,{key}+\" \"+id);}};\
         var $RefreshSig$=function(){{return __diffpackRefreshRuntime?__diffpackRefreshRuntime.createSignatureFunctionForTransform():function(type){{return type;}};}};\n"
    )
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
      // Dev error overlay: a Rust build error is surfaced in the browser instead of
      // killing the dev server; a subsequent good build clears it. (No-op when the
      // overlay script is absent, e.g. a non-HTML client.)
      if(msg.type==="build-error"){{if(window.__diffpackOverlay)window.__diffpackOverlay.showBuild(msg);return;}}
      if(msg.type==="build-ok"){{if(window.__diffpackOverlay)window.__diffpackOverlay.clear();return;}}
      if(msg.type==="reload"){{if(window.__diffpackOverlay)window.__diffpackOverlay.clear();location.reload();return;}}
      if(msg.type==="rsc-refresh"){{
        if(window.__diffpackOverlay)window.__diffpackOverlay.clear();
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
      if(msg.type==="prune"){{
        // Modules removed from the graph: fire their import.meta.hot.prune callbacks
        // and forget them, without a reload. No-op if the runtime is not up yet.
        var prt=globalThis[{global}];
        if(prt&&prt.prune)prt.prune(msg.ids||[]);
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
          await rt.hmrApply(msg.ids);
          // react-refresh's boundary accept ENQUEUES a debounced performReactRefresh
          // (~30ms timer), which is otherwise the dominant browser-side HMR latency.
          // Flush it synchronously now so the DOM updates this task, not a frame+ later.
          if(RT&&RT.performReactRefresh)RT.performReactRefresh();
          // A successful hot update supersedes any error overlay (e.g. a runtime error
          // the edit just fixed).
          if(window.__diffpackOverlay)window.__diffpackOverlay.clear();
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

/// The dev-only error overlay: a CLASSIC inline script that catches build errors
/// (pushed over the HMR WebSocket by [`client_script`]), uncaught runtime errors,
/// and unhandled promise rejections, and renders a full-screen overlay showing the
/// message plus a source-mapped stack.
///
/// React 19 re-throws an uncaught render/hydration error to `window.onerror`, so
/// nothing in the app entry needs instrumenting; the overlay just listens on the
/// browser globals. Stack frames pointing at a generated chunk (`/client.js:LINE:COL`)
/// are mapped back to their original source with a tiny in-browser VLQ base64
/// source-map consumer that fetches the chunk's sibling `.map` (which the dev bundler
/// emits with `diffpack:///` project-relative `sources` and inline `sourcesContent`).
/// Diffpack's dev maps are line-granular (one token per generated line, at column 0),
/// so the overlay honestly shows the original file + line and DROPS the column rather
/// than fabricating a precision the map does not carry.
///
/// Like [`client_script`], the overlay's own `<script>` node removes itself during
/// parse so React 19 hydrates a `<head>` identical to what it server-rendered, and
/// the overlay DOM is created LAZILY (only on the first error, appended to
/// `document.body`), so it can never reintroduce a hydration mismatch.
pub fn overlay_script() -> String {
    // No interpolation is needed (the overlay fetches relative to `location`), so this
    // is a plain raw literal rather than a `format!` — no brace escaping.
    OVERLAY_SCRIPT.to_string()
}

/// The overlay client body. Publishes `window.__diffpackOverlay` with `showBuild`,
/// `showRuntime`, `clear`, and `mapStack`.
const OVERLAY_SCRIPT: &str = r#"(function(){
  var self=document.currentScript;
  // VLQ base64 alphabet + decoder (source-map v3 `mappings` are base64-VLQ).
  var B64="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  function decodeVlq(str){
    var out=[],shift=0,value=0;
    for(var i=0;i<str.length;i++){
      var digit=B64.indexOf(str.charAt(i));
      if(digit<0)continue;
      var cont=digit&32;digit&=31;
      value+=digit<<shift;
      if(cont){shift+=5;}
      else{var neg=value&1;value>>=1;out.push(neg?-value:value);value=0;shift=0;}
    }
    return out;
  }
  // Parse the `mappings` string into per-generated-line segments. genColumn resets
  // each line; sourceIndex/origLine/origColumn are cumulative across the whole map.
  function parseMappings(mappings){
    var lines=(mappings||"").split(";");
    var srcIdx=0,origLine=0,origCol=0,result=[];
    for(var i=0;i<lines.length;i++){
      var segs=lines[i].split(","),genCol=0,lineSegs=[];
      for(var j=0;j<segs.length;j++){
        if(!segs[j])continue;
        var f=decodeVlq(segs[j]);
        if(f.length>=1)genCol+=f[0];
        if(f.length>=4){srcIdx+=f[1];origLine+=f[2];origCol+=f[3];lineSegs.push({genCol:genCol,src:srcIdx,line:origLine,col:origCol});}
      }
      result.push(lineSegs);
    }
    return result;
  }
  function makeConsumer(map){
    var parsed=parseMappings(map.mappings),sources=map.sources||[];
    return function(genLine,genCol){
      var segs=parsed[genLine-1];
      if(!segs||!segs.length)return null;
      var best=segs[0];
      for(var i=0;i<segs.length;i++){if(segs[i].genCol<=genCol)best=segs[i];else break;}
      var src=sources[best.src]||"";
      // Strip the diffpack:/// label to a project-relative path for display.
      return {source:src.replace(/^diffpack:\/\/\//,""),line:best.line+1};
    };
  }
  // Rewrite every `URL:LINE:COL` frame in a stack to `originalSource:LINE` by
  // fetching each unique chunk's sibling .map (served off disk by the dev proxy).
  var FRAME="((?:https?://|/)[^\\s()]+?):(\\d+):(\\d+)";
  async function mapStack(stack){
    if(!stack)return stack;
    var urls={},re=new RegExp(FRAME,"g"),m;
    while((m=re.exec(stack)))urls[m[1]]=1;
    var consumers={};
    await Promise.all(Object.keys(urls).map(async function(u){
      try{
        var path=u;try{path=new URL(u,location.href).pathname;}catch(_){}
        var res=await fetch(path+".map");
        if(!res.ok)return;
        consumers[u]=makeConsumer(await res.json());
      }catch(_){}
    }));
    return stack.replace(new RegExp(FRAME,"g"),function(whole,u,line,col){
      var c=consumers[u];if(!c)return whole;
      var pos=c(parseInt(line,10),parseInt(col,10));
      // Line-granular map: original file + line, column dropped honestly.
      return pos?pos.source+":"+pos.line:whole;
    });
  }
  var overlayEl=null;
  function ensureOverlay(){
    if(overlayEl&&overlayEl.parentNode)return overlayEl;
    overlayEl=document.createElement("div");
    overlayEl.id="__diffpack-overlay";
    overlayEl.setAttribute("style","position:fixed;inset:0;z-index:2147483647;background:rgba(18,18,18,0.96);color:#e8e8e8;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;padding:24px;overflow:auto;white-space:pre-wrap;box-sizing:border-box;");
    var parent=document.body||document.documentElement;
    parent.appendChild(overlayEl);
    return overlayEl;
  }
  function header(text){
    var h=document.createElement("div");
    h.setAttribute("style","color:#ff6b6b;font-weight:bold;font-size:15px;margin-bottom:12px;");
    h.textContent=text;
    return h;
  }
  function showBuild(payload){
    var el=ensureOverlay();el.textContent="";
    el.appendChild(header("Build error"));
    var body=document.createElement("div");
    body.textContent=(payload&&payload.message)||"unknown build error";
    el.appendChild(body);
  }
  function showRuntime(err){
    var el=ensureOverlay();el.textContent="";
    el.appendChild(header("Runtime error"));
    var msg=document.createElement("div");
    msg.setAttribute("style","margin-bottom:12px;");
    msg.textContent=(err&&(err.message||String(err)))||"unknown error";
    el.appendChild(msg);
    var stackEl=document.createElement("div");
    var raw=(err&&err.stack)||"";
    stackEl.textContent=raw;
    el.appendChild(stackEl);
    if(raw)mapStack(raw).then(function(mapped){stackEl.textContent=mapped;}).catch(function(){});
  }
  function clear(){if(overlayEl&&overlayEl.parentNode)overlayEl.parentNode.removeChild(overlayEl);overlayEl=null;}
  window.__diffpackOverlay={showBuild:showBuild,showRuntime:showRuntime,clear:clear,mapStack:mapStack};
  window.addEventListener("error",function(ev){
    var err=ev.error||{message:ev.message,stack:ev.filename?ev.filename+":"+ev.lineno+":"+ev.colno:""};
    showRuntime(err);
  });
  window.addEventListener("unhandledrejection",function(ev){
    var r=ev.reason;
    showRuntime(r&&typeof r==="object"?r:{message:String(r)});
  });
  // Remove this inline node during parse (before hydration) so React sees a clean
  // <head>; the listeners and window.__diffpackOverlay persist after removal.
  if(self)self.remove();
})();
"#;

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

/// Composes NEXT's own Fast Refresh runtime: the react-refresh CORE that ships
/// inside `next/dist/compiled/react-refresh`, plus `@next/react-refresh-utils`'
/// `internal/helpers.js` — the boundary helpers Next's webpack/rspack loaders call.
/// Neither file is reimplemented; each runs verbatim as the CommonJS module it is,
/// under a `require` that only answers the one specifier `helpers.js` asks for (and
/// throws, naming the specifier, for anything else rather than handing back a stub).
///
/// Next's helper API is not @vitejs/plugin-react's, so the published
/// `window.$RefreshRuntime$` is an ADAPTER over it — the surface
/// [`fast_refresh_footer`] and [`client_script`] call, expressed in Next's terms:
///
/// * `registerExportsForReactRefresh(id, exports)` — Next takes the same two
///   arguments in the opposite order.
/// * `validateRefreshBoundaryAndEnqueueUpdate(id, prev, next)` — Next splits this
///   into `isReactRefreshBoundary` (are the new exports still all components?) and
///   `shouldInvalidateReactRefreshBoundary(prevSignature, nextSignature)` (did the
///   export shape change?). Both return a reason string exactly where Next's own
///   module runtime would call `hot.invalidate()`, which is what the footer does
///   with it. The signatures are compared through `getFamilyByType`, and a re-run
///   registers the new component types under the SAME family ids, so an unchanged
///   boundary compares equal and its state is preserved.
///
/// `scheduleUpdate` is deliberately not used: it defers `performReactRefresh`
/// through webpack's `module.hot.status()`, which diffpack's runtime does not
/// implement, and the dev client already flushes `performReactRefresh` itself
/// straight after applying an update.
fn composed_next_runtime(core: &str, helpers: &str) -> String {
    format!(
        r#"(function(){{
var process={{env:{{NODE_ENV:"development"}}}};
var __core={{exports:{{}}}};
(function(module,exports){{
{core}
}})(__core,__core.exports);
var __helpers={{exports:{{}}}};
(function(module,exports,require){{
{helpers}
}})(__helpers,__helpers.exports,function(specifier){{
  if(specifier==="next/dist/compiled/react-refresh/runtime")return __core.exports;
  throw new Error("[diffpack hmr] Next's Fast Refresh helpers required an unexpected module: "+specifier);
}});
var RT=__core.exports;
var H=__helpers.exports.default;
window.$RefreshRuntime$={{
  register:RT.register,
  injectIntoGlobalHook:RT.injectIntoGlobalHook,
  createSignatureFunctionForTransform:RT.createSignatureFunctionForTransform,
  performReactRefresh:RT.performReactRefresh,
  registerExportsForReactRefresh:function(id,moduleExports){{return H.registerExportsForReactRefresh(moduleExports,id);}},
  validateRefreshBoundaryAndEnqueueUpdate:function(id,prevExports,nextExports){{
    if(!H.isReactRefreshBoundary(nextExports))return "Could not Fast Refresh (not all exports are components)";
    if(prevExports&&H.shouldInvalidateReactRefreshBoundary(H.getRefreshBoundarySignature(prevExports),H.getRefreshBoundarySignature(nextExports)))return "Could not Fast Refresh (the module's export signature changed)";
    return undefined;
  }},
  __hmr_import:function(specifier){{return import(specifier);}},
}};
}})();
"#
    )
}

/// Resolves a path under `node_modules` the way Node does: from `start`, try
/// `<dir>/node_modules/<relative>` walking up to the filesystem root, and take the
/// first hit. The single-directory lookup this replaces was wrong for every
/// workspace layout (yarn/pnpm/npm workspaces hoist shared packages to the
/// REPOSITORY root, so `apps/web/node_modules` holds almost nothing) — cal.com's
/// `apps/web` has four packages there and everything else one level up.
fn resolve_in_node_modules(start: &Path, relative: &str) -> Option<PathBuf> {
    for directory in start.ancestors() {
        let candidate = directory.join("node_modules").join(relative);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Locates and prepares the React Fast Refresh runtime from the project's installed
/// packages, handling three layouts:
///
/// * **@vitejs/plugin-react, self-contained** (older plugin-react / the v6 line): a
///   single `dist/refresh-runtime.js` that already bundles the react-refresh core
///   and the plugin's boundary helpers — handled by [`refresh_runtime_source`].
/// * **@vitejs/plugin-react, split** (plugin-react >= 4.3): the core `react-refresh`
///   CJS runtime plus the plugin's `dist/refreshUtils.js` — composed by
///   [`composed_cjs_runtime`].
/// * **Next's own**: `next/dist/compiled/react-refresh` plus
///   `next/dist/compiled/@next/react-refresh-utils` — composed and adapted by
///   [`composed_next_runtime`]. A Next app has Fast Refresh natively and does not
///   depend on @vitejs/plugin-react (not one app in the pinned Next corpus does), so
///   without this `diffpack dev` could not run a Next project at all.
///
/// The plugin-react layouts are tried first: the footer and dev client are written
/// against that API, so where a project has it, it is the direct path rather than an
/// adapter. Next's runtime is equivalent in capability, not a downgrade.
///
/// The raw `react-refresh` package ALONE is intentionally not accepted: it lacks
/// the boundary helpers (`registerExportsForReactRefresh` and the invalidation
/// check) that the HMR footer calls, so serving it would leave
/// `window.$RefreshRuntime$` half-formed and fail at update time.
pub fn find_refresh_runtime(project_root: &Path) -> Result<String, String> {
    let read = |path: &Path| {
        std::fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
    };
    let find = |relative: &str| resolve_in_node_modules(project_root, relative);

    if let Some(bundled) = find("@vitejs/plugin-react/dist/refresh-runtime.js") {
        return Ok(refresh_runtime_source(&read(&bundled)?));
    }

    if let (Some(core), Some(utils)) = (
        find("react-refresh/cjs/react-refresh-runtime.development.js"),
        find("@vitejs/plugin-react/dist/refreshUtils.js"),
    ) {
        return Ok(composed_cjs_runtime(&read(&core)?, &read(&utils)?));
    }

    if let (Some(core), Some(helpers)) = (
        find("next/dist/compiled/react-refresh/cjs/react-refresh-runtime.development.js"),
        find("next/dist/compiled/@next/react-refresh-utils/dist/internal/helpers.js"),
    ) {
        return Ok(composed_next_runtime(&read(&core)?, &read(&helpers)?));
    }

    let searched = project_root
        .ancestors()
        .map(|directory| directory.join("node_modules").display().to_string())
        .collect::<Vec<_>>()
        .join("\n  ");
    Err(format!(
        "React Fast Refresh runtime not found. Looked for the self-contained \
         @vitejs/plugin-react/dist/refresh-runtime.js, for the split layout \
         (react-refresh/cjs/react-refresh-runtime.development.js + \
         @vitejs/plugin-react/dist/refreshUtils.js), and for Next's own \
         (next/dist/compiled/react-refresh + \
         next/dist/compiled/@next/react-refresh-utils), in each of:\n  {searched}\n\
         Install next or @vitejs/plugin-react so dev-mode Fast Refresh has its \
         client runtime."
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

    /// A pinned corpus app whose `node_modules` holds a REAL `next` install. The
    /// Fast Refresh tests run against those files, never a hand-written stand-in.
    /// `None` when the corpus has not been fetched.
    fn real_next_node_modules() -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("integration/e2e/apps/next-hello-world/node_modules");
        path.join("next/dist/compiled/@next/react-refresh-utils/dist/internal/helpers.js")
            .is_file()
            .then_some(path)
    }

    /// The Fast Refresh runtime must be resolved the way Node resolves anything
    /// under `node_modules`: from the project directory UPWARD. Looking only in
    /// `<project>/node_modules` failed on every workspace layout, because
    /// yarn/pnpm/npm hoist shared packages to the repository root — cal.com's
    /// `apps/web/node_modules` holds four packages and nothing else, so
    /// `diffpack dev` died before serving a single request.
    #[test]
    fn the_refresh_runtime_is_resolved_up_the_node_modules_chain() {
        let Some(node_modules) = real_next_node_modules() else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        // A workspace: packages live at the root, the app is two levels down.
        #[cfg(unix)]
        std::os::unix::fs::symlink(&node_modules, directory.path().join("node_modules")).unwrap();
        #[cfg(not(unix))]
        return;
        let app = directory.path().join("apps/web");
        std::fs::create_dir_all(&app).unwrap();

        let source = find_refresh_runtime(&app)
            .expect("the hoisted install one level up must be found");
        assert!(
            source.contains("window.$RefreshRuntime$"),
            "the located runtime must publish the global the client reads"
        );
    }

    /// A Next app has Fast Refresh natively and does NOT depend on
    /// @vitejs/plugin-react — not one app in the pinned Next corpus does — so
    /// `diffpack dev` on a Next project has to compose Next's OWN runtime.
    ///
    /// Next's helper API differs from plugin-react's (opposite argument order on
    /// `registerExportsForReactRefresh`; a boundary check split across
    /// `isReactRefreshBoundary` + `shouldInvalidateReactRefreshBoundary`), so this
    /// EXECUTES the composed source and checks the adapted decisions rather than
    /// grepping for names: an unchanged component boundary must be swappable
    /// (`undefined`), and a module whose exports are not components must report a
    /// reason so the footer falls back to `hot.invalidate`.
    #[test]
    fn next_own_fast_refresh_runtime_is_composed_and_answers_the_boundary_check() {
        let Some(node_modules) = real_next_node_modules() else {
            return;
        };
        let mut node = std::process::Command::new("node");
        node.env_remove("FORCE_COLOR");
        if node.arg("--version").output().is_err() {
            return;
        }
        let core = std::fs::read_to_string(
            node_modules.join("next/dist/compiled/react-refresh/cjs/react-refresh-runtime.development.js"),
        )
        .unwrap();
        let helpers = std::fs::read_to_string(
            node_modules.join("next/dist/compiled/@next/react-refresh-utils/dist/internal/helpers.js"),
        )
        .unwrap();
        let runtime = composed_next_runtime(&core, &helpers);

        let directory = tempfile::tempdir().unwrap();
        let runtime_path = directory.path().join("refresh-runtime.js");
        std::fs::write(&runtime_path, &runtime).unwrap();
        let driver = directory.path().join("driver.cjs");
        std::fs::write(
            &driver,
            "const fs=require('node:fs');\n\
             global.window=global;global.self=global;\n\
             (0,eval)(fs.readFileSync(process.argv[2],'utf8'));\n\
             const RT=global.$RefreshRuntime$;\n\
             const api=['register','injectIntoGlobalHook','createSignatureFunctionForTransform',\n\
             'performReactRefresh','registerExportsForReactRefresh','validateRefreshBoundaryAndEnqueueUpdate'];\n\
             for(const k of api) if(typeof RT[k]!=='function') throw new Error('missing '+k);\n\
             function A(){return null;}\n\
             const prev={default:A};\n\
             RT.registerExportsForReactRefresh('mod.jsx',prev);\n\
             function A2(){return null;}\n\
             const next={default:A2};\n\
             RT.registerExportsForReactRefresh('mod.jsx',next);\n\
             const same=RT.validateRefreshBoundaryAndEnqueueUpdate('mod.jsx',prev,next);\n\
             const notComponents=RT.validateRefreshBoundaryAndEnqueueUpdate('data.jsx',{x:1},{x:2});\n\
             console.log(JSON.stringify({same:same===undefined,notComponents:typeof notComponents}));\n",
        )
        .unwrap();
        let mut run = std::process::Command::new("node");
        run.env_remove("FORCE_COLOR");
        let output = run.arg(&driver).arg(&runtime_path).output().unwrap();
        assert!(
            output.status.success(),
            "the composed Next runtime must load and run: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "{\"same\":true,\"notComponents\":\"string\"}\n",
            "an unchanged component boundary swaps hot; a non-component module reports a reason"
        );
    }

    /// With no Fast Refresh runtime installed anywhere, the failure names every
    /// layout it looked for AND every directory it looked in — the diagnostic that
    /// makes a workspace misconfiguration diagnosable instead of mysterious.
    #[test]
    fn a_missing_refresh_runtime_names_every_layout_and_directory_searched() {
        let directory = tempfile::tempdir().unwrap();
        let app = directory.path().join("apps/web");
        std::fs::create_dir_all(&app).unwrap();
        let error = find_refresh_runtime(&app).unwrap_err();
        assert!(error.contains("@vitejs/plugin-react/dist/refresh-runtime.js"), "{error}");
        assert!(error.contains("@vitejs/plugin-react/dist/refreshUtils.js"), "{error}");
        assert!(error.contains("@next/react-refresh-utils"), "{error}");
        assert!(
            error.contains(&app.join("node_modules").display().to_string()),
            "the project's own node_modules must be listed: {error}"
        );
        assert!(
            error.contains(
                &directory
                    .path()
                    .join("node_modules")
                    .display()
                    .to_string()
            ),
            "every ancestor searched must be listed: {error}"
        );
    }

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
    fn overlay_script_installs_listeners_and_a_vlq_consumer() {
        // The dev error overlay must catch both uncaught errors and unhandled
        // rejections, publish its control surface, and carry an in-browser VLQ
        // base64 source-map consumer so generated-chunk frames resolve to source.
        let script = overlay_script();
        assert!(script.contains("window.__diffpackOverlay"), "must publish the overlay: {script}");
        assert!(
            script.contains("addEventListener(\"error\""),
            "must listen for uncaught errors: {script}"
        );
        assert!(
            script.contains("addEventListener(\"unhandledrejection\""),
            "must listen for unhandled rejections: {script}"
        );
        // VLQ base64 source-map consumer markers: the decoder and its alphabet.
        assert!(script.contains("function decodeVlq"), "must define a VLQ decoder: {script}");
        assert!(
            script.contains("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"),
            "must carry the VLQ base64 alphabet: {script}"
        );
        assert!(script.contains("function mapStack"), "must map stacks to original source: {script}");
        // The overlay DOM is created lazily (never at parse time) so it cannot cause a
        // React 19 hydration mismatch, and the script node self-removes.
        assert!(
            script.contains("document.createElement(\"div\")") && script.contains("self.remove()"),
            "overlay DOM must be lazy and the script node must self-remove: {script}"
        );
    }

    #[test]
    fn client_script_dispatches_build_error_and_build_ok() {
        // The HMR client transports Rust build errors to the overlay (show) and the
        // recovery signal that clears it (build-ok), so a syntax error no longer kills
        // dev — it is surfaced in the browser and cleared on the next good build.
        let script = client_script("/__diffpack_hmr/ws");
        assert!(
            script.contains("msg.type===\"build-error\"") && script.contains("showBuild(msg)"),
            "must show the overlay on a build error: {script}"
        );
        assert!(
            script.contains("msg.type===\"build-ok\"") && script.contains("__diffpackOverlay.clear()"),
            "must clear the overlay on build recovery: {script}"
        );
    }

    #[test]
    fn runtime_methods_expose_the_full_import_meta_hot_api() {
        // The per-module `module.hot` (which `import.meta.hot` rewrites to) must carry
        // the complete Vite HMR surface, not silent no-op stubs.
        let m = RUNTIME_METHODS;
        for member in [
            "accept(", "acceptExports(", "dispose(", "prune(", "invalidate(", "decline(",
            "on(event,cb)", "off(event,cb)", "send(",
        ] {
            assert!(m.contains(member), "runtime must define hot.{member}: {m}");
        }
        // `data` is the persisted object; it must be the SAME `__hmrData[id]` that
        // survives dispose + re-run (state preservation across a hot update).
        assert!(
            m.contains("const data=__hmrData[id]||(__hmrData[id]=Object.create(null))"),
            "hot.data must be the persisted per-module object: {m}"
        );
        // `__disposeModule` reads `__hmrData[id]` (to hand it to disposers) but must
        // NEVER delete or reassign it, or state would not survive the re-run.
        let dispose = m
            .split("function __disposeModule")
            .nth(1)
            .and_then(|rest| rest.split("function ").next())
            .unwrap_or("");
        assert!(
            !dispose.contains("delete __hmrData") && !dispose.contains("__hmrData[id]="),
            "dispose must not clear __hmrData (data must persist across updates): {dispose}"
        );
    }

    #[test]
    fn decline_forces_a_full_reload_and_is_honored_by_apply() {
        // decline() records the flag; the apply walk turns any update reaching a
        // declined module into a full reload rather than a hot swap.
        let m = RUNTIME_METHODS;
        assert!(
            m.contains("decline(){__hmrEntry(id).declined=true;}"),
            "decline must set the declined flag: {m}"
        );
        assert!(
            m.contains("if(e&&e.declined){__hmrReload"),
            "the apply walk must full-reload when it reaches a declined module: {m}"
        );
    }

    #[test]
    fn prune_registers_callbacks_and_the_runtime_can_fire_them() {
        // prune(cb) registers a pruner; __hmrPrune fires them, emits vite:beforePrune,
        // and forgets the module (cache + entry + data), so a re-add starts clean.
        let m = RUNTIME_METHODS;
        assert!(m.contains("prune(cb){__hmrEntry(id).pruners.push(cb);}"), "prune must register: {m}");
        let body = m
            .split("function __hmrPrune")
            .nth(1)
            .and_then(|rest| rest.split("function ").next())
            .unwrap_or("");
        assert!(body.contains("vite:beforePrune"), "prune must emit vite:beforePrune: {body}");
        for cleared in ["delete __cache[id]", "delete __hmrEntries[id]", "delete __hmrData[id]"] {
            assert!(body.contains(cleared), "prune must forget the module ({cleared}): {body}");
        }
    }

    #[test]
    fn apply_emits_before_and_after_update_events() {
        // on('vite:beforeUpdate'/'vite:afterUpdate') must observe a hot update: apply
        // brackets its work with the two lifecycle events.
        let m = RUNTIME_METHODS;
        assert!(m.contains("vite:beforeUpdate"), "apply must emit vite:beforeUpdate: {m}");
        assert!(m.contains("vite:afterUpdate"), "apply must emit vite:afterUpdate: {m}");
        assert!(
            m.contains("vite:beforeFullReload"),
            "a full reload must emit vite:beforeFullReload: {m}"
        );
    }

    /// The hot-update propagation walk must be LINEAR in the graph, not quadratic.
    ///
    /// `__importers` used to scan the whole module map on every call, and it is called
    /// once per node of the walk. On an example app that is invisible; on a real one it
    /// is fatal. cal.com's client graph is ~6,000 modules, and one edit to a component
    /// whose boundary did not self-accept made the browser tab stop responding entirely
    /// — no update, no error, just a wedged main thread.
    ///
    /// The assertion is structural, not a stopwatch: `__maps` and `__chunks` are
    /// Proxies that count how many times they are ENUMERATED. A scan-per-lookup
    /// implementation enumerates once per walked node (4,000 times here); the indexed
    /// one enumerates once, ever.
    #[test]
    fn the_update_walk_indexes_importers_instead_of_rescanning_the_graph() {
        let mut node = std::process::Command::new("node");
        node.env_remove("FORCE_COLOR");
        if node.arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let harness = directory.path().join("walk.mjs");
        // A deep import chain: module i imports module i-1, and only the last module
        // (the entry) accepts — so an edit to module 0 must climb every node.
        let prelude = r#"
const N = 4000;
let enumerations = 0;
const counting = (target) => new Proxy(target, {
  ownKeys(t) { enumerations++; return Reflect.ownKeys(t); },
});
const rawMaps = Object.create(null);
for (let i = 0; i < N; i++) rawMaps[i] = i === 0 ? {} : { "./prev": i - 1 };
const __maps = counting(rawMaps);
const __chunks = counting(Object.create(null));
const __modules = Object.create(null);
const __cache = Object.create(null);
const __entryId = N - 1;
function __require(id) { return (__cache[id] ||= { exports: {} }).exports; }
function __hmrPending() { return undefined; }
"#;
        let driver = r#"
__hmrEntry(__entryId).selfAccept = true;
const applied = await __hmrApply([0]);
console.log(JSON.stringify({ applied, enumerations }));
"#;
        std::fs::write(&harness, format!("{prelude}{RUNTIME_METHODS}{driver}")).unwrap();
        let mut run = std::process::Command::new("node");
        run.env_remove("FORCE_COLOR");
        let output = run.arg(&harness).output().unwrap();
        assert!(
            output.status.success(),
            "the update walk must run: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let got: serde_json::Value =
            serde_json::from_str(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
        assert_eq!(got["applied"], true, "the walk must reach the accepting boundary: {got}");
        let enumerations = got["enumerations"].as_u64().unwrap();
        assert!(
            enumerations <= 4,
            "the importer index must be built once, not rescanned per node \
             (enumerated {enumerations} times over a 4,000-module chain)"
        );
    }

    #[test]
    fn on_off_manage_a_shared_event_bus() {
        // hot.on registers per-module; hot.off removes by identity; __hmrEmit delivers
        // an event to EVERY module's listeners (Vite's emitter is shared).
        let m = RUNTIME_METHODS;
        assert!(
            m.contains("on(event,cb){const e=__hmrEntry(id);(e.listeners[event]||(e.listeners[event]=[])).push(cb);}"),
            "on must append to the per-module listener list: {m}"
        );
        assert!(
            m.contains("off(event,cb){") && m.contains("l.splice(i,1)"),
            "off must remove the listener by identity: {m}"
        );
        assert!(
            m.contains("function __hmrEmit(event,payload){for(const k in __hmrEntries)"),
            "emit must fan out to every module's listeners: {m}"
        );
    }

    #[test]
    fn client_script_handles_the_prune_message() {
        let script = client_script("/__diffpack_hmr/ws");
        assert!(
            script.contains("msg.type===\"prune\"") && script.contains("prt.prune(msg.ids"),
            "the HMR client must route a prune message to the runtime: {script}"
        );
    }

    #[test]
    fn a_single_named_component_export_is_a_refresh_boundary() {
        // `export const Navbar = () => {...}` — the shape that broke on real apps.
        let boundary = is_refresh_boundary(
            Path::new("/app/src/components/Navbar.tsx"),
            &["Navbar".to_string()],
            "export const Navbar = () => null",
            crate::parser::JsxExtensions::default(),
        );
        assert!(boundary, "a lone uppercase named component export must be a boundary");
    }

    #[test]
    fn a_next_js_component_module_is_a_refresh_boundary() {
        // Under Next, `.js` IS a component extension; treating it as plain JS gave
        // every Next `.js` page a full page reload on every edit instead of a
        // state-preserving swap.
        let arguments = (
            Path::new("/app/components/Gallery.js"),
            ["Gallery".to_string()],
            "export const Gallery = () => null",
        );
        assert!(
            is_refresh_boundary(
                arguments.0,
                &arguments.1,
                arguments.2,
                crate::parser::JsxExtensions::NextJs,
            ),
            "a Next `.js` component module must be a Fast Refresh boundary"
        );
        assert!(
            !is_refresh_boundary(
                arguments.0,
                &arguments.1,
                arguments.2,
                crate::parser::JsxExtensions::JsxAndTsxOnly,
            ),
            "under the Vite rule a `.js` module is plain JS, never a boundary"
        );
    }

    /// A module body carrying oxc's refresh instrumentation, wrapped in a function so
    /// the preamble's `var` bindings are module-scoped exactly as they are inside a
    /// real factory.
    fn instrumented_module(key: &str, component: &str) -> String {
        format!(
            "function {component}_of_{}(){{\n{}\nfunction {component}(){{return null;}}\n\
             $RefreshReg$({component},\"{component}\");\nreturn {component};\n}}\n",
            key.replace(['/', '.', '-'], "_"),
            fast_refresh_preamble(key),
        )
    }

    /// Two DIFFERENT modules that each define a component with the same local name
    /// must register under DIFFERENT Fast Refresh family ids.
    ///
    /// oxc's transform emits `$RefreshReg$(_c, "Foo")` — the local name and nothing
    /// else — and react-refresh keys families in ONE global map, so an unscoped id
    /// merges unrelated components into one family. This executes both module bodies
    /// against a recording runtime and asserts the ids differ; the module key is what
    /// has to make them differ, not the component name.
    #[test]
    fn two_modules_with_the_same_component_name_register_under_different_family_ids() {
        let mut node = std::process::Command::new("node");
        node.env_remove("FORCE_COLOR");
        if node.arg("--version").output().is_err() {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let harness = directory.path().join("scoped.mjs");
        let driver = format!(
            "const seen=[];\n\
             globalThis.window=globalThis;\n\
             globalThis.$RefreshRuntime$={{register:(type,id)=>seen.push(id),\
             createSignatureFunctionForTransform:()=>(t)=>t}};\n\
             {}{}\
             Foo_of__a_list_tsx();\nFoo_of__b_table_tsx();\n\
             console.log(JSON.stringify(seen));\n",
            instrumented_module("/a/list.tsx", "Foo"),
            instrumented_module("/b/table.tsx", "Foo"),
        );
        std::fs::write(&harness, driver).unwrap();
        let mut run = std::process::Command::new("node");
        run.env_remove("FORCE_COLOR");
        let output = run.arg(&harness).output().unwrap();
        assert!(
            output.status.success(),
            "the instrumented modules must run: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ids: Vec<String> =
            serde_json::from_slice(&output.stdout).expect("the driver prints the registered ids");
        assert_eq!(ids.len(), 2, "both modules must register their component: {ids:?}");
        assert_ne!(
            ids[0], ids[1],
            "two modules' same-named components must NOT share a Fast Refresh family: {ids:?}"
        );
        assert!(
            ids[0].starts_with("/a/list.tsx") && ids[1].starts_with("/b/table.tsx"),
            "each id must be namespaced by its own module key: {ids:?}"
        );
    }

    /// The consequence the scoping exists to prevent, checked against the REAL
    /// react-refresh runtime: a fresh page load must leave react-refresh with NOTHING
    /// pending.
    ///
    /// `register(type, id)` treats a second registration under a known id as a hot
    /// update of the first, so with unscoped ids a bundle merely LOADING queues
    /// phantom updates for every duplicated component name. The first real edit then
    /// makes `performReactRefresh` swap hundreds of live component types for unrelated
    /// ones, and React's `scheduleRefresh` -> `flushSyncWork` loop never terminates —
    /// on cal.com the browser tab wedged and the update never appeared. Both halves
    /// are asserted so the test has teeth: unscoped ids MUST queue an update, scoped
    /// ones MUST NOT.
    #[test]
    fn a_fresh_load_queues_no_phantom_refresh_update_for_duplicated_component_names() {
        let Some(node_modules) = real_next_node_modules() else {
            return;
        };
        let mut node = std::process::Command::new("node");
        node.env_remove("FORCE_COLOR");
        if node.arg("--version").output().is_err() {
            return;
        }
        let core = std::fs::read_to_string(node_modules.join(
            "next/dist/compiled/react-refresh/cjs/react-refresh-runtime.development.js",
        ))
        .unwrap();
        let helpers = std::fs::read_to_string(
            node_modules
                .join("next/dist/compiled/@next/react-refresh-utils/dist/internal/helpers.js"),
        )
        .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let runtime_path = directory.path().join("refresh-runtime.js");
        std::fs::write(&runtime_path, composed_next_runtime(&core, &helpers)).unwrap();
        let driver_path = directory.path().join("driver.cjs");
        let driver = format!(
            "const fs=require('node:fs');\n\
             global.window=global;global.self=global;\n\
             (0,eval)(fs.readFileSync(process.argv[2],'utf8'));\n\
             const RT=global.$RefreshRuntime$;\n\
             // Unscoped: exactly what an unnamespaced `$RefreshReg$` does.\n\
             function Bare1(){{return null;}}\nfunction Bare2(){{return null;}}\n\
             RT.register(Bare1,'Foo');RT.register(Bare2,'Foo');\n\
             const unscoped=RT.performReactRefresh();\n\
             {}{}\
             Foo_of__a_list_tsx();\nFoo_of__b_table_tsx();\n\
             const scoped=RT.performReactRefresh();\n\
             console.log(JSON.stringify({{unscoped:unscoped!==null,scoped:scoped!==null}}));\n",
            instrumented_module("/a/list.tsx", "Foo"),
            instrumented_module("/b/table.tsx", "Foo"),
        );
        std::fs::write(&driver_path, driver).unwrap();
        let mut run = std::process::Command::new("node");
        run.env_remove("FORCE_COLOR");
        let output = run.arg(&driver_path).arg(&runtime_path).output().unwrap();
        assert!(
            output.status.success(),
            "the refresh runtime driver must run: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "{\"unscoped\":true,\"scoped\":false}\n",
            "unscoped ids must queue a phantom update; module-scoped ids must queue none"
        );
    }
}
