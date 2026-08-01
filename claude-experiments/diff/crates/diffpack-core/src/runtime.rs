//! Framework-neutral JavaScript registry runtime assembly.

use crate::async_graph::AsyncModules;
use crate::bundle::{DenseModuleId, ModuleMapping, RenderedBundle};
use crate::emission::ModuleFormat;
use crate::text_edit::quote;

/// Where a changed module lives in the current emitted chunk plan.
#[derive(Debug, Clone)]
pub struct HmrLocation {
    pub module_id: String,
    pub runtime_id: usize,
    pub chunk_file: String,
}

/// Optional host-supplied hot-update protocol fragments.
pub struct RuntimeHotPolicy<'a> {
    pub require_dynamic: &'a str,
    pub hot_install: &'a str,
    pub methods: &'a str,
    pub runtime_return: &'a str,
    pub reimport_guard: &'a str,
    pub server_control: &'a str,
}

#[derive(Debug, Clone)]
pub struct OwnedRuntimeHotPolicy {
    pub require_dynamic: String,
    pub hot_install: String,
    pub methods: String,
    pub runtime_return: String,
    pub reimport_guard: String,
    pub server_control: String,
}

impl OwnedRuntimeHotPolicy {
    pub fn borrowed(&self) -> RuntimeHotPolicy<'_> {
        RuntimeHotPolicy {
            require_dynamic: &self.require_dynamic,
            hot_install: &self.hot_install,
            methods: &self.methods,
            runtime_return: &self.runtime_return,
            reimport_guard: &self.reimport_guard,
            server_control: &self.server_control,
        }
    }
}

pub struct RuntimePolicyModule<'a> {
    pub id: &'a str,
    pub source: &'a str,
}

#[derive(Clone, Copy)]
pub struct RuntimePolicyRequest<'a> {
    pub format: ModuleFormat,
    pub is_main: bool,
    pub hmr: bool,
    pub entry_id: &'a str,
    pub entry_runtime_id: usize,
    pub any_async: bool,
    pub base: &'a str,
    pub chunk_files: &'a [String],
    pub modules: &'a [RuntimePolicyModule<'a>],
    pub browser_process_shim: bool,
}

#[derive(Debug, Default)]
pub struct RuntimePolicyOutput {
    pub entry_preludes: Vec<String>,
    pub compatibility_prelude: Option<String>,
    pub browser_require_native: Option<String>,
    pub hot: Option<OwnedRuntimeHotPolicy>,
}

pub trait RuntimeIntegrationPolicy: Send + Sync {
    fn configure(&self, request: RuntimePolicyRequest<'_>) -> RuntimePolicyOutput;

    fn flat_entry_prelude(&self, _browser_process_shim: bool) -> Option<String> {
        None
    }
}

/// Ordered composition of independently-owned runtime policies.
#[derive(Default)]
pub struct RuntimePolicyChain {
    policies: Vec<std::sync::Arc<dyn RuntimeIntegrationPolicy>>,
}

impl RuntimePolicyChain {
    pub fn new(policies: Vec<std::sync::Arc<dyn RuntimeIntegrationPolicy>>) -> Self {
        Self { policies }
    }
}

impl RuntimeIntegrationPolicy for RuntimePolicyChain {
    fn configure(&self, request: RuntimePolicyRequest<'_>) -> RuntimePolicyOutput {
        let mut combined = RuntimePolicyOutput::default();
        for policy in &self.policies {
            let output = policy.configure(request);
            combined.entry_preludes.extend(output.entry_preludes);
            if output.compatibility_prelude.is_some() {
                combined.compatibility_prelude = output.compatibility_prelude;
            }
            if output.browser_require_native.is_some() {
                combined.browser_require_native = output.browser_require_native;
            }
            if output.hot.is_some() {
                combined.hot = output.hot;
            }
        }
        combined
    }

    fn flat_entry_prelude(&self, browser_process_shim: bool) -> Option<String> {
        let preludes = self
            .policies
            .iter()
            .filter_map(|policy| policy.flat_entry_prelude(browser_process_shim))
            .collect::<Vec<_>>();
        (!preludes.is_empty()).then(|| preludes.join("\n"))
    }
}

#[derive(Debug, Default)]
pub struct NoRuntimeIntegrationPolicy;

impl RuntimeIntegrationPolicy for NoRuntimeIntegrationPolicy {
    fn configure(&self, _request: RuntimePolicyRequest<'_>) -> RuntimePolicyOutput {
        RuntimePolicyOutput::default()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_registry_runtime(
    entry_id: &str,
    graph_entry: DenseModuleId,
    roots: &[DenseModuleId],
    runtime_ids: &[Option<usize>],
    async_modules: &AsyncModules,
    is_main: bool,
    format: ModuleFormat,
    prelude: String,
    prerequisite_loads: String,
    modules: String,
    maps: String,
    chunks: String,
    mappings: Vec<ModuleMapping>,
    require_native: String,
    hmr_policy: Option<RuntimeHotPolicy<'_>>,
) -> RenderedBundle {
    let runtime_key = quote(&format!("__diffpack_runtime:{entry_id}"));
    // Only the main chunk names a module in its tail (it evaluates the entry
    // and returns its exports); a split chunk only registers factories, so it
    // has no single "entry" to identify.
    let entry_runtime_id =
        runtime_ids[graph_entry].expect("the entry module must have a deterministic runtime ID");
    // In ESM output (Node or browser) a split chunk is a real module, loaded
    // for its REGISTRATION side effect: `require.dynamic` imports the file and
    // then resolves the requested module by runtime id out of the shared
    // registry, so one chunk can carry several roots (and shared code that is
    // nobody's root) without any of them having to be its default export.
    // The host supplies `requireNative`; environments without native loading may
    // provide a load-safe throw-on-use stub so dead host-specific code can load,
    // so the stub lets property reads and construction succeed (the module
    // LOADS), but throws a clear, specifically-named error the moment that
    // dead code actually CALLS into the built-in — it never fabricates a
    // value. Protocol probes (`then`/`Symbol.toPrimitive`/iterators) return
    // `undefined` so the stub is neither mistaken for a thenable nor silently
    // coerced. In CJS output both go through the host `require`, as before.
    // With an async module in the build, a dynamically imported target may be
    // one; `import()` already yields a promise, so `require.dynamic` resolves
    // through `__requireAsync` and the awaited namespace is fully initialised.
    // `__chunkQuery` (defined in the runtime prelude below) is the query the
    // ENTRY chunk's own module URL carries, propagated to every chunk it
    // dynamically imports. It is empty for a normal load; it matters when a
    // host deliberately re-imports the entry under a fresh URL to get a FRESH
    // module graph — as a reload worker may do
    // (`import(entry + "?v=" + mtime)` after dropping the runtime globals). The
    // registry lives on `globalThis`, so without propagating the query the new
    // entry instance builds a new registry while every already-imported chunk
    // stays in Node's ESM cache and never re-runs its `__register`, and the
    // first `import()` of one of those chunks resolves instantly to a module
    // that registered into the DISCARDED runtime — `__require` then throws
    // "Module is not loaded: <id>".
    let require_dynamic_esm = if async_modules.any {
        r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null)return import(chunk[0]+__chunkQuery).then(()=>__requireAsync(chunk[1]));return __requireAsync(chunk[1]);};"#
    } else {
        r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null)return import(chunk[0]+__chunkQuery).then(()=>__require(chunk[1]));return __require(chunk[1]);};"#
    };
    let require_dynamic = match format {
        ModuleFormat::Esm | ModuleFormat::BrowserEsm => require_dynamic_esm,
        ModuleFormat::Cjs => {
            r#"require.dynamic=specifier=>{const chunk=__chunks[id][specifier];if(chunk===undefined)return require(specifier);if(chunk[0]!==null){if(typeof requireNative!=="function")throw new Error("Dynamic chunks require a CommonJS host");requireNative(chunk[0]);}return __require(chunk[1]);};"#
        }
    };
    let require_dynamic = hmr_policy
        .as_ref()
        .map_or(require_dynamic, |policy| policy.require_dynamic);
    let hot_install = hmr_policy.as_ref().map_or("", |policy| policy.hot_install);
    let hmr_methods = hmr_policy.as_ref().map_or("", |policy| policy.methods);
    let runtime_return = hmr_policy.as_ref().map_or_else(
        || {
            if async_modules.any {
                "return {register:__register,require:__require,requireAsync:__requireAsync};"
                    .to_string()
            } else {
                "return {register:__register,require:__require};".to_string()
            }
        },
        |policy| policy.runtime_return.to_string(),
    );
    // ASYNC MODULES. A module that top-level-`await`s renders as an `async`
    // factory, so calling it returns a promise instead of running to
    // completion. `__pending[id]` holds that promise until the module has
    // finished initialising; a rejected one is deliberately LEFT in place so a
    // later importer rejects too rather than reading a half-built namespace.
    //
    // `__require` stays synchronous and keeps returning `module.exports` — the
    // namespace object is created (and its getters installed) by the factory's
    // synchronous prefix, before its first `await`, so its identity is stable
    // and every existing caller is unaffected. Waiting is the caller's job, via
    // `require.esmAsync`/`require.async` (emitted at an async module's own
    // import sites) or `__requireAsync` (the chunk tail and `require.dynamic`).
    // Both first run `require`, which is what populates `__pending`, and only
    // then look the pending promise up.
    //
    // Every line here is emitted only when the build actually has an async
    // module, so an ordinary bundle's runtime is byte-for-byte what it was.
    let (require_async, require_async_runtime, factory_call) = if async_modules.any {
        (
            "  require.async=specifier=>{const target=__maps[id][specifier],value=require(specifier),pending=target===undefined?undefined:__pending[target];return pending?pending.then(()=>value):value;};\n  require.esmAsync=specifier=>{const target=__maps[id][specifier],namespace=require.esm(specifier),pending=target===undefined?undefined:__pending[target];return pending?pending.then(()=>namespace):namespace;};\n",
            "const __pending=Object.create(null);\nfunction __requireAsync(id){const exports=__require(id),pending=__pending[id];return pending?pending.then(()=>exports):exports;}\n",
            "const __result=factory(module,module.exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal);\n  if(__result&&typeof __result.then===\"function\")__pending[id]=__result.then(()=>{delete __pending[id];});",
        )
    } else {
        (
            "",
            "",
            "factory(module,module.exports,require,__toESM,__export,__reExport,__import,__dynamic,__esmNamespace,__seal);",
        )
    };
    let reimport_guard = hmr_policy
        .as_ref()
        .map_or("", |policy| policy.reimport_guard);
    let server_control = hmr_policy
        .as_ref()
        .map_or("", |policy| policy.server_control);
    // `__toESM` decides whether a required module is ALREADY an ES namespace or a
    // CommonJS `module.exports` that needs interop. That decision is made on
    // `__esmNamespaces`, a brand only `__esmNamespace` (i.e. only diffpack's own
    // ESM emit) can add to, plus a null-prototype `Symbol.toStringTag === "Module"`
    // test for a namespace the HOST produced. It is deliberately NOT made on
    // `__esModule`: that is a convention marker any CommonJS file may stamp on its
    // own `exports` — tslib's UMD build and every TypeScript package published with
    // `importHelpers` do — and treating it as proof of ESM handed such a module
    // straight through, so `import x from "tslib"` threw "does not provide an export
    // named default". Node's ESM-imports-CJS rule ignores `__esModule` entirely:
    // `default` is `module.exports`, which is what the interop below builds.
    //
    // Three properties the interop must hold, each of which was a defect:
    //
    //  * IDEMPOTENT AND STABLE. `__toESM` is not called once per module but once
    //    per import site, and `export * as ns from "cjs"` re-runs it on every
    //    read of `ns`. `__cjsNamespaces` keys the wrapper by the `module.exports`
    //    it wraps, so one CommonJS module has exactly one namespace object (as in
    //    Node), and `__isESM` recognises a wrapper (via `__cjsInterops`) so
    //    re-wrapping a wrapper is a no-op instead of nesting `default.default`.
    //    Keying by `module.exports` cannot cover `module.exports = 42` — a
    //    WeakMap takes no primitive key — so a static import goes through
    //    `require.esm`, which keys by the MODULE ID (`__idNamespaces`) and is the
    //    only identity that exists for every value shape. Both halves matter and
    //    neither alone is enough: id-keying alone would give two modules that
    //    each `module.exports = 42` one shared namespace under a value-keyed
    //    cache, and exports-keying alone gives ONE module a fresh namespace per
    //    read (`ns.legacy === ns.legacy` was `false` against Node's `true`).
    //  * STRICT ABOUT NAMED EXPORTS. `import { missing } from "./legacy.cjs"` is a
    //    hard error in Node; it must not evaluate to `undefined` here. The wrapper
    //    is therefore NOT exempt from `__import`'s check — the check consults the
    //    live `module.exports` and throws when the name is on neither.
    //  * LIVE, NOT A SNAPSHOT. The wrapper's enumerable keys are copied from
    //    `module.exports` at wrap time, which in an ESM<->CJS cycle is a
    //    PARTIALLY populated object. `__syncCJS` re-copies on every later
    //    `__toESM` of the same exports, and `__import` reads through to the live
    //    `module.exports`, so a key the module assigns after the wrap is visible
    //    rather than permanently missing.
    // Whether the module this chunk EVALUATES is async, and so whether the
    // chunk's own wrapper has to await it. `requireAsync` returns a promise
    // only for a module still initialising, so the wrapper's `await` is what
    // makes the chunk's default export the module's FINISHED namespace.
    let awaits_evaluation = if is_main {
        async_modules.is_async(graph_entry)
    } else {
        matches!(roots, [only] if async_modules.is_async(*only))
    };
    let require_entry = if awaits_evaluation {
        "requireAsync"
    } else {
        "require"
    };
    // See `require_dynamic_esm`: the entry chunk's own query, re-attached to
    // every chunk URL so one runtime instance only ever loads chunk instances
    // that register into it. Empty (and therefore inert) unless the host
    // imported the entry with a query. CommonJS output has no `import.meta`
    // and no such protocol, so it declares nothing.
    let chunk_query = if format.is_esm() {
        "const __chunkQuery=(()=>{const __q=import.meta.url.indexOf(\"?\");return __q<0?\"\":import.meta.url.slice(__q);})();\n"
    } else {
        ""
    };
    let tail = if is_main {
        format!(
            r#"const __runtime=globalThis[{runtime_key}]??=(()=>{{
const __modules=Object.create(null),__maps=Object.create(null),__chunks=Object.create(null),__cache=Object.create(null);
{chunk_query}const __exportStates=new WeakMap(),__esmNamespaces=new WeakSet(),__cjsNamespaces=new WeakMap(),__cjsInterops=new WeakMap(),__cjsOrigins=new WeakMap(),__idNamespaces=Object.create(null);
function __esmNamespace(){{const namespace=Object.create(null);Object.defineProperty(namespace,Symbol.toStringTag,{{value:"Module"}});__esmNamespaces.add(namespace);return namespace;}}
function __seal(namespace){{const movable=Reflect.ownKeys(namespace).filter(key=>typeof key==="string"&&Object.getOwnPropertyDescriptor(namespace,key).configurable);const sorted=[...movable].sort();if(movable.some((key,index)=>key!==sorted[index])){{const descriptors={{}};for(const key of movable){{descriptors[key]=Object.getOwnPropertyDescriptor(namespace,key);delete namespace[key];}}for(const key of sorted)Object.defineProperty(namespace,key,descriptors[key]);}}for(const key of Reflect.ownKeys(namespace)){{const descriptor=Object.getOwnPropertyDescriptor(namespace,key);if(descriptor?.configurable)Object.defineProperty(namespace,key,{{configurable:false}});}}Object.preventExtensions(namespace);}}
function __exportState(target){{let state=__exportStates.get(target);if(!state){{state={{explicit:new Set(),stars:new Map(),ambiguous:new Set()}};__exportStates.set(target,state);}}return state;}}
function __export(target,name,getter){{const state=__exportState(target);const descriptor=Object.getOwnPropertyDescriptor(target,name);if(descriptor?.configurable)delete target[name];if(!Object.prototype.hasOwnProperty.call(target,name))Object.defineProperty(target,name,{{enumerable:true,configurable:true,get:getter}});state.explicit.add(name);state.stars.delete(name);state.ambiguous.delete(name);}}
function __reExport(target,source){{const state=__exportState(target);for(const key of Object.keys(source)){{if(key==="default"||key==="__esModule"||state.explicit.has(key)||state.ambiguous.has(key))continue;const previous=state.stars.get(key);if(previous&&previous!==source){{delete target[key];state.stars.delete(key);state.ambiguous.add(key);continue;}}if(!previous){{Object.defineProperty(target,key,{{enumerable:true,configurable:true,get:()=>source[key]}});state.stars.set(key,source);}}}}}}
function __holdsProperties(value){{return value!==null&&value!==undefined&&(typeof value==="object"||typeof value==="function");}}
function __origin(exports,specifier){{if(__holdsProperties(exports)&&!__cjsOrigins.has(exports))__cjsOrigins.set(exports,specifier);return exports;}}
function __isESM(value){{if(!value||(typeof value!=="object"&&typeof value!=="function"))return false;if(__esmNamespaces.has(value)||__cjsInterops.has(value))return true;return Object.getPrototypeOf(value)===null&&value[Symbol.toStringTag]==="Module";}}
function __syncCJS(namespace,value){{if(__holdsProperties(value))for(const key of Object.keys(value))if(key!=="default"&&!Object.prototype.hasOwnProperty.call(namespace,key))__export(namespace,key,()=>value[key]);return namespace;}}
function __transpiledESM(value){{
  if(!__holdsProperties(value)||!Object.prototype.hasOwnProperty.call(value,"default"))return false;
  try{{return !!value.__esModule;}}catch{{return false;}}
}}
function __toESM(value){{
  if(__isESM(value))return value;
  const cached=__cjsNamespaces.get(value);
  if(cached)return __syncCJS(cached,value);
  const namespace=Object.create(null);
  Object.defineProperty(namespace,"__esModule",{{value:true}});
  // The `__esModule` interop. A CommonJS module that BOTH stamps `__esModule` and owns
  // a `default` property was compiled down from ESM (TypeScript / Babel / SWC output —
  // which is most of npm), so its `default` IS the module's default export and
  // `import X from` must bind that function, not the exports object wrapping it. Every
  // common module interop does this, and it is the reason the marker
  // exists; without it `import CredentialsProvider from "next-auth/providers/
  // credentials"` binds `{{__esModule:true,default:fn}}` and calling it throws
  // "is not a function" — which is exactly how cal.com's next-auth config died.
  //
  // Without a `default` property the marker says nothing about what a default import
  // should be, so the Node rule stands: the default export is `module.exports`.
  if(__transpiledESM(value))__export(namespace,"default",()=>value.default);
  else __export(namespace,"default",()=>value);
  __syncCJS(namespace,value);
  __cjsInterops.set(namespace,{{exports:value}});
  if(__holdsProperties(value))__cjsNamespaces.set(value,namespace);
  return namespace;
}}
function __namespaceOf(id,value){{
  if(__holdsProperties(value))return __toESM(value);
  const cached=__idNamespaces[id];
  if(cached)return cached;
  const namespace=__toESM(value);
  __idNamespaces[id]=namespace;
  return namespace;
}}
function __import(namespace,name){{
  if(Object.prototype.hasOwnProperty.call(namespace,name))return namespace[name];
  const interop=__cjsInterops.get(namespace);
  if(interop&&__holdsProperties(interop.exports)&&Object.prototype.hasOwnProperty.call(interop.exports,name)){{const exports=interop.exports;__export(namespace,name,()=>exports[name]);return exports[name];}}
  const origin=__cjsOrigins.get(namespace)??(interop?__cjsOrigins.get(interop.exports):undefined);
  throw new SyntaxError("The requested module "+(origin===undefined?"(unknown)":JSON.stringify(origin))+" does not provide an export named "+JSON.stringify(name));
}}
function __dynamic(require,specifier){{return Promise.resolve().then(()=>require.dynamic(specifier)).then(exports=>__toESM(__origin(exports,specifier)));}}
function __register(modules,maps,chunks){{Object.assign(__modules,modules);Object.assign(__maps,maps);Object.assign(__chunks,chunks);}}
function __require(id){{
  if(__cache[id])return __cache[id].exports;
  const factory=__modules[id];
  if(!factory)throw new Error("Module is not loaded: "+id);
  const module={{exports:{{}}}};
  __cache[id]=module;
  {hot_install}
  const require=specifier=>{{const target=__maps[id][specifier];if(target===undefined){{if(requireNative)return __origin(requireNative(specifier),specifier);throw new Error("Cannot resolve "+specifier+" from "+id);}}return __origin(__require(target),specifier);}};
  require.esm=specifier=>{{const target=__maps[id][specifier],value=require(specifier);return target===undefined?__toESM(value):__namespaceOf(target,value);}};
{require_async}  {require_dynamic}
  {factory_call}
  return module.exports;
}}
{require_async_runtime}{require_native}
{hmr_methods}
{runtime_return}
}})();
{server_control}
__runtime.register(__newModules,__newMaps,__newChunks);
const __queued=globalThis[{runtime_key}+":pending"];
if(__queued){{for(let __i=0;__i<__queued.length;__i++)__runtime.register(__queued[__i][0],__queued[__i][1],__queued[__i][2]);__queued.length=0;}}
{reimport_guard}
return __runtime.{require_entry}({entry_runtime_id});"#
        )
    } else {
        // A split chunk always REGISTERS; whether it also evaluates depends on
        // how it can be consumed, and there are two ways.
        //
        // `require.dynamic` evaluates the requested module by runtime id, so
        // registration alone is enough for it. But a chunk is ALSO imported
        // directly as an ES module: the generated SSR router does
        // `import manifest from "./_tanstack-start-manifest_v.mjs"` and reads
        // the factory off the default export. That consumer needs the default
        // export to BE the root's namespace, so a chunk with exactly one root
        // evaluates it and returns its exports.
        //
        // A chunk with several roots, or a purely shared chunk with none, has
        // no single namespace that could stand for it. Nothing imports those
        // directly (only `require.dynamic` and prerequisite headers name them),
        // and evaluating a root the caller did not ask for would run its side
        // effects early, so they register and return the runtime.
        let evaluate = match roots {
            [only] => {
                let root_runtime_id =
                    runtime_ids[*only].expect("a chunk root must have a deterministic runtime ID");
                format!("return __runtime.{require_entry}({root_runtime_id});")
            }
            _ => "return __runtime;".to_string(),
        };
        // ORDER-INDEPENDENT REGISTRATION. A chunk may execute BEFORE the chunk that
        // builds the runtime: the document loads them as separate scripts, and nothing
        // in HTML guarantees which runs first (react-dom even marks its bootstrap tag
        // `async`). Throwing there made document order load-bearing, and it broke three
        // separate ways on cal.com before this: a chunk racing the entry threw, so no
        // page hydrated.
        //
        // So an early chunk QUEUES its registration and the runtime drains the queue the
        // moment it is created — the same shape webpack uses (`webpackChunk.push`), and
        // the reason webpack chunks can be loaded in any order at all.
        //
        // Its default export is `undefined` on that path, because evaluating the root
        // needs the registry. Nothing reads it there: `require.dynamic` resolves by
        // runtime id and the RSC seam's `__webpack_chunk_load__` discards the namespace.
        // The one consumer that DOES read it — the generated SSR router importing the
        // generated manifest chunks may run in a single server bundle whose entry has
        // always executed first, so it still gets the evaluated root below.
        format!(
            r#"const __runtime=globalThis[{runtime_key}];
if(!__runtime){{
(globalThis[{runtime_key}+":pending"]??=[]).push([__newModules,__newMaps,__newChunks]);
}}else{{
__runtime.register(__newModules,__newMaps,__newChunks);
{reimport_guard}
{evaluate}
}}"#
        )
    };
    // The registry runtime is identical across formats; only the module
    // boundary differs. CJS assigns the entry's exports to `module.exports`.
    // Both ESM variants bind them to a local and re-export as the default. The
    // host prelude supplies any native module bridge required by its ESM goal.
    // A chunk that evaluates an ASYNC module can only publish its finished
    // namespace by awaiting it, so its wrapper becomes an async IIFE and the
    // chunk itself top-level-`await`s — legal in an ES module (which is why
    // `emit_with_options` still refuses top-level await in CommonJS output),
    // and it makes the emitted file's own evaluation async exactly as the
    // source module graph's is.
    let (open_wrapper, close_wrapper) = if awaits_evaluation {
        ("await (async()=>{", "})()")
    } else {
        ("(()=>{", "})()")
    };
    let code = if format.is_esm() {
        format!(
            r#"{prelude}{prerequisite_loads}const __diffpackEntry={open_wrapper}
"use strict";
const __newModules={{{modules}}};
const __newMaps={{{maps}}};
const __newChunks={{{chunks}}};
{tail}
{close_wrapper};
export default __diffpackEntry;
"#
        )
    } else {
        format!(
            r#"{prelude}module.exports=(()=>{{
"use strict";
{prerequisite_loads}const __newModules={{{modules}}};
const __newMaps={{{maps}}};
const __newChunks={{{chunks}}};
{tail}
}})();
"#
        )
    };
    RenderedBundle {
        code,
        mappings,
        map_json: None,
    }
}
