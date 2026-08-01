//! Native `.next` server-entry compilation.

use std::path::Path;

use diffpack_core::{EmitOptions, ModuleFormat};
use diffpack_default_loader::driver_config::BuildConfig;

struct NativeEntry {
    generated: std::path::PathBuf,
    output: std::path::PathBuf,
}

/// Native Next owns its request and cache AsyncLocalStorage stores. The standalone
/// adapter's `next/headers` and `next/cache` implementations intentionally own
/// different stores, so carrying those aliases into a native route would split
/// request state across two runtimes. Bind server-only public APIs back to the
/// installed Next package while leaving client-facing shims (link/image/navigation)
/// on Diffpack's client-reference identity.
fn use_native_next_server_apis(config: &mut BuildConfig, next_root: &Path) {
    const SERVER_APIS: [&str; 4] = [
        "next/headers",
        "next/headers.js",
        "next/cache",
        "next/cache.js",
    ];
    config
        .aliases
        .retain(|(specifier, _)| !SERVER_APIS.contains(&specifier.as_str()));
    for (specifier, relative) in [
        ("next/headers", "headers.js"),
        ("next/headers.js", "headers.js"),
        ("next/cache", "cache.js"),
        ("next/cache.js", "cache.js"),
    ] {
        config.aliases.push((
            specifier.to_string(),
            next_root.join(relative).to_string_lossy().into_owned(),
        ));
    }
}

fn static_metadata_userland(
    route: &crate::artifacts::NextRouteArtifact,
    generated_dir: &Path,
) -> Result<Option<std::path::PathBuf>, String> {
    if route
        .source_path
        .extension()
        .and_then(|value| value.to_str())
        != Some("ico")
    {
        return Ok(None);
    }
    let bytes = std::fs::read(&route.source_path)
        .map_err(|error| format!("cannot read {}: {error}", route.source_path.display()))?;
    let literal = bytes
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let output = generated_dir.join("favicon-userland.ts");
    std::fs::write(
        &output,
        format!(
            "const bytes=new Uint8Array([{literal}]);\n\
             export function GET(){{return new Response(bytes,{{headers:{{\
             'content-type':'image/x-icon','cache-control':'public, max-age=0, must-revalidate'\
             }}}});}}\n"
        ),
    )
    .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
    Ok(Some(output))
}

/// Discover all roots in one configured environment, then select and emit each root
/// independently. Shared modules are resolved, parsed, and transformed exactly once.
fn compile_shared_entries(
    config: &BuildConfig,
    entries: &[NativeEntry],
    shared_output: &Path,
) -> Result<(), String> {
    let Some(first) = entries.first() else {
        return Ok(());
    };
    let dispatcher = first
        .generated
        .parent()
        .ok_or_else(|| format!("entry has no parent: {}", first.generated.display()))?
        .join("__diffpack_shared_entries.cjs");
    let loaders = entries
        .iter()
        .map(|entry| {
            serde_json::to_string(&entry.generated.to_string_lossy())
                .map(|path| format!("()=>require({path})"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot serialize native entry path: {error}"))?;
    std::fs::write(
        &dispatcher,
        format!("module.exports=[{}];\n", loaders.join(",")),
    )
    .map_err(|error| format!("cannot write {}: {error}", dispatcher.display()))?;

    let discover_started = std::time::Instant::now();
    let (bundler, update) = crate::compiler::discover_native_next(&dispatcher, config)?;
    for warning in diffpack_core::partition_diagnostics(
        &update.diagnostics,
        "native Next shared entry registry",
    )? {
        eprintln!("warning: {warning}");
    }
    let reachable = bundler.reachable_modules_direct();
    eprintln!(
        "native Next: discovered {} entries ({} unique modules) in {:.2?}",
        entries.len(),
        reachable.len(),
        discover_started.elapsed()
    );

    let emit_started = std::time::Instant::now();
    if let Some(parent) = shared_output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    bundler.emit_with_options(
        &reachable,
        shared_output,
        EmitOptions {
            minify: false,
            source_map: false,
            format: ModuleFormat::Cjs,
            ..EmitOptions::default()
        },
    )?;
    let shared_parent = shared_output
        .parent()
        .ok_or_else(|| format!("shared output has no parent: {}", shared_output.display()))?;
    let shared_name = shared_output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "shared output has no file name: {}",
                shared_output.display()
            )
        })?;
    for (index, entry) in entries.iter().enumerate() {
        if let Some(parent) = entry.output.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
            let below_server = parent.strip_prefix(shared_parent).map_err(|_| {
                format!(
                    "native entry output {} is not below shared output directory {}",
                    entry.output.display(),
                    shared_parent.display()
                )
            })?;
            let up = "../".repeat(below_server.components().count());
            let specifier = if up.is_empty() {
                format!("./{shared_name}")
            } else {
                format!("{up}{shared_name}")
            };
            std::fs::write(
                &entry.output,
                format!("module.exports=require({specifier:?})[{index}]();\n"),
            )
            .map_err(|error| format!("cannot write {}: {error}", entry.output.display()))?;
        }
    }
    eprintln!(
        "native Next: emitted one shared registry and {} lazy entry stubs in {:.2?}",
        entries.len(),
        emit_started.elapsed()
    );
    Ok(())
}

/// Some CommonJS hosts require the entry module synchronously even when modules below
/// it use top-level await. Such roots cannot sit behind the shared registry's lazy
/// `require` edge, so discover them together but render each selected root separately.
fn compile_selected_entries(config: &BuildConfig, entries: &[NativeEntry]) -> Result<(), String> {
    let Some((first, rest)) = entries.split_first() else {
        return Ok(());
    };
    let started = std::time::Instant::now();
    let (mut bundler, update) = crate::compiler::discover_native_next(&first.generated, config)?;
    for warning in diffpack_core::partition_diagnostics(
        &update.diagnostics,
        "native Next independently-emitted entries",
    )? {
        eprintln!("warning: {warning}");
    }
    if !rest.is_empty() {
        let roots = rest
            .iter()
            .map(|entry| entry.generated.clone())
            .collect::<Vec<_>>();
        let update = bundler.discover_additional_entries(&roots)?;
        for warning in diffpack_core::partition_diagnostics(
            &update.diagnostics,
            "native Next independently-emitted entries",
        )? {
            eprintln!("warning: {warning}");
        }
    }
    for entry in entries {
        bundler.select_entry(&entry.generated)?;
        let reachable = bundler.reachable_modules_direct();
        if let Some(parent) = entry.output.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        bundler.emit_with_options(
            &reachable,
            &entry.output,
            EmitOptions {
                minify: false,
                source_map: false,
                format: ModuleFormat::Cjs,
                ..EmitOptions::default()
            },
        )?;
    }
    eprintln!(
        "native Next: discovered and independently emitted {} async entries in {:.2?}",
        entries.len(),
        started.elapsed()
    );
    Ok(())
}

/// Compile every discovered App Router entry through Next's official route-entry
/// runtime and place the CommonJS artifacts where `next build` expects them.
pub fn compile_app_entries(
    project_root: &Path,
    dist_dir: &Path,
    next_config_output: Option<&str>,
) -> Result<(), String> {
    compile_app_entries_inner(project_root, dist_dir, next_config_output, false)
}

/// Development counterpart to [`compile_app_entries`]. It emits the same native
/// endpoint ABI while selecting Next's development source/runtime policies.
pub fn compile_app_entries_development(
    project_root: &Path,
    dist_dir: &Path,
    next_config_output: Option<&str>,
) -> Result<(), String> {
    compile_app_entries_inner(project_root, dist_dir, next_config_output, true)
}

fn compile_app_entries_inner(
    project_root: &Path,
    dist_dir: &Path,
    next_config_output: Option<&str>,
    development: bool,
) -> Result<(), String> {
    let native_ssr_bundle = compile_native_ssr_modules(project_root, dist_dir, development)?;
    let configured = if development {
        crate::next_adapter::configure_app_router_dev(
            project_root,
            "react-server",
            &crate::next_adapter::RouteScope::All,
        )?
    } else {
        crate::next_adapter::configure_app_router(project_root, "react-server")?
    };
    let mut config = configured
        .ok_or_else(|| format!("{} is not an App Router project", project_root.display()))?;
    // Native Next workers already own these singleton stores. Route entries
    // must require those exact files while still bundling the renderer and the
    // React Server Writer under Diffpack's `react-server` conditions.
    let next_root = crate::rsc_runtime_resolve::installed_package_root(project_root, "next")?;
    use_native_next_server_apis(&mut config.build, &next_root);
    // `configure_app_router` is also used by Diffpack's standalone adapter and
    // therefore aliases React directly to Next's compiled packages. A native Next
    // route has a stronger contract: use the app-page runtime's vendored facades so
    // the renderer and Flight implementation have one shared module identity.
    config.build.aliases.retain(|(specifier, _)| {
        !specifier.starts_with("react") && !specifier.starts_with("next/dist/compiled/react")
    });
    config
        .build
        .aliases
        .extend(crate::rsc_runtime_resolve::native_next_rsc_aliases(
            &next_root,
        )?);
    config
        .build
        .aliases
        .extend(crate::rsc_runtime_resolve::native_next_context_aliases(
            &next_root,
        )?);
    let mut external_singletons = Vec::new();
    for relative in [
        // The vendored React facades below read the runtime-owned `vendored` table
        // from this module. Bundling it would make its internal React imports pass
        // through the facade aliases again, creating a circular, half-initialized
        // table. Next externalizes this same app-page runtime boundary.
        "dist/server/app-render/action-async-storage.external.js",
        "dist/server/app-render/after-task-async-storage.external.js",
        "dist/server/app-render/console-async-storage.external.js",
        "dist/server/app-render/dynamic-access-async-storage.external.js",
        "dist/server/app-render/work-async-storage.external.js",
        "dist/server/app-render/work-unit-async-storage.external.js",
        "dist/server/app-render/module-loading/track-module-loading.external.js",
    ] {
        let path = next_root.join(relative);
        external_singletons.push(path.canonicalize().map_err(|error| {
            format!(
                "cannot resolve native Next singleton {}: {error}",
                path.display()
            )
        })?);
    }
    let external_named_singletons = [
        (
            "dist/server/route-modules/app-page/vendored/rsc/react-server-dom-webpack-server.js",
            vec![
                "createClientModuleProxy",
                "createTemporaryReferenceSet",
                "decodeAction",
                "decodeFormState",
                "decodeReply",
                "decodeReplyFromAsyncIterable",
                "decodeReplyFromBusboy",
                "registerClientReference",
                "registerServerReference",
                "renderToPipeableStream",
                "renderToReadableStream",
            ],
        ),
        (
            "dist/server/route-modules/app-page/vendored/rsc/react-server-dom-webpack-static.js",
            vec!["prerender", "prerenderToNodeStream"],
        ),
    ]
    .into_iter()
    .map(|(relative, names)| {
        (
            next_root.join(relative),
            names.into_iter().map(String::from).collect(),
        )
    })
    .collect();
    let webpack_runtime_singletons = vec![
        next_root
            .join("dist/server/route-modules/app-page/module.compiled.js")
            .canonicalize()
            .map_err(|error| format!("cannot resolve native Next app-page runtime: {error}"))?,
        next_root
            .join("dist/server/route-modules/app-route/module.compiled.js")
            .canonicalize()
            .map_err(|error| format!("cannot resolve native Next app-route runtime: {error}"))?,
    ];
    let mut defines = config.build.source_policy.defines().to_vec();
    defines.push((
        "process.env.__NEXT_USE_NODE_STREAMS".to_string(),
        "true".to_string(),
    ));
    // Next's outer build worker currently reaches Diffpack through its Turbopack
    // orchestration branch and therefore has `TURBOPACK` in the process environment.
    // The emitted artifact implements Next's webpack RSC reference protocol, so its
    // internal app-page dispatcher must select the matching webpack runtime.
    defines.push(("process.env.TURBOPACK".to_string(), "false".to_string()));
    config.build.source_policy = std::sync::Arc::new(crate::source_policy::NextSourcePolicy {
        defines,
        external_singletons,
        external_named_singletons,
        webpack_runtime_singletons,
    });
    let standalone_root = project_root.join(".diffpack-output");
    crate::profile::prepare_build(project_root, &standalone_root, &mut config)?;

    let mut entries = Vec::new();
    for route in crate::artifacts::discover_app_routes(project_root)? {
        if matches!(
            route.kind,
            crate::artifacts::NextRouteArtifactKind::PagesApi
                | crate::artifacts::NextRouteArtifactKind::PagesPage
        ) {
            continue;
        }
        let entry_name = route.original_name.trim_start_matches('/');
        let generated = project_root
            .join(crate::APP_ADAPTER_DIR)
            .join("native")
            .join(format!("{}.tsx", entry_name.replace('/', "__")));
        if let Some(parent) = generated.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        let source = if route.kind == crate::artifacts::NextRouteArtifactKind::AppRoute {
            let mut effective_route = route.clone();
            if let Some(userland) = static_metadata_userland(&route, generated.parent().unwrap())? {
                effective_route.source_path = userland;
            }
            crate::artifacts::native_app_route_entry_source(
                project_root,
                &effective_route,
                next_config_output,
            )?
        } else {
            crate::artifacts::native_app_page_entry_source(
                project_root,
                &route,
                &native_ssr_bundle,
            )?
        };
        std::fs::write(&generated, source)
            .map_err(|error| format!("cannot write {}: {error}", generated.display()))?;
        let output = dist_dir.join("server/app").join(format!("{entry_name}.js"));
        entries.push(NativeEntry { generated, output });
    }
    compile_shared_entries(
        &config.build,
        &entries,
        &dist_dir.join("server/diffpack-app-entries.js"),
    )?;
    compile_pages_api_entries(project_root, dist_dir, development)?;
    Ok(())
}

/// Pages server entries and Next's compiled Pages runtime must share the exact
/// application React objects. Bundling React into each entry creates a second hook
/// dispatcher; externalizing the resolved package entrypoints preserves Node's module
/// identity while still letting Diffpack own every application module around them.
fn native_pages_react_aliases(
    next_root: &Path,
) -> Result<(Vec<(String, String)>, Vec<std::path::PathBuf>), String> {
    let react_root = crate::rsc_runtime_resolve::installed_package_root(next_root, "react")?;
    let react_dom_root =
        crate::rsc_runtime_resolve::installed_package_root(next_root, "react-dom")?;
    let entries = [
        ("react/compiler-runtime", &react_root, "compiler-runtime.js"),
        ("react/jsx-dev-runtime", &react_root, "jsx-dev-runtime.js"),
        ("react/jsx-runtime", &react_root, "jsx-runtime.js"),
        ("react", &react_root, "index.js"),
        (
            "react-dom/server.browser",
            &react_dom_root,
            "server.browser.js",
        ),
        ("react-dom/server.edge", &react_dom_root, "server.edge.js"),
        ("react-dom/server", &react_dom_root, "server.node.js"),
        ("react-dom/static", &react_dom_root, "static.node.js"),
        ("react-dom/client", &react_dom_root, "client.js"),
        ("react-dom", &react_dom_root, "index.js"),
    ];
    let mut aliases = Vec::new();
    let mut external = Vec::new();
    for (specifier, package_root, relative) in entries {
        let mut path = package_root.join(relative);
        if specifier == "react-dom/server.edge" && !path.is_file() {
            // React 18 does not publish `server.edge`, but Next's compiled Pages
            // runtime names it. This is the same compatibility alias used by
            // createReactAliases in Next's webpack configuration.
            path = next_root.join("dist/build/webpack/alias/react-dom-server.js");
        }
        if !path.is_file() {
            continue;
        }
        let path = path.canonicalize().map_err(|error| {
            format!(
                "cannot resolve Pages React entry {}: {error}",
                path.display()
            )
        })?;
        aliases.push((specifier.to_string(), path.to_string_lossy().into_owned()));
        if !external.contains(&path) {
            external.push(path);
        }
    }
    if !aliases.iter().any(|(specifier, _)| specifier == "react")
        || !aliases
            .iter()
            .any(|(specifier, _)| specifier == "react-dom")
    {
        return Err(
            "native Next Pages entries require the application's react and react-dom packages"
                .to_string(),
        );
    }
    aliases.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()).then(left.cmp(right)));
    Ok((aliases, external))
}

fn compile_pages_api_entries(
    project_root: &Path,
    dist_dir: &Path,
    development: bool,
) -> Result<(), String> {
    let routes = crate::artifacts::discover_app_routes(project_root)?;
    if !routes.iter().any(|route| {
        matches!(
            route.kind,
            crate::artifacts::NextRouteArtifactKind::PagesApi
                | crate::artifacts::NextRouteArtifactKind::PagesPage
        )
    }) {
        return Ok(());
    }
    let configured = if development {
        crate::next_adapter::configure_app_router_dev(
            project_root,
            "ssr",
            &crate::next_adapter::RouteScope::All,
        )?
    } else {
        crate::next_adapter::configure_app_router(project_root, "ssr")?
    };
    let mut config = configured
        .ok_or_else(|| format!("{} is not an App Router project", project_root.display()))?;
    let next_root = crate::rsc_runtime_resolve::installed_package_root(project_root, "next")?;
    use_native_next_server_apis(&mut config.build, &next_root);
    config.build.aliases.retain(|(specifier, _)| {
        !specifier.starts_with("react") && !specifier.starts_with("next/dist/compiled/react")
    });
    // Pages Router is classic React SSR. Unlike App Router's RSC/SSR layers it
    // intentionally compiles against the application's matching React/ReactDOM pair.
    // Leaving these specifiers unaliased reproduces Next's Pages Node layer and avoids
    // mixing the App Page runtime's vendored canary with the app's ReactDOM.
    let (pages_react_aliases, external_react_modules) = native_pages_react_aliases(&next_root)?;
    config.build.aliases.extend(pages_react_aliases);
    config
        .build
        .aliases
        .extend(crate::rsc_runtime_resolve::native_pages_context_aliases(
            &next_root,
        )?);
    let mut defines = config.build.source_policy.defines().to_vec();
    defines.push(("process.env.TURBOPACK".to_string(), "false".to_string()));
    config.build.source_policy = std::sync::Arc::new(crate::source_policy::NextSourcePolicy {
        defines,
        external_singletons: external_react_modules,
        webpack_runtime_singletons: vec![
            next_root
                .join("dist/server/route-modules/pages-api/module.compiled.js")
                .canonicalize()
                .map_err(|error| {
                    format!("cannot resolve native Next Pages API runtime: {error}")
                })?,
            next_root
                .join("dist/server/route-modules/pages/module.compiled.js")
                .canonicalize()
                .map_err(|error| format!("cannot resolve native Next Pages runtime: {error}"))?,
        ],
        ..Default::default()
    });
    crate::profile::prepare_build(
        project_root,
        &project_root.join(".diffpack-output"),
        &mut config,
    )?;

    // Pages hydration is one browser graph containing `_app`, the error page,
    // and every routable page. It is intentionally compiled separately from the
    // App Router Flight client because the two runtimes have different ABIs.
    if routes
        .iter()
        .any(|route| route.kind == crate::artifacts::NextRouteArtifactKind::PagesPage)
    {
        let mut client = crate::next_pages::configure(project_root, "client", development)?
            .ok_or_else(|| {
                format!(
                    "{} has Pages routes but no Pages client profile",
                    project_root.display()
                )
            })?;
        crate::profile::prepare_build(
            project_root,
            &project_root.join(".diffpack-pages-output"),
            &mut client,
        )?;
        let entry = client
            .entry
            .as_ref()
            .ok_or_else(|| "native Next Pages client profile has no entry".to_string())?;
        let (bundler, update) = crate::compiler::discover(entry, &client.build)?;
        for warning in
            diffpack_core::partition_diagnostics(&update.diagnostics, "native Next Pages client")?
        {
            eprintln!("warning: {warning}");
        }
        let output = dist_dir.join("static/chunks/diffpack-pages.js");
        let module_output = dist_dir.join("static/chunks/diffpack-pages.module.js");
        std::fs::create_dir_all(output.parent().expect("chunk output has a parent"))
            .map_err(|error| format!("cannot create Pages client output: {error}"))?;
        bundler.emit_with_options(
            &bundler.reachable_modules_direct(),
            &module_output,
            EmitOptions {
                minify: false,
                source_map: false,
                format: ModuleFormat::BrowserEsm,
                ..EmitOptions::default()
            },
        )?;
        // Next's stock `_document` emits Pages chunks as classic deferred scripts.
        // Diffpack's browser renderer intentionally emits ESM, so adapt only the
        // entry publication statement; the graph executes eagerly in either goal.
        // Refuse an unfamiliar shape instead of shipping syntax that the browser
        // would reject or silently dropping an export contract.
        let module_source = std::fs::read_to_string(&module_output)
            .map_err(|error| format!("cannot read {}: {error}", module_output.display()))?;
        let classic_source = module_source
            .strip_suffix("export default __diffpackEntry;\n")
            .ok_or_else(|| {
                format!(
                    "{} is not a Diffpack browser entry; cannot adapt it to Next's classic script ABI",
                    module_output.display()
                )
            })?
            .replace(
                "const __chunkQuery=(()=>{const __q=import.meta.url.indexOf(\"?\");return __q<0?\"\":import.meta.url.slice(__q);})();",
                "const __chunkQuery=\"\";",
            );
        if classic_source.contains("import.meta")
            || classic_source.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with("import ") || line.starts_with("export ")
            })
        {
            return Err(format!(
                "{} still contains ESM-only syntax after classic-script adaptation",
                module_output.display()
            ));
        }
        std::fs::write(&output, classic_source)
            .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
        std::fs::remove_file(&module_output)
            .map_err(|error| format!("cannot remove {}: {error}", module_output.display()))?;
        eprintln!("native Next: emitted {}", output.display());
    }
    // Next deliberately leaves Critters external from its compiled Pages runtime.
    // It is loaded only when experimental.optimizeCss is enabled; ordinary builds
    // must not require applications to install this Next build-time dependency.
    if !config
        .build
        .server_external_packages
        .iter()
        .any(|package| package == "critters")
    {
        config
            .build
            .server_external_packages
            .push("critters".to_string());
    }

    let builtin_dir = next_root.join("dist/pages");
    let pages_dir = if project_root.join("pages").is_dir() {
        project_root.join("pages")
    } else {
        project_root.join("src/pages")
    };
    let convention = |stem: &str| {
        ["tsx", "ts", "jsx", "js"]
            .into_iter()
            .map(|extension| pages_dir.join(format!("{stem}.{extension}")))
            .find(|path| path.is_file())
    };
    let builtin_app = convention("_app").unwrap_or_else(|| builtin_dir.join("_app.js"));
    let builtin_document =
        convention("_document").unwrap_or_else(|| builtin_dir.join("_document.js"));
    let builtin_error = convention("_error").unwrap_or_else(|| builtin_dir.join("_error.js"));
    let mut entries = Vec::new();
    let mut independently_emitted = Vec::new();
    for (page, userland) in [
        ("/_app", &builtin_app),
        ("/_document", &builtin_document),
        ("/_error", &builtin_error),
    ] {
        let generated = project_root
            .join(crate::APP_ADAPTER_DIR)
            .join("native/pages")
            .join(format!("{}.tsx", page.trim_start_matches('/')));
        if let Some(parent) = generated.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            &generated,
            crate::artifacts::native_pages_entry_source(
                project_root,
                page,
                page,
                userland,
                &builtin_app,
                &builtin_document,
            )?,
        )
        .map_err(|error| format!("cannot write {}: {error}", generated.display()))?;
        let output = dist_dir
            .join("server/pages")
            .join(format!("{}.js", page.trim_start_matches('/')));
        entries.push(NativeEntry { generated, output });
    }

    for route in routes {
        if route.kind == crate::artifacts::NextRouteArtifactKind::PagesPage {
            let generated = project_root
                .join(crate::APP_ADAPTER_DIR)
                .join("native/pages")
                .join(format!(
                    "{}.tsx",
                    route
                        .original_name
                        .trim_start_matches('/')
                        .replace('/', "__")
                ));
            if let Some(parent) = generated.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
            }
            std::fs::write(
                &generated,
                crate::artifacts::native_pages_entry_source(
                    project_root,
                    &route.original_name,
                    &route.pathname,
                    &route.source_path,
                    &builtin_app,
                    &builtin_document,
                )?,
            )
            .map_err(|error| format!("cannot write {}: {error}", generated.display()))?;
            let entry_name = route.original_name.trim_start_matches('/');
            let output = dist_dir
                .join("server/pages")
                .join(format!("{entry_name}.js"));
            independently_emitted.push(NativeEntry { generated, output });
            continue;
        }
        if route.kind != crate::artifacts::NextRouteArtifactKind::PagesApi {
            continue;
        }
        let entry_name = route.original_name.trim_start_matches('/');
        let generated = project_root
            .join(crate::APP_ADAPTER_DIR)
            .join("native/pages-api")
            .join(format!("{}.tsx", entry_name.replace('/', "__")));
        if let Some(parent) = generated.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            &generated,
            crate::artifacts::native_pages_api_entry_source(project_root, &route)?,
        )
        .map_err(|error| format!("cannot write {}: {error}", generated.display()))?;
        let output = dist_dir
            .join("server/pages")
            .join(format!("{entry_name}.js"));
        entries.push(NativeEntry { generated, output });
    }
    compile_shared_entries(
        &config.build,
        &entries,
        &dist_dir.join("server/diffpack-pages-entries.js"),
    )?;
    compile_selected_entries(&config.build, &independently_emitted)
}

fn compile_native_ssr_modules(
    project_root: &Path,
    dist_dir: &Path,
    development: bool,
) -> Result<std::path::PathBuf, String> {
    let manifest_path = project_root
        .join(".diffpack-output")
        .join(crate::rsc::SERVER_REFERENCES_MANIFEST_FILE);
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&manifest_path)
            .map_err(|error| format!("cannot read {}: {error}", manifest_path.display()))?,
    )
    .map_err(|error| format!("cannot parse {}: {error}", manifest_path.display()))?;
    let entries = manifest
        .as_object()
        .ok_or_else(|| format!("{} is not an object", manifest_path.display()))?;
    let generated = project_root
        .join(crate::APP_ADAPTER_DIR)
        .join("native/ssr-modules.ts");
    if let Some(parent) = generated.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut source = String::new();
    let mut table = Vec::new();
    for (index, (resource, reference)) in entries.iter().enumerate() {
        let name = format!("M{index}");
        source.push_str(&format!("import * as {name} from {resource:?};\n"));
        let id = reference
            .get("id")
            .ok_or_else(|| format!("server reference {resource} has no id"))?;
        table.push(format!("{:?}:{name}", id.to_string().trim_matches('"')));
    }
    source.push_str(&format!("export default {{{}}};\n", table.join(",")));
    std::fs::write(&generated, source)
        .map_err(|error| format!("cannot write {}: {error}", generated.display()))?;

    let configured = if development {
        crate::next_adapter::configure_app_router_dev(
            project_root,
            "ssr",
            &crate::next_adapter::RouteScope::All,
        )?
    } else {
        crate::next_adapter::configure_app_router(project_root, "ssr")?
    };
    let mut config = configured
        .ok_or_else(|| format!("{} is not an App Router project", project_root.display()))?;
    let next_root = crate::rsc_runtime_resolve::installed_package_root(project_root, "next")?;
    use_native_next_server_apis(&mut config.build, &next_root);
    config.build.aliases.retain(|(specifier, _)| {
        !specifier.starts_with("react") && !specifier.starts_with("next/dist/compiled/react")
    });
    config
        .build
        .aliases
        .extend(crate::rsc_runtime_resolve::native_next_ssr_aliases(
            &next_root,
        )?);
    config
        .build
        .aliases
        .extend(crate::rsc_runtime_resolve::native_next_context_aliases(
            &next_root,
        )?);
    config.build.source_policy = std::sync::Arc::new(crate::source_policy::NextSourcePolicy {
        defines: config.build.source_policy.defines().to_vec(),
        webpack_runtime_singletons: vec![
            next_root
                .join("dist/server/route-modules/app-page/module.compiled.js")
                .canonicalize()
                .map_err(|error| format!("cannot resolve native Next app-page runtime: {error}"))?,
        ],
        ..Default::default()
    });
    let standalone_root = project_root.join(".diffpack-output");
    crate::profile::prepare_build(project_root, &standalone_root, &mut config)?;
    let (bundler, update) = crate::compiler::discover_native_next(&generated, &config.build)?;
    for warning in
        diffpack_core::partition_diagnostics(&update.diagnostics, "native Next SSR module table")?
    {
        eprintln!("warning: {warning}");
    }
    let reachable = bundler.reachable_modules_direct();
    let output = dist_dir.join("server/diffpack-ssr.js");
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    bundler.emit_with_options(
        &reachable,
        &output,
        EmitOptions {
            minify: false,
            source_map: false,
            format: ModuleFormat::Cjs,
            ..EmitOptions::default()
        },
    )?;
    Ok(output)
}
