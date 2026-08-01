//! Typed Next-owned artifacts shared by standalone and native-Next adapters.

use std::path::{Path, PathBuf};

use serde_json::json;

/// Route kind at the stable Diffpack/Next binding boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextRouteArtifactKind {
    AppPage,
    ImplicitAppPage,
    AppRoute,
    PagesPage,
    PagesApi,
}

/// One discovered App Router entrypoint before any transport-specific encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextRouteArtifact {
    pub pathname: String,
    pub original_name: String,
    pub source_path: PathBuf,
    pub kind: NextRouteArtifactKind,
}

/// Materializes the per-entry artifacts consumed by Next's manifest loader.
///
/// This belongs here rather than in the binding transport: paths and schemas
/// are Next semantics and are shared by every native-Next frontend.
pub struct NativeNextOutput<'a> {
    pub dist_dir: &'a Path,
    pub standalone_root: &'a Path,
}

/// Expands Next's official App Route entry template using the same values as
/// `next-app-loader`. Keeping this as a mechanical template adapter means HTTP
/// methods, request preparation, static generation, and future Next handler
/// behavior continue to come from the installed Next version.
pub fn native_app_route_entry_source(
    project_root: &Path,
    route: &NextRouteArtifact,
    next_config_output: Option<&str>,
) -> Result<String, String> {
    if route.kind != NextRouteArtifactKind::AppRoute {
        return Err(format!("{} is not an App Route", route.original_name));
    }
    let next_root = crate::rsc_runtime_resolve::installed_package_root(project_root, "next")?;
    let template_path = next_root.join("dist/esm/build/templates/app-route.js");
    let mut source = std::fs::read_to_string(&template_path)
        .map_err(|error| format!("cannot read {}: {error}", template_path.display()))?;
    // `loadEntrypoint` normally asks Next's SWC binding to resolve template-local
    // imports. Our adapter is intentionally independent of that private transform,
    // so anchor the same imports to the installed ESM distribution explicitly.
    source = anchor_next_template_imports(source, &next_root);
    let entry = route.original_name.trim_start_matches('/');
    let filename = route
        .source_path
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no UTF-8 filename", route.source_path.display()))?;
    let replacements = [
        (
            "VAR_USERLAND",
            route.source_path.to_string_lossy().into_owned(),
        ),
        ("VAR_DEFINITION_PAGE", route.original_name.clone()),
        ("VAR_DEFINITION_PATHNAME", route.pathname.clone()),
        ("VAR_DEFINITION_FILENAME", filename.to_string()),
        ("VAR_DEFINITION_BUNDLE_PATH", format!("app/{entry}")),
        (
            "VAR_RESOLVED_PAGE_PATH",
            route.source_path.to_string_lossy().into_owned(),
        ),
    ];
    for (placeholder, value) in replacements {
        source = source.replace(placeholder, &value);
    }
    let output = next_config_output
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| format!("cannot encode nextConfigOutput: {error}"))?
        .unwrap_or_else(|| "undefined".to_string());
    source = source.replace(
        "// INJECT:nextConfigOutput",
        &format!("const nextConfigOutput = {output};"),
    );
    Ok(source)
}

/// Expands Next's official Pages API entry template. Pages API handlers are a
/// separate Node request/response ABI and must never be adapted as App Routes.
pub fn native_pages_api_entry_source(
    project_root: &Path,
    route: &NextRouteArtifact,
) -> Result<String, String> {
    if route.kind != NextRouteArtifactKind::PagesApi {
        return Err(format!("{} is not a Pages API route", route.original_name));
    }
    let next_root = crate::rsc_runtime_resolve::installed_package_root(project_root, "next")?;
    let template_path = next_root.join("dist/esm/build/templates/pages-api.js");
    let mut source = std::fs::read_to_string(&template_path)
        .map_err(|error| format!("cannot read {}: {error}", template_path.display()))?;
    source = anchor_next_template_imports(source, &next_root);
    for (placeholder, value) in [
        (
            "VAR_USERLAND",
            route.source_path.to_string_lossy().into_owned(),
        ),
        ("VAR_DEFINITION_PAGE", route.original_name.clone()),
        ("VAR_DEFINITION_PATHNAME", route.pathname.clone()),
    ] {
        source = source.replace(placeholder, &value);
    }
    Ok(source)
}

/// Expands Next's official Pages entry template for both user pages and the
/// `_app`/`_document`/`_error` global entries required by hybrid projects.
pub fn native_pages_entry_source(
    project_root: &Path,
    page: &str,
    pathname: &str,
    userland: &Path,
    app: &Path,
    document: &Path,
) -> Result<String, String> {
    let next_root = crate::rsc_runtime_resolve::installed_package_root(project_root, "next")?;
    let template_path = next_root.join("dist/esm/build/templates/pages.js");
    let mut source = std::fs::read_to_string(&template_path)
        .map_err(|error| format!("cannot read {}: {error}", template_path.display()))?;
    source = anchor_next_template_imports(source, &next_root);
    for (placeholder, value) in [
        ("VAR_USERLAND", userland.to_string_lossy().into_owned()),
        ("VAR_MODULE_APP", app.to_string_lossy().into_owned()),
        (
            "VAR_MODULE_DOCUMENT",
            document.to_string_lossy().into_owned(),
        ),
        ("VAR_DEFINITION_PAGE", page.to_string()),
        ("VAR_DEFINITION_PATHNAME", pathname.to_string()),
    ] {
        source = source.replace(placeholder, &value);
    }
    Ok(source)
}

fn anchor_next_template_imports(source: String, next_root: &Path) -> String {
    source
        .replace(
            "from '../../",
            &format!("from '{}/", next_root.join("dist/esm").to_string_lossy()),
        )
        .replace(
            "from './",
            &format!(
                "from '{}/",
                next_root.join("dist/esm/build/templates").to_string_lossy()
            ),
        )
}

/// Generates the route-specific source that Diffpack compiles for Next's
/// native server output. The renderer and route-module implementation are
/// Next's official entry runtime; Diffpack supplies its loader tree and module
/// runtime exactly where Turbopack normally injects them.
pub fn native_app_page_entry_source(
    project_root: &Path,
    route: &NextRouteArtifact,
    ssr_modules_path: &Path,
) -> Result<String, String> {
    if route.kind == NextRouteArtifactKind::AppRoute {
        return Err(format!("{} is not an App Page", route.original_name));
    }
    let app_dir = if project_root.join("app").is_dir() {
        project_root.join("app")
    } else {
        project_root.join("src/app")
    }
    .canonicalize()
    .map_err(|error| format!("cannot open the app directory: {error}"))?;
    if route.kind == NextRouteArtifactKind::ImplicitAppPage {
        let (segment, builtin) = if route.pathname == "/_not-found" {
            (
                "_not-found",
                "next/dist/client/components/builtin/not-found.js",
            )
        } else {
            (
                "_global-error",
                "next/dist/client/components/builtin/app-error.js",
            )
        };
        let tree = format!(
            "['',{children:1},{{'global-error':[()=>Promise.resolve(GlobalError),{global_error:?}]}},[]]",
            children = format!(
                "{{children:[{segment:?},{{children:['__PAGE__',{{}},{{page:[()=>Promise.resolve(Builtin),{builtin:?}]}},null]}},{{}},null]}}"
            ),
            global_error = "next/dist/client/components/builtin/global-error.js",
        );
        return native_entry_module(
            project_root,
            &format!(
                "import * as Builtin from {builtin:?};\nimport * as GlobalError from 'next/dist/client/components/builtin/global-error.js';"
            ),
            &tree,
            route,
            ssr_modules_path,
        );
    }
    let page_parent = route
        .source_path
        .parent()
        .ok_or_else(|| format!("{} has no parent", route.source_path.display()))?;
    let relative_parent = page_parent.strip_prefix(&app_dir).map_err(|_| {
        format!(
            "route source {} is not below {}",
            route.source_path.display(),
            app_dir.display()
        )
    })?;

    let mut imports = Vec::new();
    let global_error = find_convention(&app_dir, "global-error");
    let global_error_specifier = global_error
        .as_ref()
        .map(|path| format!("{path:?}"))
        .unwrap_or_else(|| "'next/dist/client/components/builtin/global-error.js'".to_string());
    imports.push(format!(
        "import * as GlobalError from {global_error_specifier};"
    ));
    // Next's app-page loader always supplies these root HTTP fallbacks, even
    // when the application has not defined matching convention files. The
    // renderer can select them while prerendering an otherwise ordinary page;
    // omitting them leaves React trying to render an undefined component.
    let mut root_fallbacks = Vec::new();
    for (convention, binding, builtin) in [
        (
            "not-found",
            "NotFound",
            "next/dist/client/components/builtin/not-found.js",
        ),
        (
            "forbidden",
            "Forbidden",
            "next/dist/client/components/builtin/forbidden.js",
        ),
        (
            "unauthorized",
            "Unauthorized",
            "next/dist/client/components/builtin/unauthorized.js",
        ),
    ] {
        let specifier = find_convention(&app_dir, convention)
            .map(|path| format!("{path:?}"))
            .unwrap_or_else(|| format!("{builtin:?}"));
        imports.push(format!("import * as {binding} from {specifier};"));
        root_fallbacks.push(format!(
            "{convention:?}:[()=>Promise.resolve({binding}),{specifier}]"
        ));
    }
    let mut module_index = 0usize;
    let mut layout_modules = Vec::new();
    let mut cursor = app_dir.clone();
    for depth in 0..=relative_parent.components().count() {
        if depth > 0 {
            cursor.push(relative_parent.components().nth(depth - 1).unwrap());
        }
        if let Some(layout) = find_convention(&cursor, "layout") {
            let name = format!("M{module_index}");
            module_index += 1;
            imports.push(format!("import * as {name} from {:?};", layout));
            layout_modules.push((depth, name, layout));
        }
    }
    let page_name = format!("M{module_index}");
    imports.push(format!(
        "import * as {page_name} from {:?};",
        route.source_path
    ));

    let page_module = format!(
        "page:[()=>Promise.resolve({page_name}),{:?}]",
        route.source_path
    );
    let mut tree = format!("['__PAGE__',{{}},{{{page_module}}},[]]");
    let segments: Vec<String> = relative_parent
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    for depth in (0..=segments.len()).rev() {
        let segment = if depth == 0 { "" } else { &segments[depth - 1] };
        let mut modules = layout_modules
            .iter()
            .find(|(layout_depth, _, _)| *layout_depth == depth)
            .map(|(_, name, path)| format!("layout:[()=>Promise.resolve({name}),{path:?}]"))
            .unwrap_or_default();
        if depth == 0 {
            if !modules.is_empty() {
                modules.push(',');
            }
            modules.push_str(&format!(
                "'global-error':[()=>Promise.resolve(GlobalError),{global_error_specifier}]"
            ));
            modules.push(',');
            modules.push_str(&root_fallbacks.join(","));
        }
        tree = format!("[{segment:?},{{children:{tree}}},{{{modules}}},[]]");
    }
    native_entry_module(
        project_root,
        &imports.join("\n"),
        &tree,
        route,
        ssr_modules_path,
    )
}

fn native_entry_module(
    project_root: &Path,
    imports: &str,
    tree: &str,
    route: &NextRouteArtifact,
    ssr_modules_path: &Path,
) -> Result<String, String> {
    let references_path = project_root
        .join(".diffpack-output")
        .join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
    let references: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&references_path)
            .map_err(|error| format!("cannot read {}: {error}", references_path.display()))?,
    )
    .map_err(|error| format!("cannot parse {}: {error}", references_path.display()))?;
    let mut reference_imports = String::new();
    let mut reference_table = Vec::new();
    for (index, (resource, reference)) in references
        .as_object()
        .ok_or_else(|| format!("{} is not an object", references_path.display()))?
        .iter()
        .enumerate()
    {
        let name = format!("R{index}");
        reference_imports.push_str(&format!("import * as {name} from {resource:?};\n"));
        let id = reference
            .get("id")
            .ok_or_else(|| format!("client reference {resource} has no id"))?;
        reference_table.push(format!("{:?}:{name}", format!("rsc:{id}")));
    }
    let mut source = include_str!("next_runtime/native-app-page.js").to_string();
    let replacements = [
        ("/*__DIFFPACK_ROUTE_IMPORTS__*/", imports.to_string()),
        ("/*__DIFFPACK_REFERENCE_IMPORTS__*/", reference_imports),
        ("/*__DIFFPACK_LOADER_TREE__*/", tree.to_string()),
        (
            "/*__DIFFPACK_REFERENCE_TABLE__*/",
            reference_table.join(",\n  "),
        ),
        (
            "/*__DIFFPACK_SSR_PATH__*/",
            serde_json::to_string(&ssr_modules_path.to_string_lossy())
                .map_err(|error| format!("cannot encode SSR module path: {error}"))?,
        ),
        (
            "/*__DIFFPACK_PAGE__*/",
            serde_json::to_string(&route.original_name)
                .map_err(|error| format!("cannot encode route page: {error}"))?,
        ),
        (
            "/*__DIFFPACK_PATHNAME__*/",
            serde_json::to_string(&route.pathname)
                .map_err(|error| format!("cannot encode route pathname: {error}"))?,
        ),
    ];
    for (placeholder, value) in replacements {
        let count = source.matches(placeholder).count();
        if count != 1 {
            return Err(format!(
                "native App Page template must contain {placeholder} exactly once; found {count}"
            ));
        }
        source = source.replacen(placeholder, &value, 1);
    }
    Ok(source)
}

fn find_convention(directory: &Path, stem: &str) -> Option<PathBuf> {
    ["tsx", "ts", "jsx", "js"]
        .into_iter()
        .map(|extension| directory.join(format!("{stem}.{extension}")))
        .find(|path| path.is_file())
}

impl NativeNextOutput<'_> {
    pub fn write_route_manifests(&self, routes: &[NextRouteArtifact]) -> Result<(), String> {
        let client_files = self.write_client_assets()?;
        let pages_manifest: serde_json::Map<String, serde_json::Value> = routes
            .iter()
            .filter(|route| {
                matches!(
                    route.kind,
                    NextRouteArtifactKind::PagesPage | NextRouteArtifactKind::PagesApi
                )
            })
            .map(|route| {
                (
                    route.pathname.clone(),
                    json!(format!(
                        "pages/{}.js",
                        route.original_name.trim_start_matches('/')
                    )),
                )
            })
            .collect();
        if !pages_manifest.is_empty() {
            let server = self.dist_dir.join("server");
            std::fs::create_dir_all(&server)
                .map_err(|error| format!("cannot create {}: {error}", server.display()))?;
            let path = server.join("pages-manifest.json");
            std::fs::write(
                &path,
                serde_json::to_vec(&pages_manifest)
                    .map_err(|error| format!("cannot serialize {}: {error}", path.display()))?,
            )
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
            self.write_pages_global_partial_manifests(routes)?;
        }
        for route in routes {
            if matches!(
                route.kind,
                NextRouteArtifactKind::PagesPage | NextRouteArtifactKind::PagesApi
            ) {
                let entry = route.original_name.trim_start_matches('/');
                // Turbopack's manifest loader indexes partial manifests by the public
                // page pathname (`/` -> `index`), not by the source-relative entry
                // name. Those differ for rewired Pages modules such as Cal.com's
                // `pages/router/embed.tsx`, which serves `/embed`.
                let manifest_entry = route.pathname.trim_start_matches('/');
                let manifest_entry = if manifest_entry.is_empty() {
                    "index"
                } else {
                    manifest_entry
                };
                let directory = self.dist_dir.join("server/pages").join(manifest_entry);
                std::fs::create_dir_all(&directory)
                    .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
                let mut manifests = vec![(
                    "pages-manifest.json",
                    json!({ route.pathname.clone(): format!("pages/{entry}.js") }),
                )];
                if route.kind == NextRouteArtifactKind::PagesPage {
                    manifests.extend([
                        ("client-build-manifest.json", json!({})),
                        (
                            "build-manifest.json",
                            json!({
                                "devFiles": [], "ampDevFiles": [], "polyfillFiles": [],
                                "lowPriorityFiles": [], "rootMainFiles": client_files,
                                "pages": { route.pathname.clone(): [] },
                                "ampFirstPages": [], "rootMainFilesTree": {},
                                "pagesChunkGroupBootstrapParams": {},
                                "chunkLoadingGlobal": "DIFFPACK"
                            }),
                        ),
                        (
                            "next-font-manifest.json",
                            json!({
                                "pages": {}, "app": {}, "appUsingSizeAdjust": false,
                                "pagesUsingSizeAdjust": false
                            }),
                        ),
                    ]);
                }
                for (name, manifest) in manifests {
                    let path = directory.join(name);
                    std::fs::write(
                        &path,
                        serde_json::to_vec(&manifest).map_err(|error| {
                            format!("cannot serialize {}: {error}", path.display())
                        })?,
                    )
                    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
                }
                continue;
            }
            let entry = route.original_name.trim_start_matches('/');
            let directory = self.dist_dir.join("server/app").join(entry);
            std::fs::create_dir_all(&directory)
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;

            let mut manifests = vec![(
                "app-paths-manifest.json",
                json!({ route.original_name.clone(): format!("app/{entry}.js") }),
            )];
            if matches!(
                route.kind,
                NextRouteArtifactKind::AppPage | NextRouteArtifactKind::ImplicitAppPage
            ) {
                manifests.extend([
                    (
                        "build-manifest.json",
                        json!({
                            "devFiles": [], "ampDevFiles": [], "polyfillFiles": [],
                            "lowPriorityFiles": [], "rootMainFiles": client_files, "pages": {},
                            "ampFirstPages": [], "rootMainFilesTree": {},
                            "pagesChunkGroupBootstrapParams": {}, "chunkLoadingGlobal": "DIFFPACK"
                        }),
                    ),
                    (
                        "server-reference-manifest.json",
                        json!({ "node": {}, "edge": {} }),
                    ),
                    (
                        "next-font-manifest.json",
                        json!({
                            "pages": {}, "app": {}, "appUsingSizeAdjust": false,
                            "pagesUsingSizeAdjust": false
                        }),
                    ),
                    ("react-loadable-manifest.json", json!({})),
                ]);
            }
            for (name, value) in manifests {
                let path = directory.join(name);
                let bytes = serde_json::to_vec(&value)
                    .map_err(|error| format!("cannot serialize {}: {error}", path.display()))?;
                std::fs::write(&path, bytes)
                    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
            }
            if matches!(
                route.kind,
                NextRouteArtifactKind::AppPage | NextRouteArtifactKind::ImplicitAppPage
            ) {
                self.write_client_reference_manifest(route, entry)?;
            }
        }
        Ok(())
    }

    fn write_pages_global_partial_manifests(
        &self,
        routes: &[NextRouteArtifact],
    ) -> Result<(), String> {
        let pages_client = if self
            .dist_dir
            .join("static/chunks/diffpack-pages.js")
            .is_file()
        {
            vec!["static/chunks/diffpack-pages.js"]
        } else {
            Vec::new()
        };
        let mut pages = serde_json::Map::new();
        pages.insert("/_app".to_string(), json!(pages_client));
        pages.insert("/_error".to_string(), json!(pages_client));
        for route in routes
            .iter()
            .filter(|route| route.kind == NextRouteArtifactKind::PagesPage)
        {
            pages.insert(route.pathname.clone(), json!(pages_client));
        }
        let build = json!({
            "devFiles": [], "ampDevFiles": [], "polyfillFiles": [],
            "lowPriorityFiles": [], "rootMainFiles": [],
            "pages": pages, "ampFirstPages": [], "rootMainFilesTree": {},
            "pagesChunkGroupBootstrapParams": {}, "chunkLoadingGlobal": "DIFFPACK"
        });
        let font = json!({
            "pages": {}, "app": {}, "appUsingSizeAdjust": false,
            "pagesUsingSizeAdjust": false
        });
        for (entry, files) in [
            (
                "_app",
                vec![
                    ("build-manifest.json", build.clone()),
                    ("pages-manifest.json", json!({ "/_app": "pages/_app.js" })),
                    ("next-font-manifest.json", font.clone()),
                ],
            ),
            (
                "_document",
                vec![(
                    "pages-manifest.json",
                    json!({ "/_document": "pages/_document.js" }),
                )],
            ),
            (
                "_error",
                vec![
                    ("client-build-manifest.json", json!({})),
                    ("build-manifest.json", build),
                    (
                        "pages-manifest.json",
                        json!({ "/_error": "pages/_error.js" }),
                    ),
                    ("next-font-manifest.json", font),
                ],
            ),
        ] {
            let directory = self.dist_dir.join("server/pages").join(entry);
            std::fs::create_dir_all(&directory)
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
            for (name, manifest) in files {
                let path = directory.join(name);
                std::fs::write(
                    &path,
                    serde_json::to_vec(&manifest)
                        .map_err(|error| format!("cannot serialize {}: {error}", path.display()))?,
                )
                .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
            }
        }
        Ok(())
    }

    fn write_client_reference_manifest(
        &self,
        route: &NextRouteArtifact,
        entry: &str,
    ) -> Result<(), String> {
        let references_path = self
            .standalone_root
            .join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE);
        let references: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&references_path)
                .map_err(|error| format!("cannot read {}: {error}", references_path.display()))?,
        )
        .map_err(|error| format!("cannot parse {}: {error}", references_path.display()))?;
        let mut client_modules = serde_json::Map::new();
        let server_references_path = self
            .standalone_root
            .join(crate::rsc::SERVER_REFERENCES_MANIFEST_FILE);
        let server_references: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&server_references_path).map_err(|error| {
                format!("cannot read {}: {error}", server_references_path.display())
            })?)
            .map_err(|error| {
                format!("cannot parse {}: {error}", server_references_path.display())
            })?;
        let mut server_mapping = serde_json::Map::new();
        for (resource, reference) in references
            .as_object()
            .ok_or_else(|| format!("{} is not an object", references_path.display()))?
        {
            let mut chunks = reference
                .get("chunks")
                .and_then(serde_json::Value::as_array)
                .cloned()
                .unwrap_or_default();
            for index in (1..chunks.len()).step_by(2) {
                if let Some(file) = chunks[index].as_str() {
                    chunks[index] = json!(format!("static/chunks/{file}"));
                }
            }
            client_modules.insert(
                resource.clone(),
                json!({
                    "id": reference.get("id").cloned().unwrap_or(json!(resource)),
                    "name": "*",
                    "chunks": chunks
                }),
            );
            if let (Some(client_id), Some(server_reference)) =
                (reference.get("id"), server_references.get(resource))
            {
                server_mapping.insert(
                    client_id.to_string().trim_matches('"').to_string(),
                    json!({
                        "*": {
                            "id": server_reference.get("id").cloned().unwrap_or(json!(resource)),
                            "name": "*", "chunks": []
                        }
                    }),
                );
            }
        }
        let manifest = json!({
            "moduleLoading": { "prefix": "/_next/" },
            "ssrModuleMapping": server_mapping.clone(), "edgeSSRModuleMapping": {},
            "clientModules": client_modules, "entryCSSFiles": {},
            "rscModuleMapping": server_mapping, "edgeRscModuleMapping": {}
        });
        let path = self
            .dist_dir
            .join("server/app")
            .join(format!("{entry}_client-reference-manifest.js"));
        let source = format!(
            "globalThis.__RSC_MANIFEST=(globalThis.__RSC_MANIFEST||{{}});globalThis.__RSC_MANIFEST[{key:?}]={manifest};",
            key = route.original_name,
            manifest = manifest
        );
        std::fs::write(&path, source)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))
    }

    fn write_client_assets(&self) -> Result<Vec<String>, String> {
        let source = self.standalone_root.join("static");
        let destination = self.dist_dir.join("static/chunks");
        std::fs::create_dir_all(&destination)
            .map_err(|error| format!("cannot create {}: {error}", destination.display()))?;
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&destination)
            .map_err(|error| format!("cannot read {}: {error}", destination.display()))?
        {
            let entry = entry.map_err(|error| format!("cannot read client asset: {error}"))?;
            if entry.path().extension().and_then(|value| value.to_str()) == Some("js") {
                files.push(format!(
                    "static/chunks/{}",
                    entry.file_name().to_string_lossy()
                ));
            }
        }
        for entry in std::fs::read_dir(&source)
            .map_err(|error| format!("cannot read {}: {error}", source.display()))?
        {
            let entry = entry.map_err(|error| format!("cannot read client asset: {error}"))?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("js") {
                continue;
            }
            let name = entry.file_name();
            std::fs::copy(&path, destination.join(&name))
                .map_err(|error| format!("cannot copy {}: {error}", path.display()))?;
            let file = format!("static/chunks/{}", name.to_string_lossy());
            if !files.contains(&file) {
                files.push(file);
            }
        }
        files.sort();
        Ok(files)
    }
}

/// Discovers the route inventory consumed by native `.next` and standalone output.
pub fn discover_app_routes(project_root: &Path) -> Result<Vec<NextRouteArtifact>, String> {
    let root = project_root.canonicalize().map_err(|error| {
        format!(
            "cannot open project root {}: {error}",
            project_root.display()
        )
    })?;
    let patterns = crate::next_adapter::discover_route_patterns(&root)?.unwrap_or_default();
    let mut routes: Vec<_> = patterns
        .into_iter()
        .map(|pattern| {
            let source_path = pattern.source_path.ok_or_else(|| {
                format!("discovered route {} has no source module", pattern.url_path)
            })?;
            let source_relative = source_path
                .strip_prefix(&root)
                .unwrap_or(&source_path)
                .to_string_lossy()
                .replace('\\', "/");
            let (route_relative, is_pages_api) = if let Some(relative) = source_relative
                .strip_prefix("pages/api/")
                .or_else(|| source_relative.strip_prefix("src/pages/api/"))
            {
                (format!("api/{relative}"), true)
            } else {
                (
                    source_relative
                        .strip_prefix("app/")
                        .or_else(|| source_relative.strip_prefix("src/app/"))
                        .ok_or_else(|| {
                            format!(
                                "route source {source_relative} is not below app/, src/app/, pages/api/, or src/pages/api/"
                            )
                        })?
                        .to_string(),
                    false,
                )
            };
            let original_name = format!(
                "/{}",
                route_relative
                    .strip_suffix(".tsx")
                    .or_else(|| route_relative.strip_suffix(".ts"))
                    .or_else(|| route_relative.strip_suffix(".jsx"))
                    .or_else(|| route_relative.strip_suffix(".js"))
                    .unwrap_or(&route_relative)
            );
            let kind = if is_pages_api {
                NextRouteArtifactKind::PagesApi
            } else {
                match pattern.kind {
                crate::next_adapter::PatternKind::Page => NextRouteArtifactKind::AppPage,
                crate::next_adapter::PatternKind::Endpoint => NextRouteArtifactKind::AppRoute,
                }
            };
            Ok(NextRouteArtifact {
                pathname: pattern.url_path,
                original_name,
                source_path,
                kind,
            })
        })
        .collect::<Result<_, String>>()?;
    // Next models the root static favicon as an App Route even though there is no
    // `route.ts` source module. Keep that executable route in the shared inventory;
    // the native compiler materializes a tiny userland module around these bytes.
    let app_dir = if root.join("app").is_dir() {
        root.join("app")
    } else {
        root.join("src/app")
    };
    let favicon = app_dir.join("favicon.ico");
    if favicon.is_file() {
        routes.push(NextRouteArtifact {
            pathname: "/favicon.ico".to_string(),
            original_name: "/favicon.ico/route".to_string(),
            source_path: favicon,
            kind: NextRouteArtifactKind::AppRoute,
        });
    }
    for page in crate::next_pages::discover_route_artifacts(&root)? {
        if routes
            .iter()
            .any(|route| route.source_path == page.source_path)
        {
            continue;
        }
        let source_relative = page
            .source_path
            .strip_prefix(&root)
            .unwrap_or(&page.source_path)
            .to_string_lossy()
            .replace('\\', "/");
        let route_relative = source_relative
            .strip_prefix("pages/")
            .or_else(|| source_relative.strip_prefix("src/pages/"))
            .ok_or_else(|| format!("Pages route source {source_relative} is outside pages/"))?;
        let original_name = format!(
            "/{}",
            route_relative
                .strip_suffix(".tsx")
                .or_else(|| route_relative.strip_suffix(".ts"))
                .or_else(|| route_relative.strip_suffix(".jsx"))
                .or_else(|| route_relative.strip_suffix(".js"))
                .or_else(|| route_relative.strip_suffix(".mdx"))
                .or_else(|| route_relative.strip_suffix(".md"))
                .unwrap_or(route_relative)
        );
        routes.push(NextRouteArtifact {
            pathname: page.pathname,
            original_name,
            source_path: page.source_path,
            kind: if page.is_api {
                NextRouteArtifactKind::PagesApi
            } else {
                NextRouteArtifactKind::PagesPage
            },
        });
    }
    if routes.is_empty() {
        return Err(format!("{} has no Next routes", root.display()));
    }
    let fallback_source = routes
        .iter()
        .find(|route| route.kind == NextRouteArtifactKind::AppPage)
        .map(|route| route.source_path.clone())
        .unwrap_or_else(|| root.join("app/layout.js"));
    let has_app_routes = routes.iter().any(|route| {
        matches!(
            route.kind,
            NextRouteArtifactKind::AppPage | NextRouteArtifactKind::AppRoute
        )
    });
    for (pathname, original_name) in [
        ("/_not-found", "/_not-found/page"),
        ("/_global-error", "/_global-error/page"),
    ] {
        if has_app_routes
            && !routes
                .iter()
                .any(|route| route.original_name == original_name)
        {
            routes.push(NextRouteArtifact {
                pathname: pathname.into(),
                original_name: original_name.into(),
                source_path: fallback_source.clone(),
                kind: NextRouteArtifactKind::ImplicitAppPage,
            });
        }
    }
    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_route_inventory_keeps_page_and_handler_identity() {
        let directory = tempfile::tempdir().unwrap();
        let app = directory.path().join("app");
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"dependencies":{"next":"16.0.0"}}"#,
        )
        .unwrap();
        std::fs::create_dir_all(app.join("api/ping")).unwrap();
        std::fs::write(
            app.join("layout.tsx"),
            "export default ({children}) => children;",
        )
        .unwrap();
        std::fs::write(app.join("page.tsx"), "export default () => null;").unwrap();
        std::fs::write(app.join("favicon.ico"), [0, 0, 1, 0]).unwrap();
        std::fs::write(
            app.join("api/ping/route.ts"),
            "export function GET() { return new Response('pong'); }",
        )
        .unwrap();
        let routes = discover_app_routes(directory.path()).unwrap();
        assert!(routes.iter().any(|route| {
            route.pathname == "/"
                && route.original_name == "/page"
                && route.kind == NextRouteArtifactKind::AppPage
        }));
        assert!(routes.iter().any(|route| {
            route.pathname == "/api/ping"
                && route.original_name == "/api/ping/route"
                && route.kind == NextRouteArtifactKind::AppRoute
        }));
        assert!(routes.iter().any(|route| {
            route.pathname == "/favicon.ico"
                && route.original_name == "/favicon.ico/route"
                && route.kind == NextRouteArtifactKind::AppRoute
        }));
    }

    #[test]
    fn native_route_inventory_supports_pages_only_projects() {
        let directory = tempfile::tempdir().unwrap();
        let pages = directory.path().join("pages");
        std::fs::write(
            directory.path().join("package.json"),
            r#"{"dependencies":{"next":"16.0.0"}}"#,
        )
        .unwrap();
        std::fs::create_dir_all(pages.join("api")).unwrap();
        std::fs::write(pages.join("index.tsx"), "export default () => null;").unwrap();
        std::fs::write(
            pages.join("api/health.ts"),
            "export default (_req, res) => res.json({ok:true});",
        )
        .unwrap();

        let routes = discover_app_routes(directory.path()).unwrap();
        assert!(routes.iter().any(|route| {
            route.pathname == "/"
                && route.original_name == "/index"
                && route.kind == NextRouteArtifactKind::PagesPage
        }));
        assert!(routes.iter().any(|route| {
            route.pathname == "/api/health"
                && route.original_name == "/api/health"
                && route.kind == NextRouteArtifactKind::PagesApi
        }));
        assert!(
            !routes
                .iter()
                .any(|route| route.kind == NextRouteArtifactKind::ImplicitAppPage)
        );
    }

    #[test]
    fn native_output_writes_next_owned_route_manifests() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("static")).unwrap();
        std::fs::write(directory.path().join("static/index.html"), "hello").unwrap();
        std::fs::write(
            directory
                .path()
                .join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE),
            "{}",
        )
        .unwrap();
        std::fs::write(
            directory
                .path()
                .join(crate::rsc::SERVER_REFERENCES_MANIFEST_FILE),
            "{}",
        )
        .unwrap();
        let route = NextRouteArtifact {
            pathname: "/".into(),
            original_name: "/page".into(),
            source_path: PathBuf::from("app/page.tsx"),
            kind: NextRouteArtifactKind::AppPage,
        };
        NativeNextOutput {
            dist_dir: directory.path(),
            standalone_root: directory.path(),
        }
        .write_route_manifests(&[route])
        .unwrap();
        let root = directory.path().join("server/app/page");
        let app_paths: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("app-paths-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(app_paths["/page"], "app/page.js");
        assert!(root.join("build-manifest.json").is_file());
        assert!(root.join("server-reference-manifest.json").is_file());
        assert!(root.join("next-font-manifest.json").is_file());
    }

    #[test]
    fn pages_partial_manifests_are_keyed_by_public_path_not_source_path() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("static")).unwrap();
        let route = NextRouteArtifact {
            pathname: "/embed".into(),
            original_name: "/router/embed".into(),
            source_path: PathBuf::from("pages/router/embed.tsx"),
            kind: NextRouteArtifactKind::PagesPage,
        };
        NativeNextOutput {
            dist_dir: directory.path(),
            standalone_root: directory.path(),
        }
        .write_route_manifests(&[route])
        .unwrap();

        let partial = directory.path().join("server/pages/embed");
        assert!(partial.join("client-build-manifest.json").is_file());
        assert!(partial.join("build-manifest.json").is_file());
        assert!(partial.join("next-font-manifest.json").is_file());
        let pages: serde_json::Value =
            serde_json::from_slice(&std::fs::read(partial.join("pages-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(pages["/embed"], "pages/router/embed.js");
        assert!(!directory.path().join("server/pages/router/embed").exists());
    }

    #[test]
    fn native_entry_uses_nexts_official_page_runtime_and_a_real_loader_tree() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join(".diffpack-output")).unwrap();
        std::fs::write(
            directory
                .path()
                .join(".diffpack-output")
                .join(crate::rsc::CLIENT_REFERENCES_MANIFEST_FILE),
            "{}",
        )
        .unwrap();
        let app = directory.path().join("app/dashboard");
        std::fs::create_dir_all(&app).unwrap();
        std::fs::write(
            directory.path().join("app/layout.js"),
            "export default()=>null",
        )
        .unwrap();
        std::fs::write(app.join("page.js"), "export default()=>null").unwrap();
        let route = NextRouteArtifact {
            pathname: "/dashboard".into(),
            original_name: "/dashboard/page".into(),
            source_path: app.join("page.js").canonicalize().unwrap(),
            kind: NextRouteArtifactKind::AppPage,
        };
        let source = native_app_page_entry_source(
            directory.path(),
            &route,
            &directory.path().join("server/diffpack-ssr.js"),
        )
        .unwrap();
        assert!(source.contains("createAppPageEntrypoint"));
        assert!(source.contains("[\"dashboard\",{children:['__PAGE__'"));
        assert!(source.contains("export const handler = entrypoint.handler"));
        assert!(!source.contains("function handler(...args)"));
        assert!(source.contains("const pendingSsrModules = new Map()"));
        assert!(!source.contains("/*__DIFFPACK_"));
        assert!(source.contains("export * from 'next/dist/server/app-render/entry-base'"));
    }
}
