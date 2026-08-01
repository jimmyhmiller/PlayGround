//! Loader output records consumed by graph construction.

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};

use diffpack_core::ResourceId;
use diffpack_core::transform::{DependencyDemand, FlatModule, SourceLanguage, TransformResult};

use crate::asset::{AssetEmission, content_hash};

pub struct SpecialModule {
    pub hash: u64,
    pub code: String,
    pub flat_module: Option<FlatModule>,
    pub assets: Vec<AssetEmission>,
    pub css: Option<String>,
    pub css_source_files: Vec<PathBuf>,
    pub css_external_imports: Vec<String>,
    pub dependency_specifiers: Vec<String>,
    pub dependency_demands: Vec<DependencyDemand>,
}

/// Compiles JavaScript synthesized by a built-in loader through the caller's
/// configured core compiler and preserves the synthetic asset identity.
pub fn compile_synthetic(
    label: &Path,
    synthetic: crate::asset::SyntheticModule,
    compile: impl FnOnce(&Path, &str) -> TransformResult,
) -> SpecialModule {
    let transformed = compile(label, &synthetic.source);
    SpecialModule {
        hash: synthetic.identity,
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: synthetic.assets,
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: transformed.dependencies,
        dependency_demands: transformed.dependency_demands,
    }
}

/// Compiles a build-generated virtual module and retains its graph edges.
pub fn virtual_module(
    source: &str,
    compile: impl FnOnce(&Path, &str) -> TransformResult,
) -> SpecialModule {
    virtual_module_at(Path::new("diffpack-virtual-module.js"), source, compile)
}

pub fn virtual_module_at(
    label: &Path,
    source: &str,
    compile: impl FnOnce(&Path, &str) -> TransformResult,
) -> SpecialModule {
    let transformed = compile(label, source);
    SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: Vec::new(),
        css: None,
        css_source_files: Vec::new(),
        css_external_imports: Vec::new(),
        dependency_specifiers: transformed.dependencies,
        dependency_demands: transformed.dependency_demands,
    }
}

/// Labels maps after loader-side source rewriting so they never claim positions
/// against different text on disk.
pub fn mark_rewritten_source(
    transformed: &mut TransformResult,
    precompiled: bool,
    compatibility_rewritten: bool,
) {
    let Some(map) = transformed.map.as_mut() else {
        return;
    };
    if precompiled {
        map.mark_generated("component");
    } else if compatibility_rewritten {
        map.mark_generated("vite-replace");
    }
}

/// Dispatches framework-neutral query loaders. `None` leaves the query for an
/// integration or external policy; the composing driver reports it if nobody
/// claims it.
pub fn query_module(
    resource: &ResourceId,
    base: &str,
    asset_inline_limit: usize,
    root: Option<&Path>,
    mut compile: impl FnMut(&Path, &str) -> TransformResult,
) -> Result<Option<SpecialModule>, String> {
    let path = Path::new(&resource.path);
    let module = match crate::loader::kind(resource) {
        Some(crate::loader::LoaderKind::Url) => asset_url(
            path.to_path_buf(),
            base,
            asset_inline_limit,
            |source| compile(Path::new("diffpack-url-asset.js"), source),
            |_, _, _| Ok(None),
        )?,
        Some(crate::loader::LoaderKind::Raw) => compile_synthetic(
            Path::new("diffpack-raw-asset.js"),
            crate::asset::raw_module(path)?,
            &mut compile,
        ),
        Some(crate::loader::LoaderKind::CssMedia) => css_media(resource)?,
        Some(crate::loader::LoaderKind::Worker) => compile_synthetic(
            Path::new("diffpack-worker.js"),
            crate::asset::worker_module(path, resource.query_has_flag("inline"))?,
            &mut compile,
        ),
        Some(crate::loader::LoaderKind::Inline) => compile_synthetic(
            Path::new("diffpack-inline-asset.js"),
            crate::asset::inline_module(path)?,
            &mut compile,
        ),
        Some(crate::loader::LoaderKind::WasmInit) => compile_synthetic(
            Path::new("diffpack-wasm-init.js"),
            crate::asset::wasm_init_module(path, base, asset_inline_limit)?,
            &mut compile,
        ),
        Some(crate::loader::LoaderKind::PublicUrl) => compile_synthetic(
            Path::new("diffpack-public-url.js"),
            crate::asset::public_url_module(path, base, root)?,
            &mut compile,
        ),
        Some(crate::loader::LoaderKind::TsrSplit) | None => return Ok(None),
    };
    Ok(Some(module))
}

/// Dispatches filesystem modules claimed by the default loader. JavaScript and
/// TypeScript return `None` for the caller's ordinary compiler path.
#[allow(clippy::too_many_arguments)]
pub fn path_module(
    path: &Path,
    postcss: Option<&crate::postcss::Postcss>,
    scss_options: &crate::sass::ScssOptions,
    project_root: Option<&Path>,
    base: &str,
    asset_inline_limit: usize,
    mut compile: impl FnMut(&Path, &str) -> TransformResult,
    custom_asset: impl FnOnce(&Path, &[u8], &str) -> Result<Option<SpecialModule>, String>,
) -> Option<Result<SpecialModule, String>> {
    if crate::css::is_css_module_path(path) {
        return Some(css_module(path, postcss, |source| {
            compile(Path::new("diffpack-css-module.js"), source)
        }));
    }
    if path.extension().and_then(|value| value.to_str()) == Some("css") {
        return Some(stylesheet(path, postcss));
    }
    if crate::sass::is_scss_path(path) {
        return Some(scss(path, scss_options, postcss, |source| {
            compile(Path::new("diffpack-css-module.js"), source)
        }));
    }
    if crate::less_stylus::is_less_or_stylus_path(path) {
        return Some(less_or_stylus(path, project_root, postcss, |source| {
            compile(Path::new("diffpack-css-module.js"), source)
        }));
    }
    if crate::source_policy::is_asset_path(path) {
        return Some(asset_url(
            path.to_path_buf(),
            base,
            asset_inline_limit,
            |source| compile(Path::new("diffpack-url-asset.js"), source),
            custom_asset,
        ));
    }
    crate::source_policy::unhandled_source(path).map(|unhandled| {
        Err(crate::source_policy::unhandled_source_message(
            path, &unhandled,
        ))
    })
}

/// Loads the virtual module behind a media-qualified CSS import.
pub fn css_media(resource: &ResourceId) -> Result<SpecialModule, String> {
    let path = Path::new(&resource.path);
    if path.extension().and_then(|extension| extension.to_str()) != Some("css") {
        return Err(format!(
            "loader `?media` applies only to CSS files (requested for {})",
            resource.path
        ));
    }
    let media = resource
        .query
        .as_deref()
        .and_then(|query| query.strip_prefix("media="))
        .filter(|media| !media.trim().is_empty())
        .ok_or_else(|| {
            format!(
                "loader `?media` requires a media query value (requested for {})",
                resource.path
            )
        })?;
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let processed = crate::css::process_media_import(path, &text, media)?;
    let assets = css_assets(processed.assets);
    Ok(SpecialModule {
        hash: content_hash(processed.css.as_bytes()),
        code: String::new(),
        flat_module: None,
        assets,
        css: Some(processed.css),
        css_source_files: processed.inlined_files,
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

/// Loads a global stylesheet and records its imports, assets, and watched files.
pub fn stylesheet(
    path: &Path,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if crate::tailwind::needs_native_tailwind_compile(&text) {
        let entry = crate::css::inline_tailwind_entry(path, &text)?;
        return Ok(SpecialModule {
            hash: content_hash(entry.css.as_bytes()),
            code: String::new(),
            flat_module: None,
            assets: css_assets(entry.assets),
            css: Some(entry.css),
            css_source_files: entry.inlined_files,
            css_external_imports: entry.external_imports,
            dependency_specifiers: Vec::new(),
            dependency_demands: Vec::new(),
        });
    }
    stylesheet_from_text(path, &text, Vec::new(), postcss)
}

/// Loads global CSS already produced by an SFC or stylesheet preprocessor.
pub fn stylesheet_from_text(
    path: &Path,
    text: &str,
    extra_source_files: Vec<PathBuf>,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<SpecialModule, String> {
    let prefixed = crate::postcss::process_optional(text, path, postcss)?;
    let processed = crate::css::process_global_css(path, &prefixed)?;
    let mut identity = processed.css.clone();
    for external in &processed.external_imports {
        identity.push('\0');
        identity.push_str(external);
    }
    let dependency_specifiers = processed
        .imports
        .iter()
        .map(css_import_specifier)
        .collect::<Vec<_>>();
    let dependency_demands = dependency_specifiers
        .iter()
        .cloned()
        .map(css_import_demand)
        .collect();
    let mut css_source_files = processed.inlined_files;
    css_source_files.extend(extra_source_files);
    crate::postcss::record_config(&mut css_source_files, postcss);
    Ok(SpecialModule {
        hash: content_hash(identity.as_bytes()),
        code: String::new(),
        flat_module: None,
        assets: css_assets(processed.assets),
        css: Some(processed.css),
        css_source_files,
        css_external_imports: processed.external_imports,
        dependency_specifiers,
        dependency_demands,
    })
}

/// Loads a CSS Module while delegating compilation of its generated JavaScript
/// mapping to the caller's selected core compiler configuration.
pub fn css_module(
    path: &Path,
    postcss: Option<&crate::postcss::Postcss>,
    compile: impl FnOnce(&str) -> TransformResult,
) -> Result<SpecialModule, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    css_module_from_text(path, &text, Vec::new(), postcss, compile)
}

/// Loads CSS Module text already produced by Sass, Less, or Stylus.
pub fn css_module_from_text(
    path: &Path,
    text: &str,
    extra_source_files: Vec<PathBuf>,
    postcss: Option<&crate::postcss::Postcss>,
    compile: impl FnOnce(&str) -> TransformResult,
) -> Result<SpecialModule, String> {
    let prefixed = crate::postcss::process_optional(text, path, postcss)?;
    let processed = crate::css::process_css_module(path, &prefixed)?;
    let mut js = String::new();
    for (index, specifier) in processed.compose_imports.iter().enumerate() {
        js.push_str(&format!(
            "import __composed_{index} from {};\n",
            quote(specifier)
        ));
    }
    let has_foreign = processed.mapping.iter().any(|(_, segments)| {
        segments
            .iter()
            .any(|segment| matches!(segment, crate::css::MappingSegment::Foreign { .. }))
    });
    if has_foreign {
        js.push_str(concat!(
            "const __compose = (mapping, name, from) => {\n",
            "  const value = mapping[name];\n",
            "  if (value === undefined) {\n",
            "    throw new Error(\"composes target \\\"\" + name + \"\\\" is not exported by \" + from);\n",
            "  }\n",
            "  return value;\n",
            "};\n",
        ));
    }
    js.push_str("const __styles = {\n");
    for (name, segments) in &processed.mapping {
        let mut parts = Vec::new();
        let mut literal_run: Option<String> = None;
        for segment in segments {
            match segment {
                crate::css::MappingSegment::Literal(literal) => match &mut literal_run {
                    Some(run) => {
                        run.push(' ');
                        run.push_str(literal);
                    }
                    None => literal_run = Some(literal.clone()),
                },
                crate::css::MappingSegment::Foreign { import, name } => {
                    if let Some(run) = literal_run.take() {
                        parts.push(quote(&run));
                    }
                    parts.push(format!(
                        "__compose(__composed_{import}, {}, {})",
                        quote(name),
                        quote(&processed.compose_imports[*import])
                    ));
                }
            }
        }
        if let Some(run) = literal_run {
            parts.push(quote(&run));
        }
        js.push_str(&format!(
            "  {}: {},\n",
            quote(name),
            parts.join(" + \" \" + ")
        ));
    }
    js.push_str("};\nexport default __styles;\n");
    for (name, _) in &processed.mapping {
        if is_valid_js_identifier(name) {
            js.push_str(&format!(
                "export const {name} = __styles[{}];\n",
                quote(name)
            ));
        }
    }
    let transformed = compile(&js);
    let mut dependency_specifiers = transformed.dependencies;
    let mut dependency_demands = transformed.dependency_demands;
    for import in &processed.imports {
        let specifier = css_import_specifier(import);
        if !dependency_specifiers.contains(&specifier) {
            dependency_specifiers.push(specifier.clone());
            dependency_demands.push(css_import_demand(specifier));
        }
    }
    let mut identity = processed.css.clone();
    identity.push('\0');
    identity.push_str(&transformed.code);
    for external in &processed.external_imports {
        identity.push('\0');
        identity.push_str(external);
    }
    let mut css_source_files = processed.inlined_files;
    css_source_files.extend(extra_source_files);
    crate::postcss::record_config(&mut css_source_files, postcss);
    Ok(SpecialModule {
        hash: content_hash(identity.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets: css_assets(processed.assets),
        css: Some(processed.css),
        css_source_files,
        css_external_imports: processed.external_imports,
        dependency_specifiers,
        dependency_demands,
    })
}

/// Compiles Sass and routes its CSS through the global or CSS Module pipeline.
pub fn scss(
    path: &Path,
    options: &crate::sass::ScssOptions,
    postcss: Option<&crate::postcss::Postcss>,
    compile: impl FnOnce(&str) -> TransformResult,
) -> Result<SpecialModule, String> {
    let compiled = crate::sass::compile_file(path, options)?;
    if crate::sass::is_scss_module_path(path) {
        css_module_from_text(path, &compiled.css, compiled.loaded_files, postcss, compile)
    } else {
        stylesheet_from_text(path, &compiled.css, compiled.loaded_files, postcss)
    }
}

/// Compiles Less or Stylus and routes its CSS through the appropriate pipeline.
pub fn less_or_stylus(
    path: &Path,
    root: Option<&Path>,
    postcss: Option<&crate::postcss::Postcss>,
    compile: impl FnOnce(&str) -> TransformResult,
) -> Result<SpecialModule, String> {
    let compiled = crate::less_stylus::compile_file(path, root)?;
    if crate::less_stylus::is_css_module_path(path) {
        css_module_from_text(path, &compiled.css, compiled.loaded_files, postcss, compile)
    } else {
        stylesheet_from_text(path, &compiled.css, compiled.loaded_files, postcss)
    }
}

/// Loads an ordinary asset import. Integration-specific representations may
/// claim the bytes first; otherwise default URL/inlining policy is applied.
pub fn asset_url(
    source_path: PathBuf,
    base: &str,
    inline_limit: usize,
    compile: impl FnOnce(&str) -> TransformResult,
    custom: impl FnOnce(&Path, &[u8], &str) -> Result<Option<SpecialModule>, String>,
) -> Result<SpecialModule, String> {
    let bytes = fs::read(&source_path)
        .map_err(|error| format!("cannot read asset {}: {error}", source_path.display()))?;
    let mut imported_assets = Vec::new();
    let mut inlined_files = Vec::new();
    let tailwind_source = if source_path.extension().and_then(|value| value.to_str()) == Some("css")
    {
        let text = String::from_utf8_lossy(&bytes);
        if crate::tailwind::needs_native_tailwind_compile(&text) {
            let entry = crate::css::inline_tailwind_entry(&source_path, &text)?;
            imported_assets = entry.assets;
            inlined_files = entry.inlined_files;
            Some(entry.css)
        } else {
            None
        }
    } else {
        None
    };
    let public_name = if tailwind_source.is_some() {
        let mut hashed = bytes.clone();
        if let Some(theme) = crate::tailwind_project::app_tailwind_theme(&source_path) {
            hashed.extend_from_slice(theme.as_bytes());
        }
        crate::asset::asset_public_name(&source_path, content_hash(&hashed))
    } else {
        crate::asset::asset_public_name(&source_path, content_hash(&bytes))
    };
    if let Some(module) = custom(&source_path, &bytes, &public_name)? {
        return Ok(module);
    }
    let (source, mut assets) =
        if inline_limit > 0 && bytes.len() <= inline_limit && tailwind_source.is_none() {
            let data_uri = crate::asset::svg_data_url(&source_path, &bytes).unwrap_or_else(|| {
                format!(
                    "data:{};base64,{}",
                    crate::asset::asset_mime_type(&source_path),
                    crate::asset::base64_encode(&bytes)
                )
            });
            (
                format!("export default {};\n", quote(&data_uri)),
                Vec::new(),
            )
        } else {
            (
                format!(
                    "export default {};\n",
                    quote(&format!("{base}assets/{public_name}"))
                ),
                vec![AssetEmission {
                    source: source_path,
                    public_name,
                    tailwind_source,
                    image_variants: None,
                }],
            )
        };
    assets.extend(css_assets(imported_assets));
    let transformed = compile(&source);
    Ok(SpecialModule {
        hash: content_hash(transformed.code.as_bytes()),
        code: transformed.code,
        flat_module: transformed.flat_module,
        assets,
        css: None,
        css_source_files: inlined_files,
        css_external_imports: Vec::new(),
        dependency_specifiers: Vec::new(),
        dependency_demands: Vec::new(),
    })
}

fn css_assets(assets: Vec<crate::css::CssAsset>) -> Vec<AssetEmission> {
    assets
        .into_iter()
        .map(|asset| AssetEmission {
            source: asset.source,
            public_name: asset.public_name,
            tailwind_source: None,
            image_variants: None,
        })
        .collect()
}

fn css_import_specifier(import: &crate::css::CssImport) -> String {
    match &import.media {
        None => import.specifier.clone(),
        Some(media) => format!("{}?media={media}", import.specifier),
    }
}

fn css_import_demand(specifier: String) -> DependencyDemand {
    DependencyDemand {
        specifier,
        all: true,
        names: Vec::new(),
        dynamic: false,
        optional: false,
        require_syntax: false,
        import_syntax: true,
        eager: true,
    }
}

fn quote(value: &str) -> String {
    serde_json::to_string(value).expect("a string always serializes")
}

fn is_valid_js_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_' || first == '$')
        || !chars.all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '$'
        })
    {
        return false;
    }
    !matches!(
        name,
        "break"
            | "case"
            | "catch"
            | "class"
            | "const"
            | "continue"
            | "debugger"
            | "default"
            | "delete"
            | "do"
            | "else"
            | "enum"
            | "export"
            | "extends"
            | "false"
            | "finally"
            | "for"
            | "function"
            | "if"
            | "import"
            | "in"
            | "instanceof"
            | "new"
            | "null"
            | "return"
            | "super"
            | "switch"
            | "this"
            | "throw"
            | "true"
            | "try"
            | "typeof"
            | "var"
            | "void"
            | "while"
            | "with"
            | "yield"
            | "let"
            | "static"
            | "await"
            | "implements"
            | "interface"
            | "package"
            | "private"
            | "protected"
            | "public"
            | "arguments"
            | "eval"
    )
}

pub struct PrecompiledComponent {
    pub code: String,
    pub language: SourceLanguage,
    pub side_effects: ComponentSideEffects,
}

/// Compiles a supported SFC and routes its generated styles through the same
/// default CSS pipeline as filesystem stylesheets.
pub fn precompile_component(
    path: &Path,
    source: &str,
    project_root: Option<&Path>,
    postcss: Option<&crate::postcss::Postcss>,
) -> Result<Option<PrecompiledComponent>, String> {
    crate::sfc::precompile(path, source, project_root, |css| {
        let styles = stylesheet_from_text(path, css, Vec::new(), postcss)?;
        Ok(ComponentSideEffects {
            css: styles.css,
            css_source_files: styles.css_source_files,
            css_external_imports: styles.css_external_imports,
            assets: styles.assets,
            dependency_specifiers: styles.dependency_specifiers,
            dependency_demands: styles.dependency_demands,
        })
    })
}

#[derive(Default)]
pub struct ComponentSideEffects {
    pub css: Option<String>,
    pub css_source_files: Vec<PathBuf>,
    pub css_external_imports: Vec<String>,
    pub assets: Vec<AssetEmission>,
    pub dependency_specifiers: Vec<String>,
    pub dependency_demands: Vec<DependencyDemand>,
}

impl ComponentSideEffects {
    /// Combines JavaScript dependencies with stylesheet imports contributed by
    /// an SFC, borrowing the transform vectors for ordinary modules.
    pub fn dependencies<'a>(
        &self,
        transformed: &'a TransformResult,
    ) -> (Cow<'a, [String]>, Cow<'a, [DependencyDemand]>) {
        if self.dependency_specifiers.is_empty() {
            return (
                Cow::Borrowed(&transformed.dependencies),
                Cow::Borrowed(&transformed.dependency_demands),
            );
        }
        let mut specifiers = transformed.dependencies.clone();
        specifiers.extend(self.dependency_specifiers.iter().cloned());
        let mut demands = transformed.dependency_demands.clone();
        demands.extend(self.dependency_demands.iter().cloned());
        (Cow::Owned(specifiers), Cow::Owned(demands))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diffpack_core::transform::Target;

    #[test]
    fn css_module_owns_mapping_generation_and_compilation_seam() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("card.module.css");
        fs::write(&path, ".card { color: red; }\n").unwrap();
        let module = css_module(&path, None, |source| {
            diffpack_core::compiler::transform_module(
                Path::new("diffpack-css-module.js"),
                source,
                Target::Client,
            )
        })
        .unwrap();
        assert!(module.css.as_deref().unwrap().contains("._card_"));
        assert!(module.code.contains("__styles"));
        assert!(module.code.contains("card"));
        assert!(module.dependency_specifiers.is_empty());
    }

    #[test]
    fn global_stylesheet_owns_import_edges_and_media_queries() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("app.css");
        fs::write(
            &path,
            "@import './print.css' print;\nbody { color: red; }\n",
        )
        .unwrap();
        let module = stylesheet(&path, None).unwrap();
        assert_eq!(
            module.dependency_specifiers,
            vec!["./print.css?media=print"]
        );
        assert!(module.dependency_demands[0].all);
        assert_eq!(module.css.as_deref(), Some("body { color: red; }\n"));
    }

    #[test]
    fn asset_url_owns_inline_and_emit_policy() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("icon.svg");
        fs::write(&path, "<svg xmlns='http://www.w3.org/2000/svg'></svg>").unwrap();
        let compile = |source: &str| {
            diffpack_core::compiler::transform_module(
                Path::new("diffpack-url-asset.js"),
                source,
                Target::Server,
            )
        };
        let inline = asset_url(path.clone(), "/", 4096, compile, |_, _, _| Ok(None)).unwrap();
        assert!(
            inline.code.contains("data:image/svg+xml"),
            "{}",
            inline.code
        );
        assert!(inline.assets.is_empty());

        let emitted = asset_url(path, "/base/", 0, compile, |_, _, _| Ok(None)).unwrap();
        assert!(emitted.code.contains("/base/assets/"), "{}", emitted.code);
        assert_eq!(emitted.assets.len(), 1);
    }

    #[test]
    fn query_dispatch_owns_builtin_loaders_but_leaves_integration_queries() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("note.txt");
        fs::write(&path, "hello").unwrap();
        let compile = |path: &Path, source: &str| {
            diffpack_core::compiler::transform_module(path, source, Target::Server)
        };
        let raw = query_module(
            &ResourceId::parse(&format!("{}?raw", path.display())),
            "/",
            0,
            None,
            compile,
        )
        .unwrap()
        .unwrap();
        assert!(raw.code.contains("hello"));

        let integration = query_module(
            &ResourceId::parse("/route.tsx?tsr-split=component"),
            "/",
            0,
            None,
            compile,
        )
        .unwrap();
        assert!(integration.is_none());
    }
}
