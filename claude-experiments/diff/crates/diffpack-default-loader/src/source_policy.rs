//! File-extension ownership and diagnostics for the default loader set.

use std::path::Path;

/// Integration source rewrites applied before the generic compiler parses a module.
pub trait SourceIntegrationPolicy: Send + Sync + std::fmt::Debug {
    fn transform(
        &self,
        _path: &Path,
        _source: &str,
        _target: diffpack_core::transform::Target,
    ) -> Result<Option<String>, String> {
        Ok(None)
    }

    fn development(&self) -> Option<std::sync::Arc<dyn SourceIntegrationPolicy>> {
        None
    }

    fn defines(&self) -> &[(String, String)] {
        &[]
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoSourceIntegrationPolicy;

impl SourceIntegrationPolicy for NoSourceIntegrationPolicy {}

pub fn is_asset_path(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|value| value.to_str()),
        Some(
            "png"
                | "jpg"
                | "jpeg"
                | "gif"
                | "svg"
                | "webp"
                | "avif"
                | "ico"
                | "bmp"
                | "woff"
                | "woff2"
                | "ttf"
                | "otf"
                | "eot"
                | "mp4"
                | "webm"
                | "mp3"
                | "wav"
                | "ogg"
                | "pdf"
                | "wasm"
        )
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnhandledSource {
    NeedsCompiler {
        kind: &'static str,
        compiler: &'static str,
    },
    NativeAddon,
    NoLoader,
}

const COMPILED_SOURCE_KINDS: &[(&str, &str, &str)] = &[
    (
        "astro",
        "an Astro component",
        "the Astro compiler (@astrojs/compiler)",
    ),
    (
        "marko",
        "a Marko template",
        "the Marko compiler (@marko/compiler)",
    ),
    (
        "riot",
        "a Riot component",
        "the Riot compiler (@riotjs/compiler)",
    ),
    ("imba", "an Imba module", "the Imba compiler"),
    (
        "civet",
        "a Civet module",
        "the Civet compiler (@danielx/civet)",
    ),
    (
        "coffee",
        "a CoffeeScript module",
        "the CoffeeScript compiler",
    ),
    ("res", "a ReScript module", "the ReScript compiler"),
    ("resi", "a ReScript interface", "the ReScript compiler"),
    ("re", "a Reason module", "the Reason compiler"),
    ("rei", "a Reason interface", "the Reason compiler"),
    ("elm", "an Elm module", "the Elm compiler"),
];

pub fn unhandled_source(path: &Path) -> Option<UnhandledSource> {
    let extension = path.extension().and_then(|value| value.to_str())?;
    if matches!(
        extension,
        "js" | "jsx" | "mjs" | "cjs" | "ts" | "tsx" | "mts" | "cts" | "json" | "md" | "mdx"
    ) || crate::sfc::is_component_path(path)
    {
        return None;
    }
    if extension == "node" {
        return Some(UnhandledSource::NativeAddon);
    }
    Some(
        match COMPILED_SOURCE_KINDS
            .iter()
            .find(|(candidate, _, _)| *candidate == extension)
        {
            Some((_, kind, compiler)) => UnhandledSource::NeedsCompiler { kind, compiler },
            None => UnhandledSource::NoLoader,
        },
    )
}

pub fn unhandled_source_message(path: &Path, unhandled: &UnhandledSource) -> String {
    let file = path.display();
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    let found = "the file was found on disk: this is neither a missing import nor a JavaScript syntax error";
    match unhandled {
        UnhandledSource::NeedsCompiler { kind, compiler } => format!(
            "{file}: `.{extension}` is {kind}, not JavaScript\n  compiling it requires {compiler}; diffpack hosts no JS plugins and has no built-in `.{extension}` compiler\n  {found}\n  build this project with its own toolchain instead"
        ),
        UnhandledSource::NativeAddon => format!(
            "{file}: `.{extension}` is a prebuilt native addon, not JavaScript\n  a native addon is machine code loaded by Node's `process.dlopen`, and diffpack cannot put native code in a JavaScript bundle\n  {found}\n  build this project with its own toolchain instead"
        ),
        UnhandledSource::NoLoader => {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default();
            format!(
                "{file}: no loader handles the `.{extension}` extension\n  diffpack loads .js/.jsx/.mjs/.cjs/.ts/.tsx/.mts/.cts, .json, .md/.mdx, .css/.scss/.sass/.less/.styl/.stylus, and static assets; nothing else is parsed as JavaScript\n  the file was found on disk: this is not a missing import\n  to import its contents or its URL, use an explicit loader query: `./{name}?raw` or `./{name}?url`"
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_assets_and_source_gaps() {
        assert!(is_asset_path(Path::new("logo.svg")));
        assert!(unhandled_source(Path::new("component.tsx")).is_none());
        assert_eq!(
            unhandled_source(Path::new("addon.node")),
            Some(UnhandledSource::NativeAddon)
        );
        assert!(matches!(
            unhandled_source(Path::new("page.astro")),
            Some(UnhandledSource::NeedsCompiler { .. })
        ));
    }
}
