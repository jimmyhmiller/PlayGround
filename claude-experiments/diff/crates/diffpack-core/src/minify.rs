//! Final JavaScript chunk minification.

use std::path::PathBuf;

use oxc_allocator::Allocator;
use oxc_codegen::{Codegen, CodegenOptions};
use oxc_minifier::{Minifier, MinifierOptions};
use oxc_parser::Parser;
use oxc_span::SourceType;

/// Minifies one finished JavaScript module chunk.
pub fn chunk(code: &str, chunk_name: &str) -> Result<String, String> {
    Ok(chunk_inner(code, chunk_name, false)?.0)
}

/// Minifies a finished chunk and returns its minified-to-readable source map.
pub fn chunk_with_map(
    code: &str,
    chunk_name: &str,
) -> Result<(String, oxc_sourcemap::SourceMap<'static>), String> {
    let (minified, map) = chunk_inner(code, chunk_name, true)?;
    let map = map.ok_or_else(|| {
        format!(
            "minify: Oxc codegen returned no source map for chunk `{chunk_name}` despite \
             source-map output being requested"
        )
    })?;
    Ok((minified, map))
}

fn chunk_inner(
    code: &str,
    chunk_name: &str,
    want_map: bool,
) -> Result<(String, Option<oxc_sourcemap::SourceMap<'static>>), String> {
    let allocator = Allocator::default();
    let parsed = Parser::new(&allocator, code, SourceType::default().with_module(true)).parse();
    if parsed.panicked || !parsed.diagnostics.is_empty() {
        let detail = parsed
            .diagnostics
            .first()
            .map(ToString::to_string)
            .unwrap_or_else(|| "parser panicked".to_string());
        return Err(format!(
            "minify: cannot parse generated chunk `{chunk_name}` for minification: {detail}"
        ));
    }
    let mut program = parsed.program;
    let minified = Minifier::new(MinifierOptions::default()).minify(&allocator, &mut program);
    let mut options = CodegenOptions::minify();
    if want_map {
        options.source_map_path = Some(PathBuf::from(chunk_name));
    }
    let mut codegen = Codegen::new().with_options(options);
    if let Some(scoping) = minified.scoping {
        codegen = codegen.with_scoping(Some(scoping));
    }
    let printed = codegen.build(&program);
    Ok((printed.code, printed.map.map(|map| map.into_owned())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minifies_a_finished_module_and_can_map_it() {
        let source = "const longName = 1 + 2; export { longName };\n";
        let plain = chunk(source, "entry.mjs").unwrap();
        let (mapped, map) = chunk_with_map(source, "entry.mjs").unwrap();
        assert_eq!(plain, mapped);
        assert!(mapped.len() < source.len());
        assert!(map.get_tokens().next().is_some());
    }

    #[test]
    fn invalid_generated_javascript_names_the_chunk() {
        let error = chunk("export {", "broken.mjs").unwrap_err();
        assert!(error.contains("broken.mjs"));
    }
}
