//! Bridges our JSIR-schema AST (jsir_ast::Node) to swc's AST for code
//! generation (ast2source) and parsing (source2ast). We don't reproduce any
//! particular tool's text output; the bar is semantic sameness (the emitted JS
//! re-parses and lowers to the same IR).

mod from_swc;
pub mod jsx;
pub mod instrument;
mod to_swc;
pub use from_swc::source_to_ir;
pub use to_swc::to_program;

use swc_common::{sync::Lrc, FileName, SourceMap};
use swc_ecma_ast::{EsVersion, Program};
use swc_ecma_codegen::{text_writer::JsWriter, Config, Emitter};
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

/// Parse JS source into an swc `Program`.
pub fn parse(src: &str) -> Result<Program, String> {
    parse_with_comments(src).map(|(p, _)| p)
}

/// Parse, also returning a comment store keyed by the parsed AST's byte
/// positions (so a `codegen_with_comments` re-emits comments at the right spot).
pub fn parse_with_comments(
    src: &str,
) -> Result<(Program, swc_common::comments::SingleThreadedComments), String> {
    use swc_common::comments::SingleThreadedComments;
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(Lrc::new(FileName::Anon), src.to_string());
    let comments = SingleThreadedComments::default();
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::EsNext,
        StringInput::from(&*fm),
        Some(&comments),
    );
    let mut parser = Parser::new_from(lexer);
    let program = parser
        .parse_program()
        .map_err(|e| format!("swc parse error: {e:?}"))?;
    Ok((program, comments))
}

/// ast2source: our JSIR AST -> swc AST -> JS source.
pub fn ast2source(file: &jsir_ast::Node) -> Result<String, String> {
    let comments = to_swc::build_comments(file);
    Ok(codegen_with_comments(&to_program(file)?, Some(&comments)))
}

/// ir2source: JSIR IR -> AST (via `hir2ast`) -> swc AST -> JS source. This is
/// the output half of the round trip (`hir2ast` builds through the trait, so the
/// AST is just an internal hand-off to swc's code generator).
pub fn ir_to_source(op: &jsir_ir::Op) -> Result<String, String> {
    let node = jsir_convert::hir2ast(op).map_err(|e| format!("hir2ast: {e}"))?;
    ast2source(&node)
}

/// Emit an swc `Program` as JS source via swc's code generator.
pub fn codegen(program: &Program) -> String {
    codegen_with_comments(program, None)
}

/// Emit an swc `Program` as JS, optionally re-emitting comments from a
/// `BytePos`-keyed store (built by `to_swc::build_comments`).
pub fn codegen_with_comments(
    program: &Program,
    comments: Option<&swc_common::comments::SingleThreadedComments>,
) -> String {
    let cm: Lrc<SourceMap> = Default::default();
    let mut buf = Vec::new();
    {
        let wr = JsWriter::new(cm.clone(), "\n", &mut buf, None);
        let mut emitter = Emitter {
            cfg: Config::default(),
            cm: cm.clone(),
            comments: comments.map(|c| c as &dyn swc_common::comments::Comments),
            wr,
        };
        emitter.emit_program(program).expect("codegen");
    }
    String::from_utf8(buf).expect("utf8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swc_parse_then_codegen_roundtrips() {
        let program = parse("var x = 1;").expect("parse");
        let out = codegen(&program);
        assert!(out.contains("var x = 1"), "unexpected codegen: {out:?}");
    }
}

#[cfg(test)]
mod corpus {
    use super::*;

    /// Self-consistency gate for ast2source: the JS we emit must be valid and
    /// stable -- parsing it and re-emitting yields identical text. Proves the
    /// generated source is well-formed and structurally unambiguous, without
    /// depending on any particular formatter. (Full IR semantic round-trip
    /// arrives with the swc->JSIR parse direction.)
    #[test]
    fn ast2source_self_consistent() {
        let mut ok = Vec::new();
        let mut unsupported = Vec::new();
        let mut unstable = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            let Some(ast) = f.expected_ast_json() else { continue };
            let value: serde_json::Value = serde_json::from_str(&ast).unwrap();
            let node = jsir_ast::Node::from_json(&value).unwrap();
            match ast2source(&node) {
                Err(_) => unsupported.push(f.name.clone()),
                Ok(js1) => {
                    let js2 = parse_with_comments(&js1)
                        .map(|(p, c)| codegen_with_comments(&p, Some(&c)));
                    match js2 {
                        Ok(js2) if js2 == js1 => ok.push(f.name.clone()),
                        _ => unstable.push(f.name.clone()),
                    }
                }
            }
        }
        eprintln!(
            "ast2source: {} self-consistent, {} unsupported {:?}, {} unstable {:?}",
            ok.len(), unsupported.len(), unsupported, unstable.len(), unstable
        );
        assert!(unstable.is_empty(), "ast2source produced unstable JS: {unstable:?}");
    }
}
