use iongraph_rust_redux::compilers::ion::schema::IonJSON;
use iongraph_rust_redux::{render_ion_pass, render_svg_from_json, GraphBuilder};
use iongraph_rust_redux::{Render, ToGraph, GraphBlock, GraphInstruction};
use std::fs;
use std::path::PathBuf;

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ion-examples")
}

// =============================================================================
// Tests using the new high-level API
// =============================================================================

#[test]
fn test_simple_add() {
    let path = fixtures_path().join("simple-add.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let svg = render_ion_pass(&json_str, 0, 0).expect("Failed to render");

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""));
    assert!(svg.len() > 100, "SVG should have content");
}

#[test]
fn test_fibonacci() {
    let path = fixtures_path().join("fibonacci.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let svg = render_ion_pass(&json_str, 0, 0).expect("Failed to render");

    assert!(svg.starts_with("<svg"));
    assert!(svg.len() > 100);
}

#[test]
fn test_branching() {
    let path = fixtures_path().join("branching.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let svg = render_ion_pass(&json_str, 0, 0).expect("Failed to render");

    assert!(svg.starts_with("<svg"));
    assert!(svg.len() > 100);
}

#[test]
fn test_while_loop() {
    let path = fixtures_path().join("while-loop.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let svg = render_ion_pass(&json_str, 0, 0).expect("Failed to render");

    assert!(svg.starts_with("<svg"));
    assert!(svg.len() > 100);
}

#[test]
fn test_try_catch() {
    let path = fixtures_path().join("try-catch.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let svg = render_ion_pass(&json_str, 0, 0).expect("Failed to render");

    assert!(svg.starts_with("<svg"));
    assert!(svg.len() > 100);
}

#[test]
fn test_mega_complex_all_functions() {
    let path = fixtures_path().join("mega-complex.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str)
        .expect("Failed to parse JSON");

    // Test all functions at pass 0
    for func_idx in 0..data.functions.len() {
        let svg = render_ion_pass(&json_str, func_idx, 0)
            .expect(&format!("Failed to render function {}", func_idx));
        assert!(
            svg.starts_with("<svg"),
            "Function {} should render valid SVG",
            func_idx
        );
        assert!(
            svg.len() > 100,
            "Function {} SVG should have content",
            func_idx
        );
    }
}

#[test]
fn test_mega_complex_all_passes() {
    let path = fixtures_path().join("mega-complex.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str)
        .expect("Failed to parse JSON");

    // Test function 0 with all passes
    let func = &data.functions[0];
    for pass_idx in 0..func.passes.len() {
        let svg = render_ion_pass(&json_str, 0, pass_idx)
            .expect(&format!("Failed to render pass {}", pass_idx));
        assert!(
            svg.starts_with("<svg"),
            "Pass {} should render valid SVG",
            pass_idx
        );
    }
}

#[test]
fn test_all_ion_examples() {
    let fixtures_dir = fixtures_path();
    let entries = fs::read_dir(&fixtures_dir).expect("Failed to read fixtures directory");

    let mut count = 0;
    for entry in entries {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

            // Try to parse and render using the simple API
            if let Ok(svg) = render_svg_from_json(&json_str) {
                assert!(
                    svg.starts_with("<svg"),
                    "File {:?} should render valid SVG",
                    path.file_name()
                );
                count += 1;
            }
        }
    }

    assert!(count > 30, "Should have tested at least 30 fixtures, got {}", count);
}

// =============================================================================
// Tests for GraphBuilder API
// =============================================================================

#[test]
fn test_graph_builder_simple_block() {
    let svg = GraphBuilder::new("test")
        .block("entry")
            .instruction("const", Some("i32"))
            .instruction("return", None)
        .render_svg();

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("const"));
    assert!(svg.contains("return"));
}

#[test]
fn test_graph_builder_diamond() {
    // Build a diamond-shaped CFG:
    //       entry
    //      /     \
    //   left    right
    //      \     /
    //        exit
    let svg = GraphBuilder::new("test")
        .block("entry")
            .instruction("branch", None)
        .block("left")
            .instruction("add", Some("i32"))
        .block("right")
            .instruction("sub", Some("i32"))
        .block("exit")
            .instruction("phi", Some("i32"))
            .instruction("return", None)
        .edge("entry", "left")
        .edge("entry", "right")
        .edge("left", "exit")
        .edge("right", "exit")
        .render_svg();

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("branch"));
    assert!(svg.contains("phi"));
}

#[test]
fn test_graph_builder_loop() {
    // Build a simple loop:
    //   entry -> header -> body -> header (back edge)
    //              |
    //              v
    //            exit
    let svg = GraphBuilder::new("test")
        .block("entry")
            .instruction("const", Some("i32"))
        .block("header")
            .attr("loopheader")
            .loop_depth(1)
            .instruction("phi", Some("i32"))
            .instruction("cmp", Some("bool"))
        .block("body")
            .loop_depth(1)
            .instruction("add", Some("i32"))
        .block("exit")
            .instruction("return", None)
        .edge("entry", "header")
        .edge("header", "body")
        .edge("header", "exit")
        .back_edge("body", "header")
        .render_svg();

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("phi"));
}

#[test]
fn test_graph_builder_complex() {
    // A more complex graph with multiple blocks
    let svg = GraphBuilder::new("complex-ir")
        .block("entry")
            .instruction("alloca", Some("i32*"))
            .instruction("store", None)
        .block("check")
            .instruction("load", Some("i32"))
            .instruction("icmp", Some("bool"))
        .block("body")
            .instruction("add", Some("i32"))
        .block("exit")
            .instruction("ret", Some("i32"))
        .edge("entry", "check")
        .edge("check", "body")
        .edge("check", "exit")
        .edge("body", "exit")
        .render_svg();

    assert!(svg.starts_with("<svg"));
    assert!(svg.len() > 500); // Complex graph should produce substantial SVG
}

#[test]
fn test_graph_builder_instruction_attributes() {
    let svg = GraphBuilder::new("test")
        .block("b0")
            .instruction_with_attrs("call", Some("void"), &["may_throw", "no_inline"])
            .instruction("return", None)
        .render_svg();

    assert!(svg.starts_with("<svg"));
}

#[test]
fn test_render_svg_from_json_auto_detect() {
    let path = fixtures_path().join("simple-add.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    // render_svg_from_json auto-detects IonJSON format
    let svg = render_svg_from_json(&json_str).expect("Failed to render");

    assert!(svg.starts_with("<svg"));
}

#[test]
fn test_render_ion_pass_bounds_check() {
    let path = fixtures_path().join("simple-add.json");
    let json_str = fs::read_to_string(&path).expect("Failed to read fixture");

    // Should fail with out of bounds function index
    let result = render_ion_pass(&json_str, 999, 0);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("out of range"));

    // Should fail with out of bounds pass index
    let result = render_ion_pass(&json_str, 0, 999);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("out of range"));
}

// =============================================================================
// Trait-based API integration tests
// =============================================================================

// Example: A simple SSA-style IR
struct SsaInstr {
    op: String,
    result_type: Option<String>,
}

impl GraphInstruction for SsaInstr {
    fn opcode(&self) -> &str { &self.op }
    fn type_annotation(&self) -> Option<&str> { self.result_type.as_deref() }
}

struct SsaBlock {
    label: String,
    phis: Vec<SsaInstr>,
    body: Vec<SsaInstr>,
    terminator: SsaInstr,
    targets: Vec<String>,
}

impl GraphBlock for SsaBlock {
    type Instruction = SsaInstr;

    fn id(&self) -> &str { &self.label }

    fn successors(&self) -> Vec<String> { self.targets.clone() }

    fn instructions(&self) -> Vec<&Self::Instruction> {
        self.phis.iter()
            .chain(self.body.iter())
            .chain(std::iter::once(&self.terminator))
            .collect()
    }
}

struct SsaFunction {
    name: String,
    blocks: Vec<SsaBlock>,
}

impl ToGraph for SsaFunction {
    type Block = SsaBlock;
    fn compiler(&self) -> &str { "ssa-ir" }
    fn blocks(&self) -> Vec<&Self::Block> { self.blocks.iter().collect() }
}

#[test]
fn test_trait_api_ssa_function() {
    let func = SsaFunction {
        name: "factorial".to_string(),
        blocks: vec![
            SsaBlock {
                label: "entry".to_string(),
                phis: vec![],
                body: vec![
                    SsaInstr { op: "const 1".to_string(), result_type: Some("i64".to_string()) },
                    SsaInstr { op: "const 0".to_string(), result_type: Some("i64".to_string()) },
                ],
                terminator: SsaInstr { op: "br".to_string(), result_type: None },
                targets: vec!["loop".to_string()],
            },
            SsaBlock {
                label: "loop".to_string(),
                phis: vec![
                    SsaInstr { op: "phi".to_string(), result_type: Some("i64".to_string()) },
                    SsaInstr { op: "phi".to_string(), result_type: Some("i64".to_string()) },
                ],
                body: vec![
                    SsaInstr { op: "mul".to_string(), result_type: Some("i64".to_string()) },
                    SsaInstr { op: "add".to_string(), result_type: Some("i64".to_string()) },
                    SsaInstr { op: "icmp slt".to_string(), result_type: Some("i1".to_string()) },
                ],
                terminator: SsaInstr { op: "br cond".to_string(), result_type: None },
                // Note: back edges should go through back_edges(), not successors()
                // For this simple test, we just show the exit path
                targets: vec!["exit".to_string()],
            },
            SsaBlock {
                label: "exit".to_string(),
                phis: vec![],
                body: vec![],
                terminator: SsaInstr { op: "ret".to_string(), result_type: Some("i64".to_string()) },
                targets: vec![],
            },
        ],
    };

    // Use the Render trait
    let svg = func.render_svg();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("phi"));
    assert!(svg.contains("mul"));

    // Can also get the IR first
    let ir = func.to_universal_ir();
    assert_eq!(ir.blocks.len(), 3);
    assert_eq!(ir.blocks[0].successors, vec!["loop"]);
}
