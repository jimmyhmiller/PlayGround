//! Parse real JavaScript with SWC and lower the supported subset to the
//! `partial::js` AST so it can be partially evaluated by the generic engine.

use swc_common::sync::Lrc;
use swc_common::{FileName, SourceMap};
use swc_ecma_ast::{EsVersion, Module};
use swc_ecma_parser::lexer::Lexer;
use swc_ecma_parser::{Parser, StringInput, Syntax};

use partial::js::{Cond, DeepVal, Js, Op};
use partial::residual::Program;

mod codegen;
mod lower;
pub use lower::compile;

/// Parse + lower + specialize `src`, then emit the residual as JavaScript text.
pub fn to_js(src: &str) -> Result<String, String> {
    let funcs = compile(src)?;
    let vm = Js::new(&funcs);
    let mut prog = partial::engine::specialize(&vm, vm.start());
    partial::residual::simplify(&mut prog);
    Ok(codegen::program_to_js(&vm, &prog, vm.input_var()))
}

/// Parse + lower + specialize a JS source file. Returns the built `Js` client
/// and the simplified residual program.
pub fn specialize(src: &str) -> Result<(Js, Program<Op, Cond>), String> {
    let funcs = compile(src)?;
    let vm = Js::new(&funcs);
    let mut prog = partial::engine::specialize(&vm, vm.start());
    partial::residual::simplify(&mut prog);
    Ok((vm, prog))
}

/// Run the residual program produced from `src` on a single dynamic input.
pub fn run_residual(src: &str, input: i64) -> Result<DeepVal, String> {
    let (vm, prog) = specialize(src)?;
    Ok(vm.run_residual(&prog, input))
}

/// Run the (subset) reference interpreter on `src` for a single input.
pub fn run_reference(src: &str, input: i64) -> Result<DeepVal, String> {
    let funcs = compile(src)?;
    let vm = Js::new(&funcs);
    Ok(vm.run_reference(input))
}

/// Parse a JS source string into an SWC module, or a human-readable error.
pub fn parse(src: &str) -> Result<Module, String> {
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(Lrc::new(FileName::Custom("input.js".into())), src.to_string());
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::EsNext,
        StringInput::from(&*fm),
        None,
    );
    let mut parser = Parser::new_from(lexer);
    parser
        .parse_module()
        .map_err(|e| format!("parse error: {:?}", e.kind()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use partial::js::DeepVal;

    fn check(src: &str, inputs: &[i64]) -> Program<Op, Cond> {
        let (vm, prog) = specialize(src).expect("specialize");
        for &i in inputs {
            assert_eq!(
                vm.run_reference(i),
                vm.run_residual(&prog, i),
                "residual diverged from reference at input={i}\nsrc:\n{src}"
            );
        }
        prog
    }

    #[test]
    fn parses_a_function() {
        let m = parse("function main(input) { return input + 1; }").unwrap();
        assert_eq!(m.body.len(), 1);
    }

    #[test]
    fn arithmetic_folds() {
        let prog = check("function main(x) { return (x + 3) * 2; }", &[0, 5, -4]);
        assert_eq!(prog.blocks.len(), 1);
    }

    #[test]
    fn for_loop_over_static_array_unrolls() {
        let src = "
            function main(x) {
                let xs = [x, x + 1, 7];
                let s = 0;
                for (let i = 0; i < xs.length; i = i + 1) {
                    s = s + xs[i];
                }
                return s;
            }";
        // sum = x + (x+1) + 7 = 2x + 8
        let prog = check(src, &[0, 3, 10]);
        assert_eq!(run_residual(src, 3).unwrap(), DeepVal::Num(2 * 3 + 8));
        assert_eq!(prog.blocks.len(), 1, "static loop should fully unroll");
    }

    #[test]
    fn switch_dispatch_folds() {
        let src = "
            function op(kind, a, b) {
                switch (kind) {
                    case 0: return a + b;
                    case 1: return a - b;
                    case 2: return a * b;
                    default: return 0;
                }
            }
            function main(x) { return op(2, x, 3) + op(0, x, 1); }";
        // x*3 + (x+1)
        let prog = check(src, &[0, 4, 9]);
        assert_eq!(prog.blocks.len(), 1, "static switch kinds should fold away");
    }

    #[test]
    fn higher_order_with_lifted_arrow() {
        let src = "
            function map(xs, f) {
                let out = [];
                for (let i = 0; i < xs.length; i = i + 1) {
                    out.push(f(xs[i]));
                }
                return out;
            }
            function main(x) { return map([x, x + 1, 7], (v) => v * 2); }";
        let prog = check(src, &[0, 3, 9]);
        assert_eq!(
            run_residual(src, 3).unwrap(),
            DeepVal::Array(vec![DeepVal::Num(6), DeepVal::Num(8), DeepVal::Num(14)])
        );
        let _ = prog;
    }

    /// The Futamura projection on REAL JavaScript: an expression interpreter,
    /// written as ordinary JS and dispatching with `switch`, specialized against
    /// a static AST with a dynamic environment. The interpreter and AST vanish.
    #[test]
    fn futamura_expression_interpreter() {
        let src = r#"
            function evalNode(node, env) {
                switch (node.op) {
                    case "lit": return node.val;
                    case "var": return env[node.name];
                    case "add": return evalNode(node.l, env) + evalNode(node.r, env);
                    case "mul": return evalNode(node.l, env) * evalNode(node.r, env);
                    default: return 0;
                }
            }
            function main(x) {
                // AST for (x + 3) * (x + x)
                let ast = {
                    op: "mul",
                    l: { op: "add", l: { op: "var", name: "x" }, r: { op: "lit", val: 3 } },
                    r: { op: "add", l: { op: "var", name: "x" }, r: { op: "var", name: "x" } }
                };
                let env = { x: x };
                return evalNode(ast, env);
            }"#;
        let prog = check(src, &[-3, 0, 1, 7, 42]);
        assert_eq!(
            prog.blocks.len(),
            1,
            "interpreter + AST did not fully specialize away"
        );
        // residual computes (x + 3) * (x + x)
        for x in [-3i64, 0, 1, 7, 42] {
            assert_eq!(run_residual(src, x).unwrap(), DeepVal::Num((x + 3) * (x + x)));
        }
    }

    /// A Brainfuck interpreter written in real JS, specialized against a static
    /// BF program (with the dynamic input fed into cell 0 via `,`).
    // A Brainfuck interpreter in real JS, fed a pre-parsed BF AST (so the JS
    // stays in-subset: no string indexing). The program
    // ",++++++.>+++++++++++++[->+++++<]>." reads input, +6, prints; then computes
    // 65 with a static loop and prints it.
    const BRAINFUCK_SRC: &str = r#"
            function exec(node, tape, ptr, out, input, inptr) {
                switch (node.op) {
                    case "add": { tape[ptr] = tape[ptr] + node.n; break; }
                    case "move": { ptr = ptr + node.n; break; }
                    case "out": { out.push(tape[ptr]); break; }
                    case "in": { tape[ptr] = input[inptr]; inptr = inptr + 1; break; }
                    case "loop": {
                        while (tape[ptr] !== 0) {
                            let r = exec(node.seq, tape, ptr, out, input, inptr);
                            ptr = r[0]; inptr = r[1];
                        }
                        break;
                    }
                    case "seq": {
                        let i = 0;
                        while (i < node.body.length) {
                            let r = exec(node.body[i], tape, ptr, out, input, inptr);
                            ptr = r[0]; inptr = r[1];
                            i = i + 1;
                        }
                        break;
                    }
                }
                return [ptr, inptr];
            }
            function main(x) {
                let tape = [0, 0, 0, 0, 0, 0, 0, 0];
                let out = [];
                let input = [x];
                let program = {
                    op: "seq",
                    body: [
                        { op: "in" },
                        { op: "add", n: 6 },
                        { op: "out" },
                        { op: "move", n: 1 },
                        { op: "add", n: 13 },
                        { op: "loop", seq: { op: "seq", body: [
                            { op: "add", n: -1 },
                            { op: "move", n: 1 },
                            { op: "add", n: 5 },
                            { op: "move", n: -1 }
                        ]}},
                        { op: "move", n: 1 },
                        { op: "out" }
                    ]
                };
                let r = exec(program, tape, 0, out, input, 0);
                return out;
            }"#;

    #[test]
    fn futamura_brainfuck() {
        let src = BRAINFUCK_SRC;
        let prog = check(src, &[0, 1, 5, 65, 200]);
        assert_eq!(prog.blocks.len(), 1, "BF interpreter did not specialize away");
        for x in [0i64, 1, 5, 65, 200] {
            assert_eq!(
                run_residual(src, x).unwrap(),
                DeepVal::Array(vec![DeepVal::Num(x + 6), DeepVal::Num(65)])
            );
        }
    }

    // ---- JS code generation ----

    /// The emitted JS must at least be syntactically valid (re-parseable).
    fn assert_valid_js(js: &str) {
        parse(js).unwrap_or_else(|e| panic!("emitted invalid JS: {e}\n--- js ---\n{js}"));
    }

    /// Serialize a DeepVal the way `JSON.stringify` would, for node comparison.
    fn dv_json(v: &DeepVal) -> String {
        match v {
            DeepVal::Num(n) => n.to_string(),
            DeepVal::Bool(b) => b.to_string(),
            DeepVal::Str(s) => format!("{s:?}"),
            DeepVal::Undef => "null".to_string(), // JSON.stringify(undefined inside) -> null
            DeepVal::Null => "null".to_string(),
            DeepVal::Array(xs) => {
                let parts: Vec<String> = xs.iter().map(dv_json).collect();
                format!("[{}]", parts.join(","))
            }
            DeepVal::Object(fs) => {
                let parts: Vec<String> =
                    fs.iter().map(|(k, v)| format!("{k:?}:{}", dv_json(v))).collect();
                format!("{{{}}}", parts.join(","))
            }
            DeepVal::Closure(_) => "\"<closure>\"".to_string(),
        }
    }

    /// Run emitted JS under node (if present) and compare to the reference for
    /// each input. Returns false if node is unavailable (test then only checks
    /// syntactic validity).
    fn run_under_node(js: &str, vm: &Js, inputs: &[i64]) -> bool {
        use std::process::Command;
        if Command::new("node").arg("--version").output().is_err() {
            return false;
        }
        for &inp in inputs {
            let harness = format!("{js}\nprocess.stdout.write(JSON.stringify(main({inp})));\n");
            let out = Command::new("node")
                .arg("-e")
                .arg(&harness)
                .output()
                .expect("spawn node");
            assert!(
                out.status.success(),
                "node failed on input {inp}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let got = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let want = dv_json(&vm.run_reference(inp));
            assert_eq!(got, want, "node output diverged from reference at input={inp}\njs:\n{js}");
        }
        true
    }

    #[test]
    fn emits_clean_straight_line_js() {
        let js = to_js("function main(x) { return (x + 3) * 2; }").unwrap();
        assert!(js.contains("return ((v0 + 3) * 2);"), "got:\n{js}");
        assert!(!js.contains("switch"), "straight-line should not use a trampoline:\n{js}");
        assert_valid_js(&js);
    }

    #[test]
    fn emits_and_runs_loop_js() {
        let src = "
            function main(n) {
                let s = 0;
                for (let i = 0; i < n; i = i + 1) { s = s + i; }
                return s;
            }";
        let js = to_js(src).unwrap();
        assert_valid_js(&js);
        let funcs = compile(src).unwrap();
        let vm = Js::new(&funcs);
        let ran = run_under_node(&js, &vm, &[0, 1, 5, 20]);
        assert!(ran, "node should be available in this environment");
    }

    #[test]
    fn emits_and_runs_futamura_interpreter_js() {
        let src = r#"
            function evalNode(node, env) {
                switch (node.op) {
                    case "lit": return node.val;
                    case "var": return env[node.name];
                    case "add": return evalNode(node.l, env) + evalNode(node.r, env);
                    case "mul": return evalNode(node.l, env) * evalNode(node.r, env);
                    default: return 0;
                }
            }
            function main(x) {
                let ast = { op: "mul",
                    l: { op: "add", l: { op: "var", name: "x" }, r: { op: "lit", val: 3 } },
                    r: { op: "add", l: { op: "var", name: "x" }, r: { op: "var", name: "x" } } };
                return evalNode(ast, { x: x });
            }"#;
        let js = to_js(src).unwrap();
        assert!(js.contains("return ((v0 + 3) * (v0 + v0));"), "got:\n{js}");
        assert_valid_js(&js);
        let vm = Js::new(&compile(src).unwrap());
        run_under_node(&js, &vm, &[-3, 0, 7, 42]);
    }

    #[test]
    fn emits_and_runs_brainfuck_js() {
        let src = BRAINFUCK_SRC;
        let js = to_js(src).unwrap();
        assert_valid_js(&js);
        let vm = Js::new(&compile(src).unwrap());
        run_under_node(&js, &vm, &[0, 1, 5, 65, 200]);
    }

    #[test]
    fn clear_errors_for_unsupported() {
        let err = |src: &str| specialize(src).err().expect("expected an error");
        // no main and no top-level code
        assert!(compile("function f(){}").err().unwrap().contains("main"));
        // still unsupported statements / control flow:
        assert!(err("function main(x){ for (const y of [x]) { return y; } }").contains("for.."));
        assert!(err("function main(x){ do { x = x + 1; } while (x < 3); return x; }").contains("do"));
    }

    /// Closures that capture outer variables now work (capture by value): a
    /// top-level `adder` returns a closure capturing `n`.
    #[test]
    fn capturing_closure_works() {
        let src = "
            function adder(n) { return function (y) { return y + n; }; }
            function main(x) {
                var add5 = adder(5);
                return add5(x);   // x + 5
            }";
        let (_, prog) = specialize(src).unwrap();
        assert_eq!(prog.blocks.len(), 1, "closure should fully inline:\n{}", to_js(src).unwrap());
        assert_node_equiv(src, &[0, 3, -2]);
    }

    // ---- pass-through (opaque) calls — Increment 2, validated by Node ----

    /// A call to an unmodeled global function passes through; its operands still
    /// specialize.
    #[test]
    fn passthrough_global_call() {
        let src = r#"function main(x) { return parseInt("42") + x; }"#;
        let js = to_js(src).unwrap();
        assert!(js.contains(r#"parseInt("42")"#), "got:\n{js}");
        assert_node_equiv(src, &[0, 5, -3, 100]);
    }

    /// A method call on a runtime global (`Math.floor`) passes through, with the
    /// argument (which itself uses the unmodeled `/`) specialized.
    #[test]
    fn passthrough_method_call() {
        let src = "function main(x) { return Math.floor(x / 2); }";
        let js = to_js(src).unwrap();
        assert!(js.contains("Math.floor("), "got:\n{js}");
        assert_node_equiv(src, &[0, 1, 7, 8, 99, -5]);
    }

    /// A method call on a dynamic primitive (the input) passes through.
    #[test]
    fn passthrough_method_on_dynamic() {
        let src = "function main(x) { return x.toString(); }";
        let js = to_js(src).unwrap();
        assert!(js.contains("v0.toString()"), "got:\n{js}");
        assert_node_equiv(src, &[0, 42, -7]);
    }

    /// An effectful call is evaluated *once* into a temp and the temp is reused;
    /// it is never duplicated even when the result is used multiple times.
    #[test]
    fn opaque_call_is_single_eval() {
        let src = "function main(x) { let y = Math.abs(x); return y + y; }";
        let js = to_js(src).unwrap();
        assert_eq!(js.matches("Math.abs(").count(), 1, "call should be evaluated once:\n{js}");
        assert_node_equiv(src, &[-5, 0, 9]);
    }

    /// The payoff for Increment 2: an interpreter whose dispatch and AST vanish,
    /// leaving an unmodeled builtin *call* applied to a specialized operand.
    #[test]
    fn passthrough_call_inside_interpreter() {
        let src = r#"
            function evalNode(node, env) {
                switch (node.op) {
                    case "lit": return node.val;
                    case "var": return env[node.name];
                    case "add": return evalNode(node.l, env) + evalNode(node.r, env);
                    case "sqrt": return Math.sqrt(evalNode(node.l, env));
                    default: return 0;
                }
            }
            function main(x) {
                // sqrt(x + x)
                let ast = { op: "sqrt", l: { op: "add",
                    l: { op: "var", name: "x" }, r: { op: "var", name: "x" } } };
                return evalNode(ast, { x: x });
            }"#;
        let (_, prog) = specialize(src).unwrap();
        assert_eq!(prog.blocks.len(), 1, "interpreter did not specialize away");
        let js = to_js(src).unwrap();
        assert!(js.contains("Math.sqrt("), "the builtin call should pass through:\n{js}");
        assert_node_equiv(src, &[0, 2, 8, 50]);
    }

    // ---- object escape (Increment 3) — validated by Node ----

    /// A tracked object passed to an unmodeled call escapes: it is materialized
    /// into a residual object and handed to the call.
    #[test]
    fn object_escapes_into_call() {
        let src = "function main(x) { let o = { a: x, b: 2 }; return JSON.stringify(o); }";
        let js = to_js(src).unwrap();
        assert!(js.contains("{\"a\": v0, \"b\": 2}"), "object should materialize:\n{js}");
        assert!(js.contains("JSON.stringify("), "got:\n{js}");
        assert_node_equiv(src, &[0, 5, -3]);
    }

    /// A tracked array escapes into a call the same way.
    #[test]
    fn array_escapes_into_call() {
        let src = "function main(x) { let xs = [x, x + 1]; return JSON.stringify(xs); }";
        let js = to_js(src).unwrap();
        assert!(js.contains("[v0, (v0 + 1)]"), "array should materialize:\n{js}");
        assert_node_equiv(src, &[0, 7, -2]);
    }

    /// After an object escapes, reads of its fields residualize (the callee may
    /// have mutated it), so they read the runtime object.
    #[test]
    fn read_after_escape() {
        let src = "function main(x) { let o = { a: x }; let s = JSON.stringify(o); return o.a + 1; }";
        let js = to_js(src).unwrap();
        assert!(js.contains(".a + 1)") || js.contains(".a) + 1)"), "read should be opaque:\n{js}");
        assert_node_equiv(src, &[0, 9, -4]);
    }

    /// After an object escapes, writes to its fields residualize too.
    #[test]
    fn write_after_escape() {
        let src = "function main(x) {
            let o = { a: x };
            let s = JSON.stringify(o);
            o.a = 99;
            return o.a;
        }";
        let js = to_js(src).unwrap();
        assert!(js.contains(".a = 99;"), "write should residualize:\n{js}");
        assert_node_equiv(src, &[1, 5]);
    }

    /// Alias invalidation: two locals refer to the same object; passing one to a
    /// call invalidates the *other* alias too, so reading through it residualizes
    /// (and yields the same runtime object).
    #[test]
    fn alias_invalidated_on_escape() {
        let src = "function main(x) {
            let o = { a: x };
            let p = o;
            let s = JSON.stringify(o);
            return p.a;
        }";
        assert_node_equiv(src, &[0, 8, -1]);
    }

    /// A closure passed to unmodeled code now escapes as a residual function
    /// (no longer an error).
    #[test]
    fn closure_into_call_residualizes() {
        let js = to_js("function main(x) { let f = () => x; return frob(f); }").unwrap();
        assert!(js.contains("function __rf"), "should emit a residual function:\n{js}");
        assert!(js.contains("frob("), "the call should pass through:\n{js}");
    }

    /// Run `{js}; <expr>` under node and return stdout (None if no node).
    fn node_eval_expr(js: &str, expr: &str) -> Option<String> {
        use std::process::Command;
        if Command::new("node").arg("--version").output().is_err() {
            return None;
        }
        let harness = format!("{js}\nprocess.stdout.write(String({expr}));");
        let out = Command::new("node").arg("-e").arg(&harness).output().expect("spawn node");
        assert!(out.status.success(), "node failed on `{expr}`: {}", String::from_utf8_lossy(&out.stderr));
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// A closure that ESCAPES (is returned) becomes a residual function, bound
    /// with its captured value; calling it later yields the right result.
    #[test]
    fn escaping_closure_becomes_residual_function() {
        let src = "
            function adder(n) { return function (y) { return y + n; }; }
            function main(x) { return adder(x); }   // closure escapes (returned)
        ";
        let js = to_js(src).unwrap();
        assert!(js.contains("function __rf0("), "expected a residual function:\n{js}");
        assert!(js.contains(".bind(null, v0)"), "captures should be bound:\n{js}");
        for (x, arg) in [(5, 3), (0, 10), (-2, 7)] {
            match (
                node_eval_expr(src, &format!("main({x})({arg})")),
                node_eval_expr(&js, &format!("main({x})({arg})")),
            ) {
                (Some(o), Some(s)) => assert_eq!(o, s, "diverged at main({x})({arg})\n{js}"),
                _ => return,
            }
        }
    }

    /// A closure stored in an object that escapes also becomes a residual fn.
    #[test]
    fn closure_in_escaped_object() {
        let src = "function main(x) { return { get: function () { return x * 2; } }; }";
        let js = to_js(src).unwrap();
        assert!(js.contains("function __rf0("), "expected a residual function:\n{js}");
        for x in [5, 0, -3] {
            match (
                node_eval_expr(src, &format!("main({x}).get()")),
                node_eval_expr(&js, &format!("main({x}).get()")),
            ) {
                (Some(o), Some(s)) => assert_eq!(o, s, "diverged at main({x}).get()\n{js}"),
                _ => return,
            }
        }
    }

    // ---- pass-through (opaque) operators, validated by Node ----

    /// Run a JS source (defining `main`) under node for one input; `None` if no
    /// node is installed.
    fn node_value(js_defining_main: &str, input: i64) -> Option<String> {
        use std::process::Command;
        if Command::new("node").arg("--version").output().is_err() {
            return None;
        }
        let harness =
            format!("{js_defining_main}\nprocess.stdout.write(JSON.stringify(main({input})));");
        let out = Command::new("node").arg("-e").arg(&harness).output().expect("spawn node");
        assert!(out.status.success(), "node failed: {}", String::from_utf8_lossy(&out.stderr));
        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// The Node oracle for pass-through programs: the emitted (specialized) JS
    /// must agree with the *original* source for every input, both run in Node.
    fn assert_node_equiv(src: &str, inputs: &[i64]) {
        let js = to_js(src).unwrap();
        assert_valid_js(&js);
        for &i in inputs {
            match (node_value(src, i), node_value(&js, i)) {
                (Some(orig), Some(spec)) => assert_eq!(
                    orig, spec,
                    "specialized output diverged from the original at input={i}\n\
                     original: {orig}\n--- residual ---\n{js}"
                ),
                _ => return, // node unavailable: assert_valid_js already ran
            }
        }
    }

    #[test]
    fn passthrough_mod_then_fold() {
        let src = "function main(x) { return (x % 10) + 1; }";
        let js = to_js(src).unwrap();
        assert!(js.contains("((v0 % 10) + 1)"), "got:\n{js}");
        assert_node_equiv(src, &[0, 7, 10, 13, 99, -4]);
    }

    #[test]
    fn passthrough_bitwise_and_ternary() {
        let src = "function main(x) { return (x & 1) === 0 ? 100 : 200; }";
        let js = to_js(src).unwrap();
        assert!(js.contains('?') && js.contains('&'), "got:\n{js}");
        assert_node_equiv(src, &[0, 1, 2, 3, 8, 15]);
    }

    #[test]
    fn passthrough_logical_and() {
        let src = "function main(x) { return (x > 0) && (x < 10); }";
        assert_node_equiv(src, &[-5, 0, 5, 9, 10, 100]);
    }

    /// The payoff: an interpreter using an unmodeled operator (`/`). The dispatch
    /// and AST still fully specialize away; only the `/` passes through.
    #[test]
    fn passthrough_inside_interpreter() {
        let src = r#"
            function evalNode(node, env) {
                switch (node.op) {
                    case "lit": return node.val;
                    case "var": return env[node.name];
                    case "add": return evalNode(node.l, env) + evalNode(node.r, env);
                    case "div": return evalNode(node.l, env) / evalNode(node.r, env);
                    default: return 0;
                }
            }
            function main(x) {
                let ast = { op: "div",
                    l: { op: "add", l: { op: "var", name: "x" }, r: { op: "lit", val: 100 } },
                    r: { op: "lit", val: 4 } };
                return evalNode(ast, { x: x });
            }"#;
        let (_, prog) = specialize(src).unwrap();
        assert_eq!(prog.blocks.len(), 1, "interpreter did not specialize away");
        let js = to_js(src).unwrap();
        assert!(js.contains("/ 4)"), "the `/` should pass through into the residual:\n{js}");
        assert_node_equiv(src, &[0, 8, 13, -20]);
    }

    /// A heap object reaching an unmodeled operator escapes too (here string
    /// concatenation coerces the object via its runtime value).
    #[test]
    fn object_escapes_into_operator() {
        let src = r#"function main(x) { let o = { a: x }; return "" + JSON.stringify(o); }"#;
        assert_node_equiv(src, &[0, 3, -5]);
    }

    // ---- try / catch / throw — modeled as control flow, validated by Node ----

    /// A static `throw` caught locally is flattened into straight-line code: the
    /// exception becomes control flow and disappears.
    #[test]
    fn static_throw_caught_folds() {
        let src = "function main(x) { try { throw 7; } catch (e) { return e + x; } }";
        let (_, prog) = specialize(src).unwrap();
        assert_eq!(prog.blocks.len(), 1, "exception control flow should flatten:\n{}", "");
        let js = to_js(src).unwrap();
        assert!(js.contains("return (7 + v0)"), "got:\n{js}");
        assert_node_equiv(src, &[0, 5, -3]);
    }

    /// A conditional `throw` keeps the dynamic branch but still becomes ordinary
    /// control flow (no runtime try/catch in the residual).
    #[test]
    fn conditional_throw() {
        let src = "function main(x) {
            let r = 0;
            try { if (x > 0) { throw 1; } r = 2; } catch (e) { r = e + 10; }
            return r;
        }";
        let js = to_js(src).unwrap();
        assert!(!js.contains("try"), "exceptions should be compiled away:\n{js}");
        assert_node_equiv(src, &[-2, 0, 1, 5]);
    }

    /// A `throw` inside an inlined callee unwinds to the caller's `catch`.
    #[test]
    fn throw_unwinds_inlined_frame() {
        let src = "
            function fail(n) { throw n + 1; }
            function main(x) {
                try { fail(x); return 0; } catch (e) { return e + 100; }
            }";
        let js = to_js(src).unwrap();
        assert!(js.contains("return ((v0 + 1) + 100)"), "got:\n{js}");
        assert_node_equiv(src, &[0, 7, -9]);
    }

    /// `catch` with no binding works.
    #[test]
    fn catch_without_binding() {
        let src = "function main(x) { try { throw 0; } catch { return x + 1; } }";
        assert_node_equiv(src, &[0, 4, -2]);
    }

    /// An uncaught `throw` residualizes a real `throw`.
    #[test]
    fn uncaught_throw_residualizes() {
        let src = "function main(x) { throw x + 1; }";
        let js = to_js(src).unwrap();
        assert!(js.contains("throw (v0 + 1);"), "got:\n{js}");
    }

    /// Soundness boundary: an operation that may throw at runtime (an unmodeled
    /// call) inside a `try` is a hard error in this version.
    #[test]
    #[should_panic(expected = "try/catch")]
    fn opaque_call_in_try_is_an_error() {
        let _ = specialize("function main(x) { try { return Math.floor(x); } catch (e) { return 0; } }");
    }

    // ---- `var` (function-scoped, hoisted) ----

    /// A `var` declared in a `for`-init is function-scoped, so it survives the
    /// loop (here a static loop that unrolls; `i` is 3 afterwards).
    #[test]
    fn var_survives_loop() {
        let src = "function main(x) {
            var total = x;
            for (var i = 0; i < 3; i = i + 1) { total = total + i; }
            return total + i;   // i === 3 after the loop
        }";
        let (_, prog) = specialize(src).unwrap();
        assert_eq!(prog.blocks.len(), 1, "static loop should unroll:\n{}", to_js(src).unwrap());
        assert_node_equiv(src, &[0, 5, -4]); // total = x+3, plus i(=3) => x+6
    }

    /// Redeclaring a `var` refers to the same binding.
    #[test]
    fn var_redeclaration() {
        let src = "function main(x) { var a = x; var a = a + 1; return a; }";
        let js = to_js(src).unwrap();
        assert!(js.contains("return (v0 + 1)"), "redeclared var should reuse the slot:\n{js}");
        assert_node_equiv(src, &[0, 9, -2]);
    }

    /// `new` passes through as a residual constructor call (single-eval), with
    /// args specialized; property reads on the result residualize.
    #[test]
    fn new_passes_through() {
        let src = "function main(x) {
            var d = new Date(x * 1000);   // unmodeled constructor, arg specialized
            return d.getTime();           // method call on the result passes through
        }";
        let js = to_js(src).unwrap();
        assert!(js.contains("new Date((v0 * 1000))"), "got:\n{js}");
        // (Node would run it, but Date is time-dependent; just check structure +
        // that the residual is well-formed JS.)
        assert!(parse(&js).is_ok(), "emitted invalid JS:\n{js}");
    }

    /// `new` with a static arg and a deterministic result, validated by Node.
    #[test]
    fn new_array_passes_through() {
        let src = "function main(x) {
            var a = new Array(3);
            a[0] = x; a[1] = x + 1; a[2] = x + 2;
            return a;
        }";
        let js = to_js(src).unwrap();
        assert!(js.contains("new Array(3)"), "got:\n{js}");
        assert_node_equiv(src, &[0, 5, -1]);
    }

    /// `var` works across a genuinely dynamic loop too.
    #[test]
    fn var_in_dynamic_loop() {
        let src = "function main(n) {
            var sum = 0;
            for (var i = 0; i < n; i = i + 1) { sum = sum + i; }
            return sum;
        }";
        assert_node_equiv(src, &[0, 1, 5, 20]);
    }

    /// `var` hoisting: a variable is usable (as `undefined`) before its
    /// declaration line, and assignments before it refer to the same binding.
    #[test]
    fn var_hoisting_use_before_decl() {
        // `a` read before `var a` -> undefined; then assigned; nested block `var b`
        // is visible after the block (function-scoped).
        let src = "function main(x) {
            var first = a;       // hoisted: undefined
            a = x + 1;
            { var b = a + 1; }   // b is function-scoped
            var a;               // bare redeclare: no-op
            return [first, a, b];
        }";
        let js = to_js(src).unwrap();
        // `a` and `b` resolve to slots, not runtime globals:
        assert!(!js.contains(" a") && !js.contains("(b)"), "vars should not leak as globals:\n{js}");
        assert_node_equiv(src, &[0, 5, -3]); // [undefined->null, x+1, x+2]
    }

    // ---- more pure operator pass-throughs ----

    #[test]
    fn passthrough_in_and_instanceof() {
        let src = r#"function main(x) {
            var o = { k: x };
            return ("k" in o) === true;
        }"#;
        let js = to_js(src).unwrap();
        assert!(js.contains(" in "), "`in` should pass through:\n{js}");
        assert_node_equiv(src, &[0, 7]);
    }

    #[test]
    fn passthrough_void() {
        // `void e` is always undefined; wrap it so node can serialize the result.
        let src = "function main(x) { return typeof void (x + 1); }"; // "undefined"
        let js = to_js(src).unwrap();
        assert!(js.contains("void"), "got:\n{js}");
        assert_node_equiv(src, &[0, 5]);
    }

    // ---- continue ----

    /// `continue` in a `for` must run the update (else it would skip `i+1` and
    /// loop forever / miscount). Sum of even i in [0, n).
    #[test]
    fn continue_in_for_runs_update() {
        let src = "function main(n) {
            var sum = 0;
            for (var i = 0; i < n; i = i + 1) {
                if ((i % 2) !== 0) { continue; }
                sum = sum + i;
            }
            return sum;
        }";
        assert_node_equiv(src, &[0, 1, 5, 8, 11]);
    }

    /// `continue` in a `while` re-tests the condition.
    #[test]
    fn continue_in_while() {
        let src = "function main(n) {
            var i = 0;
            var sum = 0;
            while (i < n) {
                i = i + 1;
                if ((i % 3) === 0) { continue; }
                sum = sum + i;
            }
            return sum;
        }";
        assert_node_equiv(src, &[0, 3, 6, 10]);
    }

    /// `continue` inside a `switch` inside a loop targets the loop, not the switch.
    #[test]
    fn continue_through_switch() {
        let src = "function main(n) {
            var sum = 0;
            for (var i = 0; i < n; i = i + 1) {
                switch (i % 3) {
                    case 0: continue;          // skip multiples of 3
                    default: sum = sum + i;
                }
            }
            return sum;
        }";
        assert_node_equiv(src, &[0, 3, 7, 10]);
    }

    #[test]
    fn continue_outside_loop_is_an_error() {
        let err = specialize("function main(x) { continue; }").err().unwrap();
        assert!(err.contains("continue"), "got: {err}");
    }

    // ---- delete ----

    /// `delete` on a tracked object folds: the field is gone at specialization
    /// time, so a later read is `undefined` and nothing residualizes.
    #[test]
    fn delete_on_tracked_object_folds() {
        let src = r#"function main(x) {
            var o = { a: x, b: 2 };
            delete o.a;
            return [o.a, o.b];   // [undefined, 2]
        }"#;
        let (_, prog) = specialize(src).unwrap();
        assert_eq!(prog.blocks.len(), 1, "delete should fold:\n{}", to_js(src).unwrap());
        let js = to_js(src).unwrap();
        assert!(!js.contains("delete"), "delete on a tracked object should not residualize:\n{js}");
        assert_node_equiv(src, &[0, 7, -3]);
    }

    /// `delete` on an escaped object residualizes a real `delete`.
    #[test]
    fn delete_on_escaped_object_residualizes() {
        let src = r#"function main(x) {
            var o = { a: x, b: 2 };
            var s = JSON.stringify(o);   // o escapes
            delete o.a;
            return [s, JSON.stringify(o)];
        }"#;
        let js = to_js(src).unwrap();
        assert!(js.contains("delete v"), "delete on an escaped object should residualize:\n{js}");
        assert_node_equiv(src, &[0, 5]);
    }

    /// `delete arr[i]` leaves a hole (reads as `undefined`).
    #[test]
    fn delete_index_on_tracked_array() {
        let src = "function main(x) {
            var a = [x, x + 1, x + 2];
            delete a[1];
            return [a[0], a[1], a[2]];   // [x, undefined, x+2]
        }";
        assert_node_equiv(src, &[0, 4, -1]);
    }

    #[test]
    fn delete_as_value_is_an_error() {
        let err = specialize("function main(x) { var o = {a:x}; return delete o.a; }").err().unwrap();
        assert!(err.contains("delete"), "got: {err}");
    }

    // ---- module body / IIFE (no explicit main) ----

    /// A program with top-level `var`s and a non-capturing IIFE, plus helper
    /// functions, specializes as a synthetic `main`.
    #[test]
    fn module_with_iife_and_helpers() {
        let src = "
            function dbl(n) { return n * 2; }
            var base = 10;
            var result = (function () { return dbl(base) + 1; })();
            function main_unused() {}            // a plain helper, hoisted
            // top-level expression statement:
            var out = result + 5;
        ";
        // No explicit main -> synthesized from the top-level statements.
        let funcs = compile(src).unwrap();
        // there is a `main`
        assert!(funcs.iter().any(|f| f.name == "main"));
    }

    /// The synthesized `main` runs top-level code; a module that computes a value
    /// folds. (Uses a global so we can return something defined.)
    #[test]
    fn module_top_level_runs() {
        let src = "
            function add(a, b) { return a + b; }
            var x = add(2, 3);
            var y = (function () { return x * x; })();
            globalThis.__r = y;   // observable top-level effect, passes through
        ";
        let js = to_js(src).unwrap();
        // x and y fold to constants (5 and 25); the global write residualizes.
        assert!(js.contains("globalThis.__r = 25") || js.contains("= 25;"), "got:\n{js}");
    }
}
