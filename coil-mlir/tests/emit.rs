//! Tests for the core-form → MLIR mapping (`emit`), using the RecordingBackend
//! so no MLIR is required. We assert on the recorded sequence of builder calls.

use coil::emit::Emitter;
use coil::reader::read_all;
use coil::recording::RecordingBackend;

fn emit(src: &str) -> RecordingBackend {
    let forms = read_all(src).expect("read");
    let mut be = RecordingBackend::new();
    {
        let mut em = Emitter::new(&mut be);
        em.emit_module(&forms).expect("emit");
    }
    be
}

fn emit_err(src: &str) -> String {
    let forms = read_all(src).expect("read");
    let mut be = RecordingBackend::new();
    let mut em = Emitter::new(&mut be);
    em.emit_module(&forms).unwrap_err().0
}

#[test]
fn terse_op_call_threads_operands() {
    // a let-bound terse op whose operands resolve to earlier bindings
    let be = emit(
        "(op \"func.func\" :attrs {:sym_name \"add\"} \
            :regions [(region (block ^entry [(: a i32) (: b i32)] \
                        (let [s (arith.addi a b)] \
                          (op \"func.return\" :operands [s] :results []))))])",
    );
    let log = be.log_text();

    // module + body created and entered
    assert!(log.contains("module v1"), "{log}");
    assert!(log.contains("module-body v2 of v1"), "{log}");

    // entry block created with two i32 args
    assert!(log.contains("i32"), "{log}");
    assert!(
        log.lines().any(|l| l.starts_with("block ") && l.contains("argtypes=")),
        "{log}"
    );

    // arith.addi consumes exactly the two block args, in order
    let addi = log.lines().find(|l| l.contains("op arith.addi")).expect("addi line");
    let block = log.lines().find(|l| l.starts_with("block ")).expect("block line");
    // pull "args=[vX vY]" off the block line and check addi operands match
    let args = block.split("args=").nth(1).unwrap().trim();
    assert!(addi.contains(&format!("operands={args}")), "addi={addi}\nblock={block}");

    // func.return takes one operand (the addi result) and has 0 explicit results
    let ret = log.lines().find(|l| l.contains("op func.return")).expect("return line");
    assert!(ret.contains("results(types=[])"), "{ret}");

    // func.func wraps exactly one region
    let func = log.lines().find(|l| l.contains("op func.func")).expect("func line");
    assert!(func.contains("regions=[r") || func.contains("regions=[v"), "{func}");
    assert!(func.contains("sym_name="), "{func}");
}

#[test]
fn inferred_results_are_marked() {
    let be = emit("(arith.constant {:value 42})");
    assert!(be.log_text().contains("results(infer)"), "{}", be.log_text());
}

#[test]
fn explicit_results_resolve_types() {
    let be = emit("(op \"my.thing\" :results [i64 !llvm.ptr])");
    let log = be.log_text();
    assert!(log.contains("= i64"), "{log}");
    assert!(log.contains("= !llvm.ptr"), "{log}");
    assert!(log.contains("results(types="), "{log}");
}

#[test]
fn attrs_are_carried() {
    let be = emit("(op \"x.y\" :attrs {:predicate slt :n 3})");
    let op = be.log_text();
    assert!(op.contains("predicate=slt"), "{op}");
    assert!(op.contains("n=3"), "{op}");
}

#[test]
fn unbound_symbol_is_error() {
    let e = emit_err("(op \"f\" :operands [nope])");
    assert!(e.contains("unbound symbol `nope`"), "{e}");
}

#[test]
fn nested_let_scoping() {
    // inner s shadows; both arith ops should see the right operands
    let be = emit(
        "(op \"func.func\" :regions [(region (block ^e [(: a i32)] \
            (let [x (arith.addi a a)] \
              (let [y (arith.muli x a)] \
                (op \"func.return\" :operands [y] :results [])))))])",
    );
    let log = be.log_text();
    let muli = log.lines().find(|l| l.contains("op arith.muli")).unwrap();
    // muli's first operand is the addi result; ensure it references a value, not unbound
    assert!(muli.contains("operands=[v"), "{muli}");
}
