use microlang::{BytecodeVm, LowBitModel, Runtime, TreeWalk};

#[test]
fn scheme_define_and_recursion_on_interpreter() {
    let mut rt = Runtime::<LowBitModel>::new();
    let r = scheme::run(&mut rt, &TreeWalk, "(define (sq n) (* n n)) (sq 6)");
    assert_eq!(rt.print(r), "36");
}

#[test]
fn scheme_runs_on_bytecode_jit() {
    let mut rt = Runtime::<LowBitModel>::new();
    let vm = BytecodeVm::<LowBitModel>::new();
    let r = scheme::run(&mut rt, &vm, "(define (fact n) (if (< n 2) 1 (* n (fact (- n 1))))) (fact 6)");
    assert_eq!(rt.print(r), "720");
}

#[test]
fn scheme_let_and_cond() {
    let mut rt = Runtime::<LowBitModel>::new();
    let r = scheme::run(&mut rt, &TreeWalk, "(let ((a 3) (b 4)) (cond ((< a b) (* a a)) (else b)))");
    assert_eq!(rt.print(r), "9");
}
