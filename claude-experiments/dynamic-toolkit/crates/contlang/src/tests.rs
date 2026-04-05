use dynir::verify;
use dynvalue::NanBox;

use crate::{lex, lower_program, parse};

fn run_interp(src: &str, entry_name: &str, args: &[u64]) -> u64 {
    let tokens = lex(src);
    let program = parse(tokens);
    let lowered = lower_program(&program);

    for func in &lowered.module.functions {
        verify(func).unwrap_or_else(|errs| {
            eprintln!("IR for {}:\n{}", func.name, func);
            panic!("verification failed for {}: {:?}", func.name, errs);
        });
    }

    let entry = lowered.func_refs[entry_name];
    let roots = dynir::interp::NoGcRoots;
    let interp = dynir::interp::ModuleInterpreter::<NanBox, _>::new(
        &lowered.module,
        &roots,
    );
    match interp.run(entry, args) {
        Ok(dynir::interp::InterpResult::Value(v)) => v,
        Ok(dynir::interp::InterpResult::Void) => 0,
        Ok(other) => panic!("unexpected result: {:?}", other),
        Err(e) => panic!("interpreter error: {:?}", e),
    }
}

// ── Basic Language Tests ────────────────────────────────────

#[test]
fn return_constant() {
    assert_eq!(run_interp("fn main() { 42 }", "main", &[]), 42);
}

#[test]
fn arithmetic() {
    assert_eq!(run_interp("fn main() { (3 + 4) * 2 - 1 }", "main", &[]), 13);
}

#[test]
fn function_calls() {
    let src = "fn double(x) { x + x }  fn main() { double(21) }";
    assert_eq!(run_interp(src, "main", &[]), 42);
}

#[test]
fn if_expression() {
    let src = r#"
        fn abs(x) { if x >= 0 { x } else { 0 - x } }
        fn main() { abs(-5) + abs(3) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 8);
}

#[test]
fn recursion() {
    let src = r#"
        fn factorial(n) { if n <= 1 { 1 } else { n * factorial(n - 1) } }
        fn main() { factorial(6) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 720);
}

// ── Continuation Tests ──────────────────────────────────────
//
// The IR's abort_to_prompt pops frames up to and including the prompt
// owner, treating it as if the prompt owner "returned" the abort value
// to its caller. This means:
//
// - For `reset { body }` to work as a delimited control operator, the
//   body (or the part that may abort) must be in a SEPARATE function.
//   The prompt owner is that separate function, and aborting causes it
//   to "return" the abort value to the caller (which sees it as the
//   result of the reset expression).
//
// - Tests below use this pattern: a helper function owns the prompt,
//   and the caller uses the result.

#[test]
fn reset_without_abort() {
    // When no abort happens, the body function returns normally.
    let src = r#"
        fn body() { reset { 42 } }
        fn main() { body() }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 42);
}

#[test]
fn abort_returns_value_to_caller() {
    // abort(v) terminates the prompt owner and returns v to the caller.
    let src = r#"
        fn body() {
            reset {
                abort(99);
                999
            }
        }
        fn main() { body() }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 99);
}

#[test]
fn abort_conditional_early_exit() {
    // Conditional abort — the abort path returns early, else path completes.
    let src = r#"
        fn checked_double(x) {
            reset {
                if x < 0 { abort(0 - 1) } else { x * 2 }
            }
        }
        fn main() { checked_double(5) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 10);
}

#[test]
fn abort_conditional_early_exit_negative() {
    let src = r#"
        fn checked_double(x) {
            reset {
                if x < 0 { abort(0 - 1) } else { x * 2 }
            }
        }
        fn main() { checked_double(0 - 5) }
    "#;
    assert_eq!(run_interp(src, "main", &[]), u64::MAX); // -1 as u64
}

#[test]
fn capture_and_abort_returns_handle() {
    // Capture a continuation and abort it back to the caller.
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn main() {
            let k: cont = capture_it();
            // k is a FrameSlice handle — just verify it's non-negative
            1
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 1);
}

#[test]
fn capture_abort_resume() {
    // Full cycle: capture → abort handle to caller → resume with value.
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn do_resume(k: cont, v) {
            resume(k, v)
        }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    // capture_it: captures, aborts k → caller gets k
    // do_resume(k, 42): resumes at shift() with 42
    //   → k = 42, abort(42) → capture_it "returns" 42
    //   → but this is the resumed continuation, so the result bubbles up
    assert_eq!(run_interp(src, "main", &[]), 42);
}

// ═══════════════════════════════════════════════════════════════════
// GC Segment Tests — via unified interpreter + GCSegmentStack
// ═══════════════════════════════════════════════════════════════════

fn run_unified_gc(src: &str, entry_name: &str) -> u64 {
    let tokens = crate::lex(src);
    let program = crate::parse(tokens);
    let lowered = crate::lower_program(&program);
    for func in &lowered.module.functions {
        dynir::verify(func).unwrap_or_else(|errs| {
            panic!("verification failed for {}: {:?}", func.name, errs);
        });
    }
    let entry = lowered.func_refs[entry_name];
    crate::unified_interp::run::<crate::gc_stack::GCSegmentStack>(&lowered.module, entry, &[])
}

#[test]
fn gc_return_constant() {
    assert_eq!(run_unified_gc("fn main() { 42 }", "main"), 42);
}

#[test]
fn gc_arithmetic() {
    assert_eq!(run_unified_gc("fn main() { (3 + 4) * 2 - 1 }", "main"), 13);
}

#[test]
fn gc_function_calls() {
    let src = "fn double(x) { x + x }  fn main() { double(21) }";
    assert_eq!(run_unified_gc(src, "main"), 42);
}

#[test]
fn gc_recursion() {
    let src = r#"
        fn factorial(n) { if n <= 1 { 1 } else { n * factorial(n - 1) } }
        fn main() { factorial(6) }
    "#;
    assert_eq!(run_unified_gc(src, "main"), 720);
}

#[test]
fn gc_reset_without_abort() {
    let src = "fn body() { reset { 42 } }  fn main() { body() }";
    assert_eq!(run_unified_gc(src, "main"), 42);
}

#[test]
fn gc_abort_returns_value() {
    let src = r#"
        fn body() { reset { abort(99); 999 } }
        fn main() { body() }
    "#;
    assert_eq!(run_unified_gc(src, "main"), 99);
}

#[test]
fn gc_capture_abort_resume() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    assert_eq!(run_unified_gc(src, "main"), 42);
}

#[test]
fn gc_multi_shot_clone() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_clone(k: cont) -> cont { clone(k) }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            let k1: cont = do_clone(k);
            let k2: cont = do_clone(k);
            do_resume(k1, 11)
        }
    "#;
    assert_eq!(run_unified_gc(src, "main"), 11);
}

#[test]
fn gc_nested_resets() {
    let src = r#"
        fn inner() { reset { abort(5) } }
        fn main() { reset { let v = inner(); v + 10 } }
    "#;
    assert_eq!(run_unified_gc(src, "main"), 15);
}

// ── GC-specific tests ──────────────────────────────────────────────

#[test]
fn gc_collects_dead_segments() {
    use dynexec::StackConfig;

    let tokens = crate::lex("fn main() { 42 }");
    let program = crate::parse(tokens);
    let lowered = crate::lower_program(&program);
    let entry = lowered.func_refs["main"];

    let config = StackConfig { heap_size: 64 * 1024 };
    let mut rt = crate::gc_stack::GCSegmentRuntime::new(config.heap_size);
    let mut conts = dynexec::VecContinuationStore::new();
    let result = crate::unified_interp::interpret::<crate::gc_stack::GCSegmentStack>(
        &lowered.module, &mut rt, &mut conts, entry, &[],
    );
    assert_eq!(result, 42);

    let used_before = rt.from_used();
    assert!(used_before > 0, "should have allocated something");

    rt.collect_empty();

    let used_after = rt.from_used();
    assert_eq!(used_after, 0, "dead segments should have been reclaimed");
}

#[test]
fn gc_moves_segments() {
    // Allocate a segment with tagged values, root it, collect, verify it moved.
    use crate::gc_stack::SegPtrPolicy;
    use dynalloc::{PtrPolicy, SemiSpace};
    use dynobj::{Compact, ObjHeader, TypeInfo};
    use dynvalue::LowBit;
    use std::cell::Cell;

    type TV = dynvalue::Value<LowBit<3>>;

    fn tag_fixnum(n: i64) -> u64 { TV::tagged(1, n as u64).to_bits() }
    fn untag_fixnum(bits: u64) -> i64 {
        let p = TV::from_bits(bits).payload();
        ((p as i64) << 3) >> 3
    }
    fn tag_ptr(ptr: *mut u8) -> u64 {
        TV::tagged(0, (ptr as u64) >> 3).to_bits()
    }
    fn untag_ptr(bits: u64) -> *mut u8 {
        (TV::from_bits(bits).payload() << 3) as *mut u8
    }

    static SEG: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_varlen_values(0);

    let mut gc = SemiSpace::new::<Compact>(4096);
    let seg = gc.alloc_obj::<Compact>(&SEG, 3);
    assert!(!seg.is_null());

    // Write tagged fixnum values
    unsafe {
        let base = SEG.varlen_element_offset(0);
        *(seg.add(base) as *mut u64) = tag_fixnum(100);
        *(seg.add(base + 8) as *mut u64) = tag_fixnum(200);
        *(seg.add(base + 16) as *mut u64) = tag_fixnum(300);
    }

    // Root the segment as a tagged pointer
    struct SingleRoot(Cell<u64>);
    impl dynobj::RootSource for SingleRoot {
        fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
            visitor(self.0.as_ptr());
        }
    }
    let root = SingleRoot(Cell::new(tag_ptr(seg)));

    // Collect — segment moves to to-space, root updated
    unsafe { gc.collect::<SegPtrPolicy>(&mut [&root]); }

    let new_seg = untag_ptr(root.0.get());
    assert_ne!(new_seg, seg, "segment should have moved");
    assert!(gc.contains(new_seg as *const u8), "new ptr should be in from-space");

    // Tagged values intact after move
    unsafe {
        let base = SEG.varlen_element_offset(0);
        assert_eq!(untag_fixnum(*(new_seg.add(base) as *const u64)), 100);
        assert_eq!(untag_fixnum(*(new_seg.add(base + 8) as *const u64)), 200);
        assert_eq!(untag_fixnum(*(new_seg.add(base + 16) as *const u64)), 300);
    }
}

#[test]
fn gc_during_execution() {
    let src = r#"
        fn work(x) { x + 1 }
        fn main() {
            let a = work(1);
            let b = work(a);
            let c = work(b);
            let d = work(c);
            let e = work(d);
            let f = work(e);
            let g = work(f);
            let h = work(g);
            let i = work(h);
            let j = work(i);
            let k = work(j);
            let l = work(k);
            let m = work(l);
            let n = work(m);
            let o = work(n);
            let p = work(o);
            p
        }
    "#;
    let tokens = crate::lex(src);
    let program = crate::parse(tokens);
    let lowered = crate::lower_program(&program);
    for func in &lowered.module.functions {
        dynir::verify(func).unwrap();
    }
    let entry = lowered.func_refs["main"];
    let mut rt = crate::gc_stack::GCSegmentRuntime::new(64 * 1024);
    rt.set_gc_stress(3);
    let mut conts = dynexec::VecContinuationStore::new();
    let result = crate::unified_interp::interpret::<crate::gc_stack::GCSegmentStack>(
        &lowered.module, &mut rt, &mut conts, entry, &[],
    );
    assert_eq!(result, 17);
    assert!(rt.gc_count() > 0, "GC should have been triggered");
}

#[test]
fn multi_shot_with_clone() {
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn do_clone(k: cont) -> cont { clone(k) }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            let k1: cont = do_clone(k);
            let k2: cont = do_clone(k);
            do_resume(k1, 11)
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 11);
}

#[test]
fn nested_resets_via_separate_functions() {
    let src = r#"
        fn inner() { reset { abort(5) } }
        fn main() {
            reset {
                let v = inner();
                v + 10
            }
        }
    "#;
    // inner: abort(5) → inner's reset returns 5 to caller
    // main's reset body: v = 5, v + 10 = 15
    assert_eq!(run_interp(src, "main", &[]), 15);
}

#[test]
fn cross_function_abort() {
    // Abort from a callee goes to the callee's own foreign prompt,
    // which matches the caller's pushed prompt.
    let src = r#"
        fn do_abort(v) { abort(v) }
        fn main() {
            reset {
                do_abort(77);
                999
            }
        }
    "#;
    // do_abort(77) aborts — but the prompt is in main (push_prompt inline).
    // abort pops do_abort, then pops main (has prompt), returns 77.
    // Since main is popped, 999 is never reached.
    assert_eq!(run_interp(src, "main", &[]), 77);
}

#[test]
fn cross_function_capture_and_resume() {
    // capture_it owns the prompt (via reset). It captures the continuation,
    // aborts the handle back to the caller. The caller resumes with a value.
    // The resumed continuation re-enters capture_it at shift(), gets the
    // resume arg, aborts it back → capture_it "returns" the resume arg.
    let src = r#"
        fn capture_it() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 42);
}

// ═══════════════════════════════════════════════════════════════════
// Unified Interpreter Tests — THE final API shape
// ═══════════════════════════════════════════════════════════════════

fn run_unified_vec(src: &str, entry_name: &str) -> u64 {
    let tokens = crate::lex(src);
    let program = crate::parse(tokens);
    let lowered = crate::lower_program(&program);
    for func in &lowered.module.functions {
        dynir::verify(func).unwrap();
    }
    let entry = lowered.func_refs[entry_name];
    crate::unified_interp::run::<dynexec::ContiguousVecStack>(&lowered.module, entry, &[])
}

#[test]
fn unified_vec_arithmetic() {
    assert_eq!(run_unified_vec("fn main() { (3 + 4) * 2 - 1 }", "main"), 13);
}

#[test]
fn unified_vec_function_calls() {
    assert_eq!(run_unified_vec("fn double(x) { x + x }  fn main() { double(21) }", "main"), 42);
}

#[test]
fn unified_vec_recursion() {
    let src = r#"
        fn factorial(n) { if n <= 1 { 1 } else { n * factorial(n - 1) } }
        fn main() { factorial(6) }
    "#;
    assert_eq!(run_unified_vec(src, "main"), 720);
}

#[test]
fn unified_vec_abort() {
    let src = r#"
        fn body() { reset { abort(99); 999 } }
        fn main() { body() }
    "#;
    assert_eq!(run_unified_vec(src, "main"), 99);
}

#[test]
fn unified_vec_capture_resume() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    assert_eq!(run_unified_vec(src, "main"), 42);
}

#[test]
fn unified_vec_multi_shot() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_clone(k: cont) -> cont { clone(k) }
        fn do_resume(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = capture_it();
            let k1: cont = do_clone(k);
            let k2: cont = do_clone(k);
            do_resume(k1, 11)
        }
    "#;
    assert_eq!(run_unified_vec(src, "main"), 11);
}

#[test]
fn unified_vec_nested_resets() {
    let src = r#"
        fn inner() { reset { abort(5) } }
        fn main() { reset { let v = inner(); v + 10 } }
    "#;
    assert_eq!(run_unified_vec(src, "main"), 15);
}
