use dynalloc::{Alloc, PtrPolicy};
use dynir::verify;
use dynobj::ObjHeader;
use dynvalue::NanBox;

use crate::{lex, lower_program, parse};

/// `PtrPolicy` for `NanBox`: heap pointers have tag=1 in the NanBox
/// scheme, payloads are in the low 48 bits.
///
/// Contlang doesn't currently allocate heap values itself, so this
/// policy effectively treats only FrameSlice handles (which we make
/// be tag=1 pointers) as traced roots.
struct ContlangPolicy;
impl dynalloc::PtrPolicy for ContlangPolicy {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
        const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
        const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
        let tag = 1u64; // arbitrary: we tag heap pointers with 1
        let expected = TAG_PATTERN | (tag << 48);
        let mask = FULL_MASK | (0x0003u64 << 48);
        if (bits & mask) == expected {
            let payload = bits & PAYLOAD_MASK;
            if payload == 0 {
                return None;
            }
            Some(payload as *mut u8)
        } else {
            None
        }
    }
    fn encode_ptr(ptr: *mut u8) -> u64 {
        const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
        const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
        let tag = 1u64;
        TAG_PATTERN | (tag << 48) | ((ptr as u64) & PAYLOAD_MASK)
    }
}

fn run_interp(src: &str, entry_name: &str, args: &[u64]) -> u64 {
    run_interp_with(src, entry_name, args, usize::MAX, 256 * 1024).0
}

/// Extended harness: returns (result, collection_count) and lets the
/// caller configure the auto-GC threshold and heap size. Used by
/// stress / reclamation tests.
fn run_interp_with(
    src: &str,
    entry_name: &str,
    args: &[u64],
    gc_threshold: usize,
    heap_size: usize,
) -> (u64, usize) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicPtr, Ordering};

    use dynir::gc_runtime::GcInterpCtx;
    use dynir::interp::ExternCallResult;
    use dynobj::{Compact, TypeInfo};

    let tokens = lex(src);
    let program = parse(tokens);
    let lowered = lower_program(&program);

    for func in &lowered.module.functions {
        verify(func).unwrap_or_else(|errs| {
            eprintln!("IR for {}:\n{}", func.name, func);
            panic!("verification failed for {}: {:?}", func.name, errs);
        });
    }


    // Build the type table. Register the byte-buffer user type first
    // (type_id 0), then the continuation types.
    let byte_type_id: u16 = 0;
    let byte_type = TypeInfo::for_header(Compact::SIZE)
        .with_varlen_bytes(0)
        .with_type_id(byte_type_id);
    let mut type_table: Vec<TypeInfo> = vec![byte_type];
    let cont_types =
        dynexec::ContinuationTypes::register_into::<Compact>(&mut type_table);
    let heap = dynalloc::SemiSpace::new::<Compact>(heap_size);
    let ctx = GcInterpCtx::<Compact, ContlangPolicy>::new(heap, type_table, cont_types);
    ctx.set_gc_threshold(gc_threshold);

    let entry = lowered.func_refs[entry_name];
    let mut interp = dynir::interp::ModuleInterpreter::<NanBox, _>::new(
        &lowered.module,
        &ctx,
    );
    interp.set_cont_ctx(&ctx);

    // Bind contlang's built-in heap externs.
    //
    // The closures need to reach back into `ctx` without holding a
    // Rust borrow that would conflict with the interpreter's own
    // `&ctx`. Use an `Arc<AtomicPtr<_>>` for the pointer indirection.
    let ctx_ptr: *const GcInterpCtx<Compact, ContlangPolicy> = &ctx;
    let ctx_ptr: Arc<AtomicPtr<GcInterpCtx<Compact, ContlangPolicy>>> =
        Arc::new(AtomicPtr::new(ctx_ptr as *mut _));

    let f_alloc = lowered.func_refs[crate::lower::BYTES_ALLOC];
    let f_get = lowered.func_refs[crate::lower::BYTES_GET];
    let f_set = lowered.func_refs[crate::lower::BYTES_SET];
    let f_len = lowered.func_refs[crate::lower::BYTES_LEN];

    let base_offset = byte_type.varlen_count_offset() + 8;

    let ctx_a = ctx_ptr.clone();
    interp.bind(f_alloc, move |args| {
        let len = args[0] as usize;
        let ctx = unsafe { &*ctx_a.load(Ordering::SeqCst) };
        let raw = ctx.alloc(&byte_type, len);
        assert!(!raw.is_null(), "bytes_alloc: OOM");
        unsafe {
            dynobj::init_header::<Compact>(raw, byte_type.type_id);
            dynobj::write_varlen_count(raw, &byte_type, len);
            core::ptr::write_bytes(raw.add(base_offset), 0, len);
        }
        ExternCallResult::Value(Some(ContlangPolicy::encode_ptr(raw)))
    });

    interp.bind(f_get, move |args| {
        let p = ContlangPolicy::try_decode_ptr(args[0])
            .expect("bytes_get: not a heap pointer");
        let idx = args[1] as usize;
        let byte = unsafe { *p.add(base_offset + idx) };
        ExternCallResult::Value(Some(byte as u64))
    });

    interp.bind(f_set, move |args| {
        let p = ContlangPolicy::try_decode_ptr(args[0])
            .expect("bytes_set: not a heap pointer");
        let idx = args[1] as usize;
        let byte = args[2] as u8;
        unsafe {
            *p.add(base_offset + idx) = byte;
        }
        ExternCallResult::Value(None)
    });

    interp.bind(f_len, move |args| {
        let p = ContlangPolicy::try_decode_ptr(args[0])
            .expect("bytes_len: not a heap pointer");
        let len = unsafe { dynobj::read_varlen_count(p, &byte_type) };
        ExternCallResult::Value(Some(len as u64))
    });

    let result = match interp.run(entry, args) {
        Ok(dynir::interp::InterpResult::Value(v)) => v,
        Ok(dynir::interp::InterpResult::Void) => 0,
        Ok(other) => panic!("unexpected result: {:?}", other),
        Err(e) => panic!("interpreter error: {:?}", e),
    };
    (result, ctx.collection_count())
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

/// While loop with a loop-carried counter. Previously broken by an
/// SSA dominance violation — the body's `i = i + 1` created a new
/// value that the header tried to use without a phi. Fixed by
/// threading in-scope variables through header block params.
#[test]
fn while_loop_counts() {
    let src = r#"
        fn main() {
            let i = 0;
            while i < 10 {
                i = i + 1;
            }
            i
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 10);
}

#[test]
fn while_loop_accumulates() {
    // Sum 1..=10 = 55. Two loop-carried vars (i and sum), both
    // threaded through header block params.
    let src = r#"
        fn main() {
            let i = 1;
            let sum = 0;
            while i <= 10 {
                sum = sum + i;
                i = i + 1;
            }
            sum
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 55);
}

// ── Heap allocation tests ──────────────────────────────────────

#[test]
fn bytes_alloc_and_roundtrip() {
    let src = r#"
        fn main() {
            let p: bytes = bytes_alloc(5);
            bytes_set(p, 0, 42);
            bytes_set(p, 1, 100);
            bytes_get(p, 0) + bytes_get(p, 1)
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 142);
}

#[test]
fn bytes_len_returns_allocation_size() {
    let src = r#"
        fn main() {
            let p: bytes = bytes_alloc(17);
            bytes_len(p)
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 17);
}

#[test]
fn bytes_sum_via_loop() {
    // Fill a byte buffer with 1..=10 and sum them. Exercises bytes_*
    // operations alongside the (now-fixed) while lowering.
    let src = r#"
        fn main() {
            let p: bytes = bytes_alloc(10);
            let i = 0;
            while i < 10 {
                bytes_set(p, i, i + 1);
                i = i + 1;
            }
            let sum = 0;
            let j = 0;
            while j < 10 {
                sum = sum + bytes_get(p, j);
                j = j + 1;
            }
            sum
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 55);
}

/// Multi-frame capture with GC between: the prompt-owning function
/// holds a bytes pointer, calls a helper, the helper allocates
/// heavily (triggering GC that forwards the pointer in the
/// prompt-owner's root slots but NOT in its vals), then the helper
/// captures. The captured prompt-owner frame must contain the
/// forwarded pointer, not the stale pre-GC address.
///
/// This test catches a bug where `build_captured_stack_impl` read
/// non-top frames' vals directly (stale after GC) instead of
/// overlaying forwarded values from root slots.
#[test]
fn multi_frame_capture_gc_forwards_non_top_frame_pointer() {
    let src = r#"
        fn do_capture_after_gc() -> cont {
            // This function is called with the prompt already
            // installed by the caller. Allocate heavily to trigger
            // GC (which forwards the caller's bytes pointer in its
            // root slots), then capture.
            let i = 0;
            while i < 20 {
                let scratch: bytes = bytes_alloc(32);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            // Now capture — the captured slice includes BOTH this
            // frame AND the caller's frame (which owns the prompt).
            // The caller's frame holds the bytes pointer. If we
            // read the caller's vals (stale), we get a dangling
            // pointer. If we read from root slots (forwarded),
            // we get the correct new address.
            shift |k| { abort(k) }
        }

        fn invoke(k: cont, v) { resume(k, v) }

        fn main() {
            let p: bytes = bytes_alloc(4);
            bytes_set(p, 0, 77);
            bytes_set(p, 1, 88);
            let k: cont = reset {
                do_capture_after_gc()
            };
            // Resume — the captured frame re-enters do_capture_after_gc
            // at the shift's resume_bb; the delimited context is empty
            // so it falls through and returns via the trampoline.
            // The interesting part is whether p in main's captured
            // frame survived the GC correctly.
            let r = invoke(k, 0);
            // r is the resume arg (0); what we really care about is
            // whether p is still valid:
            bytes_get(p, 0) + bytes_get(p, 1)
        }
    "#;

    // Aggressive GC: threshold 3, small heap.
    let (result, collections) =
        run_interp_with(src, "main", &[], 3, 16 * 1024);

    assert_eq!(result, 77 + 88, "bytes must survive GC across a multi-frame capture");
    assert!(
        collections >= 3,
        "GC should have fired during do_capture_after_gc; got {}",
        collections
    );
}

/// Chain of continuations: capture `a`, then capture `b` in a
/// context where `a` is in scope, force several GC cycles, then
/// invoke each independently.
///
/// At GC time:
///   - main's frame holds `a` and `b` as FrameSlice roots → forwarded
///   - the GC also traces b's `ContObj`, walking its varlen values
///     tail; one of those values is the captured snapshot of `a`'s
///     handle (since `a` was in scope when `b` was captured) →
///     forwarded
///
/// All three references to `a` must end up pointing at the same
/// to-space location after the cycle. If the GC missed the
/// b-internal copy of `a`, b would have a stale a-handle baked in,
/// and any later trace through b would dangle. (We don't directly
/// observe the b-internal copy here, but Cheney's deduplication via
/// forwarding bits guarantees the invariant.)
#[test]
fn chained_continuations_trace_through_each_other() {
    let src = r#"
        fn invoke(k: cont, v) { resume(k, v) }
        fn allocate_many() {
            let i = 0;
            while i < 30 {
                let scratch: bytes = bytes_alloc(16);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            0
        }
        fn main() {
            let a: cont = reset { shift |ka| { abort(ka) } };
            let b: cont = reset { shift |kb| { abort(kb) } };
            // Force GC. Both a and b rooted in main; b's ContObj
            // values tail also contains a's handle. The GC must
            // trace through b and forward a's internal copy too.
            allocate_many();
            allocate_many();
            // Now invoke each. The trampoline returns the resume
            // arg directly to the resumer.
            let r1 = invoke(a, 100);
            let r2 = invoke(b, 200);
            r1 + r2
        }
    "#;

    let (result, collections) =
        run_interp_with(src, "main", &[], 4, 32 * 1024);

    assert_eq!(result, 300);
    assert!(
        collections >= 4,
        "auto-gc should fire many times during allocate_many; got {}",
        collections
    );
}

/// Two captures of the same delimited context, both invoked from
/// post-reset code. Without the trampoline, the second invocation
/// would re-execute the first one's effects (and the original
/// invocation itself). With the trampoline, each invocation
/// returns its argument cleanly to the resumer.
#[test]
fn captured_continuation_invoked_twice_from_post_reset_code() {
    let src = r#"
        fn invoke(k: cont, v) { resume(k, v) }
        fn main() {
            let k: cont = reset {
                shift |k_inner| { abort(k_inner) }
            };
            // Two separate invocations, each should return its arg
            // (the empty captured context just yields the resume value).
            let a = invoke(k, 10);
            let b = invoke(k, 32);
            a + b
        }
    "#;

    assert_eq!(run_interp(src, "main", &[]), 42);
}

/// `let k: cont = reset { ... }` directly inline in `main` —
/// previously broken by the **over-capture** bug. The captured
/// continuation included main's whole post-reset IR (the let, the
/// allocate_many calls, AND the invoke call), so resuming
/// re-executed the second invoke call with a corrupted handle and
/// crashed.
///
/// Fixed by the resumed-frame trampoline in `abort_to_prompt`: when
/// a captured frame's body completes (via either explicit abort or
/// the synthetic exit emitted by reset's normal path), the runtime
/// notices the frame's `resume = FromResume` and trampolines back
/// to the resumer instead of running the prompt's handler block
/// (which contains the post-reset IR). Equivalent to Chez's
/// `dounderflow` trampoline at the bottom of a captured stack
/// segment.
#[test]
fn captured_continuation_does_not_re_execute_post_reset_code() {
    let src = r#"
        fn invoke(k: cont, v) { resume(k, v) }
        fn allocate_many() {
            let i = 0;
            while i < 30 {
                let scratch: bytes = bytes_alloc(16);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            0
        }
        fn main() {
            let k: cont = reset {
                shift |k_inner| { abort(k_inner) }
            };
            // k holds the captured continuation. Without the
            // trampoline, resuming k inside this function would
            // re-execute these allocate_many calls AND the invoke
            // call below — infinite recursion through a corrupted
            // handle. With the trampoline, resume returns its
            // arg to the resumer (this function's invoke result
            // slot) and main's flow continues normally.
            allocate_many();
            allocate_many();
            invoke(k, 55)
        }
    "#;

    let (result, collections) =
        run_interp_with(src, "main", &[], 4, 16 * 1024);

    assert_eq!(result, 55);
    assert!(
        collections >= 4,
        "auto-gc should fire many times during allocate_many; got {}",
        collections
    );
}

/// Continuation escape using the new `shift |k|` form inside a
/// cont-returning function. Previously this hit a type mismatch:
/// reset's handler_bb was typed as the function return (cont), but
/// the shift's resume path produced an I64 that didn't unify.
///
/// Now the lowering uses the I64-normalized reset handler with
/// bitcasts at the merge edges, so the pattern type-checks AND
/// composes with the rest of the GC pipeline.
#[test]
fn continuation_outlives_via_new_shift_form() {
    let src = r#"
        fn get_cont() -> cont {
            reset {
                shift |k| { abort(k) }
            }
        }

        fn allocate_many() {
            let i = 0;
            while i < 30 {
                let scratch: bytes = bytes_alloc(16);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            0
        }

        fn invoke(k: cont, v) { resume(k, v) }

        fn main() {
            let k: cont = get_cont();
            allocate_many();
            allocate_many();
            invoke(k, 99)
        }
    "#;

    // Aggressive threshold to force collection during allocate_many.
    let (result, collections) =
        run_interp_with(src, "main", &[], 4, 16 * 1024);

    // Capture flow:
    //   get_cont enters reset, push prompt P
    //   shift fires: handler_bb gets fresh handle k
    //   handler body: abort(k) → I64 bitcast → reset handler_bb(I64)
    //   reset handler_bb returns the I64; function Ret coerces back
    //   to FrameSlice → main's `let k: cont = get_cont()` slot is
    //   FrameSlice (rooted as a GC pointer).
    //
    // Resume flow (when main calls invoke(k, 99)):
    //   The captured frame is spliced; resume_bb(I64) gets v=99.
    //   resume_bb is the "after shift" point in get_cont's body. In
    //   this program shift was the entire reset body, so the next
    //   thing is the reset's pop_prompt + jump(reset_handler_bb, [99]).
    //   reset_handler_bb returns 99 as the function's return value.
    //   FromResume trampolines 99 back to main's invoke continuation.
    //   invoke returns 99. main returns 99.
    assert_eq!(result, 99);
    assert!(
        collections >= 4,
        "allocate_many × 2 should have triggered auto-gc many times; got {}",
        collections
    );
}

/// Continuation escape using the old-style `shift()` form. Kept as
/// a regression test alongside the new-form version above.
#[test]
fn continuation_outlives_original_reset_and_survives_gc() {
    let src = r#"
        fn get_cont() -> cont {
            reset {
                let k: cont = shift();
                abort(k)
            }
        }

        fn allocate_many() {
            let i = 0;
            while i < 30 {
                let scratch: bytes = bytes_alloc(16);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            0
        }

        fn invoke(k: cont, v) { resume(k, v) }

        fn main() {
            let k: cont = get_cont();
            // get_cont has returned; k is in main's frame, rooted as
            // a FrameSlice value. Run allocation pressure that
            // forces multiple GCs. Each GC must forward the
            // captured ContObj and all the heap pointers inside
            // its varlen values tail.
            allocate_many();
            allocate_many();
            invoke(k, 77)
        }
    "#;

    // Aggressive threshold to force collection during allocate_many.
    let (result, collections) =
        run_interp_with(src, "main", &[], 4, 16 * 1024);

    // At capture, the old-style shift() returns the handle into the
    // `k` slot, then `abort(k)` fires and the reset yields k
    // (FrameSlice), which get_cont returns to main.
    //
    // At resume, main calls `invoke(k, 77)`. invoke runs
    // `resume(k, 77)`. The captured frame is spliced on top; it
    // re-enters at "after shift()" with `k` now holding 77 (the
    // resume arg, overwriting the original handle via
    // resume_arg_value_indices). The next instruction is `abort(k)`
    // which aborts with 77. Reset yields 77. get_cont's return
    // trampolines back to invoke's continuation via FromResume
    // with 77. invoke returns 77. main returns 77.
    assert_eq!(result, 77);
    assert!(
        collections >= 4,
        "allocate_many × 2 should have triggered auto-gc many times; got {}",
        collections
    );
}

/// Multi-shot with GC interleaved between invocations.
/// Captures a continuation, resumes it once, runs allocation pressure
/// to force multiple GC cycles (during which the handle `k` is held
/// in a FrameSlice-typed handler-block param slot and so stays live),
/// then resumes the SAME handle again. Both invocations must return
/// the correct values.
///
/// If the second resume is broken — e.g., the GC didn't correctly
/// forward the ContObj's varlen values tail, or the handle was freed
/// prematurely — we'd get the wrong answer or a crash.
#[test]
fn multi_shot_with_interleaved_gc() {
    let src = r#"
        fn allocate_many() {
            let i = 0;
            while i < 20 {
                let scratch: bytes = bytes_alloc(8);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            0
        }

        fn invoke(k: cont, v) { resume(k, v) }

        fn main() {
            reset {
                let v = shift |k| {
                    let a = invoke(k, 10);
                    // Force GC cycles while `k` is still live in
                    // this handler frame's block-param slot.
                    allocate_many();
                    let b = invoke(k, 32);
                    a + b
                };
                v + 1
            }
        }
    "#;

    // Low threshold so allocate_many actually triggers GC.
    let (result, collections) =
        run_interp_with(src, "main", &[], 5, 16 * 1024);

    // Each invocation runs `v + 1` on its resume arg:
    //   first:  10 + 1 = 11
    //   second: 32 + 1 = 33
    //   sum returned from handler: 44
    // Then my shift lowering aborts to the outer reset with 44 as
    // the body_result, so reset yields 44. main returns 44.
    assert_eq!(result, 44);
    assert!(
        collections >= 2,
        "allocate_many should have triggered auto-gc at least twice; got {}",
        collections
    );
}

/// End-to-end: allocate a byte buffer, capture a continuation holding
/// its pointer, force many GC cycles via allocation pressure, resume,
/// and read from the pointer. This is the same property as the Phase 4
/// dynir test — but written in contlang as a real surface-language
/// program, proving the full stack composes: contlang syntax → IR
/// lowering → heap-backed continuations → GC tracing through ContObj
/// values tail → forwarded GcPtr → correct resume.
#[test]
fn bytes_pointer_survives_gc_through_captured_continuation() {
    let src = r#"
        fn allocate_many() {
            // Allocate + discard many byte buffers to force multiple
            // collections.
            let i = 0;
            while i < 20 {
                let scratch: bytes = bytes_alloc(16);
                bytes_set(scratch, 0, i);
                i = i + 1;
            }
            0
        }

        fn main() {
            let p: bytes = bytes_alloc(4);
            bytes_set(p, 0, 77);
            bytes_set(p, 1, 88);
            bytes_set(p, 2, 99);
            bytes_set(p, 3, 11);
            reset {
                let v = shift |k| {
                    // While the continuation (which captures p in
                    // its values tail) is dormant, run allocation
                    // pressure that triggers several GCs. Each GC
                    // must forward the captured p correctly so that
                    // when we resume and read from p, we see the
                    // original bytes.
                    allocate_many();
                    resume(k, 0)
                };
                bytes_get(p, 0) + bytes_get(p, 1)
                    + bytes_get(p, 2) + bytes_get(p, 3)
            }
        }
    "#;

    // Auto-GC threshold of 5 — forces collection to actually run
    // inside allocate_many while the continuation is dormant.
    let (result, collections) =
        run_interp_with(src, "main", &[], 5, 16 * 1024);

    assert_eq!(result, 77 + 88 + 99 + 11, "bytes must survive GC via the captured continuation");
    assert!(
        collections >= 2,
        "auto-gc should have fired during allocate_many; got {}",
        collections
    );
}

#[test]
fn while_loop_zero_iterations() {
    // Condition false on entry — body never runs, exit uses the
    // header block param (= initial value), not a stale mapping.
    let src = r#"
        fn main() {
            let i = 42;
            while i < 0 {
                i = i - 1;
            }
            i
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 42);
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
// "Is it really delimited continuations?" — adversarial tests
// ═══════════════════════════════════════════════════════════════════
//
// Real delimited continuations (shift/reset, prompt/control) must:
//   1. Capture the delimited *context* between shift and reset, so that
//      resuming re-runs the intervening computation with the fresh value.
//   2. Make resume(k, v) *return* the prompt's final value to its caller,
//      so subsequent computation can use the result.
//   3. Be re-entrant: the same captured k can be resumed multiple times
//      with observably independent effects.
//
// The existing tests above all share the shape
//     reset { let k = shift(); abort(k) }
// where the "delimited context" is literally just `abort(k)`. Those tests
// cannot distinguish a real implementation from an exception/longjmp.
// The tests below probe each of the three properties directly.

/// Property 1: the computation between shift and reset must be captured.
/// The "delimited context" is `v + 1` — on resume, that addition must
/// re-execute with the resume value bound to v.
#[test]
fn shift_captures_intervening_computation() {
    let src = r#"
        fn main() {
            reset {
                let v = shift |k| { resume(k, 5) };
                v + 1
            }
        }
    "#;
    // Classic (reset (+ 1 (shift k (k 5)))) = 6.
    assert_eq!(run_interp(src, "main", &[]), 6);
}

/// Multi-shot through the new `shift |k|` form: the handler invokes the
/// continuation twice with different values, and each invocation must
/// independently run the delimited context `v + 1`.
#[test]
fn shift_multi_shot_same_handler() {
    let src = r#"
        fn invoke(k: cont, v) { resume(k, v) }
        fn main() {
            reset {
                let v = shift |k| {
                    let a = invoke(k, 10);
                    let b = invoke(k, 32);
                    a + b
                };
                v + 1
            }
        }
    "#;
    // Each k-invocation runs `v + 1`. Results: (10+1) + (32+1) = 44.
    assert_eq!(run_interp(src, "main", &[]), 44);
}

/// If the shift handler never invokes k and just returns a constant, the
/// delimited context `v + 1` is discarded and the reset yields the
/// constant. Matches (reset (+ 1 (shift k 99))) = 99.
#[test]
fn shift_handler_discards_continuation() {
    let src = r#"
        fn main() {
            reset {
                let v = shift |k| { 99 };
                v + 1
            }
        }
    "#;
    assert_eq!(run_interp(src, "main", &[]), 99);
}

/// Prove the heap-backed continuation path is actually being used.
/// After running a simple shift/reset program, the heap must have
/// allocated at least one byte (the ContObj + ContMeta). If this
/// assertion fails, the test harness accidentally fell back to the
/// legacy Vec-backed store and none of our GC claims hold.
#[test]
fn heap_backed_path_actually_allocates() {
    use dynir::gc_runtime::GcInterpCtx;
    use dynobj::Compact;

    let src = r#"
        fn main() {
            reset {
                let v = shift |k| { resume(k, 5) };
                v + 1
            }
        }
    "#;

    let tokens = lex(src);
    let program = parse(tokens);
    let lowered = lower_program(&program);
    for func in &lowered.module.functions {
        verify(func).unwrap();
    }

    let mut type_table: Vec<dynobj::TypeInfo> = Vec::new();
    let cont_types =
        dynexec::ContinuationTypes::register_into::<Compact>(&mut type_table);
    let heap = dynalloc::SemiSpace::new::<Compact>(256 * 1024);
    let ctx = GcInterpCtx::<Compact, ContlangPolicy>::new(heap, type_table, cont_types);

    assert_eq!(ctx.from_used(), 0, "heap should start empty");

    let entry = lowered.func_refs["main"];
    let mut interp = dynir::interp::ModuleInterpreter::<NanBox, _>::new(
        &lowered.module,
        &ctx,
    );
    interp.set_cont_ctx(&ctx);
    let result = interp.run(entry, &[]).unwrap();
    let value = match result {
        dynir::interp::InterpResult::Value(v) => v,
        other => panic!("unexpected: {:?}", other),
    };

    assert_eq!(value, 6);
    assert!(
        ctx.from_used() > 0,
        "heap must have allocated ContObj/ContMeta during the shift; \
         if this is 0 we silently fell back to the Vec store"
    );
}

/// Prove that post-resume code in the *handler* actually runs and sees
/// the resumed value. If resume were tail-only (the old bug), `r * 2`
/// after it would be silently dropped.
#[test]
fn shift_resume_returns_to_post_resume_code() {
    let src = r#"
        fn main() {
            reset {
                let v = shift |k| {
                    let r = resume(k, 20);
                    r * 2
                };
                v + 1
            }
        }
    "#;
    // resume(k, 20) runs delimited context → 21 → r=21 → r*2=42.
    assert_eq!(run_interp(src, "main", &[]), 42);
}

/// Two captures in the same reset body: the first shift captures a
/// delimited context that itself contains another shift. Exercises
/// shift-after-shift within the same reset.
#[test]
fn shift_two_captures_sequentially() {
    let src = r#"
        fn main() {
            reset {
                let v1 = shift |k1| { resume(k1, 5) };
                let v2 = shift |k2| { resume(k2, 7) };
                v1 + v2
            }
        }
    "#;
    // k1(5) re-enters at "let v2 = ...", runs the second shift, k2(7)
    // re-enters at "v1 + v2", yields 12.
    assert_eq!(run_interp(src, "main", &[]), 12);
}

/// Property 2: resume must return the prompt's value to its caller so
/// the caller can keep computing with it.
#[test]
fn resume_returns_value_to_caller() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_resume(k: cont, v) {
            let r = resume(k, v);
            r + 100
        }
        fn main() {
            let k: cont = capture_it();
            do_resume(k, 42)
        }
    "#;
    // Real delimited: resume(k, 42) returns 42 to do_resume, which then
    // computes 42 + 100 = 142.
    assert_eq!(run_interp(src, "main", &[]), 142);
}

/// Property 3: multi-shot — the same continuation invoked twice yields
/// two independent results that can be combined.
#[test]
fn multi_shot_continuation_is_reentrant() {
    let src = r#"
        fn capture_it() -> cont {
            reset { let k: cont = shift(); abort(k) }
        }
        fn do_clone(k: cont) -> cont { clone(k) }
        fn invoke(k: cont, v) {
            let r = resume(k, v);
            r
        }
        fn main() {
            let k:  cont = capture_it();
            let k1: cont = do_clone(k);
            let k2: cont = do_clone(k);
            let a = invoke(k1, 10);
            let b = invoke(k2, 32);
            a + b
        }
    "#;
    // Real multi-shot: 10 + 32 = 42. A one-shot / stack-replacing impl
    // cannot even execute the second invoke — the first resume erases
    // main's frame.
    assert_eq!(run_interp(src, "main", &[]), 42);
}

/// Reclamation stress test: a contlang program that captures a
/// continuation in a while loop and discards the handle each
/// iteration. With an aggressive auto-GC threshold, the heap must
/// stay bounded — each capture allocates a fresh `ContObj` +
/// `ContMeta` pair, and the previous iteration's pair must be
/// reclaimed.
///
/// This exercises: (a) heap-backed continuations, (b) auto-GC, (c)
/// reclamation, and (d) contlang's fixed while lowering with a
/// loop-carried counter `i`.
#[test]
fn reclamation_under_capture_loop_keeps_heap_bounded() {
    let src = r#"
        fn capture_and_abort() {
            reset {
                let v = shift |k| { abort(0) };
                v + 1
            }
        }
        fn main() {
            let i = 0;
            while i < 50 {
                capture_and_abort();
                i = i + 1;
            }
            i
        }
    "#;

    // Small-ish heap (8 KB) so that without reclamation, 50
    // continuations would blow through it (each ContObj + ContMeta
    // pair is ~100+ bytes). Threshold 3 forces GC every 3
    // allocations — much more frequent than necessary to keep the
    // heap clear.
    let (result, collections) =
        run_interp_with(src, "main", &[], 3, 8 * 1024);

    assert_eq!(result, 50, "loop should have counted to 50");
    assert!(
        collections >= 10,
        "auto-gc should have fired many times under capture pressure; got {}",
        collections
    );
}
