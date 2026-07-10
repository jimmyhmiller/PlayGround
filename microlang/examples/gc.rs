//! The fusion point, with a MOVING collector: objects that survive a collection
//! are relocated, so a mutator holding a bare pointer across an allocation
//! dangles. The fix is the handle: publish the value to the shadow stack and
//! re-read it after the collection. The compiler does exactly this for the form
//! it is expanding, which is the direct remedy for the clojure-jvm form-609
//! relocation bug.

use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};

use microlang::{LowBitModel, Runtime, TreeWalk, Val};

fn main() {
    // ── 1. the relocation mechanism ─────────────────────────
    let mut rt = Runtime::<LowBitModel>::new();
    let one = rt.encode(Val::Int(1));
    let nil = rt.encode(Val::Nil);
    let list = rt.cons(one, nil);
    let stale = list; // a bare pointer the collector cannot see
    let handle = rt.root(list); // published to the shadow stack
    for _ in 0..5 {
        rt.cons(one, one); // garbage, to make the move visible
    }
    rt.collect(&None);
    let moved = handle.get(&rt);
    println!(
        "relocation: 0x{stale:x} -> 0x{moved:x}   ({} objects relocated)",
        rt.relocated
    );
    println!("  handle re-read : {}", rt.print(moved)); // correct: (1)

    let prev = take_hook();
    set_hook(Box::new(|_| {}));
    let bad = catch_unwind(AssertUnwindSafe(|| rt.as_cons(stale)));
    set_hook(prev);
    println!(
        "  stale pointer  : {}",
        if bad.is_err() {
            "USE-AFTER-MOVE (loud — the object moved)"
        } else {
            "(unexpected: no error)"
        }
    );
    rt.pop_root();

    // ── 2. the compiler under a relocating macro ────────────
    // `firstof` forces a GC that relocates the form `(f (firstof (40)) (h))`
    // mid-analysis. The compiler re-reads the form through its root and
    // re-derives the sibling `(h)` from the relocated parent — so it works.
    let cs = TreeWalk;
    let mut rt2 = Runtime::<LowBitModel>::new();
    let r = microlang::sexpr::eval_str(&mut rt2, 
        &cs,
        r#"
        (def h (fn () 99))
        (def f (fn (a b) (+ a b)))
        (defmacro firstof (x) (list 1 2 3) (gc) (first x))
        (f (firstof (40)) (h))
        "#,
    );
    println!(
        "\ncompiler under a relocating macro:\n  (f (firstof (40)) (h)) => {}   [{} objects relocated]",
        rt2.print(r),
        rt2.relocated
    );
    println!(
        "\nThe form moved while being compiled; every access went through a root,\n\
         so nothing dangled. Caching a bare `list_to_vec(form)` across the macro\n\
         would have been form-609. This is the value axis (a moving heap) and the\n\
         execution axis (re-entrant compilation) fused at one seam."
    );
}
