#[test]
fn probe_let_linearity_laundering() {
    use crate::rust_surface::check_program;
    // double-free through a let-bound linear Own: should be REJECTED (o used twice).
    let dbl = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
               dbl : Unit\nfn dbl() { let o = alloc(Zero); let u = free(o); free(o) }\n\
               main : Unit\nfn main() { dbl }\n";
    match check_program(dbl) {
        Ok(_)  => eprintln!("PROBE: ACCEPTED (double-free through let — LAUNDERING HOLE CONFIRMED)"),
        Err(e) => eprintln!("PROBE: rejected (good): {e:?}"),
    }
    // sanity: single free should be accepted.
    let ok = "%builtin Nat Nat\nenum Nat { Zero : Nat, Succ : Nat -> Nat }\n\
              one : Unit\nfn one() { let o = alloc(Zero); free(o) }\n\
              main : Unit\nfn main() { one }\n";
    match check_program(ok) {
        Ok(_)  => eprintln!("PROBE single-free: accepted (good)"),
        Err(e) => eprintln!("PROBE single-free: REJECTED (unexpected): {e:?}"),
    }
}
