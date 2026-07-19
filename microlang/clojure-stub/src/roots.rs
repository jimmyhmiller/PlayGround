//! The shadow-stack handle discipline for the frontend's compile-time data.
//!
//! WHY THIS EXISTS. The frontend threads heap pointers (forms, datums) through
//! macro expansion — and expansion EVALUATES code, which reaches a safepoint,
//! which can collect and RELOCATE those forms. A bare `u64` held across such a
//! call dangles; the moving collector catches it loudly (`use-after-move:
//! … points into a collected space`). `docs/STAGE_D_MIGRATION.md` states the
//! remedy: "the fix is the handle: publish the value to the shadow stack and
//! re-read it (`root_get`) after anything that may allocate."
//!
//! WHAT IS ACTUALLY HAZARDOUS — the scope this type is aimed at. Allocation
//! NEVER collects in this heap (Stage D: two bump spaces, Cheney evacuation at
//! safepoints only; exhaustion panics). So a value needs a handle exactly when
//! it is held across a call that can reach a SAFEPOINT — in this frontend that
//! means `CodeSpace::invoke` / `CodeSpace::eval_ir` and the things that reach
//! them: macro expansion, `-force-spine`, and a `require` that loads and runs a
//! file. NOT across every `rt.alloc`/`rt.vec_to_list`. Values that are
//! IMMEDIATES (`Sym`, int, bool, nil — see `model.rs`'s `enc_sym`) never move at
//! all, which is why the compiler's `Sym`-keyed tables need no rooting.
//!
//! WHAT MAKES IT STRUCTURAL, rather than a sprinkling of `push_root` calls:
//!   * A value can only be read back through `get`, which needs the `Runtime` —
//!     so re-reading at the point of use is the ONLY idiom the type offers.
//!   * A function that takes `&mut RootVec` CANNOT be handed a bare `Vec<u64>`.
//!     That is the borrow-checker-adjacent enforcement: `expand_each`'s callers
//!     must root before they can call it, and the type system says so.
//!   * `release` is explicit (popping needs `&mut Runtime`, which an RAII `Drop`
//!     cannot hold while the body is also using `&mut rt`). Forgetting it is a
//!     LOUD panic in `Drop`, not a silently leaked shadow slot.

use microlang::{Runtime, ValueModel};

/// A run of shadow-stack slots holding heap values across safepoints. See the
/// module docs for the discipline and why it is shaped this way.
pub struct RootVec {
    base: usize,
    len: usize,
    released: bool,
}

impl RootVec {
    /// Root `items`, in order; this frame owns shadow slots `base..base+len`.
    pub fn new<M: ValueModel>(rt: &mut Runtime<M>, items: &[u64]) -> RootVec {
        let base = rt.root_depth();
        for &v in items {
            rt.push_root(v);
        }
        RootVec { base, len: items.len(), released: false }
    }

    /// A single rooted value — one form held across a call.
    pub fn one<M: ValueModel>(rt: &mut Runtime<M>, v: u64) -> RootVec {
        RootVec::new(rt, &[v])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Re-read slot `i`: the value's CURRENT address, after any collection.
    pub fn get<M: ValueModel>(&self, rt: &Runtime<M>, i: usize) -> u64 {
        assert!(i < self.len, "RootVec::get: index {i} out of range (len {})", self.len);
        rt.root_get(self.base + i)
    }

    /// Overwrite slot `i` (e.g. with the expansion of the form it held).
    pub fn set<M: ValueModel>(&self, rt: &mut Runtime<M>, i: usize, v: u64) {
        assert!(i < self.len, "RootVec::set: index {i} out of range (len {})", self.len);
        rt.set_root(self.base + i, v);
    }

    /// A SNAPSHOT of the slots' current addresses.
    ///
    /// Valid only until the next safepoint. The one legal use is to feed an
    /// allocation-only builder — `vec_to_list`, `make_vector`, `alloc_*` — which
    /// cannot collect (see the module docs) and so cannot invalidate it. Anything
    /// that can invoke must read through `get` instead.
    pub fn snapshot<M: ValueModel>(&self, rt: &Runtime<M>) -> Vec<u64> {
        (0..self.len).map(|i| self.get(rt, i)).collect()
    }

    /// Pop this frame's slots. Frames must be released in LIFO order — the
    /// shadow stack IS a stack, and truncating below a still-open inner frame
    /// would leave it reading slots it no longer owns.
    pub fn release<M: ValueModel>(mut self, rt: &mut Runtime<M>) {
        assert_eq!(
            rt.root_depth(),
            self.base + self.len,
            "RootVec::release: not the top shadow frame (releases must be LIFO)"
        );
        rt.truncate_roots(self.base);
        self.released = true;
    }
}

impl Drop for RootVec {
    fn drop(&mut self) {
        // An unreleased frame leaks shadow slots: the collector keeps tracing
        // them, and every later frame's base is off by the leak. Loud, per
        // project law — but never while already unwinding, which would abort the
        // process and bury the original panic.
        if !self.released && !std::thread::panicking() {
            panic!(
                "RootVec dropped without release(): {} shadow slot(s) leaked at base {}",
                self.len, self.base
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use microlang::{LowBitModel, Runtime, Val};

    #[test]
    #[should_panic(expected = "RootVec dropped without release()")]
    fn unreleased_frame_is_a_loud_panic() {
        let mut rt = Runtime::<LowBitModel>::new();
        let v = rt.encode(Val::Int(1));
        let _leaked = RootVec::one(&mut rt, v); // never released
    }

    #[test]
    fn get_re_reads_the_slot_after_a_move() {
        let mut rt = Runtime::<LowBitModel>::new();
        let s = rt.alloc(microlang::Obj::Str("hello".to_string()));
        let bits = <<LowBitModel as ValueModel>::R as microlang::Repr>::enc_ref(s);
        let f = RootVec::one(&mut rt, bits);
        let before = f.get(&rt, 0);
        rt.collect(&None); // a full moving collection relocates it
        let after = f.get(&rt, 0);
        assert_ne!(before, after, "the object should have MOVED (else this proves nothing)");
        assert_eq!(rt.str_view(after), Some("hello"), "the handle must yield the CURRENT address");
        f.release(&mut rt);
    }
}
