//! The speculation + deopt axis: one program, three swappable policies.
//!
//! `Speculative` is a dispatch strategy that wraps a fallback dispatch plus a
//! `SpeculationPolicy`. On a guard hit it uses a cached target; on a miss it
//! DEOPTS — reconciles with the real receiver type via the fallback, so the
//! result never changes — and the policy decides whether to re-arm or blacklist
//! the site. Swapping the policy changes only the speculation behavior, never
//! the answer.

use microlang::{
    AlwaysMonomorphic, BlacklistAfter, LowBitModel, Megamorphic, NeverSpeculate, Runtime,
    SpeculationPolicy, Speculative, TreeWalk,
};

const DEFS: &str = r#"
    (defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
    (defmethod area Square (fn (s) (+ (field s 0) (field s 0))))
    (def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))
"#;

fn run(label: &str, policy: impl SpeculationPolicy + 'static, list: &str) {
    let mut rt = Runtime::<LowBitModel>::new();
    let spec = Speculative::new(Megamorphic::new(), policy);
    let counters = spec.counters();
    rt.set_dispatch(Box::new(spec));
    let cs = TreeWalk;
    let r = microlang::sexpr::eval_str(&mut rt, &cs, &format!("{DEFS}(total {list})"));
    let s = counters.snapshot();
    println!(
        "  [{label:18}] => {}    (spec-hits {}, deopts {}, fallbacks {})",
        rt.print(r),
        s.spec_hits,
        s.deopts,
        s.fallbacks
    );
}

fn main() {
    let poly = "(list (record 'Circle 3) (record 'Square 4) (record 'Circle 5) (record 'Square 6))";
    let mono = "(list (record 'Circle 3) (record 'Circle 4) (record 'Circle 5) (record 'Circle 6))";

    println!("polymorphic site  [Circle Square Circle Square]  (answer 54):");
    run("NeverSpeculate", NeverSpeculate, poly);
    run("AlwaysMonomorphic", AlwaysMonomorphic, poly);
    run("BlacklistAfter(2)", BlacklistAfter(2), poly);

    println!("\nmonomorphic site  [Circle Circle Circle Circle]  (answer 86):");
    run("AlwaysMonomorphic", AlwaysMonomorphic, mono);

    println!(
        "\nSame answer every time — speculation never changes results, only which\n\
         path runs. Never = pure fallback; Always = deopt-thrash on the poly site;\n\
         Blacklist = deopt twice then give up; and on the mono site it speculates\n\
         once and hits thereafter. Speculation is a dispatch strategy, so it works\n\
         under the interpreter and the closure-compiler alike."
    );
}
