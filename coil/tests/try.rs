//! lib/try.coil — `?`/try-style early exit for Result/Option, a PURE LIBRARY MACRO
//! over block/return-from (over the loop primitive). Replaces Result/Option match
//! pyramids; the compiler knows nothing of it.

mod common;
use common::build_and_run;

const IMPORT: &str = concat!(
    "(module app)\n",
    "(import \"lib/try.coil\" :use *)\n",
    "(import \"lib/result.coil\" :use *)\n",
);

fn run(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn try_result_unwraps_ok_and_short_circuits_err() {
    // half: Ok(n/2) for even, Err(n) for odd. f chains two halvings with try!.
    let body = r#"(defn half [(n i64)] (-> (Result i64 i64))
                    (if (icmp-eq (irem n 2) 0) (Ok (idiv n 2)) (Err n)))
                  (defn f [(n i64)] (-> (Result i64 i64))
                    (try (let [a (try! (half n)) b (try! (half a))] (Ok (iadd a b)))))
                  (defn main [] (-> :i64)
                    (let [r8 (match (f 8) (Ok [v] v) (Err [e] -1))         ; 8->4->2 = Ok 6
                          r6 (match (f 6) (Ok [v] (iadd v 100)) (Err [e] e))] ; 6->3 -> Err 3
                      (iadd (imul r8 10) r6)))"#; // 6*10 + 3 = 63
    assert_eq!(run(body), 63);
}

#[test]
fn try_option_unwraps_some_and_short_circuits_none() {
    let body = r#"(defn look [(n i64)] (-> (Option i64))
                    (if (icmp-eq n 0) (None) (Some (imul n 2))))
                  (defn g [(n i64)] (-> (Option i64))
                    (try (let [a (try? (look n)) b (try? (look a))] (Some (iadd a b)))))
                  (defn main [] (-> :i64)
                    (let [g3 (match (g 3) (None [] -1) (Some [v] v))    ; 3->6->12 = Some 18
                          g0 (match (g 0) (None [] 7) (Some [v] 0))]    ; 0 -> None -> 7
                      (iadd (imul g3 10) g0)))"#; // 18*10 + 7 = 187
    assert_eq!(run(body), 187);
}
