//! `loop` / `recur` semantics + tail-call recursion for fn bodies.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn loop_no_recur_returns_body() {
    assert_eq!(eval_str("(loop [x 5] x)"), "5");
}

#[test]
fn loop_simple_countdown() {
    assert_eq!(
        eval_str("(loop [i 0 sum 0] (if (= i 10) sum (recur (+ i 1) (+ sum i))))"),
        "45"
    );
}

#[test]
fn loop_factorial() {
    assert_eq!(
        eval_str("(loop [n 5 acc 1] (if (= n 0) acc (recur (- n 1) (* acc n))))"),
        "120"
    );
}

#[test]
fn fn_recur_zero_iter() {
    let src = "\
        (def go (fn [n acc] (if (= n 0) acc (recur (- n 1) (+ acc 1))))) \
        (go 0 0)";
    assert_eq!(eval_str(src), "0");
}

#[test]
fn fn_no_recur_else() {
    // Same shape but else branch returns directly (no recur).
    let src = "\
        (def go (fn [n acc] (if (= n 0) acc (+ acc 1)))) \
        (go 1 5)";
    assert_eq!(eval_str(src), "6");
}

#[test]
fn fn_recur_one_iter() {
    let src = "\
        (def go (fn [n acc] (if (= n 0) acc (recur (- n 1) (+ acc 1))))) \
        (go 1 0)";
    assert_eq!(eval_str(src), "1");
}

#[test]
fn fn_recur_tco_in_def() {
    // (defn count-down [n acc] (if (zero? n) acc (recur (dec n) (inc acc))))
    // — but using available primitives. Pure tail recursion via recur,
    // exercising deep iteration to confirm TCO actually works.
    let src = "\
        (def go (fn [n acc] (if (= n 0) acc (recur (- n 1) (+ acc 1))))) \
        (go 10000 0)";
    assert_eq!(eval_str(src), "10000");
}

#[test]
fn recur_in_closure() {
    let src = "\
        (def count-up (fn [n] \
            (let [step (fn [i] (if (= i n) i (recur (+ i 1))))] \
              (step 0)))) \
        (count-up 5000)";
    assert_eq!(eval_str(src), "5000");
}

#[test]
fn nested_loop_innermost_wins() {
    // The inner `loop` shadows the outer for `recur`.
    assert_eq!(
        eval_str(
            "(loop [outer 0 result 0] \
                (if (= outer 3) result \
                    (let [inner-sum (loop [i 0 s 0] \
                                      (if (= i 4) s (recur (+ i 1) (+ s i))))] \
                      (recur (+ outer 1) (+ result inner-sum)))))"
        ),
        // inner-sum = 0+1+2+3 = 6, run 3 times → 18
        "18"
    );
}

#[test]
fn loop_with_zero_bindings() {
    // Useful for emulating a do-loop. We exit immediately since
    // there's no recur — the body's last expression is the result.
    assert_eq!(eval_str("(loop [] 42)"), "42");
}

#[test]
fn closure_captures_into_loop_body() {
    // The outer let's `n` is a capture in the closure; the loop
    // inside uses recur for its own iteration, but `n` flows in
    // from the surrounding env unchanged each iteration.
    let src = "\
        (let [n 3] \
          (let [f (fn [] \
                    (loop [i 0 s 0] \
                      (if (= i n) s (recur (+ i 1) (+ s i)))))] \
            (f)))";
    assert_eq!(eval_str(src), "3"); // 0+1+2 = 3
}
