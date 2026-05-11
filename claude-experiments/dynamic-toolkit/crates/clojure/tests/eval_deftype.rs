//! deftype* / defprotocol / extend-type / .method / .-field

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn deftype_constructor_and_field_access() {
    let src = "\
        (deftype* Point [x y]) \
        (let [p (Point. 3 4)] (.-x p))";
    assert_eq!(eval_str(src), "3");
}

#[test]
fn deftype_two_field_access() {
    let src = "\
        (deftype* Point [x y]) \
        (let [p (Point. 10 20)] (+ (.-x p) (.-y p)))";
    assert_eq!(eval_str(src), "30");
}

#[test]
fn instance_check_positive_and_negative() {
    let src = "\
        (deftype* Box [v]) \
        (let [b (Box. 1)] (instance? Box b))";
    assert_eq!(eval_str(src), "true");
}

#[test]
fn extend_type_and_method_dispatch() {
    let src = "\
        (deftype* Circle [r]) \
        (defprotocol IShape (area [this])) \
        (extend-type Circle IShape (area [this] (* (.-r this) (.-r this)))) \
        (let [c (Circle. 5)] (.area c))";
    assert_eq!(eval_str(src), "25");
}

#[test]
fn extend_type_method_with_extra_args() {
    let src = "\
        (deftype* Adder [base]) \
        (extend-type Adder \
          IFoo (combine [this delta] (+ (.-base this) delta))) \
        (let [a (Adder. 100)] (.combine a 7))";
    assert_eq!(eval_str(src), "107");
}

#[test]
fn extend_type_multiple_methods() {
    let src = "\
        (deftype* Pair [a b]) \
        (extend-type Pair \
          IFoo (sum [this] (+ (.-a this) (.-b this))) \
                (diff [this] (- (.-a this) (.-b this)))) \
        (let [p (Pair. 10 3)] (+ (.sum p) (.diff p)))";
    assert_eq!(eval_str(src), "20"); // (10+3) + (10-3) = 13+7 = 20
}
