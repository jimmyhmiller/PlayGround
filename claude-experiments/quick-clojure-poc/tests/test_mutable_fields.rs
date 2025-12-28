use std::fs;
use std::path::PathBuf;
/// Unit tests for mutable fields in deftype*
///
/// Tests the ^:mutable field metadata and set! on field access including:
/// - Basic mutable field set!/get
/// - Multiple mutations
/// - Multiple mutable fields
/// - Mixed mutable and immutable fields
/// - Write barriers with GC stress (gc-always mode)
/// - Linked structures with mutable pointers
/// - Counter/accumulator patterns
use std::process::Command;
use std::sync::OnceLock;

static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_binary_path() -> &'static PathBuf {
    BINARY_PATH.get_or_init(|| {
        let status = Command::new("cargo")
            .args(&["build", "--release", "--quiet"])
            .status()
            .expect("Failed to build release binary");

        assert!(status.success(), "Failed to build release binary");

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir).join("target/release/quick-clojure-poc")
    })
}

fn run_code(code: &str) -> (String, String) {
    let binary_path = get_binary_path();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let temp_path = temp_dir.path().join("test.clj");
    fs::write(&temp_path, code).expect("Failed to write temp file");

    let output = Command::new(&binary_path)
        .arg(temp_path.to_str().unwrap())
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    (stdout, stderr)
}

fn run_and_get_stdout(code: &str) -> String {
    run_code(code).0
}

/// Run code with gc-always mode enabled (GC before every allocation)
fn run_code_gc_always(code: &str) -> (String, String) {
    let binary_path = get_binary_path();

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let temp_path = temp_dir.path().join("test.clj");
    fs::write(&temp_path, code).expect("Failed to write temp file");

    let output = Command::new(&binary_path)
        .arg("--gc-always")
        .arg(temp_path.to_str().unwrap())
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    (stdout, stderr)
}

fn run_gc_always_and_get_stdout(code: &str) -> String {
    run_code_gc_always(code).0
}

// ============================================================================
// Basic mutable field tests
// ============================================================================

#[test]
fn test_mutable_field_initial_value() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(println (.-val b))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_mutable_field_set() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) 100)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100 after set!, got: {}", output);
}

#[test]
fn test_mutable_field_set_returns_value() {
    // set! should return the value that was set
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(println (set! (.-val b) 999))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "999", "set! should return 999, got: {}", output);
}

#[test]
fn test_mutable_field_multiple_sets() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 0))
(do
  (set! (.-val b) 1)
  (set! (.-val b) 2)
  (set! (.-val b) 3)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "3",
        "Expected 3 after multiple sets, got: {}",
        output
    );
}

// ============================================================================
// Counter pattern tests
// ============================================================================

#[test]
fn test_counter_increment() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 0))
(do
  (set! (.-count c) (+ (.-count c) 1))
  (println (.-count c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1 after increment, got: {}", output);
}

#[test]
fn test_counter_multiple_increments() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 0))
(do
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (println (.-count c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "5",
        "Expected 5 after 5 increments, got: {}",
        output
    );
}

#[test]
fn test_counter_with_initial_value() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 100))
(do
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (println (.-count c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "102", "Expected 102, got: {}", output);
}

#[test]
fn test_counter_decrement() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 10))
(do
  (set! (.-count c) (- (.-count c) 1))
  (set! (.-count c) (- (.-count c) 1))
  (set! (.-count c) (- (.-count c) 1))
  (println (.-count c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "7",
        "Expected 7 after 3 decrements from 10, got: {}",
        output
    );
}

// ============================================================================
// Multiple mutable fields tests
// ============================================================================

#[test]
fn test_multiple_mutable_fields() {
    let code = r#"
(deftype* Point [^:mutable x ^:mutable y])
(def p (Point. 0 0))
(do
  (set! (.-x p) 10)
  (set! (.-y p) 20)
  (println (+ (.-x p) (.-y p))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "30", "Expected 30 (10+20), got: {}", output);
}

#[test]
fn test_multiple_mutable_fields_independent() {
    let code = r#"
(deftype* Point [^:mutable x ^:mutable y])
(def p (Point. 5 5))
(do
  (set! (.-x p) 100)
  (println (.-y p)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "5",
        "Setting x should not affect y, got: {}",
        output
    );
}

#[test]
fn test_three_mutable_fields() {
    let code = r#"
(deftype* Point3D [^:mutable x ^:mutable y ^:mutable z])
(def p (Point3D. 1 2 3))
(do
  (set! (.-x p) 10)
  (set! (.-y p) 20)
  (set! (.-z p) 30)
  (println (+ (.-x p) (+ (.-y p) (.-z p)))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "60", "Expected 60 (10+20+30), got: {}", output);
}

// ============================================================================
// Mixed mutable and immutable fields tests
// ============================================================================

#[test]
fn test_mixed_fields_read_immutable() {
    let code = r#"
(deftype* MixedBox [immutable-val ^:mutable mutable-val])
(def m (MixedBox. 100 200))
(println (.-immutable-val m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100, got: {}", output);
}

#[test]
fn test_mixed_fields_read_mutable() {
    let code = r#"
(deftype* MixedBox [immutable-val ^:mutable mutable-val])
(def m (MixedBox. 100 200))
(println (.-mutable-val m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "200", "Expected 200, got: {}", output);
}

#[test]
fn test_mixed_fields_set_mutable() {
    let code = r#"
(deftype* MixedBox [immutable-val ^:mutable mutable-val])
(def m (MixedBox. 100 200))
(do
  (set! (.-mutable-val m) 999)
  (println (.-mutable-val m)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "999", "Expected 999, got: {}", output);
}

#[test]
fn test_mixed_fields_immutable_unchanged() {
    let code = r#"
(deftype* MixedBox [immutable-val ^:mutable mutable-val])
(def m (MixedBox. 100 200))
(do
  (set! (.-mutable-val m) 999)
  (println (.-immutable-val m)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "100",
        "Immutable field should be unchanged, got: {}",
        output
    );
}

#[test]
fn test_first_field_mutable_second_immutable() {
    let code = r#"
(deftype* FlippedBox [^:mutable first-val second-val])
(def f (FlippedBox. 10 20))
(do
  (set! (.-first-val f) 100)
  (println (+ (.-first-val f) (.-second-val f))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "120", "Expected 120 (100+20), got: {}", output);
}

// ============================================================================
// Storing different value types
// ============================================================================

#[test]
fn test_mutable_store_boolean_true() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) true)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "true", "Expected true, got: {}", output);
}

#[test]
fn test_mutable_store_boolean_false() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) false)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "false", "Expected false, got: {}", output);
}

#[test]
fn test_mutable_store_nil() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) nil)
  (println (= (.-val b) nil)))
"#;
    let output = run_and_get_stdout(code);
    // nil doesn't print, so we check if it equals nil
    assert_eq!(
        output, "true",
        "Expected true (val == nil), got: {}",
        output
    );
}

#[test]
fn test_mutable_store_negative() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) -100)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "-100", "Expected -100, got: {}", output);
}

#[test]
fn test_mutable_store_zero() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) 0)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "0", "Expected 0, got: {}", output);
}

// ============================================================================
// Linked structures with mutable next pointers
// ============================================================================

#[test]
fn test_linked_list_basic() {
    let code = r#"
(deftype* Node [value ^:mutable next])
(def n1 (Node. 1 nil))
(def n2 (Node. 2 nil))
(do
  (set! (.-next n1) n2)
  (println (.-value (.-next n1))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "2", "Expected 2 (value of n2), got: {}", output);
}

#[test]
fn test_linked_list_three_nodes() {
    let code = r#"
(deftype* Node [value ^:mutable next])
(def n1 (Node. 1 nil))
(def n2 (Node. 2 nil))
(def n3 (Node. 3 nil))
(do
  (set! (.-next n1) n2)
  (set! (.-next n2) n3)
  (println (.-value (.-next (.-next n1)))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "3", "Expected 3 (value of n3), got: {}", output);
}

#[test]
fn test_linked_list_rewire() {
    // Test rewiring a linked list
    let code = r#"
(deftype* Node [value ^:mutable next])
(def n1 (Node. 1 nil))
(def n2 (Node. 2 nil))
(def n3 (Node. 3 nil))
(do
  (set! (.-next n1) n2)
  (set! (.-next n1) n3)
  (println (.-value (.-next n1))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "3",
        "After rewire, n1->n3, expected 3, got: {}",
        output
    );
}

#[test]
fn test_linked_list_sum() {
    // Sum values in a linked list
    let code = r#"
(deftype* Node [value ^:mutable next])
(def sum-list
  (fn [node]
    (if (= node nil)
      0
      (+ (.-value node) (sum-list (.-next node))))))
(def n1 (Node. 10 nil))
(def n2 (Node. 20 nil))
(def n3 (Node. 30 nil))
(do
  (set! (.-next n1) n2)
  (set! (.-next n2) n3)
  (println (sum-list n1)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "60", "Expected 60 (10+20+30), got: {}", output);
}

// ============================================================================
// Multiple instances
// ============================================================================

#[test]
fn test_multiple_instances_independent() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b1 (Box. 1))
(def b2 (Box. 2))
(do
  (set! (.-val b1) 100)
  (println (.-val b2)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "2",
        "Changing b1 should not affect b2, got: {}",
        output
    );
}

#[test]
fn test_multiple_instances_both_modified() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b1 (Box. 1))
(def b2 (Box. 2))
(do
  (set! (.-val b1) 100)
  (set! (.-val b2) 200)
  (println (+ (.-val b1) (.-val b2))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "300", "Expected 300 (100+200), got: {}", output);
}

// ============================================================================
// Mutable fields in functions
// ============================================================================

#[test]
fn test_set_in_function() {
    let code = r#"
(deftype* Box [^:mutable val])
(def set-box! (fn [b v] (set! (.-val b) v)))
(def b (Box. 0))
(do
  (set-box! b 42)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "42",
        "Expected 42 after set in function, got: {}",
        output
    );
}

#[test]
fn test_increment_in_function() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def inc! (fn [c] (set! (.-count c) (+ (.-count c) 1))))
(def c (Counter. 0))
(do
  (inc! c)
  (inc! c)
  (inc! c)
  (println (.-count c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "3",
        "Expected 3 after 3 inc! calls, got: {}",
        output
    );
}

#[test]
fn test_swap_in_function() {
    // Swap two mutable fields
    let code = r#"
(deftype* Pair [^:mutable first ^:mutable second])
(def swap! (fn [p]
  (let [temp (.-first p)]
    (set! (.-first p) (.-second p))
    (set! (.-second p) temp))))
(def p (Pair. 1 2))
(do
  (swap! p)
  (println (+ (* (.-first p) 10) (.-second p))))
"#;
    let output = run_and_get_stdout(code);
    // After swap: first=2, second=1, so 2*10 + 1 = 21
    assert_eq!(output, "21", "Expected 21 (2*10+1), got: {}", output);
}

// ============================================================================
// GC stress tests (gc-always mode) - testing write barriers
// ============================================================================

#[test]
fn test_gc_always_basic_mutable() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) 100)
  (println (.-val b)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "100", "gc-always: Expected 100, got: {}", output);
}

#[test]
fn test_gc_always_counter() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 0))
(do
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (set! (.-count c) (+ (.-count c) 1))
  (println (.-count c)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "5", "gc-always: Expected 5, got: {}", output);
}

#[test]
fn test_gc_always_linked_list() {
    // This tests write barriers - storing pointers to other heap objects
    let code = r#"
(deftype* Node [value ^:mutable next])
(def sum-list
  (fn [node]
    (if (= node nil)
      0
      (+ (.-value node) (sum-list (.-next node))))))
(def n1 (Node. 10 nil))
(def n2 (Node. 20 nil))
(def n3 (Node. 30 nil))
(def n4 (Node. 40 nil))
(def n5 (Node. 50 nil))
(do
  (set! (.-next n1) n2)
  (set! (.-next n2) n3)
  (set! (.-next n3) n4)
  (set! (.-next n4) n5)
  (println (sum-list n1)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(
        output, "150",
        "gc-always: Expected 150 (10+20+30+40+50), got: {}",
        output
    );
}

#[test]
fn test_gc_always_linked_list_rewire() {
    // Test that write barriers work when rewiring pointers
    let code = r#"
(deftype* Node [value ^:mutable next])
(def n1 (Node. 1 nil))
(def n2 (Node. 2 nil))
(def n3 (Node. 3 nil))
(do
  (set! (.-next n1) n2)
  (set! (.-next n1) n3)
  (println (.-value (.-next n1))))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(
        output, "3",
        "gc-always: After rewire expected 3, got: {}",
        output
    );
}

#[test]
fn test_gc_always_multiple_mutable_fields() {
    let code = r#"
(deftype* Point [^:mutable x ^:mutable y ^:mutable z])
(def p (Point. 0 0 0))
(do
  (set! (.-x p) 10)
  (set! (.-y p) 20)
  (set! (.-z p) 30)
  (println (+ (.-x p) (+ (.-y p) (.-z p)))))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "60", "gc-always: Expected 60, got: {}", output);
}

#[test]
fn test_gc_always_many_allocations_between_sets() {
    // Allocate many objects between set! calls to stress test GC
    let code = r#"
(deftype* Box [^:mutable val])
(deftype* Dummy [a b c])
(def b (Box. 0))
(do
  (def d1 (Dummy. 1 2 3))
  (def d2 (Dummy. 4 5 6))
  (def d3 (Dummy. 7 8 9))
  (set! (.-val b) 100)
  (def d4 (Dummy. 10 11 12))
  (def d5 (Dummy. 13 14 15))
  (set! (.-val b) 200)
  (def d6 (Dummy. 16 17 18))
  (set! (.-val b) 300)
  (println (.-val b)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "300", "gc-always: Expected 300, got: {}", output);
}

#[test]
fn test_gc_always_store_new_allocation() {
    // Store a freshly allocated object into a mutable field
    // This tests write barriers with young->old generation references
    let code = r#"
(deftype* Node [value ^:mutable next])
(def head (Node. 0 nil))
(do
  (set! (.-next head) (Node. 1 nil))
  (set! (.-next (.-next head)) (Node. 2 nil))
  (set! (.-next (.-next (.-next head))) (Node. 3 nil))
  (println (.-value (.-next (.-next (.-next head))))))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "3", "gc-always: Expected 3, got: {}", output);
}

#[test]
fn test_gc_always_tree_with_mutable_children() {
    let code = r#"
(deftype* TreeNode [value ^:mutable left ^:mutable right])
(def sum-tree
  (fn [node]
    (if (= node nil)
      0
      (+ (.-value node)
         (+ (sum-tree (.-left node))
            (sum-tree (.-right node)))))))

(def leaf1 (TreeNode. 1 nil nil))
(def leaf2 (TreeNode. 2 nil nil))
(def leaf3 (TreeNode. 3 nil nil))
(def leaf4 (TreeNode. 4 nil nil))

(def branch1 (TreeNode. 10 nil nil))
(def branch2 (TreeNode. 20 nil nil))

(do
  (set! (.-left branch1) leaf1)
  (set! (.-right branch1) leaf2)
  (set! (.-left branch2) leaf3)
  (set! (.-right branch2) leaf4)

  (def root (TreeNode. 100 nil nil))
  (set! (.-left root) branch1)
  (set! (.-right root) branch2)

  (println (sum-tree root)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    // 100 + 10 + 20 + 1 + 2 + 3 + 4 = 140
    assert_eq!(output, "140", "gc-always: Expected 140, got: {}", output);
}

// ============================================================================
// Loop-based stress tests
// ============================================================================

#[test]
fn test_loop_counter() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 0))
(do
  (loop [i 0]
    (if (< i 10)
      (do
        (set! (.-count c) (+ (.-count c) 1))
        (recur (+ i 1)))
      nil))
  (println (.-count c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "10", "Expected 10 after loop, got: {}", output);
}

#[test]
fn test_loop_counter_gc_always() {
    let code = r#"
(deftype* Counter [^:mutable count])
(def c (Counter. 0))
(do
  (loop [i 0]
    (if (< i 10)
      (do
        (set! (.-count c) (+ (.-count c) 1))
        (recur (+ i 1)))
      nil))
  (println (.-count c)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(
        output, "10",
        "gc-always: Expected 10 after loop, got: {}",
        output
    );
}

#[test]
fn test_loop_build_linked_list() {
    // Build a linked list in a loop and sum it
    let code = r#"
(deftype* Node [value ^:mutable next])
(def sum-list
  (fn [node]
    (if (= node nil)
      0
      (+ (.-value node) (sum-list (.-next node))))))

(def head (Node. 0 nil))

(do
  (loop [i 1 prev head]
    (if (< i 6)
      (let [new-node (Node. i nil)]
        (set! (.-next prev) new-node)
        (recur (+ i 1) new-node))
      nil))
  (println (sum-list head)))
"#;
    let output = run_and_get_stdout(code);
    // 0 + 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(output, "15", "Expected 15, got: {}", output);
}

#[test]
fn test_loop_build_linked_list_gc_always() {
    let code = r#"
(deftype* Node [value ^:mutable next])
(def sum-list
  (fn [node]
    (if (= node nil)
      0
      (+ (.-value node) (sum-list (.-next node))))))

(def head (Node. 0 nil))

(do
  (loop [i 1 prev head]
    (if (< i 6)
      (let [new-node (Node. i nil)]
        (set! (.-next prev) new-node)
        (recur (+ i 1) new-node))
      nil))
  (println (sum-list head)))
"#;
    let output = run_gc_always_and_get_stdout(code);
    // 0 + 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(output, "15", "gc-always: Expected 15, got: {}", output);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_set_same_value() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 42))
(do
  (set! (.-val b) 42)
  (println (.-val b)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_set_then_read_multiple_times() {
    let code = r#"
(deftype* Box [^:mutable val])
(def b (Box. 0))
(do
  (set! (.-val b) 42)
  (println (+ (.-val b) (+ (.-val b) (.-val b)))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "126", "Expected 126 (42*3), got: {}", output);
}

#[test]
fn test_deeply_nested_field_access() {
    let code = r#"
(deftype* Wrapper [^:mutable inner])
(def w1 (Wrapper. nil))
(def w2 (Wrapper. nil))
(def w3 (Wrapper. nil))
(def w4 (Wrapper. 42))
(do
  (set! (.-inner w1) w2)
  (set! (.-inner w2) w3)
  (set! (.-inner w3) w4)
  (println (.-inner (.-inner (.-inner (.-inner w1))))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42 at depth 4, got: {}", output);
}

#[test]
fn test_circular_reference() {
    // Test creating a circular reference (if it doesn't crash, it's fine)
    let code = r#"
(deftype* Node [value ^:mutable next])
(def n1 (Node. 1 nil))
(def n2 (Node. 2 nil))
(do
  (set! (.-next n1) n2)
  (set! (.-next n2) n1)
  (println (.-value (.-next (.-next n1)))))
"#;
    let output = run_and_get_stdout(code);
    // n1 -> n2 -> n1, so (.-next (.-next n1)) = n1, value = 1
    assert_eq!(output, "1", "Circular ref: expected 1, got: {}", output);
}
