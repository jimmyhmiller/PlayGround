/// Unit tests for deftype* implementation
///
/// Tests the deftype special form including:
/// - Type definition
/// - Instance construction (TypeName. args)
/// - Field access (.-field obj)
/// - Nested types
/// - Error handling

use std::process::Command;
use std::fs;
use std::sync::OnceLock;
use std::path::PathBuf;

static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_binary_path() -> &'static PathBuf {
    BINARY_PATH.get_or_init(|| {
        let status = Command::new("cargo")
            .args(&["build", "--release", "--quiet"])
            .status()
            .expect("Failed to build release binary");

        assert!(status.success(), "Failed to build release binary");

        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| ".".to_string());
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

fn run_and_get_stderr(code: &str) -> String {
    run_code(code).1
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
// Basic deftype and construction tests
// ============================================================================

#[test]
fn test_deftype_basic_construction() {
    let code = r#"
(deftype* Point [x y])
(println (Point. 10 20))
"#;
    let output = run_and_get_stdout(code);
    // Should return a heap object representation (may include type name like #<user/Point@...>)
    assert!(output.contains("#<") && output.contains("@"), "Expected object output, got: {}", output);
}

#[test]
fn test_deftype_field_access_x_y() {
    let code = r#"
(deftype* Point [x y])
(def p (Point. 10 20))
(println (.-x p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "10", "Expected 10, got: {}", output);
}

#[test]
fn test_deftype_field_access_y() {
    let code = r#"
(deftype* Point [x y])
(def p (Point. 10 20))
(println (.-y p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "20", "Expected 20, got: {}", output);
}

#[test]
fn test_deftype_both_fields() {
    let code = r#"
(deftype* Point [x y])
(def p (Point. 3 7))
(println (+ (.-x p) (.-y p)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "10", "Expected 10 (3+7), got: {}", output);
}

// ============================================================================
// Arbitrary field name tests (not hardcoded)
// ============================================================================

#[test]
fn test_deftype_arbitrary_field_names() {
    let code = r#"
(deftype* Person [name age city])
(def p (Person. 100 25 999))
(println (.-age p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "25", "Expected 25, got: {}", output);
}

#[test]
fn test_deftype_first_arbitrary_field() {
    let code = r#"
(deftype* Person [name age city])
(def p (Person. 100 25 999))
(println (.-name p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100, got: {}", output);
}

#[test]
fn test_deftype_last_arbitrary_field() {
    let code = r#"
(deftype* Person [name age city])
(def p (Person. 100 25 999))
(println (.-city p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "999", "Expected 999, got: {}", output);
}

#[test]
fn test_deftype_long_field_names() {
    let code = r#"
(deftype* Config [database-connection-string max-retry-count timeout-milliseconds])
(def c (Config. 1 5 3000))
(println (.-timeout-milliseconds c))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "3000", "Expected 3000, got: {}", output);
}

// ============================================================================
// Nested types tests
// ============================================================================

#[test]
fn test_deftype_nested_types() {
    let code = r#"
(deftype* Point [x y])
(deftype* Line [start end])
(def p1 (Point. 0 0))
(def p2 (Point. 10 20))
(def line (Line. p1 p2))
(println (.-x (.-start line)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "0", "Expected 0, got: {}", output);
}

#[test]
fn test_deftype_nested_end_field() {
    let code = r#"
(deftype* Point [x y])
(deftype* Line [start end])
(def p1 (Point. 0 0))
(def p2 (Point. 10 20))
(def line (Line. p1 p2))
(println (.-y (.-end line)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "20", "Expected 20, got: {}", output);
}

// ============================================================================
// Types in functions tests
// ============================================================================

#[test]
fn test_deftype_in_function_constructor() {
    let code = r#"
(deftype* Point [x y])
(def make-point (fn [x y] (Point. x y)))
(def p (make-point 5 15))
(println (.-x p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "5", "Expected 5, got: {}", output);
}

#[test]
fn test_deftype_in_function_accessor() {
    let code = r#"
(deftype* Point [x y])
(def get-x (fn [p] (.-x p)))
(def p (Point. 42 99))
(println (get-x p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_deftype_function_with_field_operations() {
    let code = r#"
(deftype* Point [x y])
(def distance-from-origin (fn [p] (+ (.-x p) (.-y p))))
(def p (Point. 3 4))
(println (distance-from-origin p))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "7", "Expected 7, got: {}", output);
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn test_deftype_invalid_field_error() {
    let code = r#"
(deftype* Point [x y])
(def p (Point. 10 20))
(println (.-invalid p))
"#;
    let stderr = run_and_get_stderr(code);
    assert!(stderr.contains("Field 'invalid' not found"),
            "Expected error about invalid field, got: {}", stderr);
}

#[test]
fn test_deftype_invalid_field_with_type_name() {
    let code = r#"
(deftype* Point [x y])
(def p (Point. 10 20))
(println (.-z p))
"#;
    let stderr = run_and_get_stderr(code);
    // Error should mention the type name
    assert!(stderr.contains("Point"),
            "Expected error to mention type 'Point', got: {}", stderr);
}

// ============================================================================
// Multiple type definitions
// ============================================================================

#[test]
fn test_deftype_multiple_types() {
    let code = r#"
(deftype* Point [x y])
(deftype* Rectangle [width height])
(def p (Point. 1 2))
(def r (Rectangle. 100 50))
(println (+ (.-x p) (.-width r)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "101", "Expected 101 (1+100), got: {}", output);
}

#[test]
fn test_deftype_same_field_names_different_types() {
    // Two types with same field names - field lookup should use runtime type
    let code = r#"
(deftype* Point2D [x y])
(deftype* Point3D [x y z])
(def p2 (Point2D. 10 20))
(def p3 (Point3D. 100 200 300))
(println (+ (.-x p2) (.-x p3)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "110", "Expected 110 (10+100), got: {}", output);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_deftype_single_field() {
    let code = r#"
(deftype* Wrapper [value])
(def w (Wrapper. 42))
(println (.-value w))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_deftype_many_fields() {
    let code = r#"
(deftype* BigType [a b c d e f])
(def bt (BigType. 1 2 3 4 5 6))
(println (+ (.-a bt) (.-f bt)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "7", "Expected 7 (1+6), got: {}", output);
}

#[test]
fn test_deftype_field_containing_other_values() {
    // Fields can hold any tagged values including booleans
    let code = r#"
(deftype* Flags [enabled count])
(def f (Flags. true 5))
(println (.-count f))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "5", "Expected 5, got: {}", output);
}

// ============================================================================
// Deep nesting and recursive traversal tests
// ============================================================================

#[test]
fn test_deftype_binary_tree_sum() {
    // Test recursive tree traversal with field access at every level
    let code = r#"
(deftype* Node [value left right])
(def sum-tree (fn [node] (if (= node nil) 0 (+ (.-value node) (+ (sum-tree (.-left node)) (sum-tree (.-right node)))))))
(def leaf1 (Node. 1 nil nil))
(def leaf2 (Node. 2 nil nil))
(def leaf3 (Node. 3 nil nil))
(def leaf4 (Node. 4 nil nil))
(def branch1 (Node. 10 leaf1 leaf2))
(def branch2 (Node. 20 leaf3 leaf4))
(def root (Node. 100 branch1 branch2))
(println (sum-tree root))
"#;
    let output = run_and_get_stdout(code);
    // 100 + 10 + 20 + 1 + 2 + 3 + 4 = 140
    assert_eq!(output, "140", "Expected 140, got: {}", output);
}

#[test]
fn test_deftype_complete_binary_tree_31_nodes() {
    // Complete binary tree with 5 levels = 31 nodes
    let code = r#"
(deftype* Node [value left right])
(def sum-tree (fn [node] (if (= node nil) 0 (+ (.-value node) (+ (sum-tree (.-left node)) (sum-tree (.-right node)))))))
(def l1 (Node. 1 nil nil))
(def l2 (Node. 1 nil nil))
(def l3 (Node. 1 nil nil))
(def l4 (Node. 1 nil nil))
(def l5 (Node. 1 nil nil))
(def l6 (Node. 1 nil nil))
(def l7 (Node. 1 nil nil))
(def l8 (Node. 1 nil nil))
(def l9 (Node. 1 nil nil))
(def l10 (Node. 1 nil nil))
(def l11 (Node. 1 nil nil))
(def l12 (Node. 1 nil nil))
(def l13 (Node. 1 nil nil))
(def l14 (Node. 1 nil nil))
(def l15 (Node. 1 nil nil))
(def l16 (Node. 1 nil nil))
(def n1 (Node. 1 l1 l2))
(def n2 (Node. 1 l3 l4))
(def n3 (Node. 1 l5 l6))
(def n4 (Node. 1 l7 l8))
(def n5 (Node. 1 l9 l10))
(def n6 (Node. 1 l11 l12))
(def n7 (Node. 1 l13 l14))
(def n8 (Node. 1 l15 l16))
(def m1 (Node. 1 n1 n2))
(def m2 (Node. 1 n3 n4))
(def m3 (Node. 1 n5 n6))
(def m4 (Node. 1 n7 n8))
(def p1 (Node. 1 m1 m2))
(def p2 (Node. 1 m3 m4))
(def root (Node. 1 p1 p2))
(println (sum-tree root))
"#;
    let output = run_and_get_stdout(code);
    // 31 nodes, each with value 1
    assert_eq!(output, "31", "Expected 31, got: {}", output);
}

#[test]
fn test_deftype_deep_field_chain() {
    // 8 levels of nesting via left pointer
    let code = r#"
(deftype* Node [value left right])
(def n1 (Node. 1 nil nil))
(def n2 (Node. 2 n1 nil))
(def n3 (Node. 3 n2 nil))
(def n4 (Node. 4 n3 nil))
(def n5 (Node. 5 n4 nil))
(def n6 (Node. 6 n5 nil))
(def n7 (Node. 7 n6 nil))
(def n8 (Node. 8 n7 nil))
(println (.-value (.-left (.-left (.-left (.-left (.-left (.-left (.-left n8)))))))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1 at depth 8, got: {}", output);
}

// ============================================================================
// Namespace tests
// ============================================================================

#[test]
fn test_deftype_different_namespaces_no_collision() {
    // Two types with same name in different namespaces should not collide
    let code = r#"
(ns foo)
(deftype* Point [x y])
(def p1 (Point. 1 2))
(ns bar)
(deftype* Point [a b c])
(def p2 (Point. 10 20 30))
(println (.-a p2))
"#;
    let output = run_and_get_stdout(code);
    // bar/Point has 3 fields, should get 10 for field 'a'
    assert_eq!(output, "10", "Expected 10, got: {}", output);
}

#[test]
fn test_deftype_qualified_constructor_call() {
    // Should be able to call constructor with qualified name
    let code = r#"
(ns myns)
(deftype* Widget [id])
(ns other)
(def w (myns/Widget. 42))
(println (.-id w))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_deftype_same_name_different_fields_different_ns() {
    // Point in ns1 has [x y], Point in ns2 has [a b c]
    // They should be completely independent
    let code = r#"
(ns ns1)
(deftype* Point [x y])
(ns ns2)
(deftype* Point [a b c])
(def p2 (Point. 1 2 3))
(println (.-b p2))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "2", "Expected 2, got: {}", output);
}

// ============================================================================
// GC stress tests (gc-always mode)
// ============================================================================

#[test]
fn test_gc_always_binary_tree() {
    // Test binary tree with gc-always mode (GC runs before every allocation)
    // This verifies that GC can correctly scan the JIT stack and preserve live objects
    let code = r#"
(deftype* Node [value left right])
(def sum-tree (fn [node] (if (= node nil) 0 (+ (.-value node) (+ (sum-tree (.-left node)) (sum-tree (.-right node)))))))
(def leaf1 (Node. 1 nil nil))
(def leaf2 (Node. 2 nil nil))
(def leaf3 (Node. 3 nil nil))
(def leaf4 (Node. 4 nil nil))
(def branch1 (Node. 10 leaf1 leaf2))
(def branch2 (Node. 20 leaf3 leaf4))
(def root (Node. 100 branch1 branch2))
(println (sum-tree root))
"#;
    let output = run_gc_always_and_get_stdout(code);
    // 100 + 10 + 20 + 1 + 2 + 3 + 4 = 140
    assert_eq!(output, "140", "Expected 140 with gc-always mode, got: {}", output);
}

#[test]
fn test_gc_always_complete_binary_tree() {
    // Complete binary tree with 31 nodes and gc-always mode
    let code = r#"
(deftype* Node [value left right])
(def sum-tree (fn [node] (if (= node nil) 0 (+ (.-value node) (+ (sum-tree (.-left node)) (sum-tree (.-right node)))))))
(def l1 (Node. 1 nil nil))
(def l2 (Node. 1 nil nil))
(def l3 (Node. 1 nil nil))
(def l4 (Node. 1 nil nil))
(def l5 (Node. 1 nil nil))
(def l6 (Node. 1 nil nil))
(def l7 (Node. 1 nil nil))
(def l8 (Node. 1 nil nil))
(def l9 (Node. 1 nil nil))
(def l10 (Node. 1 nil nil))
(def l11 (Node. 1 nil nil))
(def l12 (Node. 1 nil nil))
(def l13 (Node. 1 nil nil))
(def l14 (Node. 1 nil nil))
(def l15 (Node. 1 nil nil))
(def l16 (Node. 1 nil nil))
(def n1 (Node. 1 l1 l2))
(def n2 (Node. 1 l3 l4))
(def n3 (Node. 1 l5 l6))
(def n4 (Node. 1 l7 l8))
(def n5 (Node. 1 l9 l10))
(def n6 (Node. 1 l11 l12))
(def n7 (Node. 1 l13 l14))
(def n8 (Node. 1 l15 l16))
(def m1 (Node. 1 n1 n2))
(def m2 (Node. 1 n3 n4))
(def m3 (Node. 1 n5 n6))
(def m4 (Node. 1 n7 n8))
(def p1 (Node. 1 m1 m2))
(def p2 (Node. 1 m3 m4))
(def root (Node. 1 p1 p2))
(println (sum-tree root))
"#;
    let output = run_gc_always_and_get_stdout(code);
    // 31 nodes, each with value 1
    assert_eq!(output, "31", "Expected 31 with gc-always mode, got: {}", output);
}
