use std::fs;
use std::path::PathBuf;
/// Unit tests for PersistentArrayMap and PersistentHashMap implementation
///
/// Tests the map data structures including:
/// - hash-map and array-map constructors
/// - get / -lookup operations
/// - assoc / -assoc operations
/// - dissoc / -dissoc operations
/// - contains? on maps
/// - keys and vals operations
/// - count on maps
/// - MapEntry creation and access
/// - Map equality
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
// hash-map constructor tests
// ============================================================================

#[test]
fn test_hash_map_empty() {
    let code = r#"
(def m (hash-map))
(println (count m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "0", "Expected 0, got: {}", output);
}

#[test]
fn test_hash_map_single_pair() {
    let code = r#"
(def m (hash-map :a 1))
(println (get m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

#[test]
fn test_hash_map_multiple_pairs() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(println (+ (get m :a) (+ (get m :b) (get m :c))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "6", "Expected 6 (1+2+3), got: {}", output);
}

#[test]
fn test_hash_map_count() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(println (count m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "3", "Expected 3, got: {}", output);
}

// ============================================================================
// array-map constructor tests
// ============================================================================

#[test]
fn test_array_map_empty() {
    let code = r#"
(def m (array-map))
(println (count m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "0", "Expected 0, got: {}", output);
}

#[test]
fn test_array_map_single_pair() {
    let code = r#"
(def m (array-map :x 42))
(println (get m :x))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_array_map_multiple_pairs() {
    let code = r#"
(def m (array-map :a 10 :b 20 :c 30))
(println (+ (get m :a) (get m :c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "40", "Expected 40 (10+30), got: {}", output);
}

#[test]
fn test_array_map_count() {
    let code = r#"
(def m (array-map :a 1 :b 2 :c 3 :d 4))
(println (count m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "4", "Expected 4, got: {}", output);
}

// ============================================================================
// get / -lookup tests
// ============================================================================

#[test]
fn test_get_existing_key() {
    let code = r#"
(def m (hash-map :foo 100))
(println (get m :foo))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100, got: {}", output);
}

#[test]
fn test_get_missing_key_returns_nil() {
    let code = r#"
(def m (hash-map :foo 100))
(println (get m :bar))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "nil", "Expected nil, got: {}", output);
}

#[test]
fn test_get_with_default() {
    let code = r#"
(def m (hash-map :foo 100))
(println (get m :bar 999))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "999", "Expected 999, got: {}", output);
}

#[test]
fn test_get_existing_key_ignores_default() {
    let code = r#"
(def m (hash-map :foo 100))
(println (get m :foo 999))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100, got: {}", output);
}

#[test]
fn test_get_with_integer_keys() {
    let code = r#"
(def m (hash-map 1 "one" 2 "two" 3 "three"))
(println (get m 2))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "two", "Expected two, got: {}", output);
}

// ============================================================================
// assoc tests
// ============================================================================

#[test]
fn test_assoc_new_key() {
    let code = r#"
(def m (hash-map :a 1))
(def m2 (assoc m :b 2))
(println (get m2 :b))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "2", "Expected 2, got: {}", output);
}

#[test]
fn test_assoc_update_key() {
    let code = r#"
(def m (hash-map :a 1))
(def m2 (assoc m :a 100))
(println (get m2 :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100, got: {}", output);
}

#[test]
fn test_assoc_original_unchanged() {
    let code = r#"
(def m (hash-map :a 1))
(def m2 (assoc m :a 100))
(println (get m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "1",
        "Expected original to be unchanged at 1, got: {}",
        output
    );
}

#[test]
fn test_assoc_multiple_keys() {
    let code = r#"
(def m (hash-map :a 1))
(def m2 (assoc m :b 2 :c 3 :d 4))
(println (+ (get m2 :a) (+ (get m2 :b) (+ (get m2 :c) (get m2 :d)))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "10", "Expected 10 (1+2+3+4), got: {}", output);
}

#[test]
fn test_assoc_on_empty_map() {
    let code = r#"
(def m (hash-map))
(def m2 (assoc m :key 42))
(println (get m2 :key))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

// ============================================================================
// dissoc tests
// ============================================================================

#[test]
fn test_dissoc_existing_key() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(def m2 (dissoc m :b))
(println (get m2 :b))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "nil", "Expected nil after dissoc, got: {}", output);
}

#[test]
fn test_dissoc_preserves_other_keys() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(def m2 (dissoc m :b))
(println (+ (get m2 :a) (get m2 :c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "4", "Expected 4 (1+3), got: {}", output);
}

#[test]
fn test_dissoc_original_unchanged() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(def m2 (dissoc m :b))
(println (get m :b))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "2",
        "Expected original to be unchanged at 2, got: {}",
        output
    );
}

#[test]
fn test_dissoc_missing_key() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(def m2 (dissoc m :c))
(println (count m2))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "2", "Expected 2 (map unchanged), got: {}", output);
}

#[test]
fn test_dissoc_count_decreases() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(def m2 (dissoc m :b))
(println (count m2))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "2", "Expected 2, got: {}", output);
}

// ============================================================================
// contains? tests
// ============================================================================

#[test]
fn test_contains_existing_key() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(println (contains? m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "true", "Expected true, got: {}", output);
}

#[test]
fn test_contains_missing_key() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(println (contains? m :c))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "false", "Expected false, got: {}", output);
}

#[test]
fn test_contains_nil_value() {
    // contains? checks for key presence, not value
    let code = r#"
(def m (hash-map :a nil))
(println (contains? m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "true",
        "Expected true (key exists even if value is nil), got: {}",
        output
    );
}

// ============================================================================
// keys and vals tests
// ============================================================================

#[test]
fn test_keys_returns_sequence() {
    let code = r#"
(def m (hash-map :a 1))
(println (first (keys m)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, ":a", "Expected :a, got: {}", output);
}

#[test]
fn test_vals_returns_sequence() {
    let code = r#"
(def m (hash-map :a 42))
(println (first (vals m)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_keys_count() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(println (count (keys m)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "3", "Expected 3, got: {}", output);
}

#[test]
fn test_vals_count() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(println (count (vals m)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "3", "Expected 3, got: {}", output);
}

#[test]
fn test_keys_empty_map() {
    let code = r#"
(def m (hash-map))
(println (keys m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "nil",
        "Expected nil for empty map keys, got: {}",
        output
    );
}

#[test]
fn test_vals_empty_map() {
    let code = r#"
(def m (hash-map))
(println (vals m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "nil",
        "Expected nil for empty map vals, got: {}",
        output
    );
}

// ============================================================================
// MapEntry tests
// ============================================================================

#[test]
fn test_map_entry_key() {
    let code = r#"
(def m (hash-map :foo 42))
(def entry (first m))
(println (key entry))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, ":foo", "Expected :foo, got: {}", output);
}

#[test]
fn test_map_entry_val() {
    let code = r#"
(def m (hash-map :foo 42))
(def entry (first m))
(println (val entry))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_map_entry_nth() {
    let code = r#"
(def m (hash-map :foo 42))
(def entry (first m))
(println (list (nth entry 0) (nth entry 1)))
"#;
    let output = run_and_get_stdout(code);
    assert!(
        output.contains(":foo") && output.contains("42"),
        "Expected entry to contain :foo and 42, got: {}",
        output
    );
}

// ============================================================================
// find tests
// ============================================================================

#[test]
fn test_find_existing_key() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(def entry (find m :a))
(println (val entry))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

#[test]
fn test_find_missing_key() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(println (find m :c))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "nil", "Expected nil, got: {}", output);
}

// ============================================================================
// Nested maps tests
// ============================================================================

#[test]
fn test_nested_maps() {
    let code = r#"
(def m (hash-map :outer (hash-map :inner 42)))
(println (get (get m :outer) :inner))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

#[test]
fn test_get_in_style_access() {
    let code = r#"
(def m (hash-map :a (hash-map :b (hash-map :c 100))))
(println (get (get (get m :a) :b) :c))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "100", "Expected 100, got: {}", output);
}

// ============================================================================
// Maps with different value types
// ============================================================================

#[test]
fn test_map_with_vector_values() {
    let code = r#"
(def m (hash-map :nums [1 2 3]))
(println (count (get m :nums)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "3", "Expected 3, got: {}", output);
}

#[test]
fn test_map_with_boolean_values() {
    let code = r#"
(def m (hash-map :enabled true :disabled false))
(println (if (get m :enabled) 1 0))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

#[test]
fn test_map_with_nil_values() {
    let code = r#"
(def m (hash-map :a nil :b 2))
(println (get m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "nil", "Expected nil, got: {}", output);
}

// ============================================================================
// Array map to hash map promotion tests
// ============================================================================

#[test]
fn test_array_map_stays_small() {
    // Small maps should stay as array maps
    let code = r#"
(def m (array-map :a 1 :b 2 :c 3 :d 4))
(println (count m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "4", "Expected 4, got: {}", output);
}

#[test]
fn test_assoc_chain() {
    // Chain multiple assocs and verify final state
    let code = r#"
(def m (hash-map))
(def m1 (assoc m :a 1))
(def m2 (assoc m1 :b 2))
(def m3 (assoc m2 :c 3))
(println (+ (get m3 :a) (+ (get m3 :b) (get m3 :c))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "6", "Expected 6 (1+2+3), got: {}", output);
}

// ============================================================================
// Map as function tests
// ============================================================================

#[test]
fn test_map_as_function() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(println (m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

#[test]
fn test_map_as_function_with_default() {
    let code = r#"
(def m (hash-map :a 1))
(println (m :missing 42))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

// ============================================================================
// Keyword as function on map tests
// ============================================================================

#[test]
fn test_keyword_as_function() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(println (:a m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

#[test]
fn test_keyword_as_function_missing() {
    let code = r#"
(def m (hash-map :a 1))
(println (:b m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "nil", "Expected nil, got: {}", output);
}

#[test]
fn test_keyword_as_function_with_default() {
    let code = r#"
(def m (hash-map :a 1))
(println (:b m 99))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "99", "Expected 99, got: {}", output);
}

// ============================================================================
// merge tests
// ============================================================================

#[test]
fn test_merge_two_maps() {
    let code = r#"
(def m1 (hash-map :a 1 :b 2))
(def m2 (hash-map :c 3 :d 4))
(def m3 (merge m1 m2))
(println (+ (get m3 :a) (+ (get m3 :b) (+ (get m3 :c) (get m3 :d)))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "10", "Expected 10 (1+2+3+4), got: {}", output);
}

#[test]
fn test_merge_overwrites() {
    let code = r#"
(def m1 (hash-map :a 1 :b 2))
(def m2 (hash-map :b 100 :c 3))
(def m3 (merge m1 m2))
(println (get m3 :b))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(
        output, "100",
        "Expected 100 (m2 overwrites), got: {}",
        output
    );
}

// ============================================================================
// select-keys tests
// ============================================================================

#[test]
fn test_select_keys() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3 :d 4))
(def m2 (select-keys m [:a :c]))
(println (count m2))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "2", "Expected 2, got: {}", output);
}

#[test]
fn test_select_keys_values() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(def m2 (select-keys m [:a :c]))
(println (+ (get m2 :a) (get m2 :c)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "4", "Expected 4 (1+3), got: {}", output);
}

// ============================================================================
// empty tests
// ============================================================================

#[test]
fn test_empty_hash_map() {
    let code = r#"
(def m (hash-map :a 1 :b 2))
(def e (empty m))
(println (count e))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "0", "Expected 0, got: {}", output);
}

// ============================================================================
// GC stress tests for maps
// ============================================================================

#[test]
fn test_gc_always_hash_map_creation() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3 :d 4 :e 5))
(println (+ (get m :a) (+ (get m :b) (+ (get m :c) (+ (get m :d) (get m :e))))))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "15", "Expected 15 (1+2+3+4+5), got: {}", output);
}

#[test]
fn test_gc_always_assoc_chain() {
    let code = r#"
(def m1 (hash-map :a 1))
(def m2 (assoc m1 :b 2))
(def m3 (assoc m2 :c 3))
(def m4 (assoc m3 :d 4))
(def m5 (assoc m4 :e 5))
(println (+ (get m5 :a) (+ (get m5 :b) (+ (get m5 :c) (+ (get m5 :d) (get m5 :e))))))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "15", "Expected 15 (1+2+3+4+5), got: {}", output);
}

#[test]
fn test_gc_always_nested_maps() {
    let code = r#"
(def m (hash-map :outer (hash-map :inner (hash-map :deep 42))))
(println (get (get (get m :outer) :inner) :deep))
"#;
    let output = run_gc_always_and_get_stdout(code);
    assert_eq!(output, "42", "Expected 42, got: {}", output);
}

// ============================================================================
// Large map tests
// ============================================================================

#[test]
fn test_many_keys() {
    let code = r#"
(def m (hash-map
  :k1 1 :k2 2 :k3 3 :k4 4 :k5 5
  :k6 6 :k7 7 :k8 8 :k9 9 :k10 10))
(println (count m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "10", "Expected 10, got: {}", output);
}

#[test]
fn test_many_keys_sum() {
    let code = r#"
(def m (hash-map
  :k1 1 :k2 2 :k3 3 :k4 4 :k5 5
  :k6 6 :k7 7 :k8 8 :k9 9 :k10 10))
(println (+ (get m :k1) (+ (get m :k5) (get m :k10))))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "16", "Expected 16 (1+5+10), got: {}", output);
}

// ============================================================================
// reduce-kv tests
// ============================================================================

#[test]
fn test_reduce_kv_sum_values() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3))
(println (reduce-kv (fn [acc k v] (+ acc v)) 0 m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "6", "Expected 6 (1+2+3), got: {}", output);
}

#[test]
fn test_reduce_kv_count_entries() {
    let code = r#"
(def m (hash-map :a 1 :b 2 :c 3 :d 4))
(println (reduce-kv (fn [acc k v] (inc acc)) 0 m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "4", "Expected 4, got: {}", output);
}

// ============================================================================
// seq tests for maps
// ============================================================================

#[test]
fn test_seq_on_map() {
    let code = r#"
(def m (hash-map :a 1))
(def s (seq m))
(println (val (first s)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

#[test]
fn test_seq_on_empty_map() {
    let code = r#"
(def m (hash-map))
(println (seq m))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "nil", "Expected nil, got: {}", output);
}

// ============================================================================
// into tests for maps
// ============================================================================

#[test]
fn test_into_hash_map() {
    let code = r#"
(def pairs [[:a 1] [:b 2]])
(def m (into (hash-map) pairs))
(println (get m :a))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "1", "Expected 1, got: {}", output);
}

// ============================================================================
// map? predicate tests
// ============================================================================

#[test]
fn test_map_predicate_hash_map() {
    let code = r#"
(println (map? (hash-map :a 1)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "true", "Expected true, got: {}", output);
}

#[test]
fn test_map_predicate_array_map() {
    let code = r#"
(println (map? (array-map :a 1)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "true", "Expected true, got: {}", output);
}

#[test]
fn test_map_predicate_vector() {
    let code = r#"
(println (map? [1 2 3]))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "false", "Expected false, got: {}", output);
}

#[test]
fn test_map_predicate_nil() {
    let code = r#"
(println (map? nil))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "false", "Expected false, got: {}", output);
}

// ============================================================================
// associative? predicate tests
// ============================================================================

#[test]
fn test_associative_hash_map() {
    let code = r#"
(println (associative? (hash-map :a 1)))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "true", "Expected true, got: {}", output);
}

#[test]
fn test_associative_vector() {
    let code = r#"
(println (associative? [1 2 3]))
"#;
    let output = run_and_get_stdout(code);
    assert_eq!(output, "true", "Expected true, got: {}", output);
}
