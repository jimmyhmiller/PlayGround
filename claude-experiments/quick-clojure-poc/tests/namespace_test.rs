/// Namespace/module system tests
///
/// Tests for Clojure's namespace system: ns, in-ns, create-ns, find-ns,
/// require, refer, use, ns-name, the-ns, ns-publics, ns-map, etc.
///
/// Tests that require unimplemented features are marked #[ignore].
///
/// Currently working:
/// - Basic ns form (create/switch namespace)
/// - def/defn in named namespaces
/// - Qualified symbol access (foo.bar/x)
/// - Cross-namespace function calls (foo.bar/fn)
/// - in-ns (runtime namespace switching)
/// - create-ns / find-ns / the-ns
/// - ns-name, ns-publics, ns-map, ns-interns, ns-refers, ns-aliases, ns-resolve
/// - (use 'ns) interning referred vars
/// - :require / :refer / :as in ns form
/// - all-ns
/// - *ns* default to user namespace
///
/// Not yet implemented:
/// - alias (namespace alias at runtime)

use std::fs;
use std::path::PathBuf;
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
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir).join("target/release/quick-clojure-poc")
    })
}

/// Run code from a temp file, return (stdout, stderr, success)
fn run_code(code: &str) -> (String, String, bool) {
    let binary_path = get_binary_path();
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let temp_path = temp_dir.path().join("test.clj");
    fs::write(&temp_path, code).expect("Failed to write temp file");

    let output = Command::new(binary_path.as_os_str())
        .arg(temp_path.to_str().unwrap())
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    (stdout, stderr, output.status.success())
}

/// Assert println output matches expected
fn assert_output(code: &str, expected: &str) {
    let (stdout, stderr, success) = run_code(code);
    assert!(
        success,
        "\nCode failed: {}\nStderr: {}",
        code, stderr
    );
    assert_eq!(
        stdout, expected,
        "\nCode: {}\nExpected: {:?}\nGot: {:?}",
        code, expected, stdout
    );
}

/// Assert code fails with an error
fn assert_error(code: &str) {
    let (_stdout, _stderr, success) = run_code(code);
    assert!(
        !success,
        "\nExpected code to fail but it succeeded: {}",
        code
    );
}

// ============================================================================
// Basic ns: create and switch namespaces
// ============================================================================

#[test]
fn ns_creates_new_namespace() {
    // Defining in a new namespace, then accessing from another
    assert_output(
        "(ns myapp.core)
(def x 42)
(ns myapp.util)
(println myapp.core/x)",
        "42",
    );
}

#[test]
fn ns_def_accessible_qualified() {
    assert_output(
        "(ns foo.bar)
(def greeting \"hello\")
(ns other)
(println foo.bar/greeting)",
        "hello",
    );
}

#[test]
fn ns_defn_callable_qualified() {
    assert_output(
        "(ns math.utils)
(defn square [n] (* n n))
(ns main)
(println (math.utils/square 5))",
        "25",
    );
}

#[test]
fn ns_multiple_defs_qualified_access() {
    assert_output(
        "(ns shapes)
(def pi 3)
(defn area [r] (* pi r r))
(ns calc)
(println (shapes/area 2))",
        "12",
    );
}

#[test]
fn ns_switch_back_and_forth() {
    assert_output(
        "(ns a)
(def x 1)
(ns b)
(def y 2)
(ns a)
(println (+ x b/y))",
        "3",
    );
}

#[test]
fn ns_local_def_shadows_other_ns() {
    // After switching back to ns a, x refers to a/x (not b/x)
    assert_output(
        "(ns a)
(def x 10)
(ns b)
(def x 20)
(ns a)
(println x)",
        "10",
    );
}

#[test]
fn ns_clojure_core_available_in_new_ns() {
    // All new namespaces get clojure.core referred in
    assert_output(
        "(ns myapp)
(println (inc 41))",
        "42",
    );
}

#[test]
fn ns_clojure_core_available_after_switch() {
    // Even after switching namespaces, core is available
    assert_output(
        "(ns first.ns)
(ns second.ns)
(println (+ 1 2))",
        "3",
    );
}

#[test]
fn ns_qualified_fn_with_args() {
    assert_output(
        "(ns greeter)
(defn greet [name] (str \"Hello, \" name \"!\"))
(ns app)
(println (greeter/greet \"World\"))",
        "Hello, World!",
    );
}

#[test]
fn ns_dotted_name() {
    assert_output(
        "(ns com.example.app)
(def version \"1.0\")
(ns main)
(println com.example.app/version)",
        "1.0",
    );
}

// ============================================================================
// ns-name: returns a symbol for the namespace's name
// ============================================================================

#[test]
fn ns_name_returns_symbol() {
    assert_output("(ns myns) (println (ns-name *ns*))", "myns");
}

#[test]
fn ns_name_user_namespace() {
    assert_output("(println (ns-name *ns*))", "user");
}

#[test]
fn ns_name_dotted_ns() {
    assert_output("(ns foo.bar.baz) (println (ns-name *ns*))", "foo.bar.baz");
}

// ============================================================================
// *ns* - current namespace var
// ============================================================================

#[test]
fn ns_star_ns_is_namespace() {
    // *ns* equality with ns-name
    assert_output("(ns foo) (println (= (ns-name *ns*) 'foo))", "true");
}

// ============================================================================
// create-ns: create a namespace from a symbol
// ============================================================================

#[test]
fn create_ns_creates_namespace() {
    assert_output(
        "(create-ns 'foo.created)
(println (find-ns 'foo.created))",
        "foo.created",
    );
}

#[test]
fn create_ns_returns_namespace() {
    assert_output(
        "(println (= (find-ns 'foo.new) nil))", "true",
    );
}

// ============================================================================
// find-ns: find a namespace by symbol
// ============================================================================

#[test]
fn find_ns_existing_namespace() {
    assert_output(
        "(ns myapp)
(println (nil? (find-ns 'myapp)))",
        "false",
    );
}

#[test]
fn find_ns_missing_returns_nil() {
    assert_output("(println (nil? (find-ns 'does.not.exist)))", "true");
}

#[test]
fn find_ns_clojure_core() {
    assert_output("(println (nil? (find-ns 'clojure.core)))", "false");
}

// ============================================================================
// in-ns: switch to a namespace at runtime (takes a quoted symbol)
// ============================================================================

#[test]
fn in_ns_switches_namespace() {
    assert_output(
        "(in-ns 'myapp)
(def x 99)
(in-ns 'user)
(println myapp/x)",
        "99",
    );
}

#[test]
fn in_ns_creates_if_not_exists() {
    assert_output(
        "(in-ns 'brand.new)
(def y 7)
(in-ns 'user)
(println brand.new/y)",
        "7",
    );
}

// ============================================================================
// the-ns: coerce to namespace
// ============================================================================

#[test]
fn the_ns_from_symbol() {
    assert_output(
        "(ns target)
(println (nil? (the-ns 'target)))",
        "false",
    );
}

#[test]
fn the_ns_throws_for_missing() {
    assert_error("(the-ns 'no.such.ns)");
}

// ============================================================================
// use: refers all public vars from a namespace into the current namespace
// ============================================================================

#[test]
fn use_makes_symbols_available_unqualified() {
    assert_output(
        "(ns lib)
(def helper 42)
(ns app)
(use 'lib)
(println helper)",
        "42",
    );
}

#[test]
fn use_makes_functions_available_unqualified() {
    assert_output(
        "(ns utils)
(defn double [x] (* 2 x))
(ns main)
(use 'utils)
(println (double 5))",
        "10",
    );
}

// ============================================================================
// ns :require - loading/requiring namespaces
// ============================================================================

#[test]
fn ns_require_basic() {
    // Basic require without alias or refer
    assert_output(
        "(ns foo)
(def x 1)
(ns bar (:require [foo]))
(println foo/x)",
        "1",
    );
}

#[test]
fn ns_require_with_alias() {
    assert_output(
        "(ns mylib)
(defn helper [x] (* x 2))
(ns myapp (:require [mylib :as ml]))
(println (ml/helper 5))",
        "10",
    );
}

#[test]
fn ns_require_with_refer() {
    assert_output(
        "(ns mylib)
(defn helper [x] (* x 2))
(def val 99)
(ns myapp (:require [mylib :refer [helper val]]))
(println (helper 5))
(println val)",
        "10\n99",
    );
}

#[test]
fn ns_require_refer_all() {
    assert_output(
        "(ns mylib)
(def a 1)
(def b 2)
(ns myapp (:require [mylib :refer :all]))
(println (+ a b))",
        "3",
    );
}

// ============================================================================
// ns-publics, ns-interns, ns-map
// ============================================================================

#[test]
fn ns_publics_returns_map() {
    assert_output(
        "(ns myns)
(def pub-var 1)
(println (contains? (ns-publics 'myns) 'pub-var))",
        "true",
    );
}

#[test]
fn ns_interns_returns_map() {
    assert_output(
        "(ns myns)
(def interned 42)
(println (contains? (ns-interns 'myns) 'interned))",
        "true",
    );
}

#[test]
fn ns_map_contains_interned_and_referred() {
    assert_output(
        "(ns myns)
(def x 1)
(println (contains? (ns-map 'myns) 'x))",
        "true",
    );
}

#[test]
fn ns_map_contains_core_fns() {
    // clojure.core is referred into every ns, so core fns should be in ns-map
    assert_output(
        "(ns myns)
(println (contains? (ns-map 'myns) 'inc))",
        "true",
    );
}

// ============================================================================
// ns-aliases: namespace aliases
// ============================================================================

#[test]
fn ns_aliases_empty_by_default() {
    assert_output(
        "(ns myns)
(println (empty? (ns-aliases 'myns)))",
        "true",
    );
}

#[test]
#[ignore = "alias not yet implemented"]
fn alias_creates_namespace_alias() {
    assert_output(
        "(ns mylib)
(defn f [x] x)
(ns myapp)
(alias 'ml 'mylib)
(println (ml/f 42))",
        "42",
    );
}

// ============================================================================
// ns-resolve: resolve a symbol in a namespace
// ============================================================================

#[test]
fn ns_resolve_finds_interned_var() {
    assert_output(
        "(ns myns)
(def myvar 1)
(println (nil? (ns-resolve 'myns 'myvar)))",
        "false",
    );
}

#[test]
fn ns_resolve_returns_nil_for_missing() {
    assert_output(
        "(ns myns)
(println (nil? (ns-resolve 'myns 'no-such-var)))",
        "true",
    );
}

// ============================================================================
// all-ns: list all namespaces
// ============================================================================

#[test]
fn all_ns_contains_user() {
    assert_output(
        "(println (some #(= (ns-name %) 'user) (all-ns)))",
        "true",
    );
}

#[test]
fn all_ns_contains_clojure_core() {
    assert_output(
        "(println (some #(= (ns-name %) 'clojure.core) (all-ns)))",
        "true",
    );
}

// ============================================================================
// ns-resolve with qualified symbols
// ============================================================================

#[test]
fn qualified_symbol_namespace_function() {
    // (namespace 'foo/bar) => "foo"
    assert_output("(println (namespace 'foo/bar))", "foo");
}

#[test]
fn qualified_symbol_name_function() {
    // (name 'foo/bar) => "bar"
    assert_output("(println (name 'foo/bar))", "bar");
}

#[test]
fn unqualified_symbol_namespace_is_nil() {
    // (namespace 'foo) => nil
    assert_output("(println (namespace 'foo))", "nil");
}
