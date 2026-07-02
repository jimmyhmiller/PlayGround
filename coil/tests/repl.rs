//! `coil repl` end-to-end: drive the real binary over pipes and assert on the
//! transcript. The child expression processes inherit the REPL's stdout, so
//! values and program output land in the same captured stream — exactly what
//! an inferior-lisp buffer sees.

use std::io::Write;
use std::process::{Command, Stdio};

/// Run a scripted REPL session; return the full transcript (prompts included).
fn repl(script: &str) -> String {
    repl_args(script, &[])
}

/// `repl`, in `--isolate` (fresh-process-per-expression) mode.
fn repl_isolate(script: &str) -> String {
    repl_args(script, &["--isolate"])
}

fn repl_args(script: &str, extra: &[&str]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_coil"))
        .arg("repl")
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn coil repl");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write script");
    drop(child.stdin.take()); // EOF ends the session
    let out = child.wait_with_output().expect("repl finishes");
    assert!(
        out.status.success(),
        "repl exited nonzero.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The transcript line printed as the result of an input (prompts precede
/// results on the same line in a piped transcript: `coil> 3`).
fn assert_result(transcript: &str, expected: &str) {
    assert!(
        transcript.contains(&format!("coil> {expected}")),
        "expected result line `coil> {expected}` in transcript:\n{transcript}"
    );
}

#[test]
fn scalar_values_print_by_type() {
    let t = repl("(iadd 1 2)\n(isub 1 2)\ntrue\n3.5\n\"hello world\"\n(cast :u64 -1)\n(cast :u8 200)\n");
    assert_result(&t, "3");
    assert_result(&t, "-1");
    assert_result(&t, "true");
    assert_result(&t, "3.5");
    assert_result(&t, "\"hello world\"");
    assert_result(&t, "18446744073709551615"); // u64 reads its bits unsigned
    assert_result(&t, "200"); // u8 zero-extends, not sign-extends
}

#[test]
fn definitions_persist_and_redefine() {
    let t = repl(
        "(defn square [(x i64)] (-> i64) (imul x x))\n\
         (square 12)\n\
         (defn square [(x i64)] (-> i64) (imul x 1000))\n\
         (square 12)\n",
    );
    assert_result(&t, "#'repl/square");
    assert_result(&t, "144");
    assert_result(&t, "12000");
}

#[test]
fn multi_line_definitions_accumulate() {
    let t = repl("(defn add3 [(x i64)] (-> i64)\n  (iadd x 3))\n(add3 4)\n");
    assert!(t.contains("....> "), "continuation prompt expected:\n{t}");
    assert_result(&t, "7");
}

#[test]
fn structs_and_sums_display_recursively() {
    let t = repl(
        "(defstruct Point [(x i64) (y i64)])\n\
         (defstruct Seg [(a Point) (b Point)])\n\
         (defn mkp [(a i64) (b i64)] (-> Point)\n\
           (let [p (alloc-stack Point)] (store! (field p x) a) (store! (field p y) b) (load p)))\n\
         (mkp 3 4)\n\
         (defn mks [] (-> Seg)\n\
           (let [s (alloc-stack Seg)] (store! (field s a) (mkp 1 2)) (store! (field s b) (mkp 3 4)) (load s)))\n\
         (mks)\n\
         (Some 5)\n\
         (Ok [bool i64] true)\n",
    );
    assert_result(&t, "(Point :x 3 :y 4)");
    assert_result(&t, "(Seg :a (Point :x 1 :y 2) :b (Point :x 3 :y 4))");
    assert_result(&t, "(Some 5)");
    assert_result(&t, "(Ok true)");
}

#[test]
fn effects_run_then_value_prints() {
    let t = repl("(do (println \"side effect\") 42)\n");
    assert!(t.contains("side effect\n"), "program output expected:\n{t}");
    assert_result(&t, "side effect"); // the effect precedes the value line
    assert!(t.contains("\n42\n"), "value line expected:\n{t}");
}

#[test]
fn void_expression_runs_and_prints_nothing() {
    // libc srand: a genuinely void call (Coil bodies can't be empty).
    let t = repl("(extern srand [:i64] (-> void))\n(srand 42)\n(iadd 1 1)\n");
    assert_result(&t, "#'repl/srand");
    // The void call produces no value line: its prompt is immediately followed
    // by the next prompt.
    assert!(
        t.contains("coil> coil> 2"),
        "void expression should print nothing:\n{t}"
    );
}

#[test]
fn errors_report_and_session_survives() {
    let t = repl(
        "(defn ok [] (-> i64) 5)\n\
         (iadd 1 true)\n\
         (defn bad [] (-> i64) true)\n\
         (ok)\n",
    );
    assert!(t.contains("arithmetic on different types"), "expression error expected:\n{t}");
    assert!(t.contains("declared return type"), "definition error expected:\n{t}");
    assert_result(&t, "5"); // the session still works after both errors
}

#[test]
fn type_command_infers_without_running() {
    let t = repl(
        "(defn square [(x i64)] (-> i64) (imul x x))\n\
         :type (square 3)\n\
         :type \"hi\"\n\
         :type (Some (mkp))\n",
    );
    assert_result(&t, "i64");
    assert_result(&t, "(slice u8)");
    assert!(t.contains("error:"), "unknown fn in :type should error:\n{t}");
}

#[test]
fn slices_display_elementwise() {
    let t = repl(
        "(import \"slice.coil\" :use *)\n\
         (defn nums [] (-> (slice i64))\n\
           (let [a (alloc-heap (array i64 3))]\n\
             (store! (index a 0) 10) (store! (index a 1) 20) (store! (index a 2) 30)\n\
             (slice-new (cast (ptr i64) a) 3)))\n\
         (nums)\n",
    );
    assert_result(&t, "[10 20 30]");
}

#[test]
fn isolate_mode_keeps_main_but_excludes_it_from_expressions() {
    let t = repl_isolate(
        "(defn main [] (-> i64) 0)\n\
         (iadd 2 2)\n\
         (main)\n",
    );
    assert_result(&t, "#'repl/main");
    assert_result(&t, "4"); // expressions still evaluate with a session main
    assert!(
        t.contains("excluded from expression programs"),
        "calling the excluded main should explain itself:\n{t}"
    );
}

#[test]
fn live_mode_session_main_coexists_with_expressions() {
    // Live evals have their own entry symbol, so a session `main` needs no
    // exclusion — expressions evaluate normally alongside it. (Calling `(main)`
    // by name is a LANGUAGE-level error in any Coil program — the resolver
    // keeps `main` as the bare entry symbol — not a REPL restriction.)
    let t = repl("(defn main [] (-> i64) 41)\n(iadd 2 2)\n");
    assert_result(&t, "#'repl/main");
    assert_result(&t, "4");
    assert!(
        !t.contains("excluded from expression programs"),
        "live mode should not exclude main:\n{t}"
    );
}

// ---- live mode: persistent bindings + state --------------------------------

#[test]
fn def_bindings_persist_rebind_and_interleave() {
    let t = repl(
        "(def n 5)\n\
         (iadd n 37)\n\
         (def n (iadd n 1))\n\
         (def s \"hello\")\n\
         (iadd n 30)\n\
         s\n\
         (def s \"rebound\")\n\
         (iadd n 100)\n",
    );
    assert_result(&t, "n = 5");
    assert_result(&t, "42");
    assert_result(&t, "n = 6");
    assert_result(&t, "s = \"hello\"");
    // n still reads 6 after s took a new slot (regression: slot collision
    // between a rebound name and the next new binding).
    assert_result(&t, "36");
    assert_result(&t, "\"hello\"");
    assert_result(&t, "s = \"rebound\"");
    assert_result(&t, "106");
}

#[test]
fn def_binds_structs_and_definitions_can_use_bindings() {
    let t = repl(
        "(defstruct Point [(x i64) (y i64)])\n\
         (defn mkp [(a i64) (b i64)] (-> Point)\n\
           (let [p (alloc-stack Point)] (store! (field p x) a) (store! (field p y) b) (load p)))\n\
         (def origin (mkp 2 3))\n\
         origin\n\
         (iadd (load (field origin x)) (load (field origin y)))\n",
    );
    assert_result(&t, "origin = (Point :x 2 :y 3)");
    assert_result(&t, "(Point :x 2 :y 3)");
    assert_result(&t, "5");
}

#[test]
fn mutable_state_via_heap_cell_persists_across_evals() {
    let t = repl(
        "(def counter (alloc-heap i64))\n\
         (store! counter 0)\n\
         (store! counter (iadd (load counter) 1))\n\
         (store! counter (iadd (load counter) 40))\n\
         (load counter)\n",
    );
    assert_result(&t, "41");
}

#[test]
fn arraylist_grows_across_evals_through_a_pointer_binding() {
    let t = repl(
        "(import \"arraylist.coil\" :use *)\n\
         (import \"alloc.coil\" :use *)\n\
         (def xs (alloc-heap (ArrayList i64)))\n\
         (store! xs (al-new [i64] (malloc-allocator)))\n\
         (al-push! xs 10)\n\
         (al-push! xs 20)\n\
         (al-len (load xs))\n",
    );
    assert_result(&t, "2");
}

#[test]
fn session_command_lists_bindings_and_reset_clears_them() {
    let t = repl(
        "(def n 7)\n\
         :session\n\
         :reset\n\
         n\n",
    );
    assert!(t.contains("; (def n) : i64"), "bindings should list in :session:\n{t}");
    assert!(
        t.contains("unbound variable 'n'") || t.contains("error"),
        "n should be gone after :reset:\n{t}"
    );
}

#[test]
fn struct_redefinition_drops_dependent_bindings() {
    // Reinterpreting an old cell with a new layout would be silent corruption;
    // the binding is dropped LOUDLY instead. (A redefinition that contradicts
    // other session code is rejected outright by whole-session validation.)
    let t = repl(
        "(defstruct P [(x i64) (y i64)])\n\
         (def q (let [p (alloc-stack P)] (store! (field p x) 1) (store! (field p y) 2) (load p)))\n\
         (defstruct P [(x i64)])\n\
         q\n",
    );
    assert_result(&t, "q = (P :x 1 :y 2)");
    assert!(t.contains("dropped binding(s) q"), "drop notice expected:\n{t}");
    assert!(t.contains("unbound variable 'q'"), "q should be gone:\n{t}");
}

#[test]
fn def_rejects_isolate_mode_and_name_clashes() {
    let t = repl_isolate("(def n 5)\n");
    assert!(t.contains("needs the live session"), "isolate def should explain:\n{t}");
    let t = repl("(defn f [] (-> i64) 1)\n(def f 2)\n");
    assert!(
        t.contains("already a session definition"),
        "def clashing with a defn should be rejected:\n{t}"
    );
}

#[test]
fn load_command_adds_a_file() {
    let dir = std::env::temp_dir().join(format!("coil_repl_load_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("lib.coil");
    std::fs::write(&path, "(module x)\n(defn triple [(n i64)] (-> i64) (imul n 3))\n").unwrap();
    let t = repl(&format!(":load {}\n(triple 14)\n", path.display()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(t.contains("loaded 1 form(s)"), "load ack expected:\n{t}");
    assert_result(&t, "42");
}
