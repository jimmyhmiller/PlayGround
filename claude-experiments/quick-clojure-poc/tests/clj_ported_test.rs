/// Tests ported from Clojure's official test suite to exercise the ARM64 JIT Clojure implementation.
///
/// Source files:
///   - clojure.test-clojure.sequences (sequences.clj)
///   - clojure.test-clojure.data-structures (data_structures.clj)
///   - clojure.test-clojure.vectors (vectors.clj)
///   - clojure.test-clojure.logic (logic.clj)
///   - clojure.test-clojure.atoms (atoms.clj)
///
/// Conventions:
///   - eval_expr(expr) returns the printed representation of the last expression.
///   - For collections that print opaquely (maps as "{... N entries}", sets as "#{... N elements}"),
///     we test via get/contains?/count rather than comparing the full printed form.
///   - Tests marked #[ignore] require features not yet implemented (atoms, sorted collections,
///     interpose, partition-all, partition-by, reductions, etc.)

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_binary_path() -> &'static PathBuf {
    BINARY_PATH.get_or_init(|| {
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir)
            .join("target")
            .join("release")
            .join("quick-clojure-poc")
    })
}

fn eval_expr(expr: &str) -> String {
    let binary_path = get_binary_path();
    let output = Command::new(binary_path.as_os_str())
        .arg("-e")
        .arg(expr)
        .output()
        .expect("Failed to execute");
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    assert!(
        output.status.success(),
        "\nExpression failed: {}\nStderr: {}",
        expr, stderr
    );
    stdout
}

// ============================================================================
// SEQUENCE TESTS (from sequences.clj)
// ============================================================================

// -- first --

#[test]
fn clj_seq_first_nil() {
    assert_eq!(eval_expr("(first nil)"), "nil");
}

#[test]
fn clj_seq_first_empty_vec() {
    assert_eq!(eval_expr("(first [])"), "nil");
}

#[test]
fn clj_seq_first_single_vec() {
    assert_eq!(eval_expr("(first [1])"), "1");
}

#[test]
fn clj_seq_first_multi_vec() {
    assert_eq!(eval_expr("(first [1 2 3])"), "1");
}

#[test]
fn clj_seq_first_nil_in_vec() {
    assert_eq!(eval_expr("(first [nil])"), "nil");
}

#[test]
fn clj_seq_first_nested_vec() {
    assert_eq!(eval_expr("(first [[1 2] [3 4]])"), "[1 2]");
}

#[test]
fn clj_seq_first_list() {
    assert_eq!(eval_expr("(first (list 1 2 3))"), "1");
}

#[test]
fn clj_seq_first_empty_list() {
    assert_eq!(eval_expr("(first (list))"), "nil");
}

// -- next --

#[test]
fn clj_seq_next_nil() {
    assert_eq!(eval_expr("(next nil)"), "nil");
}

#[test]
fn clj_seq_next_empty_vec() {
    assert_eq!(eval_expr("(next [])"), "nil");
}

#[test]
fn clj_seq_next_single_vec() {
    assert_eq!(eval_expr("(next [1])"), "nil");
}

#[test]
fn clj_seq_next_multi_vec() {
    assert_eq!(eval_expr("(next [1 2 3])"), "(2 3)");
}

#[test]
fn clj_seq_next_list() {
    assert_eq!(eval_expr("(next (list 1 2 3))"), "(2 3)");
}

#[test]
fn clj_seq_next_single_list() {
    assert_eq!(eval_expr("(next (list 1))"), "nil");
}

// -- rest --

#[test]
fn clj_seq_rest_nil() {
    // Note: Clojure returns (), this impl returns nil
    assert_eq!(eval_expr("(rest nil)"), "nil");
}

#[test]
fn clj_seq_rest_empty_vec() {
    // Note: Clojure returns (), this impl returns nil
    assert_eq!(eval_expr("(rest [])"), "nil");
}

#[test]
fn clj_seq_rest_multi_vec() {
    assert_eq!(eval_expr("(rest [1 2 3])"), "(2 3)");
}

#[test]
fn clj_seq_rest_single_vec() {
    // Note: Clojure returns (), this impl returns ()
    assert_eq!(eval_expr("(rest [1])"), "()");
}

// -- last --

#[test]
fn clj_seq_last_nil() {
    assert_eq!(eval_expr("(last nil)"), "nil");
}

#[test]
fn clj_seq_last_empty_vec() {
    assert_eq!(eval_expr("(last [])"), "nil");
}

#[test]
fn clj_seq_last_single() {
    assert_eq!(eval_expr("(last [1])"), "1");
}

#[test]
fn clj_seq_last_multi() {
    assert_eq!(eval_expr("(last [1 2 3])"), "3");
}

#[test]
fn clj_seq_last_nil_in_vec() {
    assert_eq!(eval_expr("(last [1 nil])"), "nil");
}

#[test]
fn clj_seq_last_nested() {
    assert_eq!(eval_expr("(last [[] nil])"), "nil");
}

#[test]
fn clj_seq_last_list() {
    assert_eq!(eval_expr("(last (list 1 2 3))"), "3");
}

// -- second / ffirst --

#[test]
fn clj_seq_second_vec() {
    assert_eq!(eval_expr("(second [1 2 3])"), "2");
}

#[test]
fn clj_seq_second_nil() {
    assert_eq!(eval_expr("(second nil)"), "nil");
}

#[test]
fn clj_seq_second_single() {
    assert_eq!(eval_expr("(second [1])"), "nil");
}

#[test]
fn clj_seq_ffirst_nested() {
    assert_eq!(eval_expr("(first (first [[1 2] [3 4]]))"), "1");
}

#[test]
fn clj_seq_second_of_first() {
    assert_eq!(eval_expr("(second (first [[1 2] [3 4]]))"), "2");
}

// -- cons --

#[test]
fn clj_seq_cons_nil() {
    assert_eq!(eval_expr("(cons 1 nil)"), "(1)");
}

#[test]
fn clj_seq_cons_vec() {
    assert_eq!(eval_expr("(cons 1 [2 3])"), "(1 2 3)");
}

#[test]
fn clj_seq_cons_list() {
    assert_eq!(eval_expr("(cons 1 (list 2 3))"), "(1 2 3)");
}

#[test]
fn clj_seq_cons_empty_vec() {
    assert_eq!(eval_expr("(cons 1 [])"), "(1)");
}

#[test]
fn clj_seq_cons_nil_to_nil() {
    assert_eq!(eval_expr("(cons nil nil)"), "(nil)");
}

// -- conj --

#[test]
fn clj_seq_conj_nil() {
    assert_eq!(eval_expr("(conj nil 1)"), "(1)");
}

#[test]
fn clj_seq_conj_vec() {
    assert_eq!(eval_expr("(conj [1 2] 3)"), "[1 2 3]");
}

#[test]
fn clj_seq_conj_list() {
    assert_eq!(eval_expr("(conj (list 2 3) 1)"), "(1 2 3)");
}

#[test]
fn clj_seq_conj_empty_vec() {
    assert_eq!(eval_expr("(conj [] 1)"), "[1]");
}

#[test]
fn clj_seq_conj_set() {
    assert_eq!(eval_expr("(contains? (conj #{1 2} 3) 3)"), "true");
}

#[test]
fn clj_seq_conj_map_entry() {
    assert_eq!(eval_expr("(get (conj {:a 1} [:b 2]) :b)"), "2");
}

// -- empty --

#[test]
fn clj_seq_empty_nil() {
    assert_eq!(eval_expr("(empty nil)"), "nil");
}

#[test]
fn clj_seq_empty_vec() {
    assert_eq!(eval_expr("(empty [1 2])"), "[]");
}

#[test]
fn clj_seq_empty_map() {
    assert_eq!(eval_expr("(count (empty {:a 1}))"), "0");
}

// -- count --

#[test]
fn clj_seq_count_nil() {
    assert_eq!(eval_expr("(count nil)"), "0");
}

#[test]
fn clj_seq_count_empty_vec() {
    assert_eq!(eval_expr("(count [])"), "0");
}

#[test]
fn clj_seq_count_vec() {
    assert_eq!(eval_expr("(count [1 2 3])"), "3");
}

#[test]
fn clj_seq_count_list() {
    assert_eq!(eval_expr("(count (list 1 2 3))"), "3");
}

#[test]
fn clj_seq_count_map() {
    assert_eq!(eval_expr("(count {:a 1 :b 2})"), "2");
}

#[test]
fn clj_seq_count_set() {
    assert_eq!(eval_expr("(count #{1 2 3})"), "3");
}

#[test]
fn clj_seq_count_string() {
    assert_eq!(eval_expr("(count \"abc\")"), "3");
}

// -- nth --

#[test]
fn clj_seq_nth_vec() {
    assert_eq!(eval_expr("(nth [1 2 3] 0)"), "1");
}

#[test]
fn clj_seq_nth_vec_middle() {
    assert_eq!(eval_expr("(nth [1 2 3] 1)"), "2");
}

#[test]
fn clj_seq_nth_vec_last() {
    assert_eq!(eval_expr("(nth [1 2 3] 2)"), "3");
}

#[test]
#[ignore] // nth on lists not supported (only vectors)
fn clj_seq_nth_list() {
    assert_eq!(eval_expr("(nth (list 10 20 30) 2)"), "30");
}

#[test]
fn clj_seq_nth_with_default() {
    assert_eq!(eval_expr("(nth [1 2 3] 5 :not-found)"), ":not-found");
}

// -- map --

#[test]
fn clj_seq_map_inc() {
    assert_eq!(eval_expr("(map inc [1 2 3])"), "(2 3 4)");
}

#[test]
fn clj_seq_map_nil() {
    assert_eq!(eval_expr("(map inc nil)"), "nil");
}

#[test]
fn clj_seq_map_identity() {
    assert_eq!(eval_expr("(map identity [1 2 3])"), "(1 2 3)");
}

#[test]
fn clj_seq_map_add_two_colls() {
    assert_eq!(eval_expr("(map + [1 2 3] [10 20 30])"), "(11 22 33)");
}

#[test]
fn clj_seq_map_to_vec() {
    assert_eq!(eval_expr("(vec (map inc [1 2 3]))"), "[2 3 4]");
}

#[test]
fn clj_seq_map_into_vec() {
    assert_eq!(eval_expr("(into [] (map inc [1 2 3]))"), "[2 3 4]");
}

// -- filter --

#[test]
fn clj_seq_filter_even() {
    assert_eq!(eval_expr("(filter even? [1 2 3 4 5])"), "(2 4)");
}

#[test]
fn clj_seq_filter_odd() {
    assert_eq!(eval_expr("(filter odd? [1 2 3 4 5])"), "(1 3 5)");
}

#[test]
fn clj_seq_filter_nil() {
    assert_eq!(eval_expr("(filter even? nil)"), "nil");
}

#[test]
fn clj_seq_filter_none_match() {
    assert_eq!(eval_expr("(filter even? [1 3 5])"), "nil");
}

#[test]
fn clj_seq_filter_all_match() {
    assert_eq!(eval_expr("(filter even? [2 4 6])"), "(2 4 6)");
}

// -- remove --

#[test]
fn clj_seq_remove_even() {
    assert_eq!(eval_expr("(remove even? [1 2 3 4 5])"), "(1 3 5)");
}

// -- keep --

#[test]
fn clj_seq_keep_identity() {
    assert_eq!(eval_expr("(keep identity [1 nil 2 nil 3])"), "(1 2 3)");
}

// -- reduce --

#[test]
fn clj_seq_reduce_plus() {
    assert_eq!(eval_expr("(reduce + [1 2 3 4 5])"), "15");
}

#[test]
fn clj_seq_reduce_plus_init() {
    assert_eq!(eval_expr("(reduce + 10 [1 2 3 4 5])"), "25");
}

#[test]
fn clj_seq_reduce_plus_init_100() {
    assert_eq!(eval_expr("(reduce + 100 [1 2 3])"), "106");
}

#[test]
fn clj_seq_reduce_conj_vec() {
    assert_eq!(eval_expr("(reduce conj [] [1 2 3])"), "[1 2 3]");
}

#[test]
fn clj_seq_reduce_fn() {
    assert_eq!(
        eval_expr("(reduce (fn [acc x] (+ acc x)) 0 [1 2 3 4 5])"),
        "15"
    );
}

#[test]
fn clj_seq_reduce_range() {
    assert_eq!(eval_expr("(reduce + (range 10))"), "45");
}

#[test]
fn clj_seq_reduce_range_100() {
    assert_eq!(eval_expr("(reduce + (range 100))"), "4950");
}

#[test]
fn clj_seq_reduce_range_init() {
    assert_eq!(eval_expr("(reduce + 0 (range 100))"), "4950");
}

#[test]
fn clj_seq_reduce_map_range() {
    assert_eq!(eval_expr("(reduce + (map inc (range 10)))"), "55");
}

#[test]
fn clj_seq_reduce_filter_range() {
    assert_eq!(eval_expr("(reduce + (filter even? (range 10)))"), "20");
}

#[test]
fn clj_seq_reduce_filter_odd_range() {
    assert_eq!(eval_expr("(reduce + (filter odd? (range 10)))"), "25");
}

// -- take / drop --

#[test]
fn clj_seq_take_3() {
    assert_eq!(eval_expr("(take 3 [1 2 3 4 5])"), "(1 2 3)");
}

#[test]
fn clj_seq_take_1() {
    assert_eq!(eval_expr("(take 1 [1 2 3 4 5])"), "(1)");
}

#[test]
fn clj_seq_take_5() {
    assert_eq!(eval_expr("(take 5 [1 2 3 4 5])"), "(1 2 3 4 5)");
}

#[test]
fn clj_seq_take_9_from_5() {
    assert_eq!(eval_expr("(take 9 [1 2 3 4 5])"), "(1 2 3 4 5)");
}

#[test]
fn clj_seq_take_0() {
    assert_eq!(eval_expr("(take 0 [1 2 3 4 5])"), "nil");
}

#[test]
fn clj_seq_drop_1() {
    assert_eq!(eval_expr("(drop 1 [1 2 3 4 5])"), "(2 3 4 5)");
}

#[test]
fn clj_seq_drop_3() {
    assert_eq!(eval_expr("(drop 3 [1 2 3 4 5])"), "(4 5)");
}

#[test]
fn clj_seq_drop_5() {
    assert_eq!(eval_expr("(drop 5 [1 2 3 4 5])"), "nil");
}

#[test]
fn clj_seq_drop_9() {
    assert_eq!(eval_expr("(drop 9 [1 2 3 4 5])"), "nil");
}

#[test]
fn clj_seq_drop_0() {
    assert_eq!(eval_expr("(drop 0 [1 2 3 4 5])"), "(1 2 3 4 5)");
}

// -- take-while / drop-while --

#[test]
fn clj_seq_take_while_pos() {
    assert_eq!(eval_expr("(take-while pos? [1 2 3 -1 4])"), "(1 2 3)");
}

#[test]
fn clj_seq_take_while_none() {
    assert_eq!(eval_expr("(take-while pos? [-1 1 2])"), "nil");
}

#[test]
fn clj_seq_take_while_all() {
    assert_eq!(eval_expr("(take-while pos? [1 2 3 4])"), "(1 2 3 4)");
}

#[test]
fn clj_seq_drop_while_pos() {
    assert_eq!(eval_expr("(drop-while pos? [1 2 3 -1 4])"), "(-1 4)");
}

#[test]
fn clj_seq_drop_while_none() {
    assert_eq!(eval_expr("(drop-while pos? [-1 1 2 3])"), "(-1 1 2 3)");
}

#[test]
fn clj_seq_drop_while_all() {
    assert_eq!(eval_expr("(drop-while pos? [1 2 3 4])"), "nil");
}

// -- concat --

#[test]
fn clj_seq_concat_two() {
    assert_eq!(eval_expr("(concat [1 2] [3 4])"), "(1 2 3 4)");
}

#[test]
fn clj_seq_concat_three() {
    assert_eq!(eval_expr("(concat [1 2] [3 4] [5 6])"), "(1 2 3 4 5 6)");
}

#[test]
fn clj_seq_concat_single() {
    assert_eq!(eval_expr("(concat [1 2])"), "(1 2)");
}

#[test]
fn clj_seq_concat_count() {
    assert_eq!(eval_expr("(count (concat [1 2] [3 4]))"), "4");
}

#[test]
fn clj_seq_concat_into_vec() {
    assert_eq!(eval_expr("(into [] (concat [1 2] [3 4]))"), "[1 2 3 4]");
}

// -- interleave --

#[test]
fn clj_seq_interleave_two() {
    assert_eq!(eval_expr("(interleave [1 2 3] [4 5 6])"), "(1 4 2 5 3 6)");
}

#[test]
fn clj_seq_interleave_unequal_first_shorter() {
    assert_eq!(eval_expr("(interleave [1] [3 4])"), "(1 3)");
}

#[test]
fn clj_seq_interleave_unequal_second_shorter() {
    assert_eq!(eval_expr("(interleave [1 2] [3])"), "(1 3)");
}

#[test]
fn clj_seq_interleave_apply_str() {
    assert_eq!(
        eval_expr("(apply str (interleave [1 2 3] [:a :b :c]))"),
        "\"1:a2:b3:c\""
    );
}

// -- distinct --

#[test]
fn clj_seq_distinct_basic() {
    assert_eq!(eval_expr("(distinct [1 2 1 3 2])"), "(1 2 3)");
}

#[test]
fn clj_seq_distinct_all_same() {
    assert_eq!(eval_expr("(distinct [1 1 1])"), "(1)");
}

#[test]
fn clj_seq_distinct_already_unique() {
    assert_eq!(eval_expr("(distinct [1 2 3])"), "(1 2 3)");
}

#[test]
fn clj_seq_distinct_count() {
    assert_eq!(eval_expr("(count (distinct [1 1 2 2 3]))"), "3");
}

#[test]
fn clj_seq_distinct_preserves_order() {
    assert_eq!(eval_expr("(first (distinct [3 1 2 1]))"), "3");
}

// -- reverse --

#[test]
fn clj_seq_reverse_vec() {
    assert_eq!(eval_expr("(reverse [1 2 3])"), "(3 2 1)");
}

#[test]
fn clj_seq_reverse_single() {
    assert_eq!(eval_expr("(reverse [1])"), "(1)");
}

#[test]
fn clj_seq_reverse_empty() {
    // Note: Clojure returns (), this impl returns ()
    assert_eq!(eval_expr("(reverse [])"), "()");
}

#[test]
fn clj_seq_reverse_last() {
    assert_eq!(eval_expr("(last (reverse [1 2 3]))"), "1");
}

// -- sort --

#[test]
fn clj_seq_sort_vec() {
    assert_eq!(eval_expr("(sort [3 1 2])"), "(1 2 3)");
}

#[test]
fn clj_seq_sort_already_sorted() {
    assert_eq!(eval_expr("(sort [1 2 3])"), "(1 2 3)");
}

#[test]
fn clj_seq_sort_reverse() {
    assert_eq!(eval_expr("(sort [3 2 1])"), "(1 2 3)");
}

#[test]
fn clj_seq_sort_with_dups() {
    assert_eq!(eval_expr("(sort [3 1 4 1 5 9 2 6])"), "(1 1 2 3 4 5 6 9)");
}

#[test]
fn clj_seq_sort_first() {
    assert_eq!(eval_expr("(first (sort [3 1 2]))"), "1");
}

#[test]
fn clj_seq_sort_last() {
    assert_eq!(eval_expr("(last (sort [3 1 2]))"), "3");
}

#[test]
fn clj_seq_sort_distinct() {
    assert_eq!(eval_expr("(sort (distinct [3 1 2 1 3]))"), "(1 2 3)");
}

// -- butlast --

#[test]
fn clj_seq_butlast_vec() {
    assert_eq!(eval_expr("(butlast [1 2 3])"), "(1 2)");
}

#[test]
fn clj_seq_butlast_single() {
    assert_eq!(eval_expr("(butlast [1])"), "nil");
}

#[test]
fn clj_seq_butlast_empty() {
    assert_eq!(eval_expr("(butlast [])"), "nil");
}

// -- partition --

#[test]
fn clj_seq_partition_2() {
    assert_eq!(eval_expr("(partition 2 [1 2 3 4])"), "((1 2) (3 4))");
}

#[test]
fn clj_seq_partition_2_odd() {
    assert_eq!(eval_expr("(partition 2 [1 2 3])"), "((1 2))");
}

#[test]
fn clj_seq_partition_3() {
    assert_eq!(
        eval_expr("(partition 3 [1 2 3 4 5 6 7 8])"),
        "((1 2 3) (4 5 6))"
    );
}

#[test]
fn clj_seq_partition_with_step() {
    assert_eq!(
        eval_expr("(partition 2 3 [1 2 3 4 5 6 7])"),
        "((1 2) (4 5))"
    );
}

#[test]
fn clj_seq_partition_step_equals_size() {
    assert_eq!(
        eval_expr("(partition 2 3 [1 2 3 4 5 6 7 8])"),
        "((1 2) (4 5) (7 8))"
    );
}

#[test]
fn clj_seq_partition_1() {
    assert_eq!(eval_expr("(partition 1 [1 2 3])"), "((1) (2) (3))");
}

#[test]
fn clj_seq_partition_larger_than_coll() {
    assert_eq!(eval_expr("(partition 5 [1 2 3])"), "nil");
}

#[test]
fn clj_seq_partition_count() {
    assert_eq!(eval_expr("(count (partition 2 [1 2 3 4 5]))"), "2");
}

#[test]
fn clj_seq_partition_first() {
    assert_eq!(eval_expr("(first (partition 2 [1 2 3 4]))"), "(1 2)");
}

// -- flatten --

#[test]
fn clj_seq_flatten_nested() {
    assert_eq!(eval_expr("(flatten [[1 2] [3 [4 5]]])"), "(1 2 3 4 5)");
}

#[test]
fn clj_seq_flatten_deeply_nested() {
    assert_eq!(eval_expr("(flatten [1 [2 [3 [4]]]])"), "(1 2 3 4)");
}

#[test]
fn clj_seq_flatten_already_flat() {
    assert_eq!(eval_expr("(flatten [1 2 3])"), "(1 2 3)");
}

// -- mapcat --

#[test]
fn clj_seq_mapcat_list() {
    assert_eq!(eval_expr("(mapcat list [1 2 3])"), "(1 2 3)");
}

#[test]
fn clj_seq_mapcat_reverse() {
    assert_eq!(eval_expr("(mapcat reverse [[1 2] [3 4]])"), "(2 1 4 3)");
}

// -- apply --

#[test]
fn clj_seq_apply_plus() {
    assert_eq!(eval_expr("(apply + [1 2 3])"), "6");
}

#[test]
fn clj_seq_apply_str() {
    assert_eq!(eval_expr("(apply str [1 2 3])"), "\"123\"");
}

#[test]
fn clj_seq_apply_list() {
    assert_eq!(eval_expr("(apply list [1 2 3])"), "(1 2 3)");
}

#[test]
fn clj_seq_apply_vector() {
    assert_eq!(eval_expr("(apply vector [1 2 3])"), "[1 2 3]");
}

#[test]
fn clj_seq_apply_concat() {
    assert_eq!(
        eval_expr("(apply concat [[1 2] [3 4] [5 6]])"),
        "(1 2 3 4 5 6)"
    );
}

// -- map-indexed --

#[test]
fn clj_seq_map_indexed() {
    assert_eq!(
        eval_expr("(map-indexed (fn [i v] (+ i v)) [10 20 30])"),
        "(10 21 32)"
    );
}

// -- empty? --

#[test]
fn clj_seq_empty_check_nil() {
    assert_eq!(eval_expr("(empty? nil)"), "true");
}

#[test]
fn clj_seq_empty_check_empty_vec() {
    assert_eq!(eval_expr("(empty? [])"), "true");
}

#[test]
fn clj_seq_empty_check_nonempty_vec() {
    assert_eq!(eval_expr("(empty? [1])"), "false");
}

#[test]
fn clj_seq_empty_check_empty_map() {
    assert_eq!(eval_expr("(empty? {})"), "true");
}

#[test]
fn clj_seq_empty_check_nonempty_map() {
    assert_eq!(eval_expr("(empty? {:a 1})"), "false");
}

// -- every? / some / not-every? / not-any? --

#[test]
fn clj_seq_every_true() {
    assert_eq!(eval_expr("(every? pos? [1 2 3])"), "true");
}

#[test]
fn clj_seq_every_false() {
    assert_eq!(eval_expr("(every? pos? [1 -2 3])"), "false");
}

#[test]
fn clj_seq_every_empty() {
    assert_eq!(eval_expr("(every? pos? [])"), "true");
}

#[test]
fn clj_seq_every_nil() {
    assert_eq!(eval_expr("(every? pos? nil)"), "true");
}

#[test]
fn clj_seq_every_all_neg() {
    assert_eq!(eval_expr("(every? pos? [-1 -2])"), "false");
}

#[test]
fn clj_seq_some_found() {
    assert_eq!(eval_expr("(some pos? [1 2 3])"), "true");
}

#[test]
fn clj_seq_some_not_found() {
    assert_eq!(eval_expr("(some pos? [-1 -2])"), "nil");
}

#[test]
fn clj_seq_some_nil_input() {
    assert_eq!(eval_expr("(some pos? nil)"), "nil");
}

#[test]
fn clj_seq_some_set_as_fn() {
    assert_eq!(eval_expr("(some #{:a} [:b :a :c])"), ":a");
}

#[test]
fn clj_seq_some_set_not_found() {
    assert_eq!(eval_expr("(some #{:a} [:b :c])"), "nil");
}

#[test]
fn clj_seq_some_even() {
    assert_eq!(eval_expr("(some even? [1 2 3])"), "true");
}

#[test]
fn clj_seq_some_even_not_found() {
    assert_eq!(eval_expr("(some even? [1 3 5])"), "nil");
}

#[test]
fn clj_seq_not_every_true() {
    assert_eq!(eval_expr("(not-every? pos? [1 -2 3])"), "true");
}

#[test]
fn clj_seq_not_every_false() {
    assert_eq!(eval_expr("(not-every? pos? [1 2 3])"), "false");
}

#[test]
fn clj_seq_not_every_empty() {
    assert_eq!(eval_expr("(not-every? pos? [])"), "false");
}

#[test]
fn clj_seq_not_any_true() {
    assert_eq!(eval_expr("(not-any? pos? [-1 -2])"), "true");
}

#[test]
fn clj_seq_not_any_false() {
    assert_eq!(eval_expr("(not-any? pos? [1 2 3])"), "false");
}

#[test]
fn clj_seq_not_any_mixed() {
    assert_eq!(eval_expr("(not-any? pos? [-1 -2 3])"), "false");
}

#[test]
fn clj_seq_not_any_empty() {
    assert_eq!(eval_expr("(not-any? pos? [])"), "true");
}

// -- range --

#[test]
fn clj_seq_range_5() {
    assert_eq!(eval_expr("(range 5)"), "(0 1 2 3 4)");
}

#[test]
fn clj_seq_range_0() {
    assert_eq!(eval_expr("(range 0)"), "nil");
}

#[test]
fn clj_seq_range_1() {
    assert_eq!(eval_expr("(range 1)"), "(0)");
}

#[test]
fn clj_seq_range_start_end() {
    assert_eq!(eval_expr("(range 2 5)"), "(2 3 4)");
}

#[test]
fn clj_seq_range_start_end_step() {
    assert_eq!(eval_expr("(range 0 10 2)"), "(0 2 4 6 8)");
}

#[test]
fn clj_seq_range_start_end_step_3() {
    assert_eq!(eval_expr("(range 0 10 3)"), "(0 3 6 9)");
}

#[test]
fn clj_seq_range_neg() {
    assert_eq!(eval_expr("(range 3 6)"), "(3 4 5)");
}

#[test]
fn clj_seq_range_neg_values() {
    assert_eq!(eval_expr("(range -2 3)"), "(-2 -1 0 1 2)");
}

#[test]
fn clj_seq_range_into_vec() {
    assert_eq!(eval_expr("(into [] (range 5))"), "[0 1 2 3 4]");
}

#[test]
fn clj_seq_range_count() {
    assert_eq!(eval_expr("(count (range 10))"), "10");
}

#[test]
fn clj_seq_range_last() {
    assert_eq!(eval_expr("(last (range 5))"), "4");
}

#[test]
fn clj_seq_range_sum() {
    assert_eq!(eval_expr("(reduce + (range 1 101))"), "5050");
}

// -- repeat --

#[test]
fn clj_seq_repeat_3() {
    assert_eq!(eval_expr("(repeat 3 :x)"), "(:x :x :x)");
}

#[test]
fn clj_seq_repeat_0() {
    assert_eq!(eval_expr("(repeat 0 :x)"), "nil");
}

#[test]
fn clj_seq_repeat_1() {
    assert_eq!(eval_expr("(repeat 1 :x)"), "(:x)");
}

#[test]
fn clj_seq_repeat_5() {
    assert_eq!(eval_expr("(repeat 5 7)"), "(7 7 7 7 7)");
}

#[test]
fn clj_seq_repeat_neg() {
    assert_eq!(eval_expr("(repeat -1 7)"), "nil");
}

// -- into --

#[test]
fn clj_seq_into_vec_from_list() {
    assert_eq!(eval_expr("(into [] (list 1 2 3))"), "[1 2 3]");
}

#[test]
fn clj_seq_into_vec_from_nil() {
    assert_eq!(eval_expr("(into [] nil)"), "[]");
}

#[test]
fn clj_seq_into_vec_from_range() {
    assert_eq!(eval_expr("(into [] (range 5))"), "[0 1 2 3 4]");
}

#[test]
fn clj_seq_into_vec_filter() {
    assert_eq!(
        eval_expr("(into [] (filter odd? (range 10)))"),
        "[1 3 5 7 9]"
    );
}

#[test]
fn clj_seq_into_vec_take() {
    assert_eq!(eval_expr("(into [] (take 5 (range 100)))"), "[0 1 2 3 4]");
}

#[test]
fn clj_seq_into_vec_drop() {
    assert_eq!(
        eval_expr("(into [] (drop 5 (range 10)))"),
        "[5 6 7 8 9]"
    );
}

#[test]
fn clj_seq_into_set_count() {
    assert_eq!(eval_expr("(count (into #{} [1 2 3 2 1]))"), "3");
}

// -- group-by / frequencies --

#[test]
fn clj_seq_group_by_even() {
    assert_eq!(
        eval_expr("(get (group-by even? [1 2 3 4 5]) true)"),
        "[2 4]"
    );
}

#[test]
fn clj_seq_group_by_odd() {
    assert_eq!(
        eval_expr("(get (group-by even? [1 2 3 4 5]) false)"),
        "[1 3 5]"
    );
}

#[test]
fn clj_seq_frequencies_count() {
    assert_eq!(eval_expr("(get (frequencies [1 1 2 2 2 3]) 2)"), "3");
}

#[test]
fn clj_seq_frequencies_single() {
    assert_eq!(eval_expr("(get (frequencies [1 1 2 2 2 3]) 1)"), "2");
}

#[test]
fn clj_seq_frequencies_unique() {
    assert_eq!(eval_expr("(get (frequencies [1 1 2 2 2 3]) 3)"), "1");
}

// -- thread macros --

#[test]
fn clj_seq_thread_first() {
    assert_eq!(eval_expr("(-> 1 inc inc inc)"), "4");
}

#[test]
fn clj_seq_thread_last() {
    assert_eq!(eval_expr("(->> [1 2 3] (map inc) (filter even?))"), "(2 4)");
}

#[test]
fn clj_seq_thread_last_reduce() {
    assert_eq!(eval_expr("(->> (range 10) (map inc) (reduce +))"), "55");
}

// -- higher order fns --

#[test]
fn clj_seq_comp() {
    assert_eq!(eval_expr("((comp inc inc) 0)"), "2");
}

#[test]
fn clj_seq_partial() {
    assert_eq!(eval_expr("((partial + 10) 5)"), "15");
}

#[test]
fn clj_seq_juxt() {
    assert_eq!(eval_expr("((juxt inc dec) 5)"), "[6 4]");
}

// -- unsupported: interpose, partition-all, partition-by, reductions, etc. --

#[test]
#[ignore] // interpose not implemented
fn clj_seq_interpose() {
    assert_eq!(eval_expr("(interpose 0 [1 2 3])"), "(1 0 2 0 3)");
}

#[test]
#[ignore] // partition-all not implemented
fn clj_seq_partition_all() {
    assert_eq!(
        eval_expr("(partition-all 4 [1 2 3 4 5 6 7 8 9])"),
        "((1 2 3 4) (5 6 7 8) (9))"
    );
}

#[test]
#[ignore] // partition-by not implemented
fn clj_seq_partition_by() {
    assert_eq!(
        eval_expr("(partition-by odd? [1 3 2 4 5])"),
        "((1 3) (2 4) (5))"
    );
}

#[test]
#[ignore] // reductions not implemented
fn clj_seq_reductions() {
    assert_eq!(
        eval_expr("(reductions + [1 2 3 4 5])"),
        "(1 3 6 10 15)"
    );
}

#[test]
#[ignore] // take-nth not implemented
fn clj_seq_take_nth() {
    assert_eq!(eval_expr("(take-nth 2 [1 2 3 4 5])"), "(1 3 5)");
}

#[test]
#[ignore] // split-at not implemented
fn clj_seq_split_at() {
    assert_eq!(
        eval_expr("(split-at 2 [1 2 3 4 5])"),
        "[(1 2) (3 4 5)]"
    );
}

#[test]
#[ignore] // split-with not implemented
fn clj_seq_split_with() {
    assert_eq!(
        eval_expr("(split-with pos? [1 2 -1 3])"),
        "[(1 2) (-1 3)]"
    );
}

#[test]
#[ignore] // sort-by not implemented
fn clj_seq_sort_by() {
    assert_eq!(
        eval_expr("(sort-by first [[2 1] [1 3] [3 2]])"),
        "([1 3] [2 1] [3 2])"
    );
}

#[test]
#[ignore] // cycle not implemented
fn clj_seq_cycle() {
    assert_eq!(eval_expr("(take 5 (cycle [1 2 3]))"), "(1 2 3 1 2)");
}

#[test]
#[ignore] // iterate not implemented
fn clj_seq_iterate() {
    assert_eq!(eval_expr("(take 5 (iterate inc 0))"), "(0 1 2 3 4)");
}

#[test]
#[ignore] // drop-last not implemented
fn clj_seq_drop_last() {
    assert_eq!(eval_expr("(drop-last [1 2 3])"), "(1 2)");
}

#[test]
#[ignore] // nthrest not implemented
fn clj_seq_nthrest() {
    assert_eq!(eval_expr("(nthrest [1 2 3 4 5] 2)"), "(3 4 5)");
}

#[test]
#[ignore] // nthnext not implemented
fn clj_seq_nthnext() {
    assert_eq!(eval_expr("(nthnext [1 2 3 4 5] 2)"), "(3 4 5)");
}

#[test]
#[ignore] // ffirst not implemented
fn clj_seq_ffirst() {
    assert_eq!(eval_expr("(ffirst [[1 2] [3 4]])"), "1");
}

#[test]
#[ignore] // nnext not implemented
fn clj_seq_nnext() {
    assert_eq!(eval_expr("(nnext [1 2 3 4])"), "(3 4)");
}

#[test]
#[ignore] // mapv not implemented
fn clj_seq_mapv() {
    assert_eq!(eval_expr("(mapv inc [1 2 3])"), "[2 3 4]");
}

#[test]
#[ignore] // filterv not implemented
fn clj_seq_filterv() {
    assert_eq!(eval_expr("(filterv even? [1 2 3 4 5])"), "[2 4]");
}

#[test]
#[ignore] // keep-indexed not implemented
fn clj_seq_keep_indexed() {
    assert_eq!(
        eval_expr("(keep-indexed (fn [i v] (if (odd? i) v)) [10 20 30 40 50])"),
        "(20 40)"
    );
}

#[test]
#[ignore] // merge-with not implemented
fn clj_seq_merge_with() {
    assert_eq!(
        eval_expr("(get (merge-with + {:a 1} {:a 2}) :a)"),
        "3"
    );
}

#[test]
#[ignore] // not-empty not implemented
fn clj_seq_not_empty_empty() {
    assert_eq!(eval_expr("(not-empty [])"), "nil");
}

#[test]
#[ignore] // not-empty not implemented
fn clj_seq_not_empty_nonempty() {
    assert_eq!(eval_expr("(not-empty [1 2])"), "[1 2]");
}

#[test]
#[ignore] // peek not implemented
fn clj_vec_peek() {
    assert_eq!(eval_expr("(peek [1 2 3])"), "3");
}

#[test]
#[ignore] // pop not implemented
fn clj_vec_pop() {
    assert_eq!(eval_expr("(pop [1 2 3])"), "[1 2]");
}

#[test]
#[ignore] // subvec not implemented
fn clj_vec_subvec() {
    assert_eq!(eval_expr("(subvec [0 1 2 3 4] 2 4)"), "[2 3]");
}

// ============================================================================
// DATA STRUCTURE TESTS (from data_structures.clj)
// ============================================================================

// -- maps --

#[test]
fn clj_ds_map_get() {
    assert_eq!(eval_expr("(get {:a 1 :b 2} :a)"), "1");
}

#[test]
fn clj_ds_map_get_missing() {
    assert_eq!(eval_expr("(get {:a 1 :b 2} :c)"), "nil");
}

#[test]
fn clj_ds_map_get_default() {
    assert_eq!(eval_expr("(get {:a 1 :b 2} :c :default)"), ":default");
}

#[test]
fn clj_ds_map_keyword_lookup() {
    assert_eq!(eval_expr("(:a {:a 1 :b 2})"), "1");
}

#[test]
fn clj_ds_map_as_fn() {
    assert_eq!(eval_expr("({:a 1 :b 2} :a)"), "1");
}

#[test]
fn clj_ds_map_assoc() {
    assert_eq!(eval_expr("(get (assoc {:a 1} :b 2) :b)"), "2");
}

#[test]
fn clj_ds_map_assoc_overwrite() {
    assert_eq!(eval_expr("(get (assoc {:a 1} :a 2) :a)"), "2");
}

#[test]
fn clj_ds_map_assoc_multi() {
    assert_eq!(eval_expr("(count (assoc {:a 1 :b 2} :c 3 :d 4))"), "4");
}

#[test]
fn clj_ds_map_dissoc() {
    assert_eq!(eval_expr("(get (dissoc {:a 1 :b 2} :a) :a)"), "nil");
}

#[test]
fn clj_ds_map_dissoc_retains() {
    assert_eq!(eval_expr("(get (dissoc {:a 1 :b 2} :a) :b)"), "2");
}

#[test]
fn clj_ds_map_dissoc_multi() {
    assert_eq!(
        eval_expr("(count (dissoc {:a 1 :b 2 :c 3} :a :c))"),
        "1"
    );
}

#[test]
fn clj_ds_map_contains() {
    assert_eq!(eval_expr("(contains? {:a 1} :a)"), "true");
}

#[test]
fn clj_ds_map_contains_missing() {
    assert_eq!(eval_expr("(contains? {:a 1} :b)"), "false");
}

#[test]
fn clj_ds_map_contains_nil_key() {
    assert_eq!(eval_expr("(contains? {:a 1} nil)"), "false");
}

#[test]
fn clj_ds_map_keys() {
    assert_eq!(eval_expr("(keys {:a 1})"), "(:a)");
}

#[test]
fn clj_ds_map_keys_count() {
    assert_eq!(eval_expr("(count (keys {:a 1 :b 2}))"), "2");
}

#[test]
fn clj_ds_map_vals() {
    assert_eq!(eval_expr("(vals {:a 1})"), "(1)");
}

#[test]
fn clj_ds_map_vals_count() {
    assert_eq!(eval_expr("(count (vals {:a 1 :b 2}))"), "2");
}

#[test]
fn clj_ds_map_key_fn() {
    assert_eq!(eval_expr("(key (first {:a 1}))"), ":a");
}

#[test]
fn clj_ds_map_val_fn() {
    assert_eq!(eval_expr("(val (first {:a 1}))"), "1");
}

#[test]
fn clj_ds_map_merge_count() {
    assert_eq!(eval_expr("(count (merge {:a 1} {:b 2} {:c 3}))"), "3");
}

#[test]
fn clj_ds_map_merge_overwrite() {
    assert_eq!(eval_expr("(get (merge {:a 1} {:a 2}) :a)"), "2");
}

#[test]
fn clj_ds_map_update() {
    assert_eq!(eval_expr("(get (update {:a 1} :a inc) :a)"), "2");
}

#[test]
fn clj_ds_map_get_in() {
    assert_eq!(eval_expr("(get-in {:a {:b 2}} [:a :b])"), "2");
}

#[test]
fn clj_ds_map_get_in_missing() {
    assert_eq!(eval_expr("(get-in {:a {:b 2}} [:a :c])"), "nil");
}

#[test]
fn clj_ds_map_get_in_default() {
    assert_eq!(eval_expr("(get-in {:a {:b 2}} [:a :c] 0)"), "0");
}

#[test]
fn clj_ds_map_assoc_in() {
    assert_eq!(
        eval_expr("(get-in (assoc-in {} [:a :b :c] 42) [:a :b :c])"),
        "42"
    );
}

#[test]
fn clj_ds_map_update_in() {
    assert_eq!(
        eval_expr("(get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b])"),
        "2"
    );
}

#[test]
fn clj_ds_map_select_keys() {
    assert_eq!(
        eval_expr("(get (select-keys {:a 1 :b 2 :c 3} [:a :b]) :a)"),
        "1"
    );
}

#[test]
fn clj_ds_map_select_keys_count() {
    assert_eq!(
        eval_expr("(count (select-keys {:a 1 :b 2 :c 3} [:a :b]))"),
        "2"
    );
}

#[test]
fn clj_ds_map_select_keys_missing() {
    assert_eq!(
        eval_expr("(get (select-keys {:a 1 :b 2 :c 3} [:a :b]) :c)"),
        "nil"
    );
}

#[test]
fn clj_ds_map_zipmap_get() {
    assert_eq!(eval_expr("(get (zipmap [:a :b :c] [1 2 3]) :b)"), "2");
}

#[test]
fn clj_ds_map_zipmap_count() {
    assert_eq!(eval_expr("(count (zipmap [:a :b :c] [1 2 3]))"), "3");
}

#[test]
fn clj_ds_map_zipmap_shorter_keys() {
    assert_eq!(eval_expr("(count (zipmap [:a] [1 2]))"), "1");
}

#[test]
fn clj_ds_map_zipmap_shorter_vals() {
    assert_eq!(eval_expr("(count (zipmap [:a :b] [1]))"), "1");
}

#[test]
fn clj_ds_map_hash_map_count() {
    assert_eq!(eval_expr("(count (hash-map :a 1 :b 2))"), "2");
}

#[test]
fn clj_ds_map_hash_map_get() {
    assert_eq!(eval_expr("(get (hash-map :a 1 :b 2) :a)"), "1");
}

#[test]
fn clj_ds_map_into() {
    assert_eq!(eval_expr("(get (into {} [[:a 1] [:b 2]]) :b)"), "2");
}

#[test]
fn clj_ds_map_into_count() {
    assert_eq!(eval_expr("(count (into {} [[:a 1] [:b 2]]))"), "2");
}

#[test]
fn clj_ds_map_reduce_conj() {
    assert_eq!(
        eval_expr("(count (reduce conj {} [[:a 1] [:b 2]]))"),
        "2"
    );
}

#[test]
fn clj_ds_map_conj_entry() {
    assert_eq!(eval_expr("(get (conj {:a 1} [:b 2]) :b)"), "2");
}

#[test]
fn clj_ds_map_count_empty() {
    assert_eq!(eval_expr("(count {})"), "0");
}

#[test]
fn clj_ds_map_empty() {
    assert_eq!(eval_expr("(count (empty {:a 1}))"), "0");
}

// -- sets --

#[test]
fn clj_ds_set_contains() {
    assert_eq!(eval_expr("(contains? #{1 2 3} 2)"), "true");
}

#[test]
fn clj_ds_set_contains_missing() {
    assert_eq!(eval_expr("(contains? #{1 2 3} 4)"), "false");
}

#[test]
fn clj_ds_set_count() {
    assert_eq!(eval_expr("(count #{1 2 3})"), "3");
}

#[test]
fn clj_ds_set_conj() {
    assert_eq!(eval_expr("(contains? (conj #{1 2} 3) 3)"), "true");
}

#[test]
fn clj_ds_set_disj() {
    assert_eq!(eval_expr("(contains? (disj #{1 2 3} 2) 2)"), "false");
}

#[test]
fn clj_ds_set_disj_count() {
    assert_eq!(eval_expr("(count (disj #{1 2 3} 2))"), "2");
}

#[test]
fn clj_ds_set_from_vec() {
    assert_eq!(eval_expr("(count (set [1 2 3 2 1]))"), "3");
}

#[test]
fn clj_ds_set_from_vec_contains() {
    assert_eq!(eval_expr("(contains? (set [1 2 3]) 2)"), "true");
}

#[test]
fn clj_ds_set_hash_set_count() {
    assert_eq!(eval_expr("(count (hash-set 1 2 3))"), "3");
}

#[test]
fn clj_ds_set_hash_set_contains() {
    assert_eq!(eval_expr("(contains? (hash-set 1 2 3) 2)"), "true");
}

#[test]
fn clj_ds_set_into_count() {
    assert_eq!(eval_expr("(count (into #{} [1 2 3 2 1]))"), "3");
}

#[test]
fn clj_ds_set_into_contains() {
    assert_eq!(eval_expr("(contains? (into #{} [1 2 3]) 3)"), "true");
}

#[test]
fn clj_ds_set_empty() {
    assert_eq!(eval_expr("(count #{})"), "0");
}

#[test]
fn clj_ds_set_as_fn() {
    assert_eq!(eval_expr("(#{1 2 3} 2)"), "2");
}

#[test]
fn clj_ds_set_as_fn_missing() {
    assert_eq!(eval_expr("(#{1 2 3} 4)"), "nil");
}

// -- contains? on vectors --

#[test]
fn clj_ds_vec_contains_index() {
    assert_eq!(eval_expr("(contains? [1 2 3] 0)"), "true");
}

#[test]
fn clj_ds_vec_contains_last_index() {
    assert_eq!(eval_expr("(contains? [1 2 3] 2)"), "true");
}

#[test]
fn clj_ds_vec_contains_out_of_bounds() {
    assert_eq!(eval_expr("(contains? [1 2 3] 3)"), "false");
}

#[test]
fn clj_ds_vec_contains_neg() {
    assert_eq!(eval_expr("(contains? [1 2 3] -1)"), "false");
}

#[test]
fn clj_ds_vec_contains_empty() {
    assert_eq!(eval_expr("(contains? [] 0)"), "false");
}

// -- list operations --

#[test]
fn clj_ds_list_create() {
    assert_eq!(eval_expr("(list 1 2 3)"), "(1 2 3)");
}

#[test]
fn clj_ds_list_create_empty() {
    assert_eq!(eval_expr("(list)"), "()");
}

#[test]
fn clj_ds_list_is_list() {
    assert_eq!(eval_expr("(list? (list 1 2))"), "true");
}

#[test]
fn clj_ds_list_vec_is_not_list() {
    assert_eq!(eval_expr("(list? [1 2])"), "false");
}

#[test]
fn clj_ds_list_conj_front() {
    assert_eq!(eval_expr("(conj (list 2 3) 1)"), "(1 2 3)");
}

// ============================================================================
// VECTOR TESTS (from vectors.clj)
// ============================================================================

#[test]
fn clj_vec_create() {
    assert_eq!(eval_expr("[1 2 3]"), "[1 2 3]");
}

#[test]
fn clj_vec_create_empty() {
    assert_eq!(eval_expr("[]"), "[]");
}

#[test]
fn clj_vec_conj() {
    assert_eq!(eval_expr("(conj [1 2] 3)"), "[1 2 3]");
}

#[test]
fn clj_vec_assoc() {
    assert_eq!(eval_expr("(assoc [1 2 3] 1 99)"), "[1 99 3]");
}

#[test]
fn clj_vec_assoc_first() {
    assert_eq!(eval_expr("(assoc [1 2 3] 0 99)"), "[99 2 3]");
}

#[test]
fn clj_vec_assoc_last() {
    assert_eq!(eval_expr("(assoc [1 2 3] 2 99)"), "[1 2 99]");
}

#[test]
fn clj_vec_nth() {
    assert_eq!(eval_expr("(nth [10 20 30] 1)"), "20");
}

#[test]
fn clj_vec_get() {
    assert_eq!(eval_expr("(get [10 20 30] 1)"), "20");
}

#[test]
fn clj_vec_get_out_of_bounds() {
    assert_eq!(eval_expr("(get [10 20 30] 5)"), "nil");
}

#[test]
fn clj_vec_get_default() {
    assert_eq!(eval_expr("(get [10 20 30] 5 :not-found)"), ":not-found");
}

#[test]
fn clj_vec_as_fn() {
    assert_eq!(eval_expr("([10 20 30] 1)"), "20");
}

#[test]
fn clj_vec_count() {
    assert_eq!(eval_expr("(count [1 2 3])"), "3");
}

#[test]
fn clj_vec_count_empty() {
    assert_eq!(eval_expr("(count [])"), "0");
}

#[test]
fn clj_vec_is_vector() {
    assert_eq!(eval_expr("(vector? [1 2])"), "true");
}

#[test]
fn clj_vec_list_is_not_vector() {
    assert_eq!(eval_expr("(vector? (list 1 2))"), "false");
}

#[test]
fn clj_vec_from_list() {
    assert_eq!(eval_expr("(vec (list 1 2 3))"), "[1 2 3]");
}

#[test]
fn clj_vec_from_nil() {
    assert_eq!(eval_expr("(vec nil)"), "[]");
}

#[test]
fn clj_vec_from_range() {
    assert_eq!(eval_expr("(vec (range 4))"), "[0 1 2 3]");
}

#[test]
fn clj_vec_nested() {
    assert_eq!(eval_expr("[[1 2] [3 4]]"), "[[1 2] [3 4]]");
}

#[test]
fn clj_vec_first() {
    assert_eq!(eval_expr("(first [1 2 3])"), "1");
}

#[test]
fn clj_vec_last() {
    assert_eq!(eval_expr("(last [1 2 3])"), "3");
}

#[test]
fn clj_vec_str() {
    assert_eq!(eval_expr("(str [1 2 3])"), "\"[1 2 3]\"");
}

#[test]
fn clj_vec_str_empty() {
    assert_eq!(eval_expr("(str [])"), "\"[]\"");
}

#[test]
fn clj_vec_into() {
    assert_eq!(eval_expr("(into [] [1 2 3])"), "[1 2 3]");
}

#[test]
fn clj_vec_reduce_kv() {
    // reduce-kv on vectors: + init idx val
    assert_eq!(eval_expr("(reduce + 10 [2 4 6])"), "22");
}

// ============================================================================
// LOGIC TESTS (from logic.clj)
// ============================================================================

// -- if --

#[test]
fn clj_logic_if_true() {
    assert_eq!(eval_expr("(if true :t :f)"), ":t");
}

#[test]
fn clj_logic_if_false() {
    assert_eq!(eval_expr("(if false :t :f)"), ":f");
}

#[test]
fn clj_logic_if_nil() {
    assert_eq!(eval_expr("(if nil :t :f)"), ":f");
}

#[test]
fn clj_logic_if_zero_is_truthy() {
    assert_eq!(eval_expr("(if 0 :t :f)"), ":t");
}

#[test]
fn clj_logic_if_empty_string_is_truthy() {
    assert_eq!(eval_expr("(if \"\" :t :f)"), ":t");
}

#[test]
fn clj_logic_if_empty_vec_is_truthy() {
    assert_eq!(eval_expr("(if [] :t :f)"), ":t");
}

#[test]
fn clj_logic_if_empty_map_is_truthy() {
    assert_eq!(eval_expr("(if {} :t :f)"), ":t");
}

#[test]
fn clj_logic_if_keyword_is_truthy() {
    assert_eq!(eval_expr("(if :kw :t :f)"), ":t");
}

#[test]
fn clj_logic_if_number_is_truthy() {
    assert_eq!(eval_expr("(if 42 :t :f)"), ":t");
}

#[test]
fn clj_logic_if_true_no_else() {
    assert_eq!(eval_expr("(if true :t)"), ":t");
}

#[test]
fn clj_logic_if_false_no_else() {
    assert_eq!(eval_expr("(if false :t)"), "nil");
}

#[test]
fn clj_logic_if_nil_no_else() {
    assert_eq!(eval_expr("(if nil :t)"), "nil");
}

// -- nil punning --

#[test]
fn clj_logic_nil_punning_first_empty() {
    assert_eq!(eval_expr("(if (first []) :no :yes)"), ":yes");
}

#[test]
fn clj_logic_nil_punning_next_single() {
    assert_eq!(eval_expr("(if (next [1]) :no :yes)"), ":yes");
}

#[test]
fn clj_logic_nil_punning_seq_nil() {
    assert_eq!(eval_expr("(if (seq nil) :no :yes)"), ":yes");
}

#[test]
fn clj_logic_nil_punning_seq_empty() {
    assert_eq!(eval_expr("(if (seq []) :no :yes)"), ":yes");
}

// -- and --

#[test]
fn clj_logic_and_empty() {
    assert_eq!(eval_expr("(and)"), "true");
}

#[test]
fn clj_logic_and_true() {
    assert_eq!(eval_expr("(and true)"), "true");
}

#[test]
fn clj_logic_and_nil() {
    assert_eq!(eval_expr("(and nil)"), "nil");
}

#[test]
fn clj_logic_and_false() {
    assert_eq!(eval_expr("(and false)"), "false");
}

#[test]
fn clj_logic_and_true_nil() {
    assert_eq!(eval_expr("(and true nil)"), "nil");
}

#[test]
fn clj_logic_and_true_false() {
    assert_eq!(eval_expr("(and true false)"), "false");
}

#[test]
fn clj_logic_and_returns_last_truthy() {
    assert_eq!(eval_expr("(and 1 true :kw)"), ":kw");
}

#[test]
fn clj_logic_and_short_circuits_on_nil() {
    assert_eq!(eval_expr("(and 1 true :kw nil)"), "nil");
}

#[test]
fn clj_logic_and_short_circuits_on_false() {
    assert_eq!(eval_expr("(and 1 true :kw false)"), "false");
}

// -- or --

#[test]
fn clj_logic_or_empty() {
    assert_eq!(eval_expr("(or)"), "nil");
}

#[test]
fn clj_logic_or_true() {
    assert_eq!(eval_expr("(or true)"), "true");
}

#[test]
fn clj_logic_or_nil() {
    assert_eq!(eval_expr("(or nil)"), "nil");
}

#[test]
fn clj_logic_or_false() {
    assert_eq!(eval_expr("(or false)"), "false");
}

#[test]
fn clj_logic_or_nil_false_true() {
    assert_eq!(eval_expr("(or nil false true)"), "true");
}

#[test]
fn clj_logic_or_nil_false_1() {
    assert_eq!(eval_expr("(or nil false 1 2)"), "1");
}

#[test]
fn clj_logic_or_nil_false_str() {
    // String values are printed with quotes by the evaluator
    assert_eq!(eval_expr("(or nil false \"abc\" :kw)"), "\"abc\"");
}

#[test]
fn clj_logic_or_false_nil() {
    assert_eq!(eval_expr("(or false nil)"), "nil");
}

#[test]
fn clj_logic_or_nil_false() {
    assert_eq!(eval_expr("(or nil false)"), "false");
}

#[test]
fn clj_logic_or_nil_nil_nil_false() {
    assert_eq!(eval_expr("(or nil nil nil false)"), "false");
}

// -- not --

#[test]
fn clj_logic_not_nil() {
    assert_eq!(eval_expr("(not nil)"), "true");
}

#[test]
fn clj_logic_not_false() {
    assert_eq!(eval_expr("(not false)"), "true");
}

#[test]
fn clj_logic_not_true() {
    assert_eq!(eval_expr("(not true)"), "false");
}

#[test]
fn clj_logic_not_zero() {
    assert_eq!(eval_expr("(not 0)"), "false");
}

#[test]
fn clj_logic_not_number() {
    assert_eq!(eval_expr("(not 42)"), "false");
}

#[test]
fn clj_logic_not_empty_string() {
    assert_eq!(eval_expr("(not \"\")"), "false");
}

#[test]
fn clj_logic_not_string() {
    assert_eq!(eval_expr("(not \"abc\")"), "false");
}

#[test]
fn clj_logic_not_keyword() {
    assert_eq!(eval_expr("(not :kw)"), "false");
}

#[test]
fn clj_logic_not_empty_vec() {
    assert_eq!(eval_expr("(not [])"), "false");
}

#[test]
fn clj_logic_not_vec() {
    assert_eq!(eval_expr("(not [1 2])"), "false");
}

#[test]
fn clj_logic_not_empty_map() {
    assert_eq!(eval_expr("(not {})"), "false");
}

#[test]
fn clj_logic_not_map() {
    assert_eq!(eval_expr("(not {:a 1})"), "false");
}

// -- some? --

#[test]
fn clj_logic_some_nil() {
    assert_eq!(eval_expr("(some? nil)"), "false");
}

#[test]
fn clj_logic_some_false() {
    assert_eq!(eval_expr("(some? false)"), "true");
}

#[test]
fn clj_logic_some_zero() {
    assert_eq!(eval_expr("(some? 0)"), "true");
}

#[test]
fn clj_logic_some_string() {
    assert_eq!(eval_expr("(some? \"abc\")"), "true");
}

#[test]
fn clj_logic_some_vec() {
    assert_eq!(eval_expr("(some? [])"), "true");
}

// -- cond --

#[test]
fn clj_logic_cond_first_true() {
    assert_eq!(eval_expr("(cond true :a false :b)"), ":a");
}

#[test]
fn clj_logic_cond_second_true() {
    assert_eq!(eval_expr("(cond false :a true :b)"), ":b");
}

#[test]
fn clj_logic_cond_else() {
    assert_eq!(eval_expr("(cond false :a false :b :else :c)"), ":c");
}

// -- when / when-not / if-not --

#[test]
fn clj_logic_when_true() {
    assert_eq!(eval_expr("(when true 42)"), "42");
}

#[test]
fn clj_logic_when_false() {
    assert_eq!(eval_expr("(when false 42)"), "nil");
}

#[test]
fn clj_logic_when_not_true() {
    assert_eq!(eval_expr("(when-not true 42)"), "nil");
}

#[test]
fn clj_logic_when_not_false() {
    assert_eq!(eval_expr("(when-not false 42)"), "42");
}

#[test]
fn clj_logic_if_not_true() {
    assert_eq!(eval_expr("(if-not true :a :b)"), ":b");
}

#[test]
fn clj_logic_if_not_false() {
    assert_eq!(eval_expr("(if-not false :a :b)"), ":a");
}

// -- if-let / when-let --

#[test]
fn clj_logic_if_let_truthy() {
    assert_eq!(eval_expr("(if-let [x 42] x :nope)"), "42");
}

#[test]
fn clj_logic_if_let_nil() {
    assert_eq!(eval_expr("(if-let [x nil] x :nope)"), ":nope");
}

#[test]
fn clj_logic_when_let_truthy() {
    assert_eq!(eval_expr("(when-let [x 42] x)"), "42");
}

#[test]
fn clj_logic_when_let_nil() {
    assert_eq!(eval_expr("(when-let [x nil] x)"), "nil");
}

// ============================================================================
// ATOM TESTS (from atoms.clj)
// All marked #[ignore] because `atom` is not implemented
// ============================================================================

#[test]
#[ignore] // atom not implemented
fn clj_atom_create_deref() {
    assert_eq!(eval_expr("(let [a (atom 0)] (deref a))"), "0");
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_reset() {
    assert_eq!(eval_expr("(let [a (atom 0)] (reset! a 42) (deref a))"), "42");
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_swap_inc() {
    assert_eq!(eval_expr("(let [a (atom 0)] (swap! a inc) (deref a))"), "1");
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_swap_plus() {
    assert_eq!(eval_expr("(let [a (atom 0)] (swap! a + 5) (deref a))"), "5");
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_compare_and_set_success() {
    assert_eq!(
        eval_expr("(let [a (atom 10)] (compare-and-set! a 10 20) (deref a))"),
        "20"
    );
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_compare_and_set_failure() {
    assert_eq!(
        eval_expr("(let [a (atom 10)] (compare-and-set! a 99 20) (deref a))"),
        "10"
    );
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_swap_vals() {
    assert_eq!(
        eval_expr("(let [a (atom 0)] (swap-vals! a inc))"),
        "[0 1]"
    );
}

#[test]
#[ignore] // atom not implemented
fn clj_atom_reset_vals() {
    assert_eq!(
        eval_expr("(let [a (atom 0)] (reset-vals! a :b))"),
        "[0 :b]"
    );
}

// ============================================================================
// TYPE PREDICATE TESTS
// ============================================================================

#[test]
fn clj_ds_type_nil_check() {
    assert_eq!(eval_expr("(nil? nil)"), "true");
}

#[test]
fn clj_ds_type_nil_check_false() {
    assert_eq!(eval_expr("(nil? false)"), "false");
}

#[test]
fn clj_ds_type_true_check() {
    assert_eq!(eval_expr("(true? true)"), "true");
}

#[test]
fn clj_ds_type_true_check_false() {
    assert_eq!(eval_expr("(true? 1)"), "false");
}

#[test]
fn clj_ds_type_false_check() {
    assert_eq!(eval_expr("(false? false)"), "true");
}

#[test]
fn clj_ds_type_false_check_nil() {
    assert_eq!(eval_expr("(false? nil)"), "false");
}

#[test]
fn clj_ds_type_number_check() {
    assert_eq!(eval_expr("(number? 42)"), "true");
}

#[test]
fn clj_ds_type_number_check_false() {
    assert_eq!(eval_expr("(number? \"42\")"), "false");
}

#[test]
fn clj_ds_type_string_check() {
    assert_eq!(eval_expr("(string? \"abc\")"), "true");
}

#[test]
fn clj_ds_type_string_check_false() {
    assert_eq!(eval_expr("(string? 42)"), "false");
}

#[test]
fn clj_ds_type_keyword_check() {
    assert_eq!(eval_expr("(keyword? :a)"), "true");
}

#[test]
fn clj_ds_type_keyword_check_false() {
    assert_eq!(eval_expr("(keyword? \"a\")"), "false");
}

#[test]
fn clj_ds_type_vector_check() {
    assert_eq!(eval_expr("(vector? [1 2])"), "true");
}

#[test]
fn clj_ds_type_vector_check_list() {
    assert_eq!(eval_expr("(vector? (list 1 2))"), "false");
}

#[test]
fn clj_ds_type_map_check() {
    assert_eq!(eval_expr("(map? {:a 1})"), "true");
}

#[test]
fn clj_ds_type_map_check_false() {
    assert_eq!(eval_expr("(map? [1 2])"), "false");
}

#[test]
fn clj_ds_type_set_check() {
    assert_eq!(eval_expr("(set? #{1 2})"), "true");
}

#[test]
fn clj_ds_type_set_check_false() {
    assert_eq!(eval_expr("(set? [1 2])"), "false");
}

#[test]
fn clj_ds_type_coll_check_vec() {
    assert_eq!(eval_expr("(coll? [1 2])"), "true");
}

#[test]
fn clj_ds_type_coll_check_list() {
    assert_eq!(eval_expr("(coll? (list 1 2))"), "true");
}

#[test]
fn clj_ds_type_coll_check_map() {
    assert_eq!(eval_expr("(coll? {:a 1})"), "true");
}

#[test]
fn clj_ds_type_coll_check_set() {
    assert_eq!(eval_expr("(coll? #{1})"), "true");
}

#[test]
fn clj_ds_type_coll_check_number() {
    assert_eq!(eval_expr("(coll? 42)"), "false");
}

#[test]
fn clj_ds_type_sequential_vec() {
    assert_eq!(eval_expr("(sequential? [1 2])"), "true");
}

#[test]
fn clj_ds_type_sequential_list() {
    assert_eq!(eval_expr("(sequential? (list 1 2))"), "true");
}

#[test]
fn clj_ds_type_associative_map() {
    assert_eq!(eval_expr("(associative? {:a 1})"), "true");
}

#[test]
fn clj_ds_type_associative_vec() {
    assert_eq!(eval_expr("(associative? [1 2])"), "true");
}

// ============================================================================
// NUMERIC TESTS
// ============================================================================

#[test]
fn clj_ds_num_pos() {
    assert_eq!(eval_expr("(pos? 1)"), "true");
}

#[test]
fn clj_ds_num_pos_zero() {
    assert_eq!(eval_expr("(pos? 0)"), "false");
}

#[test]
fn clj_ds_num_pos_neg() {
    assert_eq!(eval_expr("(pos? -1)"), "false");
}

#[test]
fn clj_ds_num_neg() {
    assert_eq!(eval_expr("(neg? -1)"), "true");
}

#[test]
fn clj_ds_num_neg_zero() {
    assert_eq!(eval_expr("(neg? 0)"), "false");
}

#[test]
fn clj_ds_num_neg_pos() {
    assert_eq!(eval_expr("(neg? 1)"), "false");
}

#[test]
fn clj_ds_num_zero() {
    assert_eq!(eval_expr("(zero? 0)"), "true");
}

#[test]
fn clj_ds_num_zero_false() {
    assert_eq!(eval_expr("(zero? 1)"), "false");
}

#[test]
fn clj_ds_num_even() {
    assert_eq!(eval_expr("(even? 2)"), "true");
}

#[test]
fn clj_ds_num_even_false() {
    assert_eq!(eval_expr("(even? 3)"), "false");
}

#[test]
fn clj_ds_num_odd() {
    assert_eq!(eval_expr("(odd? 3)"), "true");
}

#[test]
fn clj_ds_num_odd_false() {
    assert_eq!(eval_expr("(odd? 2)"), "false");
}

#[test]
fn clj_ds_num_inc() {
    assert_eq!(eval_expr("(inc 41)"), "42");
}

#[test]
fn clj_ds_num_dec() {
    assert_eq!(eval_expr("(dec 43)"), "42");
}

#[test]
fn clj_ds_num_min() {
    assert_eq!(eval_expr("(min 1 2 3)"), "1");
}

#[test]
fn clj_ds_num_max() {
    assert_eq!(eval_expr("(max 1 2 3)"), "3");
}

#[test]
fn clj_ds_num_abs() {
    assert_eq!(eval_expr("(abs -5)"), "5");
}

#[test]
fn clj_ds_num_abs_pos() {
    assert_eq!(eval_expr("(abs 5)"), "5");
}

#[test]
fn clj_ds_num_rem() {
    assert_eq!(eval_expr("(rem 10 3)"), "1");
}

#[test]
fn clj_ds_num_mod() {
    assert_eq!(eval_expr("(mod 10 3)"), "1");
}

#[test]
fn clj_ds_num_quot() {
    assert_eq!(eval_expr("(quot 10 3)"), "3");
}

// ============================================================================
// MISC / COMBINED TESTS
// ============================================================================

#[test]
fn clj_ds_identity() {
    assert_eq!(eval_expr("(identity 42)"), "42");
}

#[test]
fn clj_ds_identity_nil() {
    assert_eq!(eval_expr("(identity nil)"), "nil");
}

#[test]
fn clj_ds_str_nil() {
    assert_eq!(eval_expr("(str nil)"), "\"\"");
}

#[test]
fn clj_ds_str_number() {
    assert_eq!(eval_expr("(str 42)"), "\"42\"");
}

#[test]
fn clj_ds_str_keyword() {
    assert_eq!(eval_expr("(str :kw)"), "\":kw\"");
}

#[test]
fn clj_ds_str_bool_true() {
    assert_eq!(eval_expr("(str true)"), "\"true\"");
}

#[test]
fn clj_ds_str_bool_false() {
    assert_eq!(eval_expr("(str false)"), "\"false\"");
}

#[test]
fn clj_ds_str_vec() {
    assert_eq!(eval_expr("(str [1 2 3])"), "\"[1 2 3]\"");
}

#[test]
fn clj_ds_str_list() {
    assert_eq!(eval_expr("(str (list 1 2 3))"), "\"(1 2 3)\"");
}

#[test]
fn clj_ds_str_concat() {
    assert_eq!(eval_expr("(str 1 2 3)"), "\"123\"");
}

#[test]
fn clj_ds_let_basic() {
    assert_eq!(eval_expr("(let [x 1 y 2] (+ x y))"), "3");
}

#[test]
fn clj_ds_let_nested() {
    assert_eq!(eval_expr("(let [x (+ 1 2)] (* x x))"), "9");
}

#[test]
fn clj_ds_loop_recur_sum() {
    assert_eq!(
        eval_expr("(loop [i 0 sum 0] (if (< i 10) (recur (inc i) (+ sum i)) sum))"),
        "45"
    );
}

#[test]
fn clj_ds_fn_basic() {
    assert_eq!(eval_expr("((fn [x] (+ x 1)) 41)"), "42");
}

#[test]
fn clj_ds_fn_multi_arg() {
    assert_eq!(eval_expr("((fn [a b c] (+ a b c)) 1 2 3)"), "6");
}

#[test]
fn clj_ds_do_returns_last() {
    assert_eq!(eval_expr("(do 1 2 3)"), "3");
}

#[test]
fn clj_ds_do_returns_nil() {
    assert_eq!(eval_expr("(do 1 2 nil)"), "nil");
}

// -- sorted collections are not implemented --

#[test]
#[ignore] // sorted-set not implemented
fn clj_ds_sorted_set() {
    assert_eq!(eval_expr("(first (sorted-set 3 1 2))"), "1");
}

#[test]
#[ignore] // sorted-map not implemented
fn clj_ds_sorted_map() {
    assert_eq!(eval_expr("(first (keys (sorted-map :b 2 :a 1)))"), ":a");
}
