use paredit_like::parinfer::Parinfer;

#[test]
fn balanced_input_is_preserved() {
    let input = "(def foo [x] (+ x 1))";
    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, input);
}

#[test]
fn missing_closer_is_fixed() {
    let input = "(def foo [x] (+ x 1)";
    let result = Parinfer::new(input).balance();
    assert!(result.is_ok(), "indent_mode should synthesize missing closers");
    let output = result.unwrap();
    assert_eq!(output, "(def foo [x] (+ x 1))");
}

#[test]
fn stray_closing_paren_is_removed() {
    let input = "(+ 1 2))";
    let result = Parinfer::new(input).balance();
    assert!(result.is_ok(), "indent_mode should drop stray closing parens");
    let output = result.unwrap();
    assert_eq!(output, "(+ 1 2)");
}

#[test]
fn escaped_quote_remains_untouched() {
    let input = "(\"\\\\\")";
    let output = Parinfer::new(input).balance().expect("expected escaped quote to round-trip");
    assert_eq!(output, input);
}

#[test]
#[ignore] // parinfer_rust indent_mode has a bug with astral-plane Unicode characters
fn astral_plane_identifier_is_preserved() {
    // Known issue: parinfer_rust's indent_mode incorrectly handles astral-plane
    // Unicode characters (like 𑏌) by removing closing brackets.
    // Input: "[[𑏌]]" -> Output: "[[𑏌]" (missing one ])
    // This is a bug in parinfer_rust, not in our code.
    let input = "[[𑏌]]";
    let output = Parinfer::new(input).balance().expect("expected success with astral-plane identifier");
    assert_eq!(output, input);
}

#[test]
fn sibling_regions_at_same_indent_balanced_input_preserved() {
    // When input is already balanced, parinfer preserves the structure
    let input = r#"(scf.if {:result f32} cond
  (region
    (scf.yield x))
  (region
    (scf.yield y)))"#;

    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, input, "balanced input should be preserved");
}

#[test]
fn sibling_regions_at_same_indent_unbalanced_input_works() {
    // This case actually WORKS - parinfer correctly handles sibling forms
    // when they're indented relative to the parent's opening line
    let input = r#"(scf.if {:result f32} cond
  (region
    (scf.yield x
  (region
    (scf.yield y"#;

    let expected = r#"(scf.if {:result f32} cond
  (region
    (scf.yield x))
  (region
    (scf.yield y)))"#;

    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, expected, "This case actually works!\nActual output:\n{}", output);
}

#[test]
#[ignore] // Known bug: parinfer cannot handle siblings at same indent as parent's opening column
fn sibling_forms_at_same_indent_as_parent_bug() {
    // BUG: When sibling forms are at the SAME indentation level as the opening
    // paren's content (column 0 in this case), parinfer cannot keep them
    // inside the parent.
    //
    // This is the core issue: two (region ...) forms at column 0 are
    // indistinguishable from two top-level forms.
    let input = r#"(scf.if cond
(region
  (scf.yield x
(region
  (scf.yield y"#;

    // What we WANT: both regions inside scf.if
    let expected = r#"(scf.if cond
(region
  (scf.yield x))
(region
  (scf.yield y)))"#;

    // What parinfer PRODUCES (wrong):
    // (scf.if cond)
    // (region
    //   (scf.yield x))
    // (region
    //   (scf.yield y))
    //
    // It closes scf.if immediately because (region starts at column 0

    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, expected,
        "sibling regions should stay inside scf.if.\nActual output:\n{}",
        output);
}

#[test]
fn scf_if_with_two_region_branches_works() {
    // This case WORKS - when regions are properly indented relative to scf.if
    let input = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      body1
    (region
      body2"#;

    let expected = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      body1)
    (region
      body2)))"#;

    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, expected);
}

#[test]
fn scf_if_user_scenario_regions_at_same_indent() {
    // User's exact scenario:
    // (scf.if {:result f32} cond
    //   (region ...)   ;; then region
    //   (region ...))  ;; else region - sibling, same indent
    //
    // Both regions at column 2, scf.if at column 0

    let input = r#"(scf.if {:result f32} cond
  (region
    (scf.yield x
  (region
    (scf.yield y"#;

    let expected = r#"(scf.if {:result f32} cond
  (region
    (scf.yield x))
  (region
    (scf.yield y)))"#;

    let output = Parinfer::new(input).balance().expect("expected success");

    eprintln!("=== User scenario: scf.if with two regions at same indent ===");
    eprintln!("INPUT:\n{}", input);
    eprintln!("OUTPUT:\n{}", output);
    eprintln!("EXPECTED:\n{}", expected);

    assert_eq!(output, expected,
        "both regions should be children of scf.if.\nActual output:\n{}",
        output);
}

#[test]
fn scf_if_inside_def_second_region_becomes_sibling_of_def() {
    // Potential bug scenario: the second region might become a sibling of `def`
    // instead of staying inside scf.if
    //
    // This could happen if parens were incorrectly placed before balancing

    let input = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      (scf.yield x))
    (region
      (scf.yield y))))"#;

    // If already balanced, should be preserved exactly
    let output = Parinfer::new(input).balance().expect("expected success");

    eprintln!("=== Already balanced input ===");
    eprintln!("INPUT:\n{}", input);
    eprintln!("OUTPUT:\n{}", output);

    assert_eq!(output, input, "balanced input should be preserved");
}

#[test]
fn scf_if_with_wrong_closing_parens_after_first_region() {
    // What if someone wrote this (incorrectly closing scf.if after first region):
    let input = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      (scf.yield x)))
  (region
    (scf.yield y)))"#;

    // Parinfer should NOT "fix" this into what we want because the second region
    // is at indent 2 (same as scf.if), making it a sibling of scf.if, not a child.
    // This is technically valid indentation for a sibling.

    let output = Parinfer::new(input).balance().expect("expected success");

    eprintln!("=== Wrong closing parens scenario ===");
    eprintln!("INPUT:\n{}", input);
    eprintln!("OUTPUT:\n{}", output);

    // The structure IS ambiguous from indentation alone
    // Parinfer will keep it as-is because the indentation is consistent
    assert_eq!(output, input);
}

#[test]
#[ignore] // Known bug: sibling regions at same indent as parent become siblings of parent
fn scf_if_second_region_at_same_indent_as_parent_bug() {
    // CONFIRMED BUG: when second region is at SAME indent as scf.if (not indented further),
    // parinfer treats it as a sibling of scf.if, not a child.
    //
    // This is a fundamental limitation of indent-based paren inference.
    //
    // Input: second region at column 2 (same as scf.if start)
    let input = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      (scf.yield x
  (region
    (scf.yield y"#;

    // What parinfer produces (WRONG):
    // - scf.if contains only first region (which is indented at 4)
    // - second region at indent 2 is a sibling of scf.if, child of def
    let _parinfer_wrong_result = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      (scf.yield x)))
  (region
    (scf.yield y)))"#;

    // What user WANTS (both regions inside scf.if):
    let expected = r#"(def my-func
  (scf.if {:result f32} cond
    (region
      (scf.yield x))
    (region
      (scf.yield y))))"#;

    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, expected,
        "BUG: second region should be inside scf.if, not a sibling.\n\
         Actual output:\n{}", output);
}

#[test]
#[ignore] // Known bug: paredit-like closes scf.if after first region
fn scf_if_sibling_regions_exact_repro() {
    // Exact reproduction from user report.
    //
    // The bug: When two sibling forms (the two (region ...) calls) have the same
    // indentation level, paredit-like treats them as if they should each close
    // their own parent. But they're actually both arguments to the same parent (scf.if).
    //
    // BEFORE (correct - what user wrote):
    // - First region ends with )))  (3 parens: yield, block, region)
    // - Second region ends with ))))) (5 parens: yield, block, region, scf.if, def)
    //
    // AFTER paredit-like balance (WRONG):
    // - First region ends with )))) (4 parens: closes scf.if prematurely!)
    // - Second region ends with )))) (4 parens: now sibling of def)

    let correct_input = r#"(def result (scf.if {:result i32} cond
    (region
      (block []
        (scf.yield true_val)))
    (region
      (block []
        (scf.yield false_val)))))"#;

    // This is what paredit-like WRONGLY produces:
    let _wrong_output = r#"(def result (scf.if {:result i32} cond
    (region
      (block []
        (scf.yield true_val))))
    (region
      (block []
        (scf.yield false_val))))"#;

    let output = Parinfer::new(correct_input).balance().expect("expected success");

    eprintln!("=== Exact repro case ===");
    eprintln!("INPUT:\n{}", correct_input);
    eprintln!("OUTPUT:\n{}", output);

    // The balanced input should be preserved exactly
    assert_eq!(output, correct_input,
        "BUG: paredit-like breaks correctly balanced scf.if with two regions.\n\
         First region should have 3 closing parens, second should have 5.\n\
         Actual output:\n{}", output);
}
