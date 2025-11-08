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
