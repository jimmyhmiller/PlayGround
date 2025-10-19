use paredit_like::parinfer::Parinfer;

#[test]
fn balanced_input_is_preserved() {
    let input = "(def foo [x] (+ x 1))";
    let output = Parinfer::new(input).balance().expect("expected success");
    assert_eq!(output, input);
}

#[test]
fn missing_closer_is_not_fixed() {
    let input = "(def foo [x] (+ x 1)";
    let result = Parinfer::new(input).balance();
    assert!(result.is_err(), "parinfer_rust does not synthesize missing closers");
}

#[test]
fn stray_closing_paren_causes_error() {
    let input = "(+ 1 2))";
    let result = Parinfer::new(input).balance();
    assert!(result.is_err(), "parinfer_rust does not drop stray closing parens");
}

#[test]
fn escaped_quote_remains_untouched() {
    let input = "(\"\\\\\")";
    let output = Parinfer::new(input).balance().expect("expected escaped quote to round-trip");
    assert_eq!(output, input);
}

#[test]
fn astral_plane_identifier_is_preserved() {
    let input = "[[𑏌]]";
    let output = Parinfer::new(input).balance().expect("expected success with astral-plane identifier");
    assert_eq!(output, input);
}
