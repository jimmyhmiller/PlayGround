use paredit_like::ast::SExpr;
use paredit_like::parser::ClojureParser;
use paredit_like::refactor::Refactorer;

fn extract_binding_texts(bindings: &[SExpr], source: &str) -> Vec<String> {
    bindings
        .iter()
        .map(|child| {
            let span = child.span();
            source[span.start.offset..span.end.offset].to_string()
        })
        .collect()
}

#[test]
fn merge_let_preserves_typed_bindings_structure() {
    let source = r#"(let [user (: User) ctx]
  (let [name (: Name) user
        email (: Email) user]
    {:name name
     :email email}))"#;

    let mut parser = ClojureParser::new().unwrap();
    let forms = parser.parse_to_sexpr(source).unwrap();
    let mut refactorer = Refactorer::new(source.to_string());

    let merged = refactorer.merge_let(&forms, 1).unwrap();

    // Should flatten into a single let form
    assert_eq!(merged.matches("(let").count(), 1, "Expected a single let form: {merged}");

    let merged_forms = parser.parse_to_sexpr(&merged).unwrap();
    let root = merged_forms.first().expect("expected root form");
    let children = match root {
        SExpr::List { children, .. } => children,
        _ => panic!("root form was not a list: {root:?}"),
    };

    // Expect: (let [bindings ...] body)
    assert!(
        matches!(children.first(), Some(SExpr::Atom { value, .. }) if value == "let"),
        "first child should be let keyword: {children:?}"
    );

    let binding_vector = match &children[1] {
        SExpr::List { open, children, .. } if *open == '[' => children,
        other => panic!("expected binding vector, found {other:?}"),
    };

    let binding_texts = extract_binding_texts(binding_vector, &merged);
    assert_eq!(
        binding_texts,
        vec![
            "user",
            "(: User)",
            "ctx",
            "name",
            "(: Name)",
            "user",
            "email",
            "(: Email)",
            "user"
        ],
        "merged bindings did not preserve typed annotation structure"
    );
}

#[test]
fn merge_all_lets_handles_nested_typed_bindings_and_values() {
    let source = r#"(let [user (: User) ctx]
  (let [session (: Session) ctx]
    (let [role (: Role) session
          label (str role)]
      (println user session role label))))"#;

    let mut parser = ClojureParser::new().unwrap();
    let forms = parser.parse_to_sexpr(source).unwrap();
    let mut refactorer = Refactorer::new(source.to_string());

    let merged = refactorer.merge_all_lets(&forms).unwrap();

    // Should collapse all nested lets into a single binding vector
    assert_eq!(merged.matches("(let").count(), 1, "Expected a single let form: {merged}");

    let merged_forms = parser.parse_to_sexpr(&merged).unwrap();
    let root = merged_forms.first().expect("expected root form");
    let children = match root {
        SExpr::List { children, .. } => children,
        _ => panic!("root form was not a list: {root:?}"),
    };

    let binding_vector = match &children[1] {
        SExpr::List { open, children, .. } if *open == '[' => children,
        other => panic!("expected binding vector, found {other:?}"),
    };

    let binding_texts = extract_binding_texts(binding_vector, &merged);
    assert_eq!(
        binding_texts,
        vec![
            "user",
            "(: User)",
            "ctx",
            "session",
            "(: Session)",
            "ctx",
            "role",
            "(: Role)",
            "session",
            "label",
            "(str role)"
        ],
        "merge_all_lets lost or reordered bindings with typed annotations"
    );
}
