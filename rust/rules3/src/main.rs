#![feature(box_syntax, box_patterns)]

// #![allow(dead_code)]

mod interpreter;
use interpreter::*;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use rustyline::validate::{MatchingBracketValidator};
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline_derive::{Completer, Helper, Highlighter, Hinter};



#[derive(Completer, Helper, Highlighter, Hinter)]
struct InputValidator {
    validator: MatchingBracketValidator,
}

impl Validator for InputValidator {
    fn validate(&self, ctx: &mut ValidationContext) -> Result<ValidationResult, ReadlineError> {
        self.validator.validate(ctx)
    }
}


fn main() {

    let rule_sub = Rule {
        left: read("-(?a ?b)"),
        right: read("builtin/-(?a ?b)"),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule_plus = Rule {
        left: read("+(?a ?b)"),
        right: read("builtin/+(?a ?b)"),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule_mult = Rule {
        left: ("*", "?a", "?b").into(),
        right: ("builtin/*", "?a", "?b").into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule1 = Rule {
        left: ("fact", 0).into(),
        right: 1.into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule2 = Rule {
        left: read("fact(?n)"),
        right: read("*(?n fact(-(?n 1)))"),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule3 = Rule {
        left: vec![("stuff", "?x")].into(),
        right: vec![("thing", "?x")].into(),
        in_scope: "main".to_string(),
        out_scope: "main".to_string(),
    };
    let rule4 = Rule {
        left: "stuff".into(),
        right: "things".into(),
        in_scope: "history".to_string(),
        out_scope: "history".to_string(),
    };
    // let rule4 = Rule {
    //     left: read("{
    //         phase: rewrite,
    //         scope: main,
    //         sub_expr: ?sub,
    //         new_sub_expr: ?new_sub
    //     }"),
    //     right: ("builtin/println", ("quote", ("?sub", "=>", "?new_sub"))).into(),
    //     in_scope: "meta".to_string(),
    //     out_scope: "io".to_string(),
    // };
    // let rule5 = Rule {
    //     left: read("{
    //         phase: rewrite,
    //         scope: main,
    //         expr: ?expr,
    //         new_expr: ?new_expr,
    //     }"),
    //     right: ("builtin/println", ("quote", ("?expr", "=>", "?new_expr"))).into(),
    //     in_scope: "meta".to_string(),
    //     out_scope: "io".to_string(),
    // };
   
    /*
    
    builtin/add-rule(quote({
        left: add-rule({left: ?x, right: ?y})
        right: builtin/add-rule(quote({
            left: ?x,
            right: ?y,
            in_scope: main,
            out_scope: main,
        }))
        in_scope: main,
        out_scope: main,
    }))
    
    add-rule(quote({
        left: [?x => ?y]
        right: add-rule(quote({
            left: ?x,
            right: ?y
        })),
    }))


    builtin/add-rule(quote({
        left: {expr: ?x, new_expr: ?y, scope: main},
        right: do {
            builtin/println(quote(?x) => quote(?y))
        }
        in_scope: meta,
        out_scope: io,
    }))

    builtin/add-rule(quote({
        left: {sub_expr: ?x, new_sub_expr: ?y, scope: main},
        right: do {
            builtin/println(quote(?x) => quote(?y))
        }
        in_scope: meta,
        out_scope: io,
    }))

    builtin/add-rule(quote({
        left: {sub_expr: ?x, new_sub_expr: ?y, scope: main},
        right: builtin/read-line()
        in_scope: meta,
        out_scope: io,
    }))


    // Need to either make `and` or `as` so I can match and name
    builtin/set-scope(quote(@history), [])
    builtin/add-rule(quote({
        left: {expr: ?expr, new_expr: ?new_expr, sub_expr: ?sub_expr, new_sub_expr: ?new_sub_expr, scope: main},
        right: 
            builtin/set-scope(quote(@history), builtin/push-back(quote({expr: ?expr,
                                                                  new_expr: ?new_expr,
                                                                  sub_expr: ?sub_expr,
                                                                  new_sub_expr: ?new_sub_expr,
                                                                  scope: main})
                                                            , @history))
        in_scope: meta,
        out_scope: io,
    }))

    builtin/add-rule(quote({
        left: test(),
        right: builtin/set-scope(quote(@history), builtin/push-back({scope: main2}, @history)),
        in_scope: main,
        out_scope: main,
    }))


    builtin/set-scope(@history, [])
   
    */




    let mut program = Program::new(
        vec![rule_sub, rule_mult, rule_plus, rule1, rule2, rule3, rule4]
    );


    let validator = InputValidator{ validator: MatchingBracketValidator::new()};
    let mut rl = Editor::new();
    rl.set_helper(Some(validator));
    // This readline library seems capable but a bit awkward.
    // Should I spend time implementing things in it?
    // Or look at doing a socket to editor based thing?

    program.submit(read("builtin/set-scope(quote(@history), [])"));
    program.submit(read("builtin/add-rule(quote({
        left: {expr: ?expr, new_expr: ?new_expr, sub_expr: ?sub_expr, new_sub_expr: ?new_sub_expr, scope: main},
        right: 
            builtin/set-scope(quote(@history), builtin/push-back(quote({
                                                                  expr: ?expr,
                                                                  new_expr: ?new_expr,
                                                                  sub_expr: ?sub_expr,
                                                                  new_sub_expr: ?new_sub_expr,
                                                                  scope: main})
                                                            , @history))
        in_scope: meta,
        out_scope: io,
    }))"));

    // program.submit(read("builtin/add-rule(quote({
    //     left: {sub_expr: ?x, new_sub_expr: ?y, scope: history},
    //     right: do {
    //         builtin/println(quote(?x) => quote(?y))
    //     }
    //     in_scope: meta,
    //     out_scope: io,
    // }))"));
    // program.submit(read("fact(1)"));
    
    doit();
    // // program.submit(read("@main"));
    // program.submit(read("@history"));
    // print(program.get_main());

    // loop {
    //     let readline = rl.readline(">> ");
    //     match readline {
    //         Ok(line) => {
    //             rl.add_history_entry(line.as_str());
    //             if line == "" {
    //                 continue;
    //             }
    //             program.submit(read(line.as_ref()));
    //             print(program.get_main());              
    //         },
    //         Err(ReadlineError::Interrupted) => {
    //             println!("Exiting");
    //             break
    //         },
    //         Err(ReadlineError::Eof) => {
    //             println!("Exiting");
    //             break
    //         },
    //         Err(err) => {
    //             println!("Error: {:?}", err);
    //             break
    //         }
    //     }
    // }

   
}



// need to think about nil
// need to think about repeats
// need to be able to add temporary rules
// need to think about socket/editor support
// need to think about how to do ffi
// need to distinguish between quote and exhausted
// need to make exhausted better
// need to plan for changing exhausted when adding rules
// need to figure out how rules get stored in rule scope and in interpreter
// need to add predicates
// need to consider infix operators.
// need to consider list of Expr as Expr. Useful for parsing as rules. Stream?
// need to consider if above is correct or if there should be token rules?
// need to consider if those last two make sense at all.
// need to make parser never panic
// need to figure out how to do macro like things, explicit quoting is not good enough
// need to think about how hygene would work.
// need to think about having a repl scope vs a main scope
// need to be able to make arbitrary scopes
// need to add names to rules
// need to add multiple clauses to rules
// need to add loading from a file
// need to add watching a file
// need to consider adding let




// Sooooo
// I got history working but it is absurdly absurdly slow....
// Need to start actually caring about performance and getting rid of clones