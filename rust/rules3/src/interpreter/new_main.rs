

use super::new::*;
use super::parser2::*;
use std::time::Instant;


pub fn run_new() {
    let now = Instant::now();
    let mut program = Program::new();
    // Need to set this non-manually
    // program.clause_indexes = vec![Clause{left: 14, right: 18}, Clause{left: 21, right: 25},];
    let main = &mut program.main;
    let rules = &mut program.rules;
    
    // read_new("[{type rule, name: fact, scope @main, clauses: [{left: fact(0), right: 1}, {left: fact(?n), right:*(?n, fact(-(?n 1)))}]}]", m, &mut program.symbols);
    // read_new("quote(fact(1))", m, &mut program.symbols);
    read_new("quote([{
        type: rule,
        name: fact,
        in_scopes: [@main],
        out_scopes: [@main],
        clauses: [
            {
                left: fact(0),
                right: 1,
            }, {
                left: fact(?n),
                right: builtin/*(?n fact(builtin/-(?n 1)))
            },
        ]
    }, {
        type: rule,
        name: main-rule,
        in_scopes: [@meta],
        out_scopes: [@io],
        clauses: [
            {
                left: {original_expr: ?x, new_expr: ?y},
                right: builtin/println(quote([?x => ?y]))
            }
        ]
    }])", rules, &mut program.symbols);

    
    read_new("fact(20)", main, &mut program.symbols);
    // if let Some((focus, root)) = program.main.forest.persistent_change(Expr::Symbol(2), n4.unwrap()) {
    //     let result = program.main.forest.garbage_collect(root, focus);
    //     println!("{}: {:?}", root, result);
    // }
    program.set_clause_indexes();
    // println!("{:?}", program.clause_indexes);
    program.full_step();
    // program.rewrite(0);


    program.main.garbage_collect_self();
    program.io.garbage_collect_self();
    // println!("{:?}", program.main.forest.arena.len());
    program.pretty_print_main();
    // println!("{:?}",program.main.get_focus());
    // program.pretty_print_scope(&program.rules);

    // let env = program.build_env(&program.main, 21, 0);
    // println!("{:?}", env);

    println!("{}", now.elapsed().as_micros());

    // Silly litle trick that does speed things up.
    // std::thread::spawn(move || drop(program));

}






// I have basically a graph because of parent pointers.
// That means I can't really make a persistent data structure.
// But what if everything kept a version and nodes kept a map of versions to old values?
// Then I could get to a node and look up any version of that node.
// That way I can have pointers in other scopes that know the scope they came from and the version
// of the node they are looking for.
// GC gets a bit harder, but totally solvable.