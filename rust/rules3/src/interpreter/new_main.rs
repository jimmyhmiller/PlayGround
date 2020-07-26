

use super::new::*;
use super::parser2::*;
use std::time::Instant;


pub fn run_new() {
    let now = Instant::now();
    let mut program = Program::new();
    // Need to set this non-manually
    program.clause_indexes = vec![Clause{left: 14, right: 18}, Clause{left: 21, right: 25},];
    let main = &mut program.main;
    let rules = &mut program.rules;
    
    // read_new("[{type rule, name: fact, scope @main, clauses: [{left: fact(0), right: 1}, {left: fact(?n), right:*(?n, fact(-(?n 1)))}]}]", m, &mut program.symbols);
    // read_new("quote(fact(1))", m, &mut program.symbols);
    read_new("quote([{
        type: rule,
        name: fact,
        scopes: [@main]
        clauses: [
            {
                left: fact(0),
                right: 1,
            }, {
                left: fact(?n),
                right: builtin/*(?n fact(builtin/-(?n 1)))
            },
        ]
    }])", rules, &mut program.symbols);

    read_new("fact(20)", main, &mut program.symbols);
    // if let Some((focus, root)) = program.main.forest.persistent_change(Expr::Symbol(2), n4.unwrap()) {
    //     let result = program.main.forest.garbage_collect(root, focus);
    //     println!("{}: {:?}", root, result);
    // }
    program.full_step();
    // program.rewrite(0);


    program.main.garbage_collect();
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