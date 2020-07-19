

use super::new::*;
use super::parser2::*;
use std::time::Instant;

pub fn run_new() {

    let mut program = Program::new();
    let m = &mut program.main;
    
    
    read_new("fact(20)", m, &mut program.symbols);
    // if let Some((focus, root)) = program.main.forest.persistent_change(Expr::Symbol(2), n4.unwrap()) {
    //     let result = program.main.forest.garbage_collect(root, focus);
    //     println!("{}: {:?}", root, result);
    // }
    let now = Instant::now();
    program.full_step();
    program.main.garbage_collect();
    // println!("{:?}", program.main.forest.arena.len());
    println!("{:?}", now.elapsed().as_micros());
    program.pretty_print_main();
    // println!("{:?}", program.main);


    // Silly litle trick that does speed things up.
    // std::thread::spawn(move || drop(program));

}