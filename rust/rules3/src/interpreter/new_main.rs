

use super::new::*;
use super::parser2::*;

pub fn run_new() {

    let mut program = Program::new();
    let m = &mut program.main;
    
    
    read_new("builtin/+(builtin/-(2, builtin/+(3, 4)), builtin/*(5, 6))", m, &mut program.symbols);
    // if let Some((focus, root)) = program.main.forest.persistent_change(Expr::Symbol(2), n4.unwrap()) {
    //     let result = program.main.forest.garbage_collect(root, focus);
    //     println!("{}: {:?}", root, result);
    // }
    program.full_step();
    program.pretty_print_main();
    // println!("{:?}", program.main);


    // Silly litle trick that does speed things up.
    // std::thread::spawn(move || drop(program));

}