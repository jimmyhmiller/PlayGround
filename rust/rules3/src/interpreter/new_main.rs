

use super::new::*;
use super::parser2::*;

pub fn run_new() {

    let mut program = Program::new();
    let m = &mut program.main;
    
    read_new("+(2, 3)", m);
    // if let Some((focus, root)) = program.main.forest.persistent_change(Expr::Symbol(2), n4.unwrap()) {
    //     let result = program.main.forest.garbage_collect(root, focus);
    //     println!("{}: {:?}", root, result);
    // }
    program.full_step();
    program.main.forest.print_tree(program.main.root);


    // Silly litle trick that does speed things up.
    // std::thread::spawn(move || drop(program));

}