mod ast;
mod syntax;
mod instruction;




fn main() {
    let program_lisp = program! {
        (set x 10)
        (set y 5)
        (set sum (+ (var x) (var y)))
        (if (> (var sum) 10) 
            (set result 1) 
            (set result 0))
        (while (> (var x) 0) 
            (set x (- (var x) 1)))
    };

    println!("{:#?}", program_lisp);
}
