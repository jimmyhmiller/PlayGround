mod lib;

use crate::lib::SBDebugger;

fn main() {
    let debugger = SBDebugger::create(true);
    println!("Hello, world!");
}

