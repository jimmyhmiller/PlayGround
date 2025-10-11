mod models;
mod parsers;
mod storage;
mod git;
mod cli;
mod inline;

use clap::Parser;
use cli::{Cli, execute_command};

fn main() {
    let cli = Cli::parse();

    if let Err(e) = execute_command(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
