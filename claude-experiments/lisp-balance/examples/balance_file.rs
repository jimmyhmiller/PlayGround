use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: balance_file <path>");
        return ExitCode::from(2);
    }
    let src = match fs::read_to_string(&args[1]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read {}: {}", &args[1], e);
            return ExitCode::from(1);
        }
    };
    print!("{}", lisp_balance::balance(&src));
    ExitCode::SUCCESS
}
