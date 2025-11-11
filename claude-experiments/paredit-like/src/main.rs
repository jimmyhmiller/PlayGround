pub mod ast;
pub mod cli;
pub mod parinfer;
pub mod parinfer_simple;
pub mod parser;
pub mod refactor;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands, Output};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Balance {
            file,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let parinfer = parinfer_simple::Parinfer::new(&source);
            let modified = parinfer.balance()?;

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Slurp {
            file,
            line,
            backward,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let mut refactorer = refactor::Refactorer::new(source.clone());

            let modified = if backward {
                refactorer.slurp_backward(&forms, line)?
            } else {
                refactorer.slurp_forward(&forms, line)?
            };

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Barf {
            file,
            line,
            backward,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let mut refactorer = refactor::Refactorer::new(source.clone());

            let modified = if backward {
                refactorer.barf_backward(&forms, line)?
            } else {
                refactorer.barf_forward(&forms, line)?
            };

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Splice {
            file,
            line,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let mut refactorer = refactor::Refactorer::new(source.clone());

            let modified = refactorer.splice(&forms, line)?;

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Raise {
            file,
            line,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let mut refactorer = refactor::Refactorer::new(source.clone());

            let modified = refactorer.raise(&forms, line)?;

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Wrap {
            file,
            line,
            with,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let mut refactorer = refactor::Refactorer::new(source.clone());

            let modified = refactorer.wrap(&forms, line, &with)?;

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::MergeLet {
            file,
            line,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let mut refactorer = refactor::Refactorer::new(source.clone());

            let modified = if let Some(line_num) = line {
                // Merge specific let at line
                refactorer.merge_let(&forms, line_num)?
            } else {
                // Merge all lets in the file
                refactorer.merge_all_lets(&forms)?
            };

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Indent {
            file,
            in_place,
            dry_run,
            diff,
        } => {
            let source = cli::read_file(&file)?;
            let mut parser = parser::ClojureParser::new()?;
            let forms = parser.parse_to_sexpr(&source)?;
            let indenter = refactor::Indenter::new(source.clone());

            let modified = indenter.indent(&forms)?;

            let output = Output {
                original: source,
                modified,
            };

            output.display(in_place, dry_run, diff, &file)?;
        }

        Commands::Batch {
            pattern,
            command,
            dry_run,
        } => {
            cli::process_batch(&pattern, &command, dry_run)?;
        }

        Commands::Install { path, force } => {
            cli::install_documentation(path, force)?;
        }
    }

    Ok(())
}
