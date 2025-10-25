mod cli;
mod commands;

use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Commands::Analyze(args) => commands::analyze::run(args),
        cli::Commands::Plot(plot_cmd) => commands::plot::run(plot_cmd),
    }
}
