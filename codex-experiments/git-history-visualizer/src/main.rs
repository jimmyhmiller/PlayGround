use anyhow::Result;
use clap::Parser;
use git_history_visualizer::{cli, commands};

fn main() -> Result<()> {
    let cli = cli::Cli::parse();
    match cli.command {
        cli::Commands::Analyze(args) => commands::analyze::run(args),
        cli::Commands::Plot(plot_cmds) => commands::plot::run(plot_cmds.command),
    }
}
