use anyhow::{Result, bail};

use crate::cli::{LinePlotArgs, PlotCommands, StackPlotArgs, SurvivalPlotArgs};

pub fn run(command: PlotCommands) -> Result<()> {
    match command {
        PlotCommands::Stack(args) => handle_stack(args),
        PlotCommands::Line(args) => handle_line(args),
        PlotCommands::Survival(args) => handle_survival(args),
    }
}

fn handle_stack(args: StackPlotArgs) -> Result<()> {
    bail!(
        "stack plot command is not implemented yet. Received arguments: {:?}",
        args
    );
}

fn handle_line(args: LinePlotArgs) -> Result<()> {
    bail!(
        "line plot command is not implemented yet. Received arguments: {:?}",
        args
    );
}

fn handle_survival(args: SurvivalPlotArgs) -> Result<()> {
    bail!(
        "survival plot command is not implemented yet. Received arguments: {:?}",
        args
    );
}
