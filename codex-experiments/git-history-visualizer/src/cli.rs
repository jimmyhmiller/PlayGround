use std::path::PathBuf;

use clap::{ArgAction, Args, Parser, Subcommand, ValueHint};

#[derive(Debug, Parser)]
#[command(
    name = "git-history-visualizer",
    author,
    version,
    about = "Rust-native clone of git-of-theseus with an ergonomic CLI"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Analyze a git repository and emit JSON metrics
    Analyze(AnalyzeArgs),
    /// Generate plots from analysis JSON outputs
    Plot(PlotCommandsArgs),
}

#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// Path to the git repository to analyze
    #[arg(value_hint = ValueHint::DirPath)]
    pub repo: PathBuf,

    /// Python datetime format string (default: %Y)
    #[arg(long = "cohort-format", default_value = "%Y")]
    pub cohort_format: String,

    /// Minimum seconds between analyzed commits (default: 604800, 1 week)
    #[arg(long, default_value_t = 7 * 24 * 60 * 60)]
    pub interval: u64,

    /// Glob patterns to ignore (repeatable)
    #[arg(long = "ignore", value_name = "PATTERN")]
    pub ignore_patterns: Vec<String>,

    /// Glob patterns that must match (repeatable)
    #[arg(long = "only", value_name = "PATTERN")]
    pub only_patterns: Vec<String>,

    /// Output directory for generated JSON files
    #[arg(long, value_hint = ValueHint::DirPath, default_value = ".")]
    pub outdir: PathBuf,

    /// Branch to analyze (default: master)
    #[arg(long, default_value = "master")]
    pub branch: String,

    /// Include all filetypes instead of limiting to known source lexers
    #[arg(long, action = ArgAction::SetTrue)]
    pub all_filetypes: bool,

    /// Ignore whitespace-only changes during blame
    #[arg(long, action = ArgAction::SetTrue)]
    pub ignore_whitespace: bool,

    /// Suppress progress output
    #[arg(long, action = ArgAction::SetTrue)]
    pub quiet: bool,

    /// Number of parallel blame workers
    #[arg(long, default_value_t = 2, value_name = "N")]
    pub jobs: usize,

    /// Enable git commit-graph optimization before analysis
    #[arg(long, action = ArgAction::SetTrue)]
    pub opt: bool,
}

#[derive(Debug, Subcommand)]
pub enum PlotCommands {
    /// Render a stack plot from cohorts/authors/exts JSON
    Stack(StackPlotArgs),
    /// Render a line plot from authors/cohorts JSON
    Line(LinePlotArgs),
    /// Render survival curves
    Survival(SurvivalPlotArgs),
}

#[derive(Debug, Args)]
pub struct CommonPlotArgs {
    /// Display the plot interactively
    #[arg(long, action = ArgAction::SetTrue)]
    pub display: bool,

    /// Path to write rendered image
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "plot.png")]
    pub outfile: PathBuf,
}

#[derive(Debug, Args)]
pub struct StackPlotArgs {
    #[command(flatten)]
    pub common: CommonPlotArgs,

    /// Maximum number of series to render (others are collapsed into \"other\")
    #[arg(long = "max-series", default_value_t = 20)]
    pub max_series: usize,

    /// Normalize each sample to 100%
    #[arg(long, action = ArgAction::SetTrue)]
    pub normalize: bool,

    /// Input JSON file produced by analyze step
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
}

#[derive(Debug, Args)]
pub struct LinePlotArgs {
    #[command(flatten)]
    pub common: CommonPlotArgs,

    /// Maximum number of series to render
    #[arg(long = "max-series", default_value_t = 20)]
    pub max_series: usize,

    /// Normalize each sample to 100%
    #[arg(long, action = ArgAction::SetTrue)]
    pub normalize: bool,

    /// Input JSON file produced by analyze step
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
}

#[derive(Debug, Args)]
pub struct SurvivalPlotArgs {
    #[command(flatten)]
    pub common: CommonPlotArgs,

    /// Include exponential decay fit
    #[arg(long = "exp-fit", action = ArgAction::SetTrue)]
    pub exp_fit: bool,

    /// Number of years to display on x axis
    #[arg(long, default_value_t = 5.0)]
    pub years: f64,

    /// One or more survival JSON files produced by analyze
    #[arg(value_hint = ValueHint::FilePath)]
    pub inputs: Vec<PathBuf>,
}

#[derive(Debug, Args)]
pub struct PlotCommandsArgs {
    #[command(subcommand)]
    pub command: PlotCommands,
}
