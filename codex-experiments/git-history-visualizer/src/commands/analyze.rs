use anyhow::{Result, ensure};

use crate::{
    analysis::{self, AnalyzeConfig},
    cli,
};

pub fn run(args: cli::AnalyzeArgs) -> Result<()> {
    ensure!(args.jobs > 0, "--jobs must be at least 1");

    let config = AnalyzeConfig {
        repo: args.repo,
        cohort_format: args.cohort_format,
        interval_secs: args.interval,
        ignore_patterns: args.ignore_patterns,
        only_patterns: args.only_patterns,
        outdir: args.outdir,
        branch: args.branch,
        all_filetypes: args.all_filetypes,
        ignore_whitespace: args.ignore_whitespace,
        quiet: args.quiet,
        jobs: args.jobs,
        opt: args.opt,
    };

    analysis::analyze(config)
}
