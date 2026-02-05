use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};

#[derive(Parser)]
#[command(name = "keep-running")]
#[command(about = "Human-friendly terminal session manager with dtach-style detach")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Session name for shortcut attach (e.g., `keep-running fuzzy-penguin`)
    #[arg(value_name = "SESSION")]
    pub session: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start a new session with the given command
    #[command(trailing_var_arg = true)]
    Run {
        /// Session name (auto-generated if not specified)
        #[arg(short, long)]
        name: Option<String>,

        /// Command to run
        #[arg(required = true)]
        command: Vec<String>,
    },

    /// Start a new session with your default shell
    Shell {
        /// Session name (auto-generated if not specified)
        #[arg(short, long)]
        name: Option<String>,
    },

    /// Attach to an existing session
    Attach {
        /// Session name (or prefix)
        session: String,
    },

    /// List all running sessions
    #[command(alias = "ls")]
    List,

    /// Kill a running session
    Kill {
        /// Session name (or prefix)
        session: String,
    },

    /// Start a session daemon without attaching (for scripts/testing)
    Start {
        /// Session name (auto-generated if not specified)
        #[arg(short, long)]
        name: Option<String>,

        /// Command to run
        #[arg(required = true, trailing_var_arg = true)]
        command: Vec<String>,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

pub fn parse() -> Cli {
    Cli::parse()
}

pub fn print_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "keep-running", &mut std::io::stdout());
}
