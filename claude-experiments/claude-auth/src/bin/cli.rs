use clap::{Args, Parser, Subcommand};
use std::fs;
use std::process::Command;
use tokio;

use claude_auth::{AuthError, ClaudeAuth, OAuthFlow, TokenInfo};

#[derive(Parser)]
#[command(name = "claude-auth-cli")]
#[command(about = "Claude Code subscription authentication CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Auth(AuthArgs),
    Token(TokenArgs),
}

#[derive(Args)]
struct AuthArgs {
    #[command(subcommand)]
    command: AuthCommands,
}

#[derive(Subcommand)]
enum AuthCommands {
    Login {
        #[arg(short, long, default_value = "claude-pro-max")]
        provider: String,
    },
    Manual {
        #[arg(short, long, default_value = "claude-pro-max")]
        provider: String,
        #[arg(long)]
        access_token: String,
        #[arg(long)]
        refresh_token: Option<String>,
        #[arg(long)]
        expires_in: Option<u64>,
    },
    List,
    Remove {
        #[arg(short, long)]
        provider: String,
    },
}

#[derive(Args)]
struct TokenArgs {
    #[arg(short, long, default_value = "claude-pro-max")]
    provider: String,
    #[arg(short, long)]
    output: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Auth(auth_args) => handle_auth_command(auth_args).await?,
        Commands::Token(token_args) => handle_token_command(token_args).await?,
    }

    Ok(())
}

async fn handle_auth_command(args: AuthArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        AuthCommands::Login { provider } => {
            println!("Starting OAuth flow for {}...", provider);
            
            let oauth_flow = OAuthFlow::new();
            let auth_url = oauth_flow.get_authorization_url()?;
            
            println!("Opening browser for OAuth authorization...");
            println!("URL: {}", auth_url);
            
            #[cfg(target_os = "macos")]
            let _ = Command::new("open").arg(&auth_url).spawn();
            
            #[cfg(target_os = "linux")]
            let _ = Command::new("xdg-open").arg(&auth_url).spawn();
            
            #[cfg(target_os = "windows")]
            let _ = Command::new("cmd").args(["/c", "start", &auth_url]).spawn();
            
            println!();
            println!("After authorization, you'll be redirected automatically.");
            println!("Waiting for callback...");
            
            let auth_code = oauth_flow.wait_for_callback()?;
            
            println!("Exchanging authorization code for tokens...");
            match oauth_flow.exchange_code_for_tokens(&auth_code).await {
                Ok(token_info) => {
                    let mut auth = ClaudeAuth::new()?;
                    auth.store_token(&provider, token_info)?;
                    println!("✅ Authentication successful for {}", provider);
                }
                Err(e) => {
                    eprintln!("❌ Authentication failed: {}", e);
                    eprintln!("This is expected due to Cloudflare protection on the token endpoint.");
                    eprintln!("Please use the 'manual' command to store tokens directly.");
                    std::process::exit(1);
                }
            }
        }
        AuthCommands::Manual { provider, access_token, refresh_token, expires_in } => {
            let expires_at = expires_in.map(|exp| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() + exp
            });

            let token_info = TokenInfo {
                access_token,
                refresh_token,
                expires_at,
                client_id: Some(claude_auth::auth::CLAUDE_CLIENT_ID.to_string()),
            };

            let mut auth = ClaudeAuth::new()?;
            auth.store_token(&provider, token_info)?;
            println!("✅ Tokens stored manually for {}", provider);
        }
        AuthCommands::List => {
            let auth = ClaudeAuth::new()?;
            let providers = auth.list_providers();
            
            if providers.is_empty() {
                println!("No authenticated providers found.");
            } else {
                println!("Authenticated providers:");
                for provider in providers {
                    if let Some(token_info) = auth.get_token(&provider) {
                        let status = if auth.is_token_expired(token_info) {
                            "🟡 EXPIRED"
                        } else {
                            "🟢 ACTIVE"
                        };
                        println!("  {} {}", status, provider);
                    }
                }
            }
        }
        AuthCommands::Remove { provider } => {
            let mut auth = ClaudeAuth::new()?;
            auth.remove_token(&provider)?;
            println!("✅ Removed authentication for {}", provider);
        }
    }

    Ok(())
}

async fn handle_token_command(args: TokenArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut auth = ClaudeAuth::new()?;
    
    match auth.get_valid_token(&args.provider).await {
        Ok(token) => {
            if let Some(output_file) = args.output {
                fs::write(&output_file, &token)?;
                println!("Token written to {}", output_file);
            } else {
                println!("{}", token);
            }
        }
        Err(AuthError::ConfigError(msg)) if msg.contains("No token found") => {
            eprintln!("❌ No authentication found for provider: {}", args.provider);
            eprintln!("Run 'claude-auth-cli auth login --provider {}' first.", args.provider);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("❌ Failed to get token: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}