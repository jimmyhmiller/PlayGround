use chrono::{DateTime, Local, Utc};
use clap::Parser;
use daemonize::Daemonize;
use dirs::home_dir;
use regex::Regex;
use signal_hook::consts::SIGTERM;
use signal_hook_tokio::Signals;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::net::IpAddr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use surge_ping::{Client, Config, PingIdentifier, PingSequence};
use tokio::time::sleep;
use tokio_stream::StreamExt;

#[derive(Parser)]
#[command(name = "ping-utility")]
#[command(about = "A utility that pings Google every second and logs response times")]
struct Args {
    #[arg(short, long, help = "Run as a daemon")]
    daemon: bool,
    
    #[arg(short, long, help = "Stop the daemon")]
    stop: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let log_dir = get_log_directory()?;
    create_dir_all(&log_dir)?;
    
    let pid_file = log_dir.join("ping-utility.pid");
    
    if args.stop {
        return stop_daemon(&pid_file).await;
    }
    
    // Create the client before daemonizing to avoid fd issues
    log_debug(&log_dir, "Creating ping client...").await?;
    let config = Config::default();
    let client = Client::new(&config)?;
    log_debug(&log_dir, "Ping client created successfully").await?;

    if args.daemon {
        // Write PID file for tracking
        let pid = std::process::id();
        std::fs::write(&pid_file, pid.to_string())?;
        
        // Set environment variable to indicate background mode
        std::env::set_var("DAEMONIZED", "1");
        log_debug(&log_dir, &format!("Starting in background mode with PID {}", pid)).await?;
        println!("Background mode started with PID {} - check logs in {}", pid, log_dir.display());
        println!("Use 'cargo run -- --stop' to stop the process");
    } else {
        log_debug(&log_dir, "Starting in foreground mode...").await?;
    }
    
    match run_ping_loop(client).await {
        Ok(_) => {
            log_debug(&log_dir, "Ping loop completed successfully").await?;
            Ok(())
        },
        Err(e) => {
            log_debug(&log_dir, &format!("Ping loop failed: {}", e)).await?;
            Err(e)
        }
    }
}

async fn run_ping_loop(client: Client) -> Result<(), Box<dyn std::error::Error>> {
    let log_dir = get_log_directory()?;
    log_debug(&log_dir, "Starting ping loop...").await?;
    
    let running = Arc::new(AtomicBool::new(true));
    log_debug(&log_dir, "Running flag created").await?;
    
    // Only set up signal handling if not daemonized (to avoid fd issues)
    if std::env::var("DAEMONIZED").is_err() {
        let signals = Signals::new(&[SIGTERM])?;
        let mut signals = signals.fuse();
        let running_clone = running.clone();
        
        tokio::spawn(async move {
            while let Some(signal) = signals.next().await {
                match signal {
                    SIGTERM => {
                        println!("Received SIGTERM, shutting down gracefully...");
                        running_clone.store(false, Ordering::SeqCst);
                    }
                    _ => {}
                }
            }
        });
    }

    let google_ip: IpAddr = "8.8.8.8".parse()?;
    let target_ssid = "The Flying Circus of Ontology";
    let mut on_target_network = false;
    let mut pinger_option: Option<surge_ping::Pinger> = None;
    let mut sequence = 0u16;
    
    log_debug(&log_dir, &format!("Target SSID: '{}'", target_ssid)).await?;
    log_debug(&log_dir, &format!("Target IP: {}", google_ip)).await?;

    log_debug(&log_dir, "Starting main loop with timed checks...").await?;
    
    let mut last_network_check = Instant::now();
    let mut last_status_log = Instant::now();
    let mut last_reason_log = Instant::now();
    let mut last_cleanup = Instant::now();
    
    // Do initial network check
    let current_network = get_current_wifi_network(&log_dir).await;
    on_target_network = match current_network {
        Ok(ref ssid) => {
            let is_match = ssid == target_ssid;
            log_debug(&log_dir, &format!("Initial network check - Current: '{}', Target: '{}', Match: {}", 
                ssid, target_ssid, is_match)).await?;
            is_match
        },
        Err(ref e) => {
            log_debug(&log_dir, &format!("Initial network check failed: {}", e)).await?;
            false
        },
    };
    
    if on_target_network {
        pinger_option = Some(client.pinger(google_ip, PingIdentifier(rand::random())).await);
        log_network_status(&log_dir, &format!("Connected to {} - starting pings", target_ssid)).await?;
        log_debug(&log_dir, "Pinger initialized successfully").await?;
    }
    
    while running.load(Ordering::SeqCst) {
        let loop_start = Instant::now();
        log_debug(&log_dir, &format!("Main loop iteration - on_target_network={}, has_pinger={}", 
            on_target_network, pinger_option.is_some())).await?;
        
        // Check network every 30 seconds
        if last_network_check.elapsed() >= Duration::from_secs(30) {
            log_debug(&log_dir, "Performing network check...").await?;
            let current_network = get_current_wifi_network(&log_dir).await;
            on_target_network = match current_network {
                Ok(ref ssid) => {
                    let is_match = ssid == target_ssid;
                    log_debug(&log_dir, &format!("Network check - Current: '{}', Target: '{}', Match: {}", 
                        ssid, target_ssid, is_match)).await?;
                    is_match
                },
                Err(ref e) => {
                    log_debug(&log_dir, &format!("Network check failed: {}", e)).await?;
                    false
                },
            };
            
            if on_target_network && pinger_option.is_none() {
                pinger_option = Some(client.pinger(google_ip, PingIdentifier(rand::random())).await);
                log_network_status(&log_dir, &format!("Connected to {} - starting pings", target_ssid)).await?;
                log_debug(&log_dir, "Pinger initialized successfully").await?;
            } else if !on_target_network && pinger_option.is_some() {
                pinger_option = None;
                log_network_status(&log_dir, &format!("Disconnected from {} - stopping pings", target_ssid)).await?;
                log_debug(&log_dir, "Pinger stopped").await?;
            }
            
            last_network_check = Instant::now();
        }
        
        // Ping every second when on target network
        if on_target_network && pinger_option.is_some() {
            log_debug(&log_dir, &format!("Attempting ping #{}", sequence)).await?;
            let start_time = Instant::now();
            
            // Add timeout to prevent hanging
            let ping_result = tokio::time::timeout(
                Duration::from_secs(10), 
                ping_host(pinger_option.as_mut().unwrap(), sequence)
            ).await;
            
            let response_time = start_time.elapsed();
            
            let ping_result = match ping_result {
                Ok(result) => result,
                Err(_) => {
                    log_debug(&log_dir, &format!("Ping #{} timed out after 10 seconds", sequence)).await?;
                    Err("Ping timeout".into())
                }
            };
            
            match &ping_result {
                Ok(duration) => log_debug(&log_dir, &format!("Ping #{} successful: {:?}", sequence, duration)).await?,
                Err(e) => log_debug(&log_dir, &format!("Ping #{} failed: {}", sequence, e)).await?,
            }
            
            log_ping_result(&log_dir, &ping_result, response_time).await?;
            sequence = sequence.wrapping_add(1);
            
            // Sleep for 1 second minus ping time
            let ping_duration = loop_start.elapsed();
            if ping_duration < Duration::from_secs(1) {
                log_debug(&log_dir, &format!("Sleeping for {}ms", (Duration::from_secs(1) - ping_duration).as_millis())).await?;
                sleep(Duration::from_secs(1) - ping_duration).await;
            }
        } else {
            // Sleep longer when not pinging
            log_debug(&log_dir, "Not on target network or no pinger - sleeping 5 seconds").await?;
            sleep(Duration::from_secs(5)).await;
        }
        
        // Status log every 5 minutes
        if last_status_log.elapsed() >= Duration::from_secs(300) {
            log_debug(&log_dir, &format!("Status: on_target_network={}, sequence={}, running={}", 
                on_target_network, sequence, running.load(Ordering::SeqCst))).await?;
            last_status_log = Instant::now();
        }
        
        // Reason log every 60 seconds when not pinging
        if !on_target_network && last_reason_log.elapsed() >= Duration::from_secs(60) {
            log_debug(&log_dir, "Not pinging: Not connected to target network").await?;
            last_reason_log = Instant::now();
        }
        
        // Cleanup every hour
        if last_cleanup.elapsed() >= Duration::from_secs(3600) {
            cleanup_old_logs(&log_dir).await?;
            last_cleanup = Instant::now();
        }
    }
    
    cleanup_pid_file(&log_dir.join("ping-utility.pid")).await?;
    Ok(())
}

async fn stop_daemon(pid_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !pid_file.exists() {
        eprintln!("PID file not found. Daemon may not be running.");
        return Ok(());
    }
    
    let pid_content = std::fs::read_to_string(pid_file)?;
    let pid: u32 = pid_content.trim().parse()?;
    
    unsafe {
        if libc::kill(pid as i32, SIGTERM) == 0 {
            println!("Sent SIGTERM to daemon process {}", pid);
            std::fs::remove_file(pid_file)?;
        } else {
            eprintln!("Failed to send signal to process {}", pid);
        }
    }
    
    Ok(())
}

async fn cleanup_pid_file(pid_file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if pid_file.exists() {
        std::fs::remove_file(pid_file)?;
    }
    Ok(())
}

async fn get_current_wifi_network(log_dir: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    log_debug(log_dir, "Checking current WiFi network...").await?;
    
    let output = Command::new("system_profiler")
        .arg("SPAirPortDataType")
        .arg("-detailLevel")
        .arg("basic")
        .output()?;
    
    let output_str = String::from_utf8(output.stdout)?;
    
    // Log raw output for debugging
    if output_str.is_empty() {
        log_debug(log_dir, "system_profiler returned empty output").await?;
    } else {
        log_debug(log_dir, &format!("system_profiler output length: {} bytes", output_str.len())).await?;
    }
    
    let re = Regex::new(r"Current Network Information:\s*\n\s*(.+?):")?;
    if let Some(captures) = re.captures(&output_str) {
        if let Some(ssid) = captures.get(1) {
            let network_name = ssid.as_str().trim().to_string();
            log_debug(log_dir, &format!("Detected WiFi network: '{}'", network_name)).await?;
            return Ok(network_name);
        }
    }
    
    log_debug(log_dir, "No SSID found in system_profiler output").await?;
    Err("No SSID found".into())
}

async fn ping_host(pinger: &mut surge_ping::Pinger, sequence: u16) -> Result<Duration, Box<dyn std::error::Error>> {
    let payload = [0; 56];
    
    // Can't log here since we don't have log_dir, but we can use eprintln for debugging
    if std::env::var("DAEMONIZED").is_err() {
        println!("ping_host: Starting ping sequence {}", sequence);
    }
    
    match pinger.ping(PingSequence(sequence), &payload).await {
        Ok((_, duration)) => {
            if std::env::var("DAEMONIZED").is_err() {
                println!("ping_host: Ping {} completed successfully in {:?}", sequence, duration);
            }
            Ok(duration)
        },
        Err(e) => {
            if std::env::var("DAEMONIZED").is_err() {
                println!("ping_host: Ping {} failed: {}", sequence, e);
            }
            Err(e.into())
        },
    }
}

fn get_log_directory() -> Result<PathBuf, std::io::Error> {
    let home = home_dir().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "Home directory not found")
    })?;
    Ok(home.join(".ping-utility"))
}

async fn log_ping_result(
    log_dir: &PathBuf,
    ping_result: &Result<Duration, Box<dyn std::error::Error>>,
    _total_time: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();
    let log_file = log_dir.join(format!("ping-{}.log", date_str));

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    let timestamp = now.format("%Y-%m-%d %H:%M:%S").to_string();
    let (status, ping_time) = match ping_result {
        Ok(duration) => ("SUCCESS", duration.as_millis()),
        Err(_) => ("FAILURE", 0),
    };
    
    writeln!(
        file,
        "{} | {} | {}ms",
        timestamp,
        status,
        ping_time
    )?;

    Ok(())
}

async fn log_network_status(
    log_dir: &PathBuf,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();
    let log_file = log_dir.join(format!("ping-{}.log", date_str));

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)?;

    let timestamp = now.format("%Y-%m-%d %H:%M:%S").to_string();
    
    writeln!(file, "{} | NETWORK | {}", timestamp, message)?;

    Ok(())
}

async fn log_debug(
    log_dir: &PathBuf,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let now = Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();
    let debug_log_file = log_dir.join(format!("ping-debug-{}.log", date_str));

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&debug_log_file)?;

    let timestamp = now.format("%Y-%m-%d %H:%M:%S.%3f").to_string();
    
    writeln!(file, "{} | DEBUG | {}", timestamp, message)?;
    
    // Only print to console when we have one (not daemonized)
    if std::env::var("DAEMONIZED").is_err() {
        println!("{} | DEBUG | {}", timestamp, message);
    }

    Ok(())
}

async fn cleanup_old_logs(log_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let entries = std::fs::read_dir(log_dir)?;
    let thirty_days_ago = Utc::now() - chrono::Duration::days(30);

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if let Some(file_name) = path.file_name() {
            if let Some(file_name_str) = file_name.to_str() {
                if file_name_str.starts_with("ping-") && file_name_str.ends_with(".log") {
                    let date_part = &file_name_str[5..15]; // Extract YYYY-MM-DD
                    if let Ok(file_date) = DateTime::parse_from_str(
                        &format!("{} 00:00:00 +0000", date_part),
                        "%Y-%m-%d %H:%M:%S %z",
                    ) {
                        if file_date.with_timezone(&Utc) < thirty_days_ago {
                            std::fs::remove_file(&path)?;
                            println!("Deleted old log file: {:?}", path);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
