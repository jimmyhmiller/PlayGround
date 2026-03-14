use std::env;
use std::ffi::{c_int, c_long};
use std::os::unix::process::CommandExt;
use std::process::{self, Command};
use std::time::Instant;

#[repr(C)]
struct Timeval {
    tv_sec: c_long,
    tv_usec: Suseconds,
}

#[repr(C)]
struct Rusage {
    ru_utime: Timeval,
    ru_stime: Timeval,
    ru_maxrss: c_long,
    ru_ixrss: c_long,
    ru_idrss: c_long,
    ru_isrss: c_long,
    ru_minflt: c_long,
    ru_majflt: c_long,
    ru_nswap: c_long,
    ru_inblock: c_long,
    ru_oublock: c_long,
    ru_msgsnd: c_long,
    ru_msgrcv: c_long,
    ru_nsignals: c_long,
    ru_nvcsw: c_long,
    ru_nivcsw: c_long,
}

impl Rusage {
    fn zeroed() -> Self {
        Rusage {
            ru_utime: Timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_stime: Timeval {
                tv_sec: 0,
                tv_usec: 0,
            },
            ru_maxrss: 0,
            ru_ixrss: 0,
            ru_idrss: 0,
            ru_isrss: 0,
            ru_minflt: 0,
            ru_majflt: 0,
            ru_nswap: 0,
            ru_inblock: 0,
            ru_oublock: 0,
            ru_msgsnd: 0,
            ru_msgrcv: 0,
            ru_nsignals: 0,
            ru_nvcsw: 0,
            ru_nivcsw: 0,
        }
    }
}

unsafe extern "C" {
    safe fn wait4(pid: c_int, status: *mut c_int, options: c_int, rusage: *mut Rusage) -> c_int;
    safe fn fork() -> c_int;
    unsafe fn sigaddset(set: *mut Sigset, signo: i32) -> i32;
    unsafe fn sigemptyset(set: *mut Sigset) -> i32;
    unsafe fn sigprocmask(how: i32, set: *const Sigset, oldset: *mut Sigset) -> i32;
}

#[cfg(target_os = "macos")]
type Sigset = u32;
#[cfg(target_os = "macos")]
type Suseconds = i32;

#[cfg(target_os = "macos")]
const SIGINT: i32 = 2;
#[cfg(target_os = "macos")]
const SIG_BLOCK: i32 = 1;
#[cfg(target_os = "macos")]
const SIG_SETMASK: i32 = 3;

#[cfg(target_os = "linux")]
#[repr(C)]
struct Sigset {
    __val: [usize; 16],
}

#[cfg(target_os = "linux")]
type Suseconds = c_long;

#[cfg(target_os = "linux")]
const SIGINT: i32 = 2;
#[cfg(target_os = "linux")]
const SIG_BLOCK: i32 = 0;
#[cfg(target_os = "linux")]
const SIG_SETMASK: i32 = 2;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("vtime currently supports only macOS and Linux");

fn timeval_to_secs(tv: &Timeval) -> f64 {
    tv.tv_sec as f64 + tv.tv_usec as f64 / 1_000_000.0
}

fn format_duration(secs: f64) -> String {
    if secs < 0.001 {
        format!("{:.0} µs", secs * 1_000_000.0)
    } else if secs < 1.0 {
        format!("{:.1} ms", secs * 1_000.0)
    } else if secs < 60.0 {
        format!("{:.3} s", secs)
    } else {
        let mins = (secs / 60.0) as u64;
        let remaining = secs - (mins as f64 * 60.0);
        format!("{}m {:.3}s", mins, remaining)
    }
}

fn format_bytes(bytes: i64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

struct Flags {
    memory: bool,
    io: bool,
    cpu: bool,
    all: bool,
}

fn parse_args(args: &[String]) -> Result<(Flags, usize), i32> {
    let mut flags = Flags {
        memory: false,
        io: false,
        cpu: false,
        all: false,
    };
    let mut cmd_start = 1;

    while cmd_start < args.len() {
        match args[cmd_start].as_str() {
            "--" => {
                cmd_start += 1;
                break;
            }
            "-a" | "--all" => {
                flags.all = true;
                cmd_start += 1;
            }
            "-c" | "--cpu" => {
                flags.cpu = true;
                cmd_start += 1;
            }
            "-m" | "--memory" => {
                flags.memory = true;
                cmd_start += 1;
            }
            "-d" | "--disk" => {
                flags.io = true;
                cmd_start += 1;
            }
            "-h" | "--help" => {
                print_usage();
                return Err(0);
            }
            s if s.starts_with('-') && !s.starts_with("--") && s.len() > 2 => {
                for c in s[1..].chars() {
                    match c {
                        'a' => flags.all = true,
                        'c' => flags.cpu = true,
                        'm' => flags.memory = true,
                        'd' => flags.io = true,
                        'h' => {
                            print_usage();
                            return Err(0);
                        }
                        _ => {
                            eprintln!("vtime: unknown flag '-{}'", c);
                            print_usage();
                            return Err(1);
                        }
                    }
                }
                cmd_start += 1;
            }
            _ => break,
        }
    }

    if flags.all {
        flags.memory = true;
        flags.io = true;
        flags.cpu = true;
    }

    if cmd_start >= args.len() {
        print_usage();
        return Err(1);
    }

    Ok((flags, cmd_start))
}

fn print_usage() {
    eprintln!("Usage: vtime [flags] <command> [args...]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  -a, --all       Show all details");
    eprintln!("  -c, --cpu       Show CPU utilization and context switches");
    eprintln!("  -m, --memory    Show memory usage and page faults");
    eprintln!("  -d, --disk      Show disk I/O");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (flags, cmd_start) = match parse_args(&args) {
        Ok(parsed) => parsed,
        Err(code) => process::exit(code),
    };

    let wall_start = Instant::now();

    {
        let mut sigint_mask = std::mem::MaybeUninit::<Sigset>::uninit();
        let mut old_mask = std::mem::MaybeUninit::<Sigset>::uninit();
        unsafe {
            sigemptyset(sigint_mask.as_mut_ptr());
            sigaddset(sigint_mask.as_mut_ptr(), SIGINT);
        }
        if unsafe { sigprocmask(SIG_BLOCK, sigint_mask.as_ptr(), old_mask.as_mut_ptr()) } != 0 {
            eprintln!("vtime: failed to block SIGINT");
            process::exit(1);
        }
        let old_mask = unsafe { old_mask.assume_init() };

        let pid = fork();
        if pid < 0 {
            eprintln!("vtime: fork failed");
            process::exit(1);
        }
        if pid == 0 {
            let _ = unsafe { sigprocmask(SIG_SETMASK, &old_mask, std::ptr::null_mut()) };
            let err = Command::new(&args[cmd_start])
                .args(&args[cmd_start + 1..])
                .exec();
            eprintln!("vtime: {}: {}", args[cmd_start], err);
            process::exit(127);
        }

        let mut status: i32 = 0;
        let mut rusage = Rusage::zeroed();
        let ret = wait4(pid, &mut status, 0, &mut rusage);
        let wall_elapsed = wall_start.elapsed().as_secs_f64();

        if ret < 0 {
            eprintln!("vtime: wait4 failed");
            process::exit(1);
        }

        let user_secs = timeval_to_secs(&rusage.ru_utime);
        let sys_secs = timeval_to_secs(&rusage.ru_stime);

        eprintln!();
        eprintln!("  Wall clock time ·····  {}", format_duration(wall_elapsed));
        eprintln!("  User CPU time ·······  {}", format_duration(user_secs));
        eprintln!("  System CPU time ·····  {}", format_duration(sys_secs));

        if flags.cpu {
            let cpu_total = user_secs + sys_secs;
            let cpu_pct = if wall_elapsed > 0.0 {
                (cpu_total / wall_elapsed) * 100.0
            } else {
                0.0
            };
            eprintln!("  CPU utilization ·····  {:.0}%", cpu_pct);
            eprintln!("  Voluntary ctx sw ····  {}", rusage.ru_nvcsw);
            eprintln!("  Involuntary ctx sw ··  {}", rusage.ru_nivcsw);
        }

        if flags.memory {
            let max_rss = if cfg!(target_os = "macos") {
                rusage.ru_maxrss as i64
            } else {
                (rusage.ru_maxrss as i64) * 1024
            };
            eprintln!("  Max memory (RSS) ····  {}", format_bytes(max_rss));
            eprintln!("  Page faults (major) ·  {}", rusage.ru_majflt);
            eprintln!("  Page faults (minor) ·  {}", rusage.ru_minflt);
        }

        if flags.io {
            eprintln!("  Disk reads ··········  {}", rusage.ru_inblock);
            eprintln!("  Disk writes ·········  {}", rusage.ru_oublock);
        }

        if status & 0x7f == 0 {
            process::exit((status >> 8) & 0xff);
        } else {
            let sig = status & 0x7f;
            eprintln!("  Killed by signal ····  {}", sig);
            process::exit(128 + sig);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_args;

    fn to_args(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn double_dash_stops_option_parsing() {
        let args = to_args(&["vtime", "--", "sleep", "1"]);
        let (flags, cmd_start) = parse_args(&args).expect("parse should succeed");

        assert!(!flags.all);
        assert_eq!(cmd_start, 2);
        assert_eq!(args[cmd_start], "sleep");
    }

    #[test]
    fn command_after_double_dash_can_start_with_dash() {
        let args = to_args(&["vtime", "--", "-cmd", "arg"]);
        let (_, cmd_start) = parse_args(&args).expect("parse should succeed");

        assert_eq!(cmd_start, 2);
        assert_eq!(args[cmd_start], "-cmd");
    }
}
