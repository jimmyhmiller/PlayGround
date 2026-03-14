use std::os::unix::process::CommandExt;
use std::process::Command;
use std::thread;
use std::time::Duration;

const SIGINT: i32 = 2;

unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
    fn setsid() -> i32;
}

#[test]
fn double_dash_runs_the_following_command() {
    let output = Command::new(env!("CARGO_BIN_EXE_vtime"))
        .args(["--", "sh", "-c", "exit 0"])
        .output()
        .expect("vtime should run");

    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn interrupted_run_still_prints_summary() {
    let child = unsafe {
        Command::new(env!("CARGO_BIN_EXE_vtime"))
            .args(["sleep", "10"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .pre_exec(|| {
                if setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            })
            .spawn()
            .expect("vtime should start")
    };

    thread::sleep(Duration::from_secs(1));

    let rc = unsafe { kill(-(child.id() as i32), SIGINT) };
    assert_eq!(rc, 0, "failed to signal process group");

    let output = child.wait_with_output().expect("vtime should exit");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(output.status.code(), Some(130), "stderr: {stderr}");
    assert!(stderr.contains("Wall clock time"), "stderr: {stderr}");
    assert!(stderr.contains("Killed by signal"), "stderr: {stderr}");
}
