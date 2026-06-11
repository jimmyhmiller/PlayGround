//! host-switch — toggle an /etc/hosts entry based on whether we're on the home network.
//!
//! "Home" is identified by the default gateway's MAC address (the router's hardware
//! address). This is stable, survives reboots, and — unlike the Wi-Fi SSID — requires
//! NO Location Services permission, which matters on macOS 14+/26 where the SSID reads
//! back as `<redacted>` for non-privileged callers.
//!
//! When home: the managed block is present in /etc/hosts.
//! When away (or offline): the managed block is removed.
//!
//! Zero dependencies. Just std, shelling out to `route` and `arp` (both in the base OS).

use std::fs;
use std::io::Write;
use std::process::Command;

// ─── Configuration ──────────────────────────────────────────────────────────
// The MAC address of your home router (default gateway). Captured while on the
// home network. Compared octet-by-octet, so formatting/zero-padding don't matter.
const HOME_GATEWAY_MAC: &str = "bc:07:1d:75:e2:1c";

// The line(s) to install in /etc/hosts when home.
const MANAGED_LINES: &[&str] = &["192.168.0.55\tcomputer.jimmyhmiller.com"];

const DEFAULT_HOSTS_PATH: &str = "/etc/hosts";

fn hosts_path() -> String {
    // HOST_SWITCH_HOSTS overrides the target file (used for testing).
    std::env::var("HOST_SWITCH_HOSTS").unwrap_or_else(|_| DEFAULT_HOSTS_PATH.to_string())
}
const BEGIN_MARKER: &str = "# >>> host-switch (managed) >>>";
const END_MARKER: &str = "# <<< host-switch (managed) <<<";
// ────────────────────────────────────────────────────────────────────────────

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();

    let gw_mac = default_gateway_mac();
    let detected_home = match &gw_mac {
        Some(mac) => mac_eq(mac, HOME_GATEWAY_MAC),
        None => false, // no gateway → offline / not home
    };

    // `home`/`away` force the decision (used by uninstall); otherwise auto-detect.
    let at_home = match mode.as_str() {
        "home" => true,
        "away" => false,
        _ => detected_home,
    };

    if mode == "status" {
        println!(
            "gateway MAC : {}",
            gw_mac.as_deref().unwrap_or("(none / offline)")
        );
        println!("home MAC    : {}", HOME_GATEWAY_MAC);
        println!("decision    : {}", if at_home { "HOME — block present" } else { "AWAY — block absent" });
        return;
    }

    match reconcile(at_home) {
        Ok(true) => {
            flush_dns();
            log(&format!(
                "applied: {} (gateway {})",
                if at_home { "added block" } else { "removed block" },
                gw_mac.as_deref().unwrap_or("none")
            ));
        }
        Ok(false) => { /* already in desired state — stay quiet */ }
        Err(e) => {
            log(&format!("ERROR: {e}"));
            std::process::exit(1);
        }
    }
}

/// Returns the MAC address of the default gateway, if reachable.
fn default_gateway_mac() -> Option<String> {
    // 1. Find the default gateway IP.
    let route_out = Command::new("route").args(["-n", "get", "default"]).output().ok()?;
    let route_txt = String::from_utf8_lossy(&route_out.stdout);
    let gw_ip = route_txt
        .lines()
        .find_map(|l| l.trim().strip_prefix("gateway:"))
        .map(|s| s.trim().to_string())?;

    // 2. Resolve the gateway IP to a MAC via the ARP table.
    //    Output looks like: `? (192.168.0.1) at bc:7:1d:75:e2:1c on en0 ifscope [ethernet]`
    let arp_out = Command::new("arp").args(["-n", &gw_ip]).output().ok()?;
    let arp_txt = String::from_utf8_lossy(&arp_out.stdout);
    let line = arp_txt.lines().next()?;
    let after_at = line.split(" at ").nth(1)?;
    let mac = after_at.split_whitespace().next()?;
    if mac == "(incomplete)" {
        return None;
    }
    Some(mac.to_string())
}

/// Case/zero-padding-insensitive MAC comparison (compares the six octet values).
fn mac_eq(a: &str, b: &str) -> bool {
    parse_mac(a).map_or(false, |pa| parse_mac(b).map_or(false, |pb| pa == pb))
}

fn parse_mac(s: &str) -> Option<[u8; 6]> {
    let mut out = [0u8; 6];
    let mut n = 0;
    for part in s.split(':') {
        if n >= 6 {
            return None;
        }
        out[n] = u8::from_str_radix(part.trim(), 16).ok()?;
        n += 1;
    }
    if n == 6 { Some(out) } else { None }
}

/// Make /etc/hosts match the desired state. Returns Ok(true) if a write happened.
fn reconcile(at_home: bool) -> std::io::Result<bool> {
    let original = fs::read_to_string(&hosts_path())?;
    let stripped = strip_block(&original);

    let desired = if at_home {
        let mut s = stripped.trim_end().to_string();
        s.push('\n');
        s.push_str(BEGIN_MARKER);
        s.push('\n');
        for line in MANAGED_LINES {
            s.push_str(line);
            s.push('\n');
        }
        s.push_str(END_MARKER);
        s.push('\n');
        s
    } else {
        // Keep a trailing newline like the original.
        let mut s = stripped.trim_end().to_string();
        s.push('\n');
        s
    };

    if desired == original {
        return Ok(false);
    }

    write_atomic(&hosts_path(), &desired)?;
    Ok(true)
}

/// Remove the managed block (markers inclusive) from the hosts content.
fn strip_block(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut skipping = false;
    for line in content.lines() {
        let t = line.trim();
        if t == BEGIN_MARKER {
            skipping = true;
            continue;
        }
        if t == END_MARKER {
            skipping = false;
            continue;
        }
        if skipping {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Write atomically: temp file in the same dir, then rename over the target.
fn write_atomic(path: &str, contents: &str) -> std::io::Result<()> {
    let tmp = format!("{path}.host-switch.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}

fn flush_dns() {
    let _ = Command::new("dscacheutil").arg("-flushcache").status();
    let _ = Command::new("killall").args(["-HUP", "mDNSResponder"]).status();
}

fn log(msg: &str) {
    // Goes to the LaunchDaemon's StandardOut/Err log.
    println!("[host-switch] {msg}");
    let _ = std::io::stdout().flush();
}
