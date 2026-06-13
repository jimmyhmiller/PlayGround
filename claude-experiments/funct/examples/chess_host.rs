//! chess_host — the ported chess widget (examples/widgets/chess.ft) driven by
//! a *fake* version of the jim-editor widget host, end to end.
//!
//! This is the same host surface a real widget worker provides — globals
//! (canvas_w/h), render/animation control, and a subprocess bridge with REAL
//! pipes — registered through funct's embedding API. We then drive it exactly
//! like the GUI would: click the "bot" button, click squares to move, and pump
//! `on_frame` so the widget can read Stockfish's reply off the pipe.
//!
//! Run: `cargo run --example chess_host`   (install `stockfish` for the bot)

use funct::{Fault, Funct, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Host: real subprocess pipes, so proc_spawn actually launches Stockfish.
// ---------------------------------------------------------------------------

struct Proc {
    child: Child,
    stdin: ChildStdin,
    lines: Arc<Mutex<VecDeque<String>>>,
}

#[derive(Default)]
struct ProcRegistry {
    next: i64,
    procs: HashMap<i64, Proc>,
}

impl ProcRegistry {
    fn spawn(&mut self, cmd: &str) -> i64 {
        let spawned = Command::new(cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match spawned {
            Ok(c) => c,
            Err(_) => return -1, // widget protocol: -1 = could not spawn
        };
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let lines: Arc<Mutex<VecDeque<String>>> = Arc::default();
        let sink = lines.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                sink.lock().unwrap().push_back(line);
            }
        });
        self.next += 1;
        let id = self.next;
        self.procs.insert(
            id,
            Proc {
                child,
                stdin,
                lines,
            },
        );
        id
    }
}

impl Drop for ProcRegistry {
    fn drop(&mut self) {
        for (_, p) in self.procs.iter_mut() {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
    }
}

struct Host {
    render_requested: Arc<AtomicBool>,
    animating: Arc<AtomicBool>,
}

fn install_host(vm: &mut Funct) -> Host {
    let reg: Arc<Mutex<ProcRegistry>> = Arc::default();
    let render_requested = Arc::new(AtomicBool::new(false));
    let animating = Arc::new(AtomicBool::new(false));

    let handle = |v: &Value| -> Result<i64, Fault> {
        match v {
            Value::Int(i) => Ok(*i),
            o => Err(Fault::new(format!(
                "handle must be Int, got {}",
                o.type_name()
            ))),
        }
    };

    // ---- subprocess bridge (the bit the chess widget really needs) ----
    let r = reg.clone();
    vm.register1("proc_spawn", move |cmd: String| {
        r.lock().unwrap().spawn(&cmd)
    });
    let r = reg.clone();
    vm.register_raw("proc_write", move |_vm, args| {
        let id = handle(&args[0])?;
        let line = String::from_value_str(&args[1])?;
        let mut g = r.lock().unwrap();
        let ok = match g.procs.get_mut(&id) {
            Some(p) => writeln!(p.stdin, "{line}")
                .and_then(|_| p.stdin.flush())
                .is_ok(),
            None => false,
        };
        Ok(Value::Bool(ok))
    });
    let r = reg.clone();
    vm.register_raw("proc_read", move |_vm, args| {
        let id = handle(&args[0])?;
        let g = r.lock().unwrap();
        let line = g
            .procs
            .get(&id)
            .and_then(|p| p.lines.lock().unwrap().pop_front())
            .unwrap_or_default();
        Ok(Value::str(line))
    });
    let r = reg.clone();
    vm.register_raw("proc_alive", move |_vm, args| {
        let id = handle(&args[0])?;
        let mut g = r.lock().unwrap();
        let alive = matches!(
            g.procs.get_mut(&id).map(|p| p.child.try_wait()),
            Some(Ok(None))
        );
        Ok(Value::Bool(alive))
    });
    let r = reg.clone();
    vm.register_raw("proc_kill", move |_vm, args| {
        let id = handle(&args[0])?;
        if let Some(mut p) = r.lock().unwrap().procs.remove(&id) {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
        Ok(Value::Unit)
    });

    // ---- render / animation control + misc surface ----
    let rr = render_requested.clone();
    vm.register0("request_render", move || rr.store(true, Ordering::SeqCst));
    let an = animating.clone();
    vm.register1("set_animating", move |on: bool| {
        an.store(on, Ordering::SeqCst)
    });
    vm.register1("host_env", |name: String| {
        std::env::var(name).unwrap_or_default()
    });
    vm.register1("widget_asset", |rel: String| format!("assets/{rel}"));
    vm.register1("clipboard_set", |_text: String| true);

    Host {
        render_requested,
        animating,
    }
}

// tiny helper so register_raw closures can pull a String out of a Value
trait FromValueStr {
    fn from_value_str(v: &Value) -> Result<String, Fault>;
}
impl FromValueStr for String {
    fn from_value_str(v: &Value) -> Result<String, Fault> {
        match v {
            Value::Str(s) => Ok(s.to_string()),
            o => Err(Fault::new(format!("expected Str, got {}", o.type_name()))),
        }
    }
}

// ---------------------------------------------------------------------------
// Driving the widget like a player would.
// ---------------------------------------------------------------------------

/// Algebraic square ("e2") -> board index (0 = a8, 63 = h1).
fn sq(name: &str) -> i64 {
    let b = name.as_bytes();
    let file = (b[0] - b'a') as i64;
    let rank = (b[1] - b'0') as i64;
    (8 - rank) * 8 + file
}

fn call(vm: &mut Funct, f: &str, args: Vec<Value>) -> Value {
    vm.call(f, args).unwrap_or_else(|e| panic!("{f}: {e}"))
}

fn xy(vm: &mut Funct, f: &str, arg: Value) -> (f64, f64) {
    match call(vm, f, vec![arg]) {
        Value::Tuple(t) => {
            let num = |v: &Value| match v {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                o => panic!("expected number, got {o:?}"),
            };
            (num(&t[0]), num(&t[1]))
        }
        o => panic!("expected (x,y), got {o:?}"),
    }
}

fn click_xy(vm: &mut Funct, x: f64, y: f64) {
    call(
        vm,
        "on_click",
        vec![
            Value::Float(x),
            Value::Float(y),
            Value::Bool(false),
            Value::Bool(false),
            Value::str(""),
        ],
    );
}

fn click_sq(vm: &mut Funct, idx: i64) {
    let (x, y) = xy(vm, "sq_click_xy", Value::Int(idx));
    click_xy(vm, x, y);
}

fn click_btn(vm: &mut Funct, name: &str) {
    let (x, y) = xy(vm, "btn_xy", Value::str(name));
    click_xy(vm, x, y);
}

fn moves_played(vm: &mut Funct) -> i64 {
    call(vm, "game_state", vec![]).to_json().unwrap()["moves"]
        .as_i64()
        .unwrap_or(-1)
}

fn last_san(vm: &mut Funct) -> String {
    match call(vm, "history_san", vec![]) {
        Value::List(items) => items
            .last()
            .map(|v| format!("{v}").trim_matches('"').to_string())
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Pump the animation tick until `pred` holds or we time out.
fn pump_until(vm: &mut Funct, secs: f64, mut pred: impl FnMut(&mut Funct) -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs_f64(secs);
    while Instant::now() < deadline {
        call(vm, "on_frame", vec![Value::Float(0.016)]);
        if pred(vm) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

fn print_board(vm: &mut Funct) {
    println!("    +-----------------+");
    for row in 0..8 {
        print!("  {} |", 8 - row);
        for col in 0..8 {
            let piece = match call(vm, "board_at", vec![Value::Int(row * 8 + col)]) {
                Value::Str(s) => s.to_string(),
                _ => String::new(),
            };
            let glyph = if piece.is_empty() {
                '.'
            } else {
                let t = piece.as_bytes()[1] as char;
                if piece.starts_with('w') {
                    t.to_ascii_uppercase()
                } else {
                    t.to_ascii_lowercase()
                }
            };
            print!(" {glyph}");
        }
        println!(" |");
    }
    println!("    +-----------------+");
    println!("      a b c d e f g h");
}

fn main() {
    let mut vm = Funct::new();
    // So `import "host"` inside chess.ft resolves to examples/widgets/host.ft.
    vm.set_module_root(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/widgets"));
    let host = install_host(&mut vm);
    vm.set_global("canvas_w", Value::Float(800.0));
    vm.set_global("canvas_h", Value::Float(640.0));

    println!("== load chess.ft + on_init ==");
    vm.eval(include_str!("widgets/chess.ft"))
        .expect("chess.ft compiles + runs");
    call(&mut vm, "on_init", vec![]);
    println!(
        "  host: render_requested = {}",
        host.render_requested.swap(false, Ordering::SeqCst)
    );

    // Turn the bot on via the real toolbar button — it spawns Stockfish over
    // the host's proc pipe and takes Black.
    println!("\n== enable bot (clicks the 'bot' button) ==");
    click_btn(&mut vm, "bot");
    let g = call(&mut vm, "game_state", vec![]).to_json().unwrap();
    if g["engine_ok"] == false {
        println!("  no UCI engine could be spawned — install `stockfish`. Playing human-only.");
    } else {
        println!(
            "  bot_side = {}, engine handle = {}",
            g["bot_side"], g["eng"]
        );
    }
    let bot = g["engine_ok"] != false;

    // A short opening for White; the bot answers each one.
    let opening = ["e2e4", "g1f3", "f1c4", "e1g1", "d2d3"];
    print_board(&mut vm);

    for (i, mv) in opening.iter().enumerate() {
        let (from, to) = (sq(&mv[0..2]), sq(&mv[2..4]));
        let before = moves_played(&mut vm);
        click_sq(&mut vm, from);
        click_sq(&mut vm, to);
        if moves_played(&mut vm) <= before {
            println!(
                "\n  {}. {} is not legal in this position — stopping.",
                i + 1,
                mv
            );
            break;
        }
        println!("\n== White {}. {} ({}) ==", i + 1, last_san(&mut vm), mv);

        if bot {
            let target = moves_played(&mut vm) + 1;
            if !host.animating.load(Ordering::SeqCst) {
                println!("  (game over)");
                print_board(&mut vm);
                break;
            }
            print!("  Stockfish thinking…");
            std::io::stdout().flush().ok();
            if pump_until(&mut vm, 15.0, |vm| moves_played(vm) >= target) {
                println!(" replied {}", last_san(&mut vm));
            } else {
                println!(" (no reply in time)");
                break;
            }
        }
        print_board(&mut vm);
        let g = call(&mut vm, "game_state", vec![]).to_json().unwrap();
        if g["over"] != false {
            println!("  game over: {}", g["status"]);
            break;
        }
    }

    let fen = match call(&mut vm, "current_fen", vec![]) {
        Value::Str(s) => s.to_string(),
        _ => String::new(),
    };
    let sans = match call(&mut vm, "history_san", vec![]) {
        Value::List(items) => items
            .iter()
            .map(|v| format!("{v}").trim_matches('"').to_string())
            .collect::<Vec<_>>(),
        _ => vec![],
    };
    println!("\n== final ==");
    println!("  moves: {}", sans.join(" "));
    println!("  fen:   {fen}");
}
