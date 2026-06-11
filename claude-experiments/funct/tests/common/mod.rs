//! Widget host harness: the same host surface the jim-editor widget worker
//! provides, with REAL subprocess pipes (std::process) — the UCI engine the
//! chess widget drives through proc_* is an actual child process.

use funct::{Fault, Funct, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const MAX_BUFFERED_LINES: usize = 4096;

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
    fn spawn(&mut self, cmd: &str, args: &[String]) -> i64 {
        let spawned = Command::new(cmd)
            .args(args)
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
                let mut q = sink.lock().unwrap();
                if q.len() >= MAX_BUFFERED_LINES {
                    q.pop_front();
                }
                q.push_back(line);
            }
        });
        self.next += 1;
        let id = self.next;
        self.procs.insert(id, Proc { child, stdin, lines });
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

pub struct WidgetHost {
    pub render_requested: Arc<AtomicBool>,
    pub animating: Arc<AtomicBool>,
    pub clipboard: Arc<Mutex<String>>,
}

impl WidgetHost {
    pub fn take_render_request(&self) -> bool {
        self.render_requested.swap(false, Ordering::SeqCst)
    }
}

/// Register the widget host surface into an engine.
pub fn install_widget_host(vm: &mut Funct) -> WidgetHost {
    // So `import "host"` inside chess.ft resolves to examples/widgets/host.ft.
    vm.set_module_root(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/widgets"));
    let registry: Arc<Mutex<ProcRegistry>> = Arc::default();
    let render_requested = Arc::new(AtomicBool::new(false));
    let animating = Arc::new(AtomicBool::new(false));
    let clipboard = Arc::new(Mutex::new(String::new()));

    let need = |name: &str, args: &[Value], n: usize| -> Result<(), Fault> {
        if args.len() != n {
            return Err(Fault::new(format!("{} expects {} argument(s), got {}", name, n, args.len())));
        }
        Ok(())
    };
    let handle_of = |name: &str, v: &Value| -> Result<i64, Fault> {
        match v {
            Value::Int(i) => Ok(*i),
            other => Err(Fault::new(format!("{}: handle must be Int, got {}", name, other.type_name()))),
        }
    };

    // ----- subprocess bridge (real pipes) -----
    let reg = registry.clone();
    vm.register_raw("proc_spawn", move |_vm, args| {
        let (cmd, extra): (String, Vec<String>) = match args.as_slice() {
            [Value::Str(c)] => (c.to_string(), vec![]),
            [Value::Str(c), Value::List(items)] => {
                let mut a = Vec::new();
                for it in items.iter() {
                    match it {
                        Value::Str(s) => a.push(s.to_string()),
                        other => {
                            return Err(Fault::new(format!(
                                "proc_spawn: args must be strings, got {}",
                                other.type_name()
                            )))
                        }
                    }
                }
                (c.to_string(), a)
            }
            _ => return Err(Fault::new("proc_spawn expects (cmd) or (cmd, [args])")),
        };
        Ok(Value::Int(reg.lock().unwrap().spawn(&cmd, &extra)))
    });
    let reg = registry.clone();
    vm.register_raw("proc_write", move |_vm, args| {
        need("proc_write", &args, 2)?;
        let id = handle_of("proc_write", &args[0])?;
        let line = match &args[1] {
            Value::Str(s) => s.to_string(),
            other => return Err(Fault::new(format!("proc_write: expected Str, got {}", other.type_name()))),
        };
        let mut r = reg.lock().unwrap();
        let ok = match r.procs.get_mut(&id) {
            Some(p) => writeln!(p.stdin, "{}", line).and_then(|_| p.stdin.flush()).is_ok(),
            None => false,
        };
        Ok(Value::Bool(ok))
    });
    let reg = registry.clone();
    vm.register_raw("proc_read", move |_vm, args| {
        need("proc_read", &args, 1)?;
        let id = handle_of("proc_read", &args[0])?;
        let r = reg.lock().unwrap();
        let line = r
            .procs
            .get(&id)
            .and_then(|p| p.lines.lock().unwrap().pop_front())
            .unwrap_or_default();
        Ok(Value::str(line))
    });
    let reg = registry.clone();
    vm.register_raw("proc_alive", move |_vm, args| {
        need("proc_alive", &args, 1)?;
        let id = handle_of("proc_alive", &args[0])?;
        let mut r = reg.lock().unwrap();
        let alive = match r.procs.get_mut(&id) {
            Some(p) => matches!(p.child.try_wait(), Ok(None)),
            None => false,
        };
        Ok(Value::Bool(alive))
    });
    let reg = registry.clone();
    vm.register_raw("proc_kill", move |_vm, args| {
        need("proc_kill", &args, 1)?;
        let id = handle_of("proc_kill", &args[0])?;
        let mut r = reg.lock().unwrap();
        if let Some(mut p) = r.procs.remove(&id) {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
        Ok(Value::Unit)
    });

    // ----- render / animation control -----
    let rr = render_requested.clone();
    vm.register_raw("request_render", move |_vm, _args| {
        rr.store(true, Ordering::SeqCst);
        Ok(Value::Unit)
    });
    let an = animating.clone();
    vm.register_raw("set_animating", move |_vm, args| {
        need("set_animating", &args, 1)?;
        match &args[0] {
            Value::Bool(b) => {
                an.store(*b, Ordering::SeqCst);
                Ok(Value::Unit)
            }
            other => Err(Fault::new(format!("set_animating: expected Bool, got {}", other.type_name()))),
        }
    });

    // ----- misc host surface -----
    vm.register0("time", || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    });
    vm.register1("host_env", |name: String| std::env::var(name).unwrap_or_default());
    vm.register1("widget_asset", |rel: String| format!("assets/{}", rel));
    vm.register_raw("host_log", |_vm, args| {
        let parts: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
        eprintln!("[widget] {}", parts.join(" "));
        Ok(Value::Unit)
    });
    let cb = clipboard.clone();
    vm.register_raw("clipboard_set", move |_vm, args| {
        need("clipboard_set", &args, 1)?;
        match &args[0] {
            Value::Str(s) => {
                *cb.lock().unwrap() = s.to_string();
                Ok(Value::Bool(true))
            }
            other => Err(Fault::new(format!("clipboard_set: expected Str, got {}", other.type_name()))),
        }
    });

    // The rest of the shared host.ft surface. This widget doesn't draw through
    // it, but the real app has ONE host registering the WHOLE interface for all
    // widgets — so we mirror that here. It also keeps snapshots portable: a
    // saved game references every native the imported host.ft binds, so any
    // worker restoring it must register the same surface (these four included).
    vm.register2("uniform_set", |_name: String, _value: Value| {});
    vm.register5("mask_paint", |_name: String, _x: f64, _y: f64, _r: f64, _v: f64| {});
    vm.register3("oklch", |l: f64, c: f64, h: f64| format!("oklch({l:.3} {c:.3} {h:.1})"));
    vm.register2("emit", |_kind: String, _payload: Value| {});

    WidgetHost { render_requested, animating, clipboard }
}
