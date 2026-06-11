//! style_host — a *fake* version of the jim-editor style system, just big
//! enough to show what embedding funct in the real app feels like.
//!
//! It mirrors the surface in `style-bevy/src/script_bridge.rs`: the host owns
//! a shader-uniform table + a set of mask textures, registers a handful of
//! native functions the script draws through, hands the script a persistent
//! `state` atom, and drives it one frame at a time. Nothing here touches Bevy
//! — the "GPU" is a HashMap and the "textures" are paint logs we print at the
//! end — but the embedding API calls are exactly the ones the real host makes.
//!
//! Run: `cargo run --example style_host`

use funct::{Funct, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Everything the "GPU side" of the app would hold. In jim-editor these are
/// real shader uniforms and mask textures; here they're just records we can
/// print to prove the script drove them.
#[derive(Default)]
struct Surface {
    uniforms: BTreeMap<String, String>,
    paints: Vec<String>,
    events: Vec<String>,
}

fn main() {
    let mut vm = Funct::new();
    // So `import "host"` inside glow.ft resolves to examples/widgets/host.ft.
    vm.set_module_root(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/widgets"));

    // Shared host state, captured by the native closures below.
    let surface: Arc<Mutex<Surface>> = Arc::default();
    let animating = Arc::new(AtomicBool::new(false));

    // ----- the host surface the script draws through -----
    // These are the exact analogues of style-bevy's register_fn calls.

    let s = surface.clone();
    vm.register2("uniform_set", move |name: String, value: Value| {
        s.lock().unwrap().uniforms.insert(name, format!("{}", value));
    });

    // THE headline: a 5-argument native. Before bumping the arity cap this
    // needed register_raw + hand-rolled arg parsing; now it's one line.
    let s = surface.clone();
    vm.register5(
        "mask_paint",
        move |name: String, x: f64, y: f64, radius: f64, value: f64| {
            s.lock().unwrap().paints.push(format!(
                "{name}: brush @ ({x:.0},{y:.0}) r={radius:.0} v={value:.2}"
            ));
        },
    );

    // A pure helper the script uses for color — OkLCh -> hex-ish string.
    vm.register3("oklch", |l: f64, c: f64, h: f64| {
        format!("oklch({l:.2} {c:.2} {h:.0})")
    });

    let s = surface.clone();
    vm.register2("emit", move |kind: String, payload: Value| {
        s.lock().unwrap().events.push(format!("{kind} {payload}"));
    });

    vm.register1("host_log", |msg: String| println!("  [script] {msg}"));

    let an = animating.clone();
    vm.register1("set_animating", move |on: bool| an.store(on, Ordering::SeqCst));

    // ----- globals the host injects into the script's world -----
    vm.set_global("canvas_w", Value::Float(800.0));
    vm.set_global("canvas_h", Value::Float(480.0));

    // `state` is a host-owned atom: it survives a hot reload because the new
    // script body is handed the *same* atom on the next load.
    let state = vm.make_atom(Value::Unit);
    vm.set_global("state", state);

    // ----- load + run, exactly like the real loader -----
    let src = include_str!("widgets/glow.ft");
    println!("== load glow.ft ==");
    vm.eval(src).expect("glow.ft compiles + runs top-level");
    vm.call("on_init", vec![]).expect("on_init");
    println!(
        "  host: animating = {}\n",
        animating.load(Ordering::SeqCst)
    );

    // Drive a few frames at a fixed 60fps dt, like the style ticker would.
    println!("== run 5 frames ==");
    let dt = Value::Float(1.0 / 60.0);
    for _ in 0..5 {
        vm.call("tick", vec![dt.clone()]).expect("tick");
    }
    // Pull persistent state back out, the way the host inspects it.
    println!("  host: frame_count = {}", vm.call("frame_count", vec![]).unwrap());
    report(&surface, "after 5 frames");

    // ----- hot reload: same atom, fresh code -----
    // Re-eval a patched body. `state` is untouched, so `frames` keeps counting
    // — exactly the hot-reload behavior the dust/widget scripts rely on.
    println!("\n== hot reload (brighter pulse, bigger brush) ==");
    let patched = src
        .replace("48.0", "96.0")
        .replace("0.5 + 0.5 * sin", "0.6 + 0.4 * sin");
    vm.eval(&patched).expect("patched glow.ft re-evals");
    surface.lock().unwrap().paints.clear();
    for _ in 0..3 {
        vm.call("tick", vec![dt.clone()]).expect("tick after reload");
    }
    println!(
        "  host: frame_count = {} (kept counting across the reload)",
        vm.call("frame_count", vec![]).unwrap()
    );
    report(&surface, "after reload (state persisted)");
}

fn report(surface: &Arc<Mutex<Surface>>, when: &str) {
    let s = surface.lock().unwrap();
    println!("\n-- host surface {when} --");
    println!("  uniforms:");
    for (k, v) in &s.uniforms {
        println!("    {k} = {v}");
    }
    println!("  events: {:?}", s.events);
    println!("  mask paints ({}):", s.paints.len());
    for p in s.paints.iter().take(3) {
        println!("    {p}");
    }
    if s.paints.len() > 3 {
        println!("    … {} more", s.paints.len() - 3);
    }
}
