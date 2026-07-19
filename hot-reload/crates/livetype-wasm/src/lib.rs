//! The browser demo's host.
//!
//! This is the same shape as the native `livetype-ffi-gui` demo: declare a
//! foreign interface, bind native implementations for it, load a program whose
//! `letonce` globals hold the native resources, and then run it — editing the
//! code while it runs. Only the "native toolkit" differs: instead of a
//! windowing library it is a command buffer that JavaScript paints onto a
//! `<canvas>`.
//!
//! The engine here is the interpreted (cold) tier. The LLVM tier cannot come
//! along — inkwell does not target wasm — but it does not own any live-editing
//! semantics: pause, repair, resume, migration, and verification all live in
//! the core, and `tests/differential_fuzz.rs` holds the two tiers to identical
//! observable behaviour.

use livetype_core::{
    Actor, ActorStatus, Condition, DefId, FunctionState, Instruction, Session, Turn, Value,
    Version, World,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

const INTERFACE: &str = include_str!("../../../demo/interface.lt");
const SCENE: &str = include_str!("../../../demo/scene.lt");

/// The scripted edits, in the order the demo walks them.
const SCENARIOS: &[(&str, &str)] = &[
    ("Redefine a running function", include_str!("../../../demo/edit_1_radius.lt")),
    ("Evolve a struct under live data", include_str!("../../../demo/edit_2_migrate.lt")),
    ("Reject a breaking edit", include_str!("../../../demo/edit_3_rejected.lt")),
    ("Introduce an enum", include_str!("../../../demo/edit_4_enum.lt")),
    ("Break it: add a variant", include_str!("../../../demo/edit_5_break.lt")),
    ("Repair the root cause", include_str!("../../../demo/edit_6_repair.lt")),
];

/// How many instructions one frame may take before the host gives up. A frame
/// is bounded work between two `yield`s; this only fires if an edit introduces
/// an unbounded loop, in which case the demo says so rather than hanging the tab.
const FRAME_INSTRUCTION_CAP: usize = 2_000_000;

/// The "native toolkit": what the guest drew. `pending` accumulates the frame
/// being rendered; `committed` is the last frame that ran to completion. A
/// frozen program therefore leaves the last good frame on screen instead of a
/// half-drawn one — ordinary double buffering.
#[derive(Default)]
struct Toolkit {
    canvases_opened: u64,
    frames_cleared: u64,
    circles_drawn: u64,
    pending: Vec<i32>,
    committed: Vec<i32>,
}

/// Status of the last `step_frame`, mirrored into JS as a small integer.
const STATUS_RUNNING: u32 = 0;
const STATUS_FROZEN: u32 = 1;
const STATUS_DONE: u32 = 2;
const STATUS_CAPPED: u32 = 3;

#[wasm_bindgen]
pub struct Demo {
    session: Session,
    actor: Actor,
    toolkit: Arc<Mutex<Toolkit>>,
    frames: u64,
    last_condition: String,
}

#[wasm_bindgen]
impl Demo {
    /// Boot the demo's own scene.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Demo, JsError> {
        Demo::boot(SCENE)
    }

    /// Boot a fresh world from arbitrary scene source.
    ///
    /// This is how an edit to something only a `letonce` initializer calls —
    /// `seed`, here — can be observed at all. Installing a new `seed` into the
    /// *running* world is a no-op by design: the initializer already ran and
    /// will never run again, which is exactly what `letonce` is for. To see a
    /// different `seed`, the world has to start over with it already in place.
    pub fn restart_with(scene: &str) -> Result<Demo, JsError> {
        Demo::boot(scene)
    }

    /// Declare the foreign interface, bind the canvas implementations, then
    /// load the program. The order matters — a `letonce` initializer calls
    /// `open_canvas` while the scene is being evaluated.
    fn boot(scene: &str) -> Result<Demo, JsError> {
        let toolkit = Arc::new(Mutex::new(Toolkit::default()));
        let mut session = Session::new();
        session
            .eval(INTERFACE)
            .map_err(|e| JsError::new(&format!("interface: {e}")))?;

        let kind = session
            .foreign_kind("Canvas")
            .ok_or_else(|| JsError::new("the interface did not declare `Canvas`"))?;
        {
            let tk = Arc::clone(&toolkit);
            session
                .register_foreign(
                    "open_canvas",
                    Box::new(move |_| {
                        let mut t = tk.lock().unwrap();
                        t.canvases_opened += 1;
                        Value::Foreign { kind, ptr: t.canvases_opened }
                    }),
                )
                .map_err(|e| JsError::new(&e))?;
        }
        {
            let tk = Arc::clone(&toolkit);
            session
                .register_foreign(
                    "clear",
                    Box::new(move |_| {
                        let mut t = tk.lock().unwrap();
                        t.frames_cleared += 1;
                        t.pending.clear();
                        Value::Unit
                    }),
                )
                .map_err(|e| JsError::new(&e))?;
        }
        {
            let tk = Arc::clone(&toolkit);
            session
                .register_foreign(
                    "circle",
                    Box::new(move |args| {
                        // The verifier has already checked this call against the
                        // declared signature, so a mismatch here is a host bug.
                        let [_, Value::I64(x), Value::I64(y), Value::I64(r), Value::I64(hue)] =
                            args
                        else {
                            return Value::Unit;
                        };
                        let mut t = tk.lock().unwrap();
                        t.circles_drawn += 1;
                        t.pending.extend_from_slice(&[
                            *x as i32, *y as i32, *r as i32, *hue as i32,
                        ]);
                        Value::Unit
                    }),
                )
                .map_err(|e| JsError::new(&e))?;
        }

        session
            .eval(scene)
            .map_err(|e| JsError::new(&e))?;
        let main = session
            .fn_id("main")
            .ok_or_else(|| JsError::new("the scene has no `main`"))?;
        let actor = session
            .engine
            .spawn(main, vec![])
            .map_err(|c| JsError::new(&format!("{c:?}")))?;

        Ok(Demo { session, actor, toolkit, frames: 0, last_condition: String::new() })
    }

    /// Run the program until it crosses its next `yield` — one frame. Edits
    /// applied between calls land between two instructions of this loop; the
    /// loop is never restarted.
    pub fn step_frame(&mut self) -> u32 {
        if matches!(self.actor.status, ActorStatus::Paused(_)) {
            return STATUS_FROZEN;
        }
        for _ in 0..FRAME_INSTRUCTION_CAP {
            match self.session.engine.step(&mut self.actor) {
                Turn::Progress => {}
                Turn::Yielded => {
                    self.frames += 1;
                    let mut t = self.toolkit.lock().unwrap();
                    // Commit the completed frame for painting.
                    t.committed = std::mem::take(&mut t.pending);
                    return STATUS_RUNNING;
                }
                Turn::Done => return STATUS_DONE,
                Turn::Paused => {
                    self.last_condition = match &self.actor.status {
                        ActorStatus::Paused(c) => {
                            self.session.engine.with_world(|w| describe(w, c))
                        }
                        _ => String::new(),
                    };
                    return STATUS_FROZEN;
                }
                Turn::Blocked => return STATUS_FROZEN,
            }
        }
        STATUS_CAPPED
    }

    /// The last completed frame, as flat `[x, y, r, hue]` quads.
    pub fn draw_ops(&self) -> Vec<i32> {
        self.toolkit.lock().unwrap().committed.clone()
    }

    /// Apply an edit to the live world, reporting as JSON what it did:
    /// `{ok, error, installed, unobservable}`.
    ///
    /// `unobservable` names functions the edit installed that the running
    /// program can no longer reach — `seed`, whose only caller is a `letonce`
    /// initializer that already ran. Such an edit really is installed, and the
    /// old message said so and stopped there, which reads as the edit being
    /// ignored. It needs to say why nothing moved.
    pub fn eval(&mut self, source: &str) -> String {
        let before = self.current_versions();
        let before_types = self.current_types();
        if let Err(error) = self.session.eval(source) {
            return format!("{{\"ok\":false,\"error\":{}}}", json_string(&error));
        }
        let after = self.current_versions();

        let changed: Vec<(DefId, String, bool)> = after
            .iter()
            .filter(|(id, (version, ..))| {
                before.get(id).map(|(v, ..)| v != version).unwrap_or(true)
            })
            .map(|(id, (_, name, broken))| (*id, name.clone(), *broken))
            .collect();
        // A function published Broken is not an installation, and calling it
        // one hides the whole point of the breaking step.
        let (broken, installed): (Vec<_>, Vec<_>) =
            changed.iter().partition(|(_, _, is_broken)| *is_broken);
        let installed: Vec<(DefId, String)> =
            installed.iter().map(|(id, name, _)| (*id, name.clone())).collect();

        let unobservable: Vec<String> = self.session.engine.with_world(|w| {
            match reachable_from_entry(w) {
                // Reachability is only knowable when nothing on the live path
                // calls through a function *value*; stay quiet rather than
                // guess.
                None => Vec::new(),
                Some(live) => installed
                    .iter()
                    .filter(|(id, _)| !live.contains(id))
                    .map(|(_, name)| name.clone())
                    .collect(),
            }
        });

        let types: Vec<String> = {
            let after_types = self.current_types();
            after_types
                .iter()
                .filter(|(id, (version, _))| {
                    before_types.get(id).map(|(v, _)| v != version).unwrap_or(true)
                })
                .map(|(_, (_, name))| name.clone())
                .collect()
        };

        format!(
            "{{\"ok\":true,\"installed\":[{}],\"broken\":[{}],\"types\":[{}],\"unobservable\":[{}]}}",
            json_names(installed.iter().map(|(_, n)| n.as_str())),
            json_names(broken.iter().map(|(_, n, _)| n.as_str())),
            json_names(types.iter().map(|n| n.as_str())),
            json_names(unobservable.iter().map(|n| n.as_str())),
        )
    }

    /// Every current function as `id -> (version, name, broken)`.
    fn current_versions(&self) -> BTreeMap<DefId, (Version, String, bool)> {
        self.session.engine.with_world(|w| {
            w.current_functions
                .iter()
                .filter_map(|(id, version)| {
                    w.functions.get(&(*id, *version)).map(|state| {
                        let broken = matches!(state, FunctionState::Broken { .. });
                        (*id, (*version, function_name(state), broken))
                    })
                })
                .collect()
        })
    }

    /// Every current schema as `id -> (version, name)`. A struct or enum
    /// evolving is as much a part of an edit as a function changing, and
    /// reporting only functions makes a migration step look like it did less
    /// than it did.
    fn current_types(&self) -> BTreeMap<DefId, (Version, String)> {
        self.session.engine.with_world(|w| {
            w.current_schemas
                .iter()
                .filter_map(|(id, version)| {
                    w.schemas
                        .get(&(*id, *version))
                        .map(|schema| (*id, (*version, schema.name.clone())))
                })
                .collect()
        })
    }

    /// Mark a frozen program runnable again. The next step re-executes the
    /// instruction that trapped, and re-traps cleanly if nothing was repaired.
    pub fn thaw(&mut self) {
        self.session.engine.thaw(&mut self.actor);
        self.last_condition.clear();
    }

    /// Throw the whole world away and boot a fresh one.
    pub fn reset(&mut self) -> Result<(), JsError> {
        *self = Demo::new()?;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn frozen(&self) -> bool {
        matches!(self.actor.status, ActorStatus::Paused(_))
    }

    #[wasm_bindgen(getter)]
    pub fn condition(&self) -> String {
        self.last_condition.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The counters that make the FFI story checkable: the canvas is opened
    /// once and never again, no matter how many edits land.
    #[wasm_bindgen(getter)]
    pub fn canvases_opened(&self) -> u64 {
        self.toolkit.lock().unwrap().canvases_opened
    }

    #[wasm_bindgen(getter)]
    pub fn circles_drawn(&self) -> u64 {
        self.toolkit.lock().unwrap().circles_drawn
    }

    /// The live world, for the inspector: every current definition with its
    /// version and whether it is Broken. This is read straight out of the
    /// engine, not tracked alongside it.
    pub fn world_json(&self) -> String {
        self.session.engine.with_world(|w| {
            let names = Names::of(w);
            let mut out = String::from("{\"functions\":[");
            let mut first = true;
            for (id, version) in &w.current_functions {
                let Some(state) = w.functions.get(&(*id, *version)) else {
                    continue;
                };
                let (name, broken, diagnostics) = match state {
                    FunctionState::Ready(f) => (f.name.clone(), false, Vec::new()),
                    FunctionState::Broken { name, diagnostics, .. } => (
                        name.clone(),
                        true,
                        diagnostics.iter().map(|d| names.humanize(d)).collect(),
                    ),
                };
                // `letonce` lowers to a synthetic initializer function per
                // global. They are real definitions, but they are the
                // compiler's bookkeeping, not the program the viewer is reading.
                if name.starts_with("__") {
                    continue;
                }
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&format!(
                    "{{\"name\":{},\"version\":{},\"broken\":{},\"why\":{}}}",
                    json_string(&name),
                    version.0,
                    broken,
                    json_string(&diagnostics.join("; ")),
                ));
            }
            out.push_str("],\"types\":[");
            let mut first = true;
            for (id, version) in &w.current_schemas {
                let Some(schema) = w.schemas.get(&(*id, *version)) else {
                    continue;
                };
                if !first {
                    out.push(',');
                }
                first = false;
                let shape = if schema.variants.is_empty() {
                    schema
                        .fields
                        .iter()
                        .map(|f| f.name.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    schema
                        .variants
                        .iter()
                        .map(|v| v.name.clone())
                        .collect::<Vec<_>>()
                        .join(" | ")
                };
                out.push_str(&format!(
                    "{{\"name\":{},\"version\":{},\"shape\":{}}}",
                    json_string(&schema.name),
                    version.0,
                    json_string(&shape),
                ));
            }
            out.push_str("]}");
            out
        })
    }

    /// The scene source, so the editor shows the real program rather than a
    /// copy that can drift from what is running.
    pub fn scene_source() -> String {
        SCENE.to_string()
    }

    pub fn scenario_count() -> usize {
        SCENARIOS.len()
    }

    pub fn scenario_title(index: usize) -> String {
        SCENARIOS.get(index).map(|(t, _)| t.to_string()).unwrap_or_default()
    }

    pub fn scenario_source(index: usize) -> String {
        SCENARIOS.get(index).map(|(_, s)| s.to_string()).unwrap_or_default()
    }
}

/// The functions the running program can still reach, walking direct calls from
/// `main`. `None` means "cannot say": something reachable calls through a
/// function *value*, whose callee is resolved at call time.
///
/// This is what separates "your edit is live" from "your edit is installed but
/// nothing will ever call it" — the case a `letonce`-only function like `seed`
/// lands in once its initializer has run.
fn reachable_from_entry(world: &World) -> Option<BTreeSet<DefId>> {
    let entry = world.current_functions.iter().find(|(id, version)| {
        world
            .functions
            .get(&(**id, **version))
            .is_some_and(|s| function_name(s) == "main")
    })?;

    let mut seen = BTreeSet::new();
    let mut work = vec![*entry.0];
    while let Some(id) = work.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(version) = world.current_functions.get(&id) else {
            continue;
        };
        // A Broken function has no code to walk; its callees stay unvisited,
        // which is conservative in the safe direction (fewer "unobservable"
        // claims, never a false one).
        let Some(FunctionState::Ready(f)) = world.functions.get(&(id, *version)) else {
            continue;
        };
        for instruction in &f.code {
            match instruction {
                Instruction::Call { function, .. } => work.push(*function),
                Instruction::IndirectCall { .. } => return None,
                _ => {}
            }
        }
    }
    Some(seen)
}

/// Resolves the `DefId`s the engine speaks in — correct for a runtime, useless
/// on screen — into names, including the ids the verifier embeds inside its
/// diagnostic strings ("callee 1000001 is broken").
struct Names {
    /// `(id as text, name)`, longest id first so one id can never be rewritten
    /// inside another.
    ids: Vec<(String, String)>,
}

impl Names {
    fn of(world: &World) -> Names {
        let mut ids: Vec<(String, String)> = world
            .current_functions
            .iter()
            .filter_map(|(id, version)| {
                world
                    .functions
                    .get(&(*id, *version))
                    .map(|state| (id.to_string(), function_name(state)))
            })
            .collect();
        ids.sort_by_key(|(digits, _)| std::cmp::Reverse(digits.len()));
        Names { ids }
    }

    fn humanize(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (digits, name) in &self.ids {
            out = out.replace(digits.as_str(), name);
        }
        out
    }

    fn name_of(&self, id: DefId) -> String {
        let digits = id.to_string();
        self.ids
            .iter()
            .find(|(d, _)| *d == digits)
            .map(|(_, name)| name.clone())
            .unwrap_or_else(|| format!("fn#{id}"))
    }
}

fn function_name(state: &FunctionState) -> String {
    match state {
        FunctionState::Ready(f) => f.name.clone(),
        FunctionState::Broken { name, .. } => name.clone(),
    }
}

/// Render a condition for a human.
fn describe(world: &World, condition: &Condition) -> String {
    let names = Names::of(world);
    match condition {
        Condition::BrokenFunction { function, diagnostics } => format!(
            "`{}` is broken — {}",
            names.name_of(*function),
            diagnostics
                .iter()
                .map(|d| names.humanize(d))
                .collect::<Vec<_>>()
                .join("; "),
        ),
        Condition::MissingMigration { type_id, from, to, .. } => {
            let name = world
                .current_schemas
                .get(type_id)
                .and_then(|v| world.schemas.get(&(*type_id, *v)))
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("type#{type_id}"));
            format!(
                "no migration for `{name}` v{} → v{} — install one and it resumes",
                from.0, to.0
            )
        }
        Condition::RuntimeTypeError { function, pc, message } => format!(
            "`{}` trapped at pc {pc} — {}",
            names.name_of(*function),
            names.humanize(message),
        ),
    }
}

fn json_names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    names.map(json_string).collect::<Vec<_>>().join(",")
}

/// Minimal JSON string escaping — enough for names and diagnostics, and it
/// keeps this crate's dependency list to just wasm-bindgen.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
