//! Low-fi interactive REPL for poking at a live sim.
//!
//! You pass a `Sim` you built in Rust (one of the examples, say) and
//! drop into a command line. Commands:
//!
//!   step [N]              — advance N events (default 1). Time-step;
//!                           skips past same-instant events.
//!   run T                 — run until sim time T (accepts `500ms`, `1s`, `1500000000`).
//!   now                   — print current sim clock.
//!   inspect NODE          — print slots + inbox depth of node by name.
//!   inspect all           — print summary of every node.
//!   slot NODE SLOT        — print a single slot's current value.
//!   set_param NAME EXPR   — rebind a param. EXPR is a tiny subset:
//!                           integers, floats, booleans, Exp(N), Uniform(a,b),
//!                           Bernoulli(p). Time literals (1ms, 25us) supported.
//!   params                — list currently-bound params (as debug).
//!   inject NODE TAG       — inject a Variant(tag, Nil) packet into a node.
//!   snap save NAME        — save a named snapshot of the current sim.
//!   snap list             — list saved snapshots.
//!   snap restore NAME     — roll sim back to the saved snapshot.
//!   rewind T              — roll back to sim time T via snapshot + replay
//!                           (only works if a prior snapshot covers T).
//!   metrics               — recent MetricRecorded events.
//!   events [N]            — last N events (default 20).
//!   q | quit              — exit.
//!
//! The REPL is deliberately simple — no parsing library. Commands are
//! whitespace-split; expressions use a tiny hand-rolled mini-parser.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use crate::event::Event;
use crate::expr::Expr;
use crate::sim::{NodeId, Sim, Time};
use crate::value::Value;

pub struct Repl {
    sim: Sim,
    snapshots: HashMap<String, Sim>,
}

impl Repl {
    pub fn new(sim: Sim) -> Self {
        Self { sim, snapshots: HashMap::new() }
    }

    /// Run the REPL against stdin/stdout until the user quits.
    pub fn run(&mut self) {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        let mut input = String::new();

        writeln!(out, "flow repl — type 'help' for commands, 'q' to exit").ok();
        loop {
            write!(out, "\n[t={:.3}ms] > ", self.sim.now_ns as f64 / 1_000_000.0).ok();
            out.flush().ok();
            input.clear();
            if stdin.lock().read_line(&mut input).unwrap_or(0) == 0 { break; }
            let line = input.trim();
            if line.is_empty() { continue; }

            match self.dispatch(line) {
                Ok(msg) => { if !msg.is_empty() { writeln!(out, "{}", msg).ok(); } }
                Err(e) => { writeln!(out, "error: {}", e).ok(); }
            }
        }
        writeln!(out, "bye").ok();
    }

    fn dispatch(&mut self, line: &str) -> Result<String, String> {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().ok_or("empty command")?;
        let rest: Vec<&str> = parts.collect();
        match cmd {
            "help" | "h" => Ok(HELP.into()),
            "q" | "quit" => std::process::exit(0),
            "now" => Ok(format!("{} ns ({:.3} ms)", self.sim.now_ns, self.sim.now_ns as f64 / 1_000_000.0)),
            "step" => self.cmd_step(&rest),
            "run"  => self.cmd_run(&rest),
            "inspect" => self.cmd_inspect(&rest),
            "slot" => self.cmd_slot(&rest),
            "set_param" => self.cmd_set_param(&rest),
            "params" => Ok(self.cmd_params()),
            "inject" => self.cmd_inject(&rest),
            "snap" => self.cmd_snap(&rest),
            "rewind" => self.cmd_rewind(&rest),
            "metrics" => Ok(self.cmd_metrics()),
            "events" => Ok(self.cmd_events(&rest)),
            other => Err(format!("unknown command: `{}`", other)),
        }
    }

    // --- step / run ---

    fn cmd_step(&mut self, args: &[&str]) -> Result<String, String> {
        let n: usize = args.first().map(|s| s.parse().unwrap_or(1)).unwrap_or(1);
        // Advance to the next time-advance N times.
        let start_events = self.sim.log.total_recorded;
        for _ in 0..n {
            let before = self.sim.now_ns;
            // Find the time of the next event beyond now_ns.
            let next_t = self.sim.log.events.iter().rev()
                .find_map(|_| None::<Time>)  // we don't use log; use in_flight + pending
                .or_else(|| {
                    let a = self.sim.in_flight.peek().map(|r| r.0.arrives_at_ns);
                    let b = self.sim.pending_actions.peek().map(|r| r.0.at_ns);
                    match (a, b) {
                        (Some(x), Some(y)) => Some(x.min(y)),
                        (Some(x), None) => Some(x),
                        (None, Some(y)) => Some(y),
                        (None, None) => None,
                    }
                });
            match next_t {
                Some(t) if t > before => { self.sim.run_until(t); }
                _ => break,
            }
        }
        let delta = self.sim.log.total_recorded - start_events;
        Ok(format!("+{} events; t={:.3}ms", delta, self.sim.now_ns as f64 / 1_000_000.0))
    }

    fn cmd_run(&mut self, args: &[&str]) -> Result<String, String> {
        let t_str = args.first().ok_or("usage: run <time>")?;
        let t = parse_time(t_str)?;
        let start_events = self.sim.log.total_recorded;
        self.sim.run_until(t);
        let delta = self.sim.log.total_recorded - start_events;
        Ok(format!("ran +{} events; t={:.3}ms", delta, self.sim.now_ns as f64 / 1_000_000.0))
    }

    // --- inspect ---

    fn cmd_inspect(&self, args: &[&str]) -> Result<String, String> {
        let name = args.first().ok_or("usage: inspect <node|all>")?;
        if *name == "all" {
            let mut lines = Vec::new();
            for n in self.sim.nodes.values() {
                let kind = if n.is_compound() { "compound" } else { "leaf" };
                lines.push(format!("{:3} {:<16} ({:>8}) parent={:?} inbox={} slots={}",
                    n.id.0, n.name, kind, n.parent.map(|p| p.0), n.inbox.len(), n.slots.len()));
            }
            Ok(lines.join("\n"))
        } else {
            let nid = self.find_node(name)?;
            let n = &self.sim.nodes[&nid];
            let mut out = format!("{} (id {}), {} node",
                n.name, nid.0,
                if n.is_compound() { "compound" } else { "leaf" });
            if let Some(p) = n.parent { out.push_str(&format!(", parent={}", p.0)); }
            out.push_str(&format!("\n  inbox: {} packets", n.inbox.len()));
            if let Some(cb) = &n.compound {
                out.push_str("\n  in_ports:");
                for (k, v) in &cb.in_ports { out.push_str(&format!("\n    {} -> {}", k, v.0)); }
                out.push_str("\n  out_ports:");
                for (k, v) in &cb.out_ports { out.push_str(&format!("\n    {} -> {}", k, v.0)); }
            }
            for (k, v) in &n.slots {
                out.push_str(&format!("\n  {}: {}", k, format_value(v)));
            }
            Ok(out)
        }
    }

    fn cmd_slot(&self, args: &[&str]) -> Result<String, String> {
        let name = args.first().ok_or("usage: slot <node> <slot>")?;
        let slot = args.get(1).ok_or("usage: slot <node> <slot>")?;
        let nid = self.find_node(name)?;
        let n = &self.sim.nodes[&nid];
        let v = n.slots.get(*slot).ok_or_else(|| format!("no slot `{}`", slot))?;
        Ok(format_value(v))
    }

    // --- params ---

    fn cmd_set_param(&mut self, args: &[&str]) -> Result<String, String> {
        let name = args.first().ok_or("usage: set_param <name> <expr>")?;
        let rest = args[1..].join(" ");
        if rest.is_empty() { return Err("missing value expression".into()); }
        let expr = parse_simple_expr(&rest)?;
        self.sim.params.insert((*name).to_string(), expr);
        Ok(format!("param `{}` rebound", name))
    }

    fn cmd_params(&self) -> String {
        if self.sim.params.is_empty() { return "(no params bound)".into(); }
        let mut lines = Vec::new();
        for (k, v) in &self.sim.params {
            lines.push(format!("  {} = {:?}", k, v));
        }
        lines.join("\n")
    }

    // --- inject ---

    fn cmd_inject(&mut self, args: &[&str]) -> Result<String, String> {
        let node = args.first().ok_or("usage: inject <node> <tag>")?;
        let tag = args.get(1).ok_or("usage: inject <node> <tag>")?;
        let nid = self.find_node(node)?;
        let pid = self.sim.inject(nid, Value::variant(*tag, Value::Nil), None);
        Ok(format!("injected pkt {} ({}) into {}", pid.0, tag, node))
    }

    // --- snapshots ---

    fn cmd_snap(&mut self, args: &[&str]) -> Result<String, String> {
        let sub = args.first().ok_or("usage: snap <save|list|restore> [name]")?;
        match *sub {
            "save" => {
                let name = args.get(1).ok_or("usage: snap save <name>")?;
                self.snapshots.insert((*name).to_string(), self.sim.snapshot());
                Ok(format!("saved snapshot `{}` at t={:.3}ms",
                    name, self.sim.now_ns as f64 / 1_000_000.0))
            }
            "list" => {
                if self.snapshots.is_empty() { return Ok("(no snapshots)".into()); }
                let mut lines = Vec::new();
                for (k, s) in &self.snapshots {
                    lines.push(format!("  {}  t={:.3}ms  events={}", k,
                        s.now_ns as f64 / 1_000_000.0, s.log.total_recorded));
                }
                Ok(lines.join("\n"))
            }
            "restore" => {
                let name = args.get(1).ok_or("usage: snap restore <name>")?;
                let snap = self.snapshots.get(*name).ok_or_else(|| format!("no snapshot `{}`", name))?.clone();
                self.sim.restore_from(snap);
                Ok(format!("restored to t={:.3}ms", self.sim.now_ns as f64 / 1_000_000.0))
            }
            other => Err(format!("unknown snap subcommand `{}`", other)),
        }
    }

    // --- rewind (using the newest snapshot that's ≤ t) ---

    fn cmd_rewind(&mut self, args: &[&str]) -> Result<String, String> {
        let t_str = args.first().ok_or("usage: rewind <time>")?;
        let t = parse_time(t_str)?;
        // Find the best candidate snapshot — the one with the largest
        // now_ns that's ≤ target t.
        let best = self.snapshots.iter()
            .filter(|(_, s)| s.now_ns <= t)
            .max_by_key(|(_, s)| s.now_ns)
            .map(|(k, s)| (k.clone(), s.clone()));
        let (name, snap) = best.ok_or("no snapshot covers that time; try `snap save` first")?;
        self.sim.restore_from(snap);
        self.sim.run_until(t);
        Ok(format!("rewound to t={:.3}ms via snapshot `{}`",
            self.sim.now_ns as f64 / 1_000_000.0, name))
    }

    // --- metrics / events ---

    fn cmd_metrics(&self) -> String {
        let mut lines = Vec::new();
        for e in self.sim.log.events.iter().rev() {
            if let Event::MetricRecorded { node, name, value, at_ns } = e {
                let nname = self.sim.nodes.get(node).map(|n| n.name.as_str()).unwrap_or("?");
                lines.push(format!("  {:>8.3}ms  {:<12} {} = {}",
                    *at_ns as f64 / 1_000_000.0, nname, name, format_value(value)));
                if lines.len() >= 20 { break; }
            }
        }
        if lines.is_empty() { return "(no metrics)".into(); }
        lines.reverse();
        lines.join("\n")
    }

    fn cmd_events(&self, args: &[&str]) -> String {
        let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(20);
        let mut lines = Vec::new();
        for e in self.sim.log.events.iter().rev().take(n) {
            lines.push(format!("  {}", format_event(e, &self.sim)));
        }
        lines.reverse();
        lines.join("\n")
    }

    // --- helpers ---

    fn find_node(&self, name: &str) -> Result<NodeId, String> {
        self.sim.node_by_name(name).ok_or_else(|| format!("no node named `{}`", name))
    }
}

// -----------------------------------------------------------------------------
// Simple parsing helpers
// -----------------------------------------------------------------------------

fn parse_time(s: &str) -> Result<Time, String> {
    let s = s.trim();
    let (num_part, mult) = if let Some(rest) = s.strip_suffix("ns") {
        (rest, 1u64)
    } else if let Some(rest) = s.strip_suffix("us") {
        (rest, 1_000u64)
    } else if let Some(rest) = s.strip_suffix("ms") {
        (rest, 1_000_000u64)
    } else if let Some(rest) = s.strip_suffix("s") {
        (rest, 1_000_000_000u64)
    } else {
        (s, 1u64)  // raw ns
    };
    let num: f64 = num_part.trim().parse().map_err(|_| format!("bad time `{}`", s))?;
    Ok((num * mult as f64) as Time)
}

/// Parse a tiny expression grammar for `set_param` values:
///   literals: INT, FLOAT, true, false
///   time literals: 1ms, 25us, 1s
///   distribution calls: Exp(x), Uniform(a, b), Bernoulli(p)
///   nothing else — no arithmetic, no params referencing params.
///
/// Sufficient for live-tuning tasks like "slow the service time to 200ms".
fn parse_simple_expr(s: &str) -> Result<Expr, String> {
    let s = s.trim();
    // Try time literal / number first.
    if let Ok(t) = parse_time(s) {
        // Disambiguate: if the string ends in a time suffix it's Int ns.
        if s.ends_with("ns") || s.ends_with("us") || s.ends_with("ms") || s.ends_with("s") {
            return Ok(Expr::int(t as i64));
        }
    }
    if let Ok(n) = s.parse::<i64>() { return Ok(Expr::int(n)); }
    if let Ok(f) = s.parse::<f64>() { return Ok(Expr::float(f)); }
    if s == "true" { return Ok(Expr::bool(true)); }
    if s == "false" { return Ok(Expr::bool(false)); }
    // Distribution calls: IDENT ( args )
    if let Some(paren) = s.find('(') {
        if s.ends_with(')') {
            let name = &s[..paren];
            let inner = &s[paren + 1 .. s.len() - 1];
            let args: Vec<&str> = inner.split(',').map(|a| a.trim()).collect();
            match name {
                "Exp" => {
                    if args.len() != 1 { return Err("Exp takes 1 arg (mean_ns)".into()); }
                    let mean = parse_simple_expr(args[0])?;
                    return Ok(Expr::exp_dist(mean));
                }
                "Uniform" => {
                    if args.len() != 2 { return Err("Uniform takes 2 args".into()); }
                    let lo = parse_simple_expr(args[0])?;
                    let hi = parse_simple_expr(args[1])?;
                    return Ok(Expr::uniform_int(lo, hi));
                }
                "Bernoulli" => {
                    if args.len() != 1 { return Err("Bernoulli takes 1 arg".into()); }
                    let p = parse_simple_expr(args[0])?;
                    return Ok(Expr::bernoulli(p));
                }
                _ => return Err(format!("unknown function `{}`", name)),
            }
        }
    }
    Err(format!("unparseable expression: `{}`", s))
}

fn format_value(v: &Value) -> String {
    match v {
        Value::Nil => "nil".into(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => format!("{:?}", s),
        Value::Variant { tag, payload } => {
            match payload.as_ref() {
                Value::Nil => tag.clone(),
                other => format!("{}({})", tag, format_value(other)),
            }
        }
        Value::Record(fs) => {
            let inner: Vec<String> = fs.iter()
                .map(|(k, v)| format!("{}:{}", k, format_value(v))).collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Samples(s) => format!("[Samples {}/{}]", s.items.len(), s.cap),
        Value::NodeRef(id) => format!("→{}", id.0),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(format_value).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

fn format_event(e: &Event, sim: &Sim) -> String {
    let name = |id: &NodeId| sim.nodes.get(id).map(|n| n.name.clone()).unwrap_or_else(|| format!("#{}", id.0));
    let t = |ns: u64| format!("{:>8.3}ms", ns as f64 / 1_000_000.0);
    match e {
        Event::ClockAdvanced { from_ns: _, to_ns } => format!("{} clock", t(*to_ns)),
        Event::RuleFired { node, rule, at_ns } =>
            format!("{} fire  {} · {}", t(*at_ns), name(node), rule),
        Event::SlotWritten { node, slot, value, at_ns } =>
            format!("{} slot  {}.{} = {}", t(*at_ns), name(node), slot, format_value(value)),
        Event::PacketEmitted { from, to, at_ns, arrives_at_ns, payload, .. } =>
            format!("{} emit  {}→{} ({}) arr@{:.3}ms",
                t(*at_ns), name(from), name(to), format_value(payload), *arrives_at_ns as f64 / 1_000_000.0),
        Event::PacketDelivered { to, at_ns, .. } =>
            format!("{} deliv → {}", t(*at_ns), name(to)),
        Event::PacketConsumed { by, rule, at_ns, .. } =>
            format!("{} consu {} · {}", t(*at_ns), name(by), rule),
        Event::MetricRecorded { node, name: m, value, at_ns } =>
            format!("{} metric {} {} = {}", t(*at_ns), name(node), m, format_value(value)),
        Event::NodeSpawned { node, template, parent, at_ns } =>
            format!("{} spawn {} from {} parent={:?}", t(*at_ns), name(node), template, parent.map(|p| p.0)),
        Event::NodeDespawned { node, at_ns } =>
            format!("{} desp  {}", t(*at_ns), name(node)),
    }
}

const HELP: &str = "\
commands:
  step [N]           advance N time-steps (default 1)
  run T              run until sim time T (e.g. run 500ms)
  now                show current clock
  inspect NODE       show node's slots, inbox, ports
  inspect all        summary of all nodes
  slot NODE SLOT     show one slot's value
  set_param NAME V   rebind param V = literal | Exp(n) | Uniform(a,b) | Bernoulli(p)
  params             list all params
  inject NODE TAG    inject Variant(tag, nil) into a node's inbox
  snap save NAME     save a named snapshot
  snap list          list snapshots
  snap restore NAME  roll back to a snapshot
  rewind T           roll back to time T via best snapshot
  metrics            recent MetricRecorded events
  events [N]         last N events (default 20)
  q | quit           exit";
