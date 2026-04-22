//! Serialize a simulation (structural graph + recorded event log) to
//! a self-contained HTML file that plays back the run.
//!
//! This is *not* a real visual editor. It's a proof that the
//! formalism's event log is sufficient to drive a visualization —
//! the log is the source of truth; a renderer just walks it and
//! reconstructs frames. Whatever the eventual whiteboard UI does, it
//! consumes the same stream.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::event::Event;
use crate::samples::Samples;
use crate::sim::{NodeId, Sim};
use crate::value::Value;

/// Auto-layout: circle around origin with unit radius.
fn circle_layout(nodes: &[NodeId], radius: f64) -> BTreeMap<NodeId, (f64, f64)> {
    let mut out = BTreeMap::new();
    let n = nodes.len().max(1);
    for (i, nid) in nodes.iter().enumerate() {
        let theta = (i as f64) * std::f64::consts::TAU / (n as f64);
        out.insert(*nid, (radius * theta.cos(), radius * theta.sin()));
    }
    out
}

/// Escape a string for JSON.
fn esc(s: &str) -> String {
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

fn value_to_json(v: &Value) -> String {
    match v {
        Value::Nil => "null".to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => {
            if f.is_finite() { format!("{}", f) } else { "null".to_string() }
        }
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => esc(s),
        Value::Variant { tag, payload } => format!(
            r#"{{"tag":{},"payload":{}}}"#, esc(tag), value_to_json(payload)
        ),
        Value::Record(fields) => {
            let inner: Vec<String> = fields.iter()
                .map(|(k, v)| format!("{}:{}", esc(k), value_to_json(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        Value::Samples(Samples { cap, items }) => {
            let inner: Vec<String> = items.iter().map(value_to_json).collect();
            format!(r#"{{"_samples":{{"cap":{},"items":[{}]}}}}"#,
                cap, inner.join(","))
        }
        Value::NodeRef(id) => format!(r#"{{"_noderef":{}}}"#, id.0),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(value_to_json).collect();
            format!("[{}]", inner.join(","))
        }
    }
}

fn event_to_json(ev: &Event) -> String {
    match ev {
        Event::ClockAdvanced { from_ns, to_ns } => format!(
            r#"{{"k":"clock","at":{},"to":{}}}"#, from_ns, to_ns
        ),
        Event::RuleFired { node, rule, at_ns } => format!(
            r#"{{"k":"fire","at":{},"node":{},"rule":{}}}"#,
            at_ns, node.0, esc(rule)
        ),
        Event::SlotWritten { node, slot, value, at_ns } => format!(
            r#"{{"k":"slot","at":{},"node":{},"slot":{},"value":{}}}"#,
            at_ns, node.0, esc(slot), value_to_json(value)
        ),
        Event::PacketEmitted { packet, from, to, at_ns, arrives_at_ns, payload } => format!(
            r#"{{"k":"emit","at":{},"packet":{},"from":{},"to":{},"arrives":{},"payload":{}}}"#,
            at_ns, packet.0, from.0, to.0, arrives_at_ns, value_to_json(payload)
        ),
        Event::PacketDelivered { packet, to, at_ns } => format!(
            r#"{{"k":"deliver","at":{},"packet":{},"to":{}}}"#,
            at_ns, packet.0, to.0
        ),
        Event::PacketConsumed { packet, by, rule, at_ns } => format!(
            r#"{{"k":"consume","at":{},"packet":{},"by":{},"rule":{}}}"#,
            at_ns, packet.0, by.0, esc(rule)
        ),
        Event::MetricRecorded { node, name, value, at_ns } => format!(
            r#"{{"k":"metric","at":{},"node":{},"name":{},"value":{}}}"#,
            at_ns, node.0, esc(name), value_to_json(value)
        ),
        Event::NodeSpawned { node, template, parent, at_ns } => format!(
            r#"{{"k":"spawn","at":{},"node":{},"template":{},"parent":{}}}"#,
            at_ns, node.0, esc(template),
            parent.map(|p| p.0.to_string()).unwrap_or_else(|| "null".into())
        ),
        Event::NodeDespawned { node, at_ns } => format!(
            r#"{{"k":"despawn","at":{},"node":{}}}"#,
            at_ns, node.0
        ),
    }
}

fn sim_to_json(sim: &Sim, title: &str) -> String {
    let node_ids: Vec<NodeId> = sim.nodes.keys().copied().collect();
    let layout = circle_layout(&node_ids, 260.0);

    let nodes_json: Vec<String> = sim.nodes.values().map(|n| {
        let (x, y) = layout[&n.id];
        format!(
            r#"{{"id":{},"name":{},"x":{:.1},"y":{:.1}}}"#,
            n.id.0, esc(&n.name), x, y
        )
    }).collect();

    let edges_json: Vec<String> = sim.edges.values().map(|e| format!(
        r#"{{"id":{},"from":{},"to":{}}}"#, e.id.0, e.from.0, e.to.0
    )).collect();

    let events_json: Vec<String> = sim.log.events.iter().map(event_to_json).collect();

    let duration = sim.now_ns;

    format!(
        r#"{{
"title":{},
"duration_ns":{},
"nodes":[{}],
"edges":[{}],
"events":[{}]
}}"#,
        esc(title),
        duration,
        nodes_json.join(","),
        edges_json.join(","),
        events_json.join(",")
    )
}

/// Write a self-contained HTML file that plays back this sim.
pub fn write_html<P: AsRef<Path>>(sim: &Sim, title: &str, path: P) -> std::io::Result<()> {
    let json = sim_to_json(sim, title);
    let html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", &json);
    let mut f = fs::File::create(path)?;
    f.write_all(html.as_bytes())?;
    Ok(())
}

impl Sim {
    /// Shortcut: write a self-contained HTML player next to the sim.
    pub fn write_html<P: AsRef<Path>>(&self, title: &str, path: P) -> std::io::Result<()> {
        write_html(self, title, path)
    }
}

const HTML_TEMPLATE: &str = include_str!("player.html");
