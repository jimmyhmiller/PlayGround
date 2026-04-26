use std::cmp::Reverse;
use std::collections::VecDeque;

use std::collections::BTreeMap;

use crate::event::Event;
use crate::expr::{EvalCtx, Expr};
use crate::rule::{Effect, EmitTo, MetaOp, ReturnPathOp, Rule, When};
use crate::scenario::Action;
use crate::sim::{EdgeId, Node, NodeId, Packet, Scheduled, Sim, Time};
use crate::template::EdgeEnd;
use crate::value::{Bindings, Value, match_pattern};

impl Sim {
    /// Run the simulation forward until either `deadline_ns` is reached
    /// or the world becomes quiescent (no fireable rules, no in-flight
    /// packets). Returns the number of rule firings that occurred.
    ///
    /// Records per-phase wall-clock samples into `Sim::perf_samples`
    /// (drained by the host each frame). Phases:
    ///   - `sim.run_until.total` — total wall time inside this call
    ///   - `sim.fire_rules` — inner loop firing all instant-due rules
    ///   - `sim.dispatch_actions` — scenario-action queue at this instant
    ///   - `sim.fire_timeline` — user-editable timeline events at this instant
    ///   - `sim.deliver_packets` — drains the in-flight heap at this instant
    ///
    /// Phase samples are accumulated across the whole `run_until` call
    /// (multiple instants), then pushed once per phase. That keeps the
    /// sample-vec small under heavy bursts (one entry per phase per call,
    /// not one per inner iteration).
    pub fn run_until(&mut self, deadline_ns: Time) -> usize {
        use std::time::Instant;
        let total_start = Instant::now();
        let mut fire_us: f64 = 0.0;
        let mut action_us: f64 = 0.0;
        let mut timeline_us: f64 = 0.0;
        let mut deliver_us: f64 = 0.0;

        let mut firings = 0usize;
        loop {
            // 1. Fire every rule that can fire at the current instant.
            let phase_start = Instant::now();
            let mut instant_steps = 0usize;
            while self.try_fire_one() {
                firings += 1;
                instant_steps += 1;
                if instant_steps > self.max_steps_per_instant {
                    panic!(
                        "Sim: exceeded {} rule firings at instant {} ns \
                         (runaway rules? cycle with zero-latency edges?)",
                        self.max_steps_per_instant, self.now_ns
                    );
                }
            }
            fire_us += phase_start.elapsed().as_secs_f64() * 1_000_000.0;

            // 2. Nothing more to fire. Next event is the earliest of:
            //    (a) a scheduled scenario action
            //    (b) an in-flight delivery
            //    (c) the deadline.
            let next_delivery = self.in_flight.peek().map(|r| r.0.arrives_at_ns);
            let next_action = self.pending_actions.peek().map(|r| r.0.at_ns);
            let next_timeline = self.timeline.next_pending_at_ns();

            let next_event = [next_delivery, next_action, next_timeline]
                .into_iter()
                .flatten()
                .min();

            let Some(t) = next_event else {
                if self.now_ns < deadline_ns {
                    self.log.push(Event::ClockAdvanced {
                        from_ns: self.now_ns,
                        to_ns: deadline_ns,
                    });
                    self.now_ns = deadline_ns;
                }
                self.perf_samples.push(("sim.fire_rules", fire_us));
                self.perf_samples.push(("sim.dispatch_actions", action_us));
                self.perf_samples.push(("sim.fire_timeline", timeline_us));
                self.perf_samples.push(("sim.deliver_packets", deliver_us));
                self.record_phase_us("sim.run_until.total", total_start);
                return firings;
            };

            if t > deadline_ns {
                if self.now_ns < deadline_ns {
                    self.log.push(Event::ClockAdvanced {
                        from_ns: self.now_ns,
                        to_ns: deadline_ns,
                    });
                    self.now_ns = deadline_ns;
                }
                self.perf_samples.push(("sim.fire_rules", fire_us));
                self.perf_samples.push(("sim.dispatch_actions", action_us));
                self.perf_samples.push(("sim.fire_timeline", timeline_us));
                self.perf_samples.push(("sim.deliver_packets", deliver_us));
                self.record_phase_us("sim.run_until.total", total_start);
                return firings;
            }

            if t > self.now_ns {
                self.log.push(Event::ClockAdvanced {
                    from_ns: self.now_ns,
                    to_ns: t,
                });
                self.now_ns = t;
            }

            // 3a. Apply scheduled scenario actions for this instant, in order.
            let phase_start = Instant::now();
            while let Some(head) = self.pending_actions.peek() {
                if head.0.at_ns != self.now_ns { break; }
                let p = self.pending_actions.pop().unwrap().0;
                self.apply_action(p.action);
            }
            action_us += phase_start.elapsed().as_secs_f64() * 1_000_000.0;

            // 3a'. Fire due timeline events (user-editable scenario).
            // Type-mismatched writes are silently no-op'd but still
            // mark the event fired so the queue makes progress —
            // see timeline.rs for the rationale.
            let phase_start = Instant::now();
            self.fire_due_timeline_events();
            timeline_us += phase_start.elapsed().as_secs_f64() * 1_000_000.0;

            // 3b. Deliver all packets scheduled for this instant.
            let phase_start = Instant::now();
            while let Some(head) = self.in_flight.peek() {
                if head.0.arrives_at_ns != self.now_ns { break; }
                let sched = self.in_flight.pop().unwrap().0;
                let mut pkt = sched.packet;
                pkt.from_edge = Some(sched.edge);
                if !self.nodes.contains_key(&sched.deliver_to) {
                    continue;  // despawned while in flight
                }
                // Resolve compound → inner leaf via in_ports.
                let port = self.edges.get(&sched.edge).and_then(|e| e.to_port.clone());
                let leaf = self.resolve_in_port_chain(sched.deliver_to, port);
                let Some(leaf) = leaf else { continue; };
                self.log.push(Event::PacketDelivered {
                    packet: pkt.id,
                    to: leaf,
                    at_ns: self.now_ns,
                });
                if let Some(node) = self.nodes.get_mut(&leaf) {
                    node.inbox.push_back(pkt);
                    self.fireable.insert(leaf);
                }
            }
            deliver_us += phase_start.elapsed().as_secs_f64() * 1_000_000.0;
        }
    }

    /// Attempt to fire exactly one rule. Returns true if one fired.
    ///
    /// Deterministic firing order: iterate `self.fireable` by `NodeId`
    /// (BTreeSet is sorted), rules by definition order, first fireable
    /// wins. The worklist is maintained incrementally — packet
    /// delivery / inject mark a node fireable; consuming the last
    /// packet from a non-source-rule node refreshes it back out.
    fn try_fire_one(&mut self) -> bool {
        // Snapshot the fireable set into a scratch buffer. The inner
        // loop calls `&mut self` (which mutates `fireable` itself when
        // the rule consumes a packet), so we can't iterate the set
        // directly. Vec extension is O(n) but `fireable.len()` is
        // typically far smaller than `nodes.len()` on dense graphs.
        self.fire_iter_buf.clear();
        self.fire_iter_buf.extend(self.fireable.iter().copied());
        let len = self.fire_iter_buf.len();
        for i in 0..len {
            let nid = self.fire_iter_buf[i];
            // Re-look-up the class each iteration: the previous fire
            // may have despawned this node. The id is `Copy`, no
            // alloc per iteration.
            let tid = match self.nodes.get(&nid).and_then(|n| n.class) {
                Some(t) => t,
                None => continue,
            };
            let rule_count = match self.templates.get(tid.0 as usize) {
                Some(t) => t.rules.len(),
                None => continue,
            };
            for ri in 0..rule_count {
                if self.try_fire(nid, tid, ri) {
                    return true;
                }
            }
        }
        false
    }

    fn try_fire(&mut self, nid: NodeId, tid: crate::sim::TemplateId, rule_idx: usize) -> bool {
        // Pull the rule vec out, hand a `&Rule` to the matcher/firer,
        // put it back. `mem::take` is two pointer swaps — no deep
        // clone of the `Expr`/`Effect` trees, no Arc, no allocation.
        let idx = tid.0 as usize;
        let rules = match self.templates.get_mut(idx) {
            Some(t) => std::mem::take(&mut t.rules),
            None => return false,
        };
        let fired = if let Some(rule) = rules.get(rule_idx) {
            if let Some(fire) = self.match_rule(nid, rule) {
                self.execute_fire(nid, rule, fire);
                true
            } else {
                false
            }
        } else {
            false
        };
        if let Some(t) = self.templates.get_mut(idx) {
            t.rules = rules;
        }
        fired
    }


    fn match_rule(&mut self, nid: NodeId, rule: &Rule) -> Option<MatchResult> {
        // Index of the (at most one) Input pattern. Stored as a position
        // rather than a cloned `When` so we just borrow `&rule.when[idx]`
        // when we need its contents — avoids cloning the entire pattern
        // tree on every fire attempt.
        let input_when_idx = rule
            .when
            .iter()
            .position(|w| matches!(w, When::Input { .. }));

        let inbox_len = self.nodes.get(&nid)?.inbox.len();

        // Iteration count: one attempt per inbox packet for input rules,
        // a single attempt for source rules.
        let max_pkt_idx = if input_when_idx.is_some() {
            inbox_len
        } else {
            1
        };

        for try_idx in 0..max_pkt_idx {
            let pkt_idx: Option<usize> = input_when_idx.map(|_| try_idx);

            // Re-take field-disjoint borrows each iteration so rng can be
            // reborrowed for guard eval at the bottom. `slots`/`inbox`
            // here are borrows into `self.nodes` — no clone.
            let nodes = &self.nodes;
            let edges = &self.edges;
            let params = &self.params;
            let now_ns = self.now_ns;
            let node = nodes.get(&nid)?;
            let slots = &node.slots;

            let mut bindings = Bindings::new();

            // Match the Input pattern if present.
            if let (Some(w_idx), Some(idx)) = (input_when_idx, pkt_idx) {
                let When::Input { pattern, from } = &rule.when[w_idx] else {
                    unreachable!()
                };
                let pkt = &node.inbox[idx];
                if let Some(from_name) = from {
                    let ok = pkt
                        .from_edge
                        .and_then(|eid| edges.get(&eid))
                        .and_then(|e| nodes.get(&e.from))
                        .map(|n| n.name == *from_name)
                        .unwrap_or(false);
                    if !ok {
                        continue;
                    }
                }
                if !match_pattern(&pkt.payload, pattern, &mut bindings) {
                    continue;
                }
            }

            // Match all SlotMatch patterns.
            let mut slot_ok = true;
            for w in &rule.when {
                if let When::SlotMatch { slot, pattern } = w {
                    let Some(sv) = slots.get(slot) else {
                        slot_ok = false;
                        break;
                    };
                    if !match_pattern(sv, pattern, &mut bindings) {
                        slot_ok = false;
                        break;
                    }
                }
            }
            if !slot_ok {
                continue;
            }

            // Evaluate guard, if present. IMPORTANT: guard evaluation
            // can consume RNG (e.g. Bernoulli). To preserve determinism
            // and avoid "phantom" RNG draws from rules that fail to
            // match, we only evaluate the guard on rules where
            // when-patterns already matched (this branch).
            if let Some(ref g) = rule.guard {
                let mut pstack: Vec<String> = Vec::new();
                let pkt_ref: Option<&Packet> = pkt_idx.map(|i| &node.inbox[i]);
                // Field-disjoint borrow of the RNG; coexists with the
                // `&self.nodes`/`&self.edges`/`&self.params` borrows
                // above because they're separate struct fields.
                let rng = &mut self.rng;
                let mut ctx = EvalCtx {
                    bindings: &bindings,
                    slots,
                    now_ns,
                    rng,
                    current_node: Some(nid),
                    params,
                    param_stack: &mut pstack,
                    nodes,
                    edges,
                    packet: pkt_ref,
                };
                let v = g.eval(&mut ctx);
                if !v.as_bool().expect("guard must evaluate to Bool") {
                    continue;
                }
            }

            return Some(MatchResult {
                bindings,
                consumed_pkt_idx: pkt_idx,
            });
        }

        None
    }

    fn execute_fire(&mut self, nid: NodeId, rule: &Rule, fire: MatchResult) {
        let MatchResult { bindings, consumed_pkt_idx } = fire;
        let now = self.now_ns;

        self.log.push(Event::RuleFired {
            node: nid,
            rule: rule.name.clone(),
            at_ns: now,
        });

        let consumed_pkt: Option<Packet> = if let Some(idx) = consumed_pkt_idx {
            let node = self.nodes.get_mut(&nid).unwrap();
            let pkt = node.inbox.remove(idx).expect("consume: index valid");
            self.log.push(Event::PacketConsumed {
                packet: pkt.id,
                by: nid,
                rule: rule.name.clone(),
                at_ns: now,
            });
            // Worklist maintenance: if this consume drained the inbox
            // and the class has no source-style rule, the node has no
            // pending work and can drop out of `fireable`. Effects
            // that follow may re-mark it (e.g. self-emit goes through
            // `inject` and re-inserts).
            self.refresh_fireable(nid);
            Some(pkt)
        } else {
            None
        };

        let mut bindings = bindings;
        for eff in &rule.effects {
            self.execute_effect(nid, &rule.name, eff, &mut bindings, consumed_pkt.as_ref());
        }
    }

    fn execute_effect(
        &mut self,
        nid: NodeId,
        rule_name: &str,
        eff: &Effect,
        bindings: &mut Bindings,
        consumed_pkt: Option<&Packet>,
    ) {
        let now = self.now_ns;

        match eff {
            Effect::SetSlot { slot, value } => {
                let v = self.eval_at_node(nid, bindings, value, consumed_pkt);
                self.write_slot(nid, slot, v, now);
            }
            Effect::SamplesPush { slot, value } => {
                let v = self.eval_at_node(nid, bindings, value, consumed_pkt);
                let node = self.nodes.get_mut(&nid).unwrap();
                match node.slots.get_mut(slot) {
                    Some(Value::Samples(s)) => s.push(v),
                    Some(other) => {
                        let detail = format!("slot `{}` is not Samples: {:?}", slot, other);
                        self.record_error("slot_type_mismatch", Some(nid), Some(rule_name), detail);
                    }
                    None => {
                        let detail = format!("SamplesPush: slot `{}` missing", slot);
                        self.record_error("slot_missing", Some(nid), Some(rule_name), detail);
                    }
                }
            }
            Effect::SamplesPopOldestInto { slot, into_var } => {
                let node = self.nodes.get_mut(&nid).unwrap();
                let v = match node.slots.get_mut(slot) {
                    Some(Value::Samples(s)) => match s.pop_oldest() {
                        Some(v) => Some(v),
                        None => {
                            let detail = format!("pop from empty Samples slot `{}`", slot);
                            self.record_error("samples_empty_pop", Some(nid), Some(rule_name), detail);
                            None
                        }
                    },
                    Some(other) => {
                        let detail = format!("slot `{}` is not Samples: {:?}", slot, other);
                        self.record_error("slot_type_mismatch", Some(nid), Some(rule_name), detail);
                        None
                    }
                    None => {
                        let detail = format!("SamplesPopOldestInto: slot `{}` missing", slot);
                        self.record_error("slot_missing", Some(nid), Some(rule_name), detail);
                        None
                    }
                };
                if let Some(v) = v {
                    bindings.insert(into_var.clone(), v);
                }
            }
            Effect::SamplesDropOldest { slot, n } => {
                let n_v = self.eval_at_node(nid, bindings, n, consumed_pkt);
                let n_val = match n_v.as_int() {
                    Some(n) => n,
                    None => {
                        self.record_error(
                            "expr_type_mismatch",
                            Some(nid),
                            Some(rule_name),
                            format!("SamplesDropOldest: n must be Int, got {:?}", n_v),
                        );
                        return;
                    }
                };
                let node = self.nodes.get_mut(&nid).unwrap();
                match node.slots.get_mut(slot) {
                    Some(Value::Samples(s)) => {
                        for _ in 0..n_val.max(0) {
                            if s.pop_oldest().is_none() { break; }
                        }
                    }
                    Some(other) => {
                        let detail = format!("slot `{}` is not Samples: {:?}", slot, other);
                        self.record_error("slot_type_mismatch", Some(nid), Some(rule_name), detail);
                    }
                    None => {
                        let detail = format!("SamplesDropOldest: slot `{}` missing", slot);
                        self.record_error("slot_missing", Some(nid), Some(rule_name), detail);
                    }
                }
            }
            Effect::Emit { payload, to, meta_ops, return_path_op } => {
                let v = self.eval_at_node(nid, bindings, payload, consumed_pkt);
                self.schedule_emit(
                    nid, rule_name, to.clone(), v,
                    consumed_pkt, meta_ops, return_path_op,
                    bindings,
                );
            }
            Effect::EmitToEach { payload, targets, meta_ops, return_path_op } => {
                let targets_v = self.eval_at_node(nid, bindings, targets, consumed_pkt);
                let items = match targets_v {
                    Value::List(v) => v,
                    other => {
                        self.record_error(
                            "emit_to_each_bad_targets",
                            Some(nid),
                            Some(rule_name),
                            format!("targets must be a List of NodeRef, got {:?}", other),
                        );
                        return;
                    }
                };
                let payload_v = self.eval_at_node(nid, bindings, payload, consumed_pkt);
                for t in items {
                    let nref = match t {
                        Value::NodeRef(id) => id,
                        other => {
                            self.record_error(
                                "emit_to_each_bad_targets",
                                Some(nid),
                                Some(rule_name),
                                format!("target must be NodeRef, got {:?}", other),
                            );
                            continue;
                        }
                    };
                    // Routes through the same engine path as
                    // ToTargetExpr — silent-drops when no outbound
                    // edge exists from `nid` to `nref`.
                    self.schedule_emit(
                        nid,
                        rule_name,
                        crate::rule::EmitTo::ToTargetExpr(
                            crate::expr::Expr::Lit(Value::NodeRef(nref))
                        ),
                        payload_v.clone(),
                        consumed_pkt,
                        meta_ops,
                        return_path_op,
                        bindings,
                    );
                }
            }
            Effect::RecordMetric { name, value } => {
                let v = self.eval_at_node(nid, bindings, value, consumed_pkt);
                self.log.push(Event::MetricRecorded {
                    node: nid,
                    name: name.clone(),
                    value: v,
                    at_ns: now,
                });
            }
            Effect::RecordError { kind, detail } => {
                let v = self.eval_at_node(nid, bindings, detail, consumed_pkt);
                let detail_s = match v {
                    Value::Str(s) => s,
                    // Non-string details stringify via Debug — rule
                    // authors usually pass a literal string, but any
                    // Value is accepted so you can surface a slot
                    // value without needing a sprintf-like facility.
                    other => format!("{:?}", other),
                };
                self.record_error(kind, Some(nid), Some(rule_name), detail_s);
            }
            Effect::Spawn { template, into_var } => {
                match self.try_spawn_from_template(template, Some(nid)) {
                    Ok(new_id) => {
                        if let Some(var) = into_var {
                            bindings.insert(var.clone(), Value::NodeRef(new_id));
                        }
                    }
                    Err(detail) => {
                        self.record_error("spawn_failed", Some(nid), Some(rule_name), detail);
                    }
                }
            }
            Effect::Despawn { node } => {
                let v = self.eval_at_node(nid, bindings, node, consumed_pkt);
                match v {
                    Value::NodeRef(id) => { self.despawn_node(id); }
                    other => {
                        self.record_error(
                            "expr_type_mismatch",
                            Some(nid),
                            Some(rule_name),
                            format!("Despawn: expected NodeRef, got {:?}", other),
                        );
                    }
                }
            }
        }
    }

    /// Walk up from `node` through its `parent` chain to find the
    /// nearest compound whose `out_ports[port_name]` equals the
    /// starting descendant. This is how an inner leaf's
    /// `ToOutPort("name")` is resolved to a specific compound
    /// boundary.
    fn find_enclosing_compound_for_out_port(&self, start: NodeId, port_name: &str) -> Option<NodeId> {
        let mut cur = self.nodes.get(&start)?.parent;
        let seeker = start;
        while let Some(pid) = cur {
            if let Some(p_node) = self.nodes.get(&pid) {
                if let Some(body) = &p_node.compound {
                    if body.out_ports.get(port_name) == Some(&seeker) {
                        return Some(pid);
                    }
                    // Port exists but isn't mapped to us? Keep walking
                    // upward — outer compounds may also have a same-named
                    // port and we might be their port via transitive
                    // inner-node identity.
                }
                cur = p_node.parent;
            } else {
                break;
            }
        }
        None
    }

    /// Walk a compound → inner leaf chain using the given port name at
    /// each level (same name re-used for nested compounds). Returns
    /// `None` if a port is missing or a compound has no inner node
    /// mapped for this port.
    fn resolve_in_port_chain(&self, start: NodeId, port: Option<String>) -> Option<NodeId> {
        let mut cur = start;
        loop {
            let node = self.nodes.get(&cur)?;
            match &node.compound {
                None => return Some(cur),
                Some(body) => {
                    let port_name = port.as_ref()?;
                    let inner = body.in_ports.get(port_name).copied()?;
                    cur = inner;
                }
            }
        }
    }

    /// Apply every timeline event whose `at_ns <= now_ns` and that
    /// hasn't fired yet. Each event is a moment with a list of slot
    /// writes that fire atomically. Type-mismatched writes inside an
    /// event are silently skipped (per action); the event still
    /// completes and is marked `fired` so the queue makes progress.
    /// Events stay in the vec post-fire for UI history.
    pub(crate) fn fire_due_timeline_events(&mut self) {
        let now = self.now_ns;
        for i in 0..self.timeline.events.len() {
            if self.timeline.events[i].fired { continue; }
            if self.timeline.events[i].at_ns > now { break; }
            // Boundary marker — UIs use this to drop stale in-flight
            // visuals, so the canvas reflects the new sim state
            // rather than replaying pre-change traffic.
            let event_id = self.timeline.events[i].id;
            self.log.push(Event::TimelineEventFired { event_id, at_ns: now });
            // Clone the actions list out so we can borrow `self.nodes`
            // mutably without aliasing the event itself.
            let actions = self.timeline.events[i].actions.clone();
            for act in actions {
                if let Some(n) = self.nodes.get_mut(&act.node) {
                    let ok = match (n.slots.get(&act.slot), &act.value) {
                        (Some(Value::Float(_)), Value::Float(_)) => true,
                        (Some(Value::Int(_)),   Value::Int(_))   => true,
                        (Some(Value::Bool(_)),  Value::Bool(_))  => true,
                        (Some(Value::Str(_)),   Value::Str(_))   => true,
                        (Some(Value::Nil),      _)               => true,
                        _ => false,
                    };
                    if ok {
                        n.slots.insert(act.slot.clone(), act.value.clone());
                        self.log.push(Event::SlotWritten {
                            node: act.node, slot: act.slot, value: act.value,
                            at_ns: now,
                        });
                    }
                }
            }
            self.timeline.events[i].fired = true;
        }
    }

    /// Apply one scripted scenario action at the current clock.
    fn apply_action(&mut self, action: Action) {
        match action {
            Action::Inject { node, payload, metadata, return_path } => {
                self.inject_with(node, payload, metadata, return_path);
            }
            Action::SetSlot { node, slot, value } => {
                let now = self.now_ns;
                if let Some(n) = self.nodes.get_mut(&node) {
                    n.slots.insert(slot.clone(), value.clone());
                    self.log.push(Event::SlotWritten {
                        node, slot, value, at_ns: now,
                    });
                }
            }
            Action::SetEdgeLatency { edge, latency } => {
                if let Some(e) = self.edges.get_mut(&edge) {
                    e.latency_ns = latency;
                }
            }
            Action::KillNode { node } => {
                self.despawn_node(node);
            }
            Action::SetParam { name, value } => {
                self.params.insert(name, value);
            }
        }
    }

    /// Instantiate a template. Returns the new node's id.
    ///
    /// Template edges with `EdgeEnd::Parent` require a parent to be
    /// supplied. Panics otherwise — use `try_spawn_from_template` if
    /// you want to recover.
    pub fn spawn_from_template(&mut self, template_name: &str, parent: Option<NodeId>) -> NodeId {
        self.try_spawn_from_template(template_name, parent)
            .unwrap_or_else(|e| panic!("spawn_from_template: {}", e))
    }

    /// Non-panicking variant: returns `Err(detail)` if the template is
    /// missing or its edges require a parent that wasn't supplied.
    pub fn try_spawn_from_template(
        &mut self,
        template_name: &str,
        parent: Option<NodeId>,
    ) -> Result<NodeId, String> {
        let tid = self
            .template_by_name
            .get(template_name)
            .copied()
            .ok_or_else(|| format!("no template named `{}`", template_name))?;
        let t = self.templates[tid.0 as usize].clone();

        self.next_instance_seq += 1;
        let seq = self.next_instance_seq;
        let instance_name = format!("{}_{}", t.node_name_prefix, seq);

        let new_id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = Node {
            id: new_id,
            name: instance_name.clone(),
            slots: t.slots.clone(),
            outbound: Vec::new(),
            inbox: VecDeque::new(),
            probes: t.probes.clone(),
            class: Some(tid),
            parent,
            compound: None,
        };
        self.nodes.insert(new_id, node);
        // Source-rule classes need to be in the fireable set even with
        // an empty inbox; non-source classes will get added when their
        // first packet arrives.
        self.refresh_fireable(new_id);
        self.log.push(Event::NodeSpawned {
            node: new_id,
            template: t.name.clone(),
            parent,
            at_ns: self.now_ns,
        });

        // Materialize template edges.
        for spec in &t.edges {
            let from = match spec.from {
                EdgeEnd::ThisInstance => new_id,
                EdgeEnd::Parent => match parent {
                    Some(p) => p,
                    None => return Err(format!(
                        "template `{}` has an edge with EdgeEnd::Parent but no parent was supplied",
                        template_name,
                    )),
                },
            };
            let to = match spec.to {
                EdgeEnd::ThisInstance => new_id,
                EdgeEnd::Parent => match parent {
                    Some(p) => p,
                    None => return Err(format!(
                        "template `{}` has an edge with EdgeEnd::Parent but no parent was supplied",
                        template_name,
                    )),
                },
            };
            self.add_edge(from, to, spec.latency.clone());
        }

        // Deliver bootstrap packets. These seed any self-driven loops
        // the class relies on to stay alive (e.g. a `tick(nil)` packet
        // a `rule tick` rearms each step). Done after edge creation so
        // the first packet an initial rule emits routes over its own
        // fresh self-edge.
        for payload in t.initial_packets.iter().cloned() {
            self.inject(new_id, payload);
        }

        Ok(new_id)
    }

    /// Instantiate a class with an explicit instance name. Distinct
    /// from [`try_spawn_from_template`] which auto-generates a name —
    /// the class/instance API this feeds wants user-supplied names so
    /// canvas files can address specific instances.
    pub fn instantiate(
        &mut self,
        class_name: &str,
        instance_name: &str,
    ) -> Result<NodeId, String> {
        let tid = self
            .template_by_name
            .get(class_name)
            .copied()
            .ok_or_else(|| format!("no class named `{}`", class_name))?;
        let t = self.templates[tid.0 as usize].clone();
        let new_id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let node = Node {
            id: new_id,
            name: instance_name.to_string(),
            slots: t.slots.clone(),
            outbound: Vec::new(),
            inbox: VecDeque::new(),
            probes: t.probes.clone(),
            class: Some(tid),
            parent: None,
            compound: None,
        };
        self.nodes.insert(new_id, node);
        self.refresh_fireable(new_id);
        self.log.push(Event::NodeSpawned {
            node: new_id,
            template: t.name.clone(),
            parent: None,
            at_ns: self.now_ns,
        });
        for spec in &t.edges {
            // Only ThisInstance endpoints are meaningful here — Parent
            // has no referent in the named-instance path.
            let resolve = |end: EdgeEnd| -> Result<NodeId, String> {
                match end {
                    EdgeEnd::ThisInstance => Ok(new_id),
                    EdgeEnd::Parent => Err(format!(
                        "class `{}` has an edge with EdgeEnd::Parent; \
                         instantiate() does not supply a parent",
                        class_name
                    )),
                }
            };
            let from = resolve(spec.from)?;
            let to = resolve(spec.to)?;
            self.add_edge(from, to, spec.latency.clone());
        }
        for payload in t.initial_packets.iter().cloned() {
            self.inject(new_id, payload);
        }
        Ok(new_id)
    }

    /// Build the `(metadata, return_path)` pair that an emitted packet
    /// should carry, starting from the consumed packet's values (or
    /// empty for source emits) and applying the rule's modifications.
    ///
    /// Returns `None` and records an error if a modification is
    /// malformed (e.g. popping an empty path, pushing a non-NodeRef):
    /// the caller should silently drop the emit.
    fn build_meta_and_path(
        &mut self,
        from_nid: NodeId,
        rule_name: &str,
        consumed_pkt: Option<&Packet>,
        meta_ops: &[MetaOp],
        rp_op: &ReturnPathOp,
        bindings: &Bindings,
    ) -> Option<(BTreeMap<String, Value>, Vec<NodeId>)> {
        // Inherit from consumed packet. Empty map/vec for source emits
        // are zero-heap — Vec::new() / BTreeMap::new() don't allocate.
        let mut metadata = consumed_pkt.map(|p| p.metadata.clone()).unwrap_or_default();
        let inherited_path: Vec<NodeId> = consumed_pkt.map(|p| p.return_path.clone()).unwrap_or_default();

        for op in meta_ops {
            match op {
                MetaOp::Set { key, value } => {
                    let v = self.eval_at_node(from_nid, bindings, value, consumed_pkt);
                    metadata.insert(key.clone(), v);
                }
                MetaOp::Remove { key } => { metadata.remove(key); }
            }
        }

        let return_path = match rp_op {
            ReturnPathOp::Inherit => inherited_path,
            ReturnPathOp::Push(e) => {
                let v = self.eval_at_node(from_nid, bindings, e, consumed_pkt);
                match v {
                    Value::NodeRef(id) => {
                        let mut p = Vec::with_capacity(inherited_path.len() + 1);
                        p.push(id);
                        p.extend(inherited_path);
                        p
                    }
                    other => {
                        self.record_error(
                            "return_path_push_bad_type",
                            Some(from_nid),
                            Some(rule_name),
                            format!("pushing value must be NodeRef, got {:?}", other),
                        );
                        return None;
                    }
                }
            }
            ReturnPathOp::Pop => {
                if inherited_path.is_empty() {
                    self.record_error(
                        "return_path_empty_pop",
                        Some(from_nid),
                        Some(rule_name),
                        "popping from empty return_path",
                    );
                    return None;
                }
                inherited_path[1..].to_vec()
            }
            ReturnPathOp::Replace(e) => {
                let v = self.eval_at_node(from_nid, bindings, e, consumed_pkt);
                match v {
                    Value::List(items) => {
                        let mut out = Vec::with_capacity(items.len());
                        for item in items {
                            match item {
                                Value::NodeRef(id) => out.push(id),
                                other => {
                                    self.record_error(
                                        "return_path_replace_bad_type",
                                        Some(from_nid),
                                        Some(rule_name),
                                        format!("return_path entry must be NodeRef, got {:?}", other),
                                    );
                                    return None;
                                }
                            }
                        }
                        out
                    }
                    other => {
                        self.record_error(
                            "return_path_replace_bad_type",
                            Some(from_nid),
                            Some(rule_name),
                            format!("return_path replacement must be List of NodeRef, got {:?}", other),
                        );
                        return None;
                    }
                }
            }
        };

        Some((metadata, return_path))
    }

    fn schedule_emit(
        &mut self,
        from_nid: NodeId,
        rule_name: &str,
        to: EmitTo,
        payload: Value,
        consumed_pkt: Option<&Packet>,
        meta_ops: &[MetaOp],
        rp_op: &ReturnPathOp,
        bindings: &Bindings,
    ) {
        // Build metadata + return_path first. If either is malformed,
        // record_error has already fired; drop the emit.
        let Some((metadata, return_path)) = self.build_meta_and_path(
            from_nid, rule_name, consumed_pkt, meta_ops, rp_op, bindings,
        ) else {
            return;
        };

        // Resolve outbound edge. Yields (edge_id, deliver_to): normally
        // deliver_to == edge.to, but reverse-routing (see ToTargetExpr
        // below) lets a reply travel along an existing INBOUND edge in
        // the opposite direction, in which case deliver_to = edge.from.
        //
        // Each branch borrows the `outbound` slice locally and avoids
        // holding it across `&mut self` calls.
        let (target_edge_id, deliver_to) = match to {
            EmitTo::DefaultOut => {
                // Prefer the first NON-self outbound. A node's self-loop is
                // usually a timing mechanism, not its intended downstream.
                // Fall back to first outbound if only self-loops exist. If
                // the node has NO outbound at all, silently drop (matches
                // "unwired output" semantics; common during interactive
                // authoring).
                let outs = self.outbound(from_nid);
                let pick = outs
                    .iter()
                    .find(|&&eid| self.edges[&eid].to != from_nid)
                    .or_else(|| outs.first())
                    .copied();
                match pick {
                    Some(eid) => (eid, self.edges[&eid].to),
                    None => return,
                }
            }
            EmitTo::ToTarget(name) => {
                let outs = self.outbound(from_nid);
                match outs.iter().copied().find(|eid| {
                    let to_id = self.edges[eid].to;
                    self.nodes.get(&to_id).map(|n| n.name == name).unwrap_or(false)
                }) {
                    Some(eid) => (eid, self.edges[&eid].to),
                    None => {
                        self.record_error(
                            "emit_no_edge",
                            Some(from_nid),
                            Some(rule_name),
                            format!("Emit::ToTarget(`{}`): no outbound edge with that target", name),
                        );
                        return;
                    }
                }
            }
            EmitTo::ToTargetExpr(e) => {
                // Eval first (mutates rng), then borrow outbound. The
                // borrow can't live across the eval call.
                let v = self.eval_at_node(from_nid, bindings, &e, consumed_pkt);
                let target_id = match v {
                    Value::NodeRef(id) => id,
                    Value::Str(name) => match self.node_by_name(&name) {
                        Some(id) => id,
                        None => {
                            self.record_error(
                                "emit_unknown_target",
                                Some(from_nid),
                                Some(rule_name),
                                format!("EmitTo::ToTargetExpr: no node named `{}`", name),
                            );
                            return;
                        }
                    },
                    // Nil is the "no match" signal from expressions like
                    // `OutNeighborWithSlot` that may resolve to nothing.
                    // Silently drop — not an error, expected flow.
                    Value::Nil => return,
                    other => {
                        self.record_error(
                            "emit_target_bad_type",
                            Some(from_nid),
                            Some(rule_name),
                            format!("ToTargetExpr must yield Str, NodeRef, or Nil, got {:?}", other),
                        );
                        return;
                    }
                };
                // Look for an outbound edge first.
                let outs = self.outbound(from_nid);
                if let Some(eid) = outs
                    .iter()
                    .copied()
                    .find(|eid| self.edges[eid].to == target_id)
                {
                    (eid, self.edges[&eid].to)
                } else if let Some(eid) = self
                    .outbound(target_id)
                    .iter()
                    .copied()
                    .find(|eid| self.edges[eid].to == from_nid)
                {
                    // Reverse-route: NO outbound edge exists to the target,
                    // but an INBOUND edge from the target does. A reply
                    // travels back along the request edge. This is how
                    // return_path-based replies reach clients without
                    // requiring a user-drawn response edge or a hidden
                    // auto-created one — the single wire the user drew
                    // carries both directions.
                    (eid, target_id)
                } else {
                    // Neither direction available — the target genuinely
                    // isn't reachable from this node. Silent-drop matches
                    // "unwired output" contract for interactive authoring.
                    return;
                }
            }
            EmitTo::ToOutPort(port_name) => {
                // Find the nearest ancestor compound that has this out-port
                // mapped to `from_nid`. Fan onto ALL outgoing edges whose
                // from=that compound and from_port=port_name. (Multiple
                // edges = broadcast.) Return early without setting a
                // single target edge — handled below.
                let Some(compound_nid) = self.find_enclosing_compound_for_out_port(from_nid, &port_name) else {
                    self.record_error(
                        "emit_bad_port",
                        Some(from_nid),
                        Some(rule_name),
                        format!("ToOutPort(`{}`): no enclosing compound with that out-port mapping", port_name),
                    );
                    return;
                };
                let edge_ids: Vec<EdgeId> = self
                    .outbound(compound_nid)
                    .iter()
                    .copied()
                    .filter(|eid| {
                        self.edges[eid].from_port.as_deref() == Some(port_name.as_str())
                    })
                    .collect();
                if edge_ids.is_empty() {
                    // Port is mapped but nothing is wired to it — silently drop
                    // (same contract as unwired DefaultOut).
                    return;
                }
                for eid in edge_ids {
                    let edge = self.edges[&eid].clone();
                    let latency = self.eval_latency_expr(&edge.latency_ns, compound_nid, bindings, &payload);
                    let arrives_at = self.now_ns.saturating_add(latency);
                    let pid = self.next_packet_id();
                    self.log.push(Event::PacketEmitted {
                        packet: pid,
                        from: compound_nid,
                        to: edge.to,
                        at_ns: self.now_ns,
                        arrives_at_ns: arrives_at,
                        payload: payload.clone(),
                    });
                    self.in_flight.push(Reverse(Scheduled {
                        arrives_at_ns: arrives_at,
                        packet: Packet {
                            id: pid,
                            payload: payload.clone(),
                            from_edge: None,
                            metadata: metadata.clone(),
                            return_path: return_path.clone(),
                            emitted_at_ns: self.now_ns,
                        },
                        edge: eid,
                        deliver_to: edge.to,
                    }));
                    // ToOutPort is always forward: deliver_to == edge.to by
                    // construction, so every edge is a forward traversal.
                    let seq = self.next_emit_seq;
                    self.next_emit_seq += 1;
                    if let Some(e) = self.edges.get_mut(&eid) {
                        e.last_sent_seq = Some(seq);
                    }
                }
                return;
            }
        };
        let edge = self.edges[&target_edge_id].clone();
        let latency = self.eval_latency_expr(&edge.latency_ns, from_nid, &Bindings::new(), &payload);
        let arrives_at = self.now_ns.saturating_add(latency);
        let pid = self.next_packet_id();
        self.log.push(Event::PacketEmitted {
            packet: pid,
            from: from_nid,
            to: deliver_to,
            at_ns: self.now_ns,
            arrives_at_ns: arrives_at,
            payload: payload.clone(),
        });
        self.in_flight.push(Reverse(Scheduled {
            arrives_at_ns: arrives_at,
            packet: Packet {
                id: pid,
                payload,
                from_edge: None,
                metadata,
                return_path,
                emitted_at_ns: self.now_ns,
            },
            edge: target_edge_id,
            deliver_to,
        }));
        // Stamp forward-direction traversals only. Reverse-route replies
        // (deliver_to == edge.from) travel the same edge backwards and
        // shouldn't count as "this edge was used for sending" — routing
        // decisions care about forward load, not returning traffic.
        if deliver_to == edge.to {
            let seq = self.next_emit_seq;
            self.next_emit_seq += 1;
            if let Some(e) = self.edges.get_mut(&target_edge_id) {
                e.last_sent_seq = Some(seq);
            }
        }
    }

    fn eval_at_node(
        &mut self,
        nid: NodeId,
        bindings: &Bindings,
        e: &Expr,
        packet: Option<&Packet>,
    ) -> Value {
        // Field-disjoint borrows: nodes/edges/params via shared refs,
        // rng via mut. `slots` is a borrow into `self.nodes` rather
        // than a fresh clone — the BTreeMap clone here was the bulk
        // of per-firing allocation cost on dense grids.
        let mut pstack: Vec<String> = Vec::new();
        let nodes = &self.nodes;
        let edges = &self.edges;
        let params = &self.params;
        let now_ns = self.now_ns;
        let rng = &mut self.rng;
        let slots = &nodes.get(&nid).unwrap().slots;
        let mut ctx = EvalCtx {
            bindings,
            slots,
            now_ns,
            rng,
            current_node: Some(nid),
            params,
            param_stack: &mut pstack,
            nodes,
            edges,
            packet,
        };
        e.eval(&mut ctx)
    }

    fn eval_latency_expr(
        &mut self,
        latency_ns: &Expr,
        src_nid: NodeId,
        bindings: &Bindings,
        packet_payload: &Value,
    ) -> Time {
        // Edge latency is evaluated with: source node's slots + `packet` = payload.
        // No consumed-packet is available here — latency exprs don't
        // reference meta/return_path. Bindings still get cloned because
        // we mutate them with `packet`, but slots are a borrow.
        let mut b = bindings.clone();
        b.insert("packet".to_string(), packet_payload.clone());
        let mut pstack: Vec<String> = Vec::new();
        let nodes = &self.nodes;
        let edges = &self.edges;
        let params = &self.params;
        let now_ns = self.now_ns;
        let rng = &mut self.rng;
        let slots = &nodes.get(&src_nid).unwrap().slots;
        let mut ctx = EvalCtx {
            bindings: &b,
            slots,
            now_ns,
            rng,
            current_node: Some(src_nid),
            params,
            param_stack: &mut pstack,
            nodes,
            edges,
            packet: None,
        };
        let v = latency_ns.eval(&mut ctx);
        let n = match v.as_int() {
            Some(n) => n,
            None => {
                self.record_error(
                    "edge_latency_bad_type",
                    Some(src_nid),
                    None,
                    format!("edge latency must be Int, got {:?}", v),
                );
                0
            }
        };
        if n < 0 {
            self.record_error(
                "edge_latency_negative",
                Some(src_nid),
                None,
                format!("edge latency was negative: {}", n),
            );
            return 0;
        }
        n as Time
    }

    fn write_slot(&mut self, nid: NodeId, slot: &str, value: Value, at_ns: Time) {
        let node = self.nodes.get_mut(&nid).expect("write_slot: node gone");
        if let Some(existing) = node.slots.get(slot) {
            match (existing, &value) {
                (Value::Samples(_), Value::Samples(_)) => {}
                (Value::Samples(_), other) => panic!(
                    "SetSlot: slot `{}` is Samples but tried to write {:?}", slot, other
                ),
                _ => {}
            }
        }
        node.slots.insert(slot.to_string(), value.clone());
        self.log.push(Event::SlotWritten {
            node: nid,
            slot: slot.to_string(),
            value,
            at_ns,
        });
    }
}

struct MatchResult {
    bindings: Bindings,
    consumed_pkt_idx: Option<usize>,
}
